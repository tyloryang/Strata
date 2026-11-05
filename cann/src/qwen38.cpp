// Native Qwen4Exp correctness path. Large linear projections execute on the NPU;
// normalization, routing, convolution and recurrent/attention state use host FP32.
#include "strata/cann/qwen38.hpp"
#include "strata/cann/gguf.hpp"
#include "strata/cann/ops.hpp"
#include "strata/cann/tensor_load.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <list>
#include <numeric>
#include <stdexcept>
#include <unordered_map>

namespace strata::cann {
namespace {
using Vec = std::vector<float>;
void require(bool ok, const std::string & message) {
    if (!ok) throw std::runtime_error("qwen4exp: " + message);
}
float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
float silu_host(float x) { return x * sigmoid(x); }
std::string shard_two(std::string path) {
    const std::string marker = "00001-of-00002";
    const auto p = path.rfind(marker);
    require(p != std::string::npos, "pass shard 00001-of-00002 of the original GGUF");
    path.replace(p, marker.size(), "00002-of-00002");
    return path;
}
Vec softmax_host(Vec x) {
    const float largest = *std::max_element(x.begin(), x.end());
    float sum = 0;
    for (auto & v : x) { v = std::exp(v - largest); sum += v; }
    for (auto & v : x) v /= sum;
    return x;
}
std::vector<int> top_indices(const Vec & x, int count) {
    require(count > 0 && count <= (int)x.size(), "invalid top-k");
    std::vector<int> ids(x.size());
    std::iota(ids.begin(), ids.end(), 0);
    std::partial_sort(ids.begin(), ids.begin() + count, ids.end(), [&](int a, int b) {
        return x[a] == x[b] ? a < b : x[a] > x[b];
    });
    ids.resize(count);
    return ids;
}
Vec half_round(const Vec & x) {
    std::vector<uint16_t> h(x.size());
    Vec out(x.size());
    f32_to_f16(x.data(), h.data(), x.size());
    f16_to_f32(h.data(), out.data(), x.size());
    return out;
}
}

struct Qwen38Model::Impl {
    Gguf gguf, table;
    Device device;
    Stream stream;
    int64_t cap, n, layers, hc, kh, vh, sd, conv_k, heads, kvh, hd, rot, interval, experts, top_k;
    int64_t idx_dim, idx_heads, idx_topk, vocab_count, position = 0;
    float eps, base;
    size_t budget, used = 0;
    uint64_t matmuls = 0, hits = 0, misses = 0;
    struct Cached { std::unique_ptr<Buffer> data; std::list<std::string>::iterator age; };
    std::unordered_map<std::string, Cached> weights;
    std::list<std::string> ages;
    std::unordered_map<std::string, Vec> vectors;
    struct State {
        std::deque<Vec> conv, ple;
        Vec recurrent;
        std::vector<Vec> keys, values, index;
    };
    std::vector<State> states;
    std::vector<int64_t> tokens;

    int64_t integer(const std::string & key) const { return gguf.metadata_int("qwen4exp." + key); }
    const std::vector<uint64_t> & array(const std::string & key) const {
        return gguf.metadata_uints("qwen4exp." + key);
    }
    const GgufTensor & tensor(const std::string & name) const {
        const auto * t = gguf.find(name);
        require(t != nullptr, "missing tensor " + name);
        return *t;
    }
    const Vec & vector(const std::string & name) {
        auto it = vectors.find(name);
        if (it == vectors.end()) it = vectors.emplace(name, load_tensor(gguf, tensor(name))).first;
        return it->second;
    }
    Impl(const std::string & path, int64_t capacity)
        : gguf(path), table(shard_two(path)),
          device(std::getenv("STRATA_CANN_DEVICE") ? std::stoi(std::getenv("STRATA_CANN_DEVICE")) : 0),
          stream(device), cap(capacity) {
        n = integer("embedding_length"); layers = integer("block_count");
        hc = integer("hyper_connection.count"); kh = integer("ssm.group_count");
        vh = integer("ssm.time_step_rank"); sd = integer("ssm.state_size"); conv_k = integer("ssm.conv_kernel");
        heads = integer("attention.head_count"); kvh = integer("attention.head_count_kv");
        hd = integer("attention.key_length"); rot = integer("rope.dimension_count");
        interval = integer("full_attention_interval"); experts = integer("expert_count"); top_k = integer("expert_used_count");
        idx_dim = integer("attention.indexer.key_length"); idx_heads = integer("attention.indexer.head_count");
        idx_topk = integer("attention.indexer.top_k");
        eps = gguf.metadata_flt("qwen4exp.attention.layer_norm_rms_epsilon");
        base = gguf.metadata_flt("qwen4exp.rope.freq_base");
        require(n == 2560 && layers == 48 && hc == 4 && kh == 16 && vh == 48 && sd == 128 && conv_k == 4 &&
                heads == 24 && kvh == 2 && hd == 256 && rot == 64 && interval == 4 && experts == 512 && top_k == 10 &&
                idx_dim == 128 && idx_heads == 4 && idx_topk > 0 && eps > 0 && base > 1,
                "only the original Flash-Next geometry is supported");
        require(cap > 0 && cap <= integer("context_length"), "invalid context capacity");
        require(table.find("per_layer_token_embd.weight") != nullptr, "shard 2 has no PLE table");
        require(array("ple.layer_multipliers").size() >= 3 && array("ple.head_offsets").size() == 16 &&
                array("ple.head_vocab_sizes").size() == 16 && array("attention.compress_ratios").size() == size_t(layers),
                "invalid PLE/QSA metadata");
        vocab_count = tensor("output.weight").dims.at(1);
        const double gib = std::getenv("STRATA_BUDGET_GB") ? std::stod(std::getenv("STRATA_BUDGET_GB")) : 12;
        require(std::isfinite(gib) && gib >= 3 && gib <= 48, "STRATA_BUDGET_GB must be between 3 and 48");
        budget = size_t(gib * 1073741824.0);
        states.resize(layers);
        fprintf(stderr, "native strata-cann qwen4exp: layers=%lld hidden=%lld device=%d context=%lld FP32 projections=NPU state=CPU\n",
                (long long)layers, (long long)n, device.ordinal(), (long long)cap);
    }
    Vec linear(const std::string & name, const Vec & x, int expert = -1) {
        const auto & t = tensor(name);
        require(t.dims.at(0) == x.size(), "linear input shape for " + name);
        const size_t rows = t.dims.size() > 1 ? t.dims[1] : 1, count = rows * x.size();
        if (expert >= 0) require(t.dims.size() == 3 && (uint64_t)expert < t.dims[2], "invalid expert for " + name);
        else require(t.dims.size() <= 2, "expert selection required for " + name);
        const std::string key = name + "#" + std::to_string(expert);
        auto it = weights.find(key);
        if (it == weights.end()) {
            ++misses;
            require(count * 4 <= budget, "weight exceeds HBM cache budget: " + name);
            while (used + count * 4 > budget && !ages.empty()) {
                const auto victim = ages.back(); ages.pop_back();
                used -= weights.at(victim).data->size(); weights.erase(victim);
            }
            auto host = load_tensor_slice(gguf, t, expert < 0 ? 0 : (uint64_t)expert * count, count);
            auto data = std::make_unique<Buffer>(count * 4);
            data->copy_from_host(host.data(), count * 4);
            ages.push_front(key);
            it = weights.emplace(key, Cached{std::move(data), ages.begin()}).first;
            used += count * 4;
        } else {
            ++hits;
            ages.splice(ages.begin(), ages, it->second.age);
        }
        Buffer bx(x.size()*4), by(rows*4);
        bx.copy_from_host(x.data(), x.size()*4);
        Tensor a{it->second.data->data(), DType::F32, (int64_t)x.size(), (int64_t)rows};
        Tensor b{bx.data(), DType::F32, 1, (int64_t)x.size()}, c{by.data(), DType::F32, 1, (int64_t)rows};
        matmul(stream, a, b, c);
        ++matmuls;
        Vec out(rows); by.copy_to_host(out.data(), rows*4);
        for (auto v : out) require(std::isfinite(v), "non-finite NPU projection " + name);
        return out;
    }
    Vec norm(Vec x, const Vec & gamma, size_t width) const {
        require(width && x.size()%width == 0 && (gamma.size()==width || gamma.size()==x.size()), "norm shape");
        for (size_t row=0; row<x.size(); row+=width) {
            float sum=0; for(size_t j=0;j<width;++j) sum += x[row+j]*x[row+j];
            const float scale=1/std::sqrt(sum/width+eps);
            for(size_t j=0;j<width;++j) x[row+j] *= scale * gamma[(row+j)%gamma.size()];
        }
        return x;
    }
    Vec rope(Vec x, size_t width, int64_t pos) const {
        require(width >= (size_t)rot && x.size()%width==0, "rope shape");
        for(size_t row=0;row<x.size();row+=width) for(int j=0;j<rot/2;++j) {
            const float angle=std::pow(base, -2.0f*j/rot)*pos, c=std::cos(angle), s=std::sin(angle);
            const float a=x[row+j], b=x[row+j+rot/2];
            x[row+j]=a*c-b*s; x[row+j+rot/2]=a*s+b*c;
        }
        return x;
    }
    std::pair<Vec,Vec> hc_mix(const Vec & residual,const std::string & prefix,bool inject=true) {
        const auto normalized=norm(residual,vector(prefix+"norm.weight"),n);
        auto low=linear(prefix+"down.weight",normalized);
        for(auto &v:low)v=silu_host(v/hc);
        const auto gate=linear(prefix+"up.weight",low);
        require(gate.size()==normalized.size(), "HC gate shape");
        Vec mixed(n,0);
        for(int s=0;s<hc;++s) for(int j=0;j<n;++j) mixed[j]+=normalized[s*n+j]*sigmoid(gate[s*n+j])/hc;
        return {std::move(mixed),inject?linear(prefix+"inject.weight",normalized):Vec{}};
    }
    void combine(Vec & residual,const Vec & output,const Vec & inject) const {
        require(output.size()==(size_t)n && inject.size()==(size_t)hc,"HC combine shape");
        for(int s=0;s<hc;++s) {
            const float scale=2*sigmoid(inject[s]/hc);
            for(int j=0;j<n;++j)residual[s*n+j]+=output[j]*scale;
        }
    }
    void ple(Vec & residual,int64_t token,int il) {
        const std::string p="blk."+std::to_string(il)+".";
        const uint64_t eos=integer("ple.eos_token_id"), width=integer("embedding_length_per_layer_input");
        const auto & mult=array("ple.layer_multipliers"), &offset=array("ple.head_offsets"), &vocab=array("ple.head_vocab_sizes");
        uint64_t ctx[3]={(uint64_t)token,eos,eos}; bool cut=false;
        for(int back=1;back<=2;++back) {
            cut=cut || tokens.size()<(size_t)back || (uint64_t)tokens[tokens.size()-back]==eos;
            ctx[back]=cut?eos:(uint64_t)tokens[tokens.size()-back];
        }
        Vec emb;
        for(int length=2;length<=3;++length) {
            uint64_t hash=0;for(int i=0;i<length;++i)hash^=ctx[i]*mult[i];
            for(int head=0;head<8;++head) {
                const int h=(length-2)*8+head; require(vocab[h]>0,"empty PLE hash vocabulary");
                const uint64_t row=hash%vocab[h]+offset[h];
                const auto values=load_tensor_slice(table,*table.find("per_layer_token_embd.weight"),row*width,width);
                emb.insert(emb.end(),values.begin(),values.end());
            }
        }
        const auto key=norm(linear(p+"ple_key.weight",emb),vector(p+"ple_norm_key.weight"),n);
        const auto value=linear(p+"ple_value.weight",emb);
        const auto query=norm(residual,vector(p+"ple_norm_query.weight"),n);
        require(key.size()==residual.size() && value.size()==(size_t)n,"PLE projection shape");
        Vec gated(residual.size());
        for(int s=0;s<hc;++s) {
            float score=0;for(int j=0;j<n;++j)score+=key[s*n+j]*query[s*n+j];
            score/=std::sqrt((float)n);
            const float transformed=score==0?0:std::copysign(std::sqrt(std::max(std::abs(score),1e-6f)),score);
            const float gate=sigmoid(transformed);
            for(int j=0;j<n;++j)gated[s*n+j]=value[j]*gate;
        }
        auto normalized=norm(gated,vector(p+"ple_norm_conv.weight"),n);
        const auto &kernel=vector(p+"ple_conv1d.weight");
        require(kernel.size()==residual.size()*conv_k,"PLE conv shape");
        auto &history=states[il].ple;
        const int dilation=integer("ple.ngram_size");require(dilation>0,"PLE dilation");
        for(size_t c=0;c<residual.size();++c) {
            float value=normalized[c]*kernel[c*conv_k+conv_k-1];
            for(int tap=0;tap<conv_k-1;++tap) {
                const size_t back=(conv_k-1-tap)*dilation;
                if(history.size()>=back)value+=history[history.size()-back][c]*kernel[c*conv_k+tap];
            }
            residual[c]+=gated[c]+silu_host(value);
        }
        history.push_back(std::move(normalized));
        if(history.size()>(size_t)(dilation*(conv_k-1)))history.pop_front();
    }
    Vec gdn(const Vec &x,int il) {
        const std::string p="blk."+std::to_string(il)+".";
        auto raw=linear(p+"attn_qkv.weight",x), z=linear(p+"attn_gate.weight",x);
        const auto &kernel=vector(p+"ssm_conv1d.weight");
        const size_t kd=kh*sd, vd=vh*sd;
        require(raw.size()==2*kd+vd && z.size()==vd && kernel.size()==raw.size()*conv_k,"GDN shape");
        auto &st=states[il]; Vec convolved(raw.size());
        for(size_t c=0;c<raw.size();++c) {
            float value=raw[c]*kernel[c*conv_k+conv_k-1];
            for(int back=1;back<conv_k;++back)if(st.conv.size()>=(size_t)back)
                value+=st.conv[st.conv.size()-back][c]*kernel[c*conv_k+conv_k-1-back];
            convolved[c]=silu_host(value);
        }
        st.conv.push_back(std::move(raw));if(st.conv.size()>=(size_t)conv_k)st.conv.pop_front();
        for(size_t row=0;row<2*kd;row+=sd) {
            float sum=0;for(int j=0;j<sd;++j)sum+=convolved[row+j]*convolved[row+j];
            const float scale=1/std::sqrt(sum+eps);for(int j=0;j<sd;++j)convolved[row+j]*=scale;
        }
        auto beta=linear(p+"ssm_beta.weight",x), alpha=linear(p+"ssm_alpha.weight",x);
        const auto &dt=vector(p+"ssm_dt.bias"), &a=vector(p+"ssm_a");
        require(beta.size()==(size_t)vh && alpha.size()==(size_t)vh && dt.size()==(size_t)vh && a.size()==(size_t)vh,"GDN gates");
        if(st.recurrent.empty())st.recurrent.resize(vh*sd*sd,0);
        Vec out(vd,0),delta(sd);
        for(int h=0;h<vh;++h) {
            const float *q=convolved.data()+(h%kh)*sd,*k=convolved.data()+kd+(h%kh)*sd,*v=convolved.data()+2*kd+h*sd;
            float *state=st.recurrent.data()+h*sd*sd;
            const float av=alpha[h]+dt[h], decay=std::exp((av>20?av:std::log1p(std::exp(av)))*a[h]), b=sigmoid(beta[h]);
            for(int i=0;i<sd*sd;++i)state[i]*=decay;
            for(int j=0;j<sd;++j) {
                float pred=0;for(int i=0;i<sd;++i)pred+=k[i]*state[i*sd+j];
                delta[j]=(v[j]-pred)*b;
            }
            for(int i=0;i<sd;++i)for(int j=0;j<sd;++j)state[i*sd+j]+=k[i]*delta[j];
            for(int j=0;j<sd;++j) {
                float value=0;for(int i=0;i<sd;++i)value+=q[i]*state[i*sd+j];
                out[h*sd+j]=value/std::sqrt((float)sd);
            }
        }
        out=norm(std::move(out),vector(p+"ssm_norm.weight"),sd);
        for(size_t j=0;j<out.size();++j)out[j]*=sigmoid(z[j]);
        return linear(p+"ssm_out.weight",out);
    }
    Vec attention(const Vec &x,int il) {
        const std::string p="blk."+std::to_string(il)+".";
        const auto qg=linear(p+"attn_q.weight",x);
        require(qg.size()==(size_t)(heads*2*hd),"attention q/g shape");
        Vec q(heads*hd),gate(heads*hd);
        for(int h=0;h<heads;++h)for(int j=0;j<hd;++j) {q[h*hd+j]=qg[h*2*hd+j];gate[h*hd+j]=qg[h*2*hd+hd+j];}
        q=rope(norm(std::move(q),vector(p+"attn_q_norm.weight"),hd),hd,position);
        auto k=rope(norm(linear(p+"attn_k.weight",x),vector(p+"attn_k_norm.weight"),hd),hd,position);
        auto v=linear(p+"attn_v.weight",x),idx=linear(p+"indexer.k_proj.weight",x);
        require(k.size()==(size_t)(kvh*hd) && v.size()==k.size() && idx.size()==(size_t)idx_dim,"attention kv shape");
        auto &st=states[il];st.keys.push_back(half_round(k));st.values.push_back(half_round(v));st.index.push_back(half_round(idx));
        const size_t count=st.keys.size();std::vector<int> selected;
        if(count>(size_t)(idx_topk+3)) {
            const size_t ratio=array("attention.compress_ratios").at(il);require(ratio==4,"QSA ratio must be four");
            const size_t full=count/ratio,blocks=full+(count%ratio?1:0);
            Vec pooled(blocks*idx_dim,0);
            for(size_t b=0;b<full;++b)for(size_t r=0;r<ratio;++r)for(int j=0;j<idx_dim;++j)pooled[b*idx_dim+j]+=st.index[b*ratio+r][j]/ratio;
            if(count%ratio)std::copy(st.index[0].begin(),st.index[0].end(),pooled.begin()+full*idx_dim);
            pooled=norm(std::move(pooled),vector(p+"indexer.k_norm.weight"),idx_dim);
            for(size_t b=0;b<blocks;++b) {
                Vec block(pooled.begin()+b*idx_dim,pooled.begin()+(b+1)*idx_dim);
                block=rope(std::move(block),idx_dim,b==full?0:b*ratio);
                std::copy(block.begin(),block.end(),pooled.begin()+b*idx_dim);
            }
            const auto iq=rope(norm(linear(p+"indexer.q_proj.weight",x),vector(p+"indexer.q_norm.weight"),idx_dim),idx_dim,position);
            Vec expanded(count);
            for(size_t b=0;b<blocks;++b) {
                float score=0;for(int h=0;h<idx_heads;++h) {float dot=0;for(int j=0;j<idx_dim;++j)dot+=iq[h*idx_dim+j]*pooled[b*idx_dim+j];score+=std::max(0.f,dot);}
                if(count%ratio && b==full)score+=1e9f;
                for(size_t r=0;r<ratio && b*ratio+r<count;++r)expanded[b*ratio+r]=score;
            }
            selected=top_indices(expanded,std::min<size_t>(count,idx_topk+ratio-1));
        } else {selected.resize(count);std::iota(selected.begin(),selected.end(),0);}
        Vec out(heads*hd,0);
        for(int h=0;h<heads;++h) {
            const int kv=h/(heads/kvh);Vec scores(selected.size());
            for(size_t c=0;c<selected.size();++c) {float dot=0;for(int j=0;j<hd;++j)dot+=q[h*hd+j]*st.keys[selected[c]][kv*hd+j];scores[c]=dot/std::sqrt((float)hd);}
            scores=softmax_host(std::move(scores));
            for(size_t c=0;c<selected.size();++c)for(int j=0;j<hd;++j)out[h*hd+j]+=scores[c]*st.values[selected[c]][kv*hd+j];
            for(int j=0;j<hd;++j)out[h*hd+j]*=sigmoid(gate[h*hd+j]);
        }
        return linear(p+"attn_output.weight",out);
    }
    Vec moe(const Vec &x,int il) {
        const std::string p="blk."+std::to_string(il)+".";
        const auto scores=softmax_host(linear(p+"ffn_gate_inp.weight",x));
        require(scores.size()==(size_t)experts,"MoE router shape");
        const auto ids=top_indices(scores,top_k);float sum=0;for(int id:ids)sum+=scores[id];
        Vec out(n,0);
        auto expert=[&](const std::string &suffix,int id) {
            auto gate=linear(p+"ffn_gate_"+suffix+".weight",x,id),up=linear(p+"ffn_up_"+suffix+".weight",x,id);
            require(gate.size()==up.size(),"MoE intermediate shape");
            for(size_t j=0;j<gate.size();++j)gate[j]=silu_host(gate[j])*up[j];
            return linear(p+"ffn_down_"+suffix+".weight",gate,id);
        };
        for(int id:ids) {const auto y=expert("exps",id);require(y.size()==out.size(),"MoE down shape");for(int j=0;j<n;++j)out[j]+=(scores[id]/sum)*y[j];}
        const auto shared=expert("shexp",-1);const float scale=sigmoid(linear(p+"ffn_gate_inp_shexp.weight",x).at(0));
        for(int j=0;j<n;++j)out[j]+=shared[j]*scale;
        return out;
    }
    Vec step(int64_t token,int64_t pos) {
        require(pos==position && pos<cap && token>=0 && token<vocab_count,"invalid token/position");
        const auto emb=load_tensor_slice(gguf,tensor("token_embd.weight"),token*n,n);
        Vec residual;for(int i=0;i<hc;++i)residual.insert(residual.end(),emb.begin(),emb.end());
        const auto &ple_layers=array("ple.layers");
        for(int il=0;il<layers;++il) {
            if(std::find(ple_layers.begin(),ple_layers.end(),il)!=ple_layers.end())ple(residual,token,il);
            const std::string p="blk."+std::to_string(il)+".";
            auto [x,inject]=hc_mix(residual,p+"hc_attn_");
            combine(residual,(il+1)%interval==0?attention(x,il):gdn(x,il),inject);
            auto [f,scatter]=hc_mix(residual,p+"hc_ffn_");combine(residual,moe(f,il),scatter);
            if(const char *dir=std::getenv("STRATA_CANN_TRACE")) {
                std::filesystem::create_directories(dir);
                const auto name=std::filesystem::path(dir)/("p"+std::to_string(pos)+"-l"+std::to_string(il)+".f32");
                std::ofstream out(name,std::ios::binary);out.write((const char*)residual.data(),residual.size()*4);
                require(bool(out),"cannot write layer trace");
            }
        }
        tokens.push_back(token);++position;
        return hc_mix(residual,"output_hc_",false).first;
    }
};

Qwen38Model::Qwen38Model(const std::string &path,int64_t cap):impl_(std::make_unique<Impl>(path,cap)){}
Qwen38Model::~Qwen38Model()=default;
std::vector<float> Qwen38Model::step(int64_t token,int64_t position){return impl_->step(token,position);}
Model::Top Qwen38Model::top(const std::vector<float> &hidden,int64_t k){
    const auto logits=impl_->linear("output.weight",hidden);
    if(const char *dir=std::getenv("STRATA_CANN_TRACE")) {
        const auto name=std::filesystem::path(dir)/("p"+std::to_string(impl_->position-1)+"-logits.f32");
        std::ofstream out(name,std::ios::binary);out.write((const char*)logits.data(),logits.size()*4);
        require(bool(out),"cannot write logits trace");
    }
    const auto ids=top_indices(logits,k);const double largest=logits[ids[0]];
    double sum=0;for(float value:logits)sum+=std::exp(double(value)-largest);
    Model::Top result;result.log_z=largest+std::log(sum);
    for(int id:ids)result.best.emplace_back(logits[id],id);
    fprintf(stderr,"native stats: NPU matmuls=%llu cache_hits=%llu cache_misses=%llu weight_GiB=%.3f\n",
            (unsigned long long)impl_->matmuls,(unsigned long long)impl_->hits,(unsigned long long)impl_->misses,impl_->used/1073741824.0);
    return result;
}
void Qwen38Model::reset(){impl_->tokens.clear();impl_->states.clear();impl_->states.resize(impl_->layers);impl_->position=0;}
int64_t Qwen38Model::capacity()const{return impl_->cap;}
int64_t Qwen38Model::vocab()const{return impl_->vocab_count;}
int64_t Qwen38Model::cache_bytes()const{return impl_->used;}
}

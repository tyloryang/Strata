#include "strata/cann/npu_tensor.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

using namespace strata::cann;
namespace nt = strata::cann::npu;

namespace {
void close(Stream& stream,const nt::ND& actual,const std::vector<float>& expected,const char* label,float tolerance=1e-5f) {
    const auto values=nt::read_f32(stream,actual);
    if(values.size()!=expected.size())throw std::runtime_error(std::string(label)+": size mismatch");
    float largest=0;
    for(size_t i=0;i<values.size();++i) {
        const float error=std::abs(values[i]-expected[i]);largest=std::max(largest,error);
        if(!std::isfinite(values[i]) || error>tolerance) {
            std::fprintf(stderr,"%s[%zu]: %.9g vs %.9g (tol %.9g)\n",label,i,values[i],expected[i],tolerance);
            throw std::runtime_error(std::string(label)+": numeric mismatch");
        }
    }
    std::printf("%-34s max_abs_error=%.8g PASS\n",label,largest);
}
void check(bool condition,const char* message) {if(!condition)throw std::runtime_error(message);}
}

int main() {
    setvbuf(stdout,nullptr,_IONBF,0);
    try {
        Device device(std::getenv("STRATA_CANN_DEVICE")?std::stoi(std::getenv("STRATA_CANN_DEVICE")):0);
        Stream stream(device);
        const float host[]={1,2,3,4,5,6};
        auto a=nt::from_host(host,{2,3});
        auto transpose=nt::permute(a,{1,0});
        close(stream,nt::contiguous(stream,transpose),{1,4,2,5,3,6},"strided transpose copy");
        close(stream,nt::reshape(stream,transpose,{6}),{1,4,2,5,3,6},"strided reshape");
        auto inner=nt::slice(a,1,1,2);
        close(stream,inner,{2,3,5,6},"offset noncontiguous read");
        auto destination=nt::zeros({2,4});auto middle=nt::slice(destination,1,1,2);
        nt::copy(stream,inner,middle);
        close(stream,destination,{0,2,3,0,0,5,6,0},"copy into strided destination");
        const float overlap_values[]={1,2,3,4};auto overlap=nt::from_host(overlap_values,{4});
        auto source=nt::slice(overlap,0,0,3),target=nt::slice(overlap,0,1,3);
        nt::copy(stream,source,target);
        close(stream,overlap,{1,1,2,3},"overlap copy device temporary");
        close(stream,nt::sum(stream,transpose,1,false),{5,7,9},"strided reduce sum");
        close(stream,nt::mean(stream,a,0,false),{2.5f,3.5f,4.5f},"mean");
        auto scalar=nt::sum(stream,nt::reshape(stream,a,{6}),0,false);
        check(scalar.shape.empty(),"scalar reduction did not retain rank zero");
        close(stream,scalar,{21},"rank zero reduction");
        close(stream,nt::binary(stream,a,nt::mean(stream,a,1,true),nt::Binary::Sub),{-1,0,1,-1,0,1},"broadcast subtraction");

        const float batched[]={1,2,3,4,5,6,7,8,9,10,11,12};
        const float rhs[]={1,0,1,0,1,1};
        auto ba=nt::from_host(batched,{2,2,3});
        auto bb=nt::permute(nt::from_host(rhs,{2,3}),{1,0});
        close(stream,nt::matmul(stream,ba,bb),{4,5,10,11,16,17,22,23},"batched matmul broadcast transpose",1e-4f);
        close(stream,nt::matmul(stream,a,bb),{4,5,10,11},"2D matmul transpose",1e-4f);
        close(stream,nt::cat(stream,{nt::slice(a,1,0,1),nt::slice(a,1,1,2)},1),{1,2,3,4,5,6},"cat strided tensors");
        for(int repeat=0;repeat<8;++repeat) {
            auto joined=nt::cat(stream,{nt::slice(a,0,0,1),nt::slice(a,0,1,1)},0);
            close(stream,nt::sum(stream,joined,1,false),{6,15},"cat descriptor ownership reuse");
        }
        const int64_t rows[]={2,0};auto index=nt::from_host(rows,{2},nt::Type::I64);
        close(stream,nt::index_select(stream,transpose,0,index),{3,6,1,4},"strided index_select");
        const auto [values,indices]=nt::topk(stream,a,2);
        close(stream,values,{3,2,6,5},"topk values");
        check(nt::read_i64(stream,indices)==std::vector<int64_t>({2,1,2,1}),"topk indices mismatch");
        std::puts("topk integer indices               PASS");
        for(auto type:{nt::Type::F16,nt::Type::BF16,nt::Type::I32,nt::Type::I64,nt::Type::U8})
            close(stream,nt::cast(stream,nt::cast(stream,inner,type),nt::Type::F32),{2,3,5,6},"cast strided roundtrip");
        const uint8_t bytes[]={0,1,127,255};auto byte_tensor=nt::from_host(bytes,{4},nt::Type::U8);
        check(nt::read_i64(stream,byte_tensor)==std::vector<int64_t>({0,1,127,255}),"uint8 cast/sign mismatch");
        close(stream,nt::cast(stream,byte_tensor,nt::Type::F32),{0,1,127,255},"uint8 max value conversion");

        const float signed_host[]={-2,0,2};auto signed_x=nt::from_host(signed_host,{3});
        close(stream,nt::unary(stream,signed_x,nt::Unary::Abs),{2,0,2},"abs");
        close(stream,nt::unary(stream,signed_x,nt::Unary::Sign),{-1,0,1},"sign");
        close(stream,nt::unary(stream,signed_x,nt::Unary::Relu),{0,0,2},"relu");
        close(stream,nt::unary(stream,signed_x,nt::Unary::Neg),{2,0,-2},"neg");
        close(stream,nt::unary(stream,nt::unary(stream,signed_x,nt::Unary::Exp),nt::Unary::Log),{-2,0,2},"exp log",1e-5f);
        std::vector<float> sigmoid,softplus,silu,sin,cos;
        for(float v:signed_host) {
            sigmoid.push_back(1/(1+std::exp(-v)));softplus.push_back(std::log1p(std::exp(v)));
            silu.push_back(v/(1+std::exp(-v)));sin.push_back(std::sin(v));cos.push_back(std::cos(v));
        }
        close(stream,nt::unary(stream,signed_x,nt::Unary::Sigmoid),sigmoid,"sigmoid",1e-5f);
        close(stream,nt::unary(stream,signed_x,nt::Unary::Softplus),softplus,"softplus",1e-5f);
        close(stream,nt::unary(stream,signed_x,nt::Unary::Silu),silu,"silu",1e-5f);
        close(stream,nt::unary(stream,signed_x,nt::Unary::Sin),sin,"sin",1e-5f);
        close(stream,nt::unary(stream,signed_x,nt::Unary::Cos),cos,"cos",1e-5f);
        close(stream,nt::pow_scalar(stream,signed_x,2),{4,0,4},"pow scalar");
        close(stream,nt::clamp_min(stream,signed_x,1),{1,1,2},"clamp min");
        close(stream,nt::add_scalar(stream,nt::scale(stream,signed_x,2),3),{-1,3,7},"scalar scale add");
        auto positive=nt::add_scalar(stream,nt::pow_scalar(stream,signed_x,2),1);
        close(stream,nt::unary(stream,positive,nt::Unary::Sqrt),{std::sqrt(5.f),1,std::sqrt(5.f)},"sqrt");
        close(stream,nt::unary(stream,positive,nt::Unary::Rsqrt),{1/std::sqrt(5.f),1,1/std::sqrt(5.f)},"rsqrt");
        close(stream,nt::binary(stream,positive,positive,nt::Binary::Div),{1,1,1},"division");
        close(stream,nt::binary(stream,positive,positive,nt::Binary::Mul),{25,1,25},"multiply");
        close(stream,nt::binary(stream,positive,positive,nt::Binary::Add),{10,2,10},"addition");
        const float partition=std::log(std::exp(-2.f)+1+std::exp(2.f));
        close(stream,nt::logsumexp(stream,signed_x),{partition},"logsumexp scalar");
        auto probability=nt::softmax(stream,signed_x);
        close(stream,probability,{std::exp(-2.f-partition),std::exp(-partition),std::exp(2.f-partition)},"softmax");
        std::puts("all device tensor checks passed");
        return 0;
    } catch(const std::exception& error) {
        std::fprintf(stderr,"FAILED: %s\n",error.what());return 1;
    }
}

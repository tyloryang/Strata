// Device decoder microbenchmark. Input upload, output readback and the full CPU
// correctness oracle are outside timing; dequantize_device's setup and sync are in.
#include "strata/cann/dequant_npu.hpp"
#include "strata/cann/ops.hpp"
#include "strata/cann/quants.hpp"

#include <acl/acl.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace strata::cann;

namespace {
struct Format { int type; const char* name; };
constexpr Format formats[] = {
    {18,"IQ3_XXS"},{21,"IQ3_S"},{13,"Q5_K"},{14,"Q6_K"},
    {8,"Q8_0"},{12,"Q4_K"},{16,"IQ2_XXS"},{17,"IQ2_XS"},
    {20,"IQ4_NL"},{22,"IQ2_S"},{23,"IQ4_XS"},{42,"Q2_0"},
    {0,"F32"},{1,"F16"},{30,"BF16"}
};
struct Options {
    uint64_t elements=4194304;
    int warmup=1,repeats=3;
    std::string label="current";
    std::vector<Format> selected{std::begin(formats),std::end(formats)};
};
uint64_t positive(const std::string& text,const char* label,bool zero=false) {
    size_t end=0;
    if(text.empty() || text.front()=='-') throw std::runtime_error(std::string("invalid ")+label);
    const auto value=std::stoull(text,&end);
    if(end!=text.size() || (!zero && !value)) throw std::runtime_error(std::string("invalid ")+label);
    return value;
}
Options parse(int argc,char** argv) {
    Options options;
    for(int i=1;i<argc;++i) {
        const std::string option=argv[i];
        if(option=="--help") {
            std::puts("usage: dequant_npu_bench [--elements N] [--warmup N] [--repeats N] [--types IQ3_XXS,IQ3_S,Q5_K,Q6_K] [--label old-or-new]");
            std::puts("Defaults: 4194304 elements, 1 warmup, 3 repeats, all 15 supported formats. Counts round up to full blocks.");
            std::exit(0);
        }
        if(i+1>=argc)throw std::runtime_error("missing argument for "+option);
        const std::string value=argv[++i];
        if(option=="--elements") options.elements=positive(value,"element count");
        else if(option=="--warmup" || option=="--repeats") {
            const auto count=positive(value,option.c_str(),option=="--warmup");
            if(count>1000)throw std::runtime_error("warmup/repeats must be <= 1000");
            (option=="--warmup"?options.warmup:options.repeats)=int(count);
        } else if(option=="--label") {
            if(value.empty() || value.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-")!=std::string::npos)
                throw std::runtime_error("label must use letters/digits/underscore/dot/hyphen");
            options.label=value;
        } else if(option=="--types") {
            options.selected.clear();size_t start=0;
            do {
                const auto end=value.find(',',start);
                const auto item=value.substr(start,end==std::string::npos?std::string::npos:end-start);
                const Format* found=nullptr;
                for(const auto& format:formats)if(item==format.name || item==std::to_string(format.type))found=&format;
                if(!found)throw std::runtime_error("unknown format "+item);
                if(std::any_of(options.selected.begin(),options.selected.end(),[&](const Format& f){return f.type==found->type;}))
                    throw std::runtime_error("duplicate format "+item);
                options.selected.push_back(*found);
                if(end==std::string::npos)break;
                start=end+1;
            } while(true);
        } else throw std::runtime_error("unknown option "+option);
    }
    if(options.elements>uint64_t(std::numeric_limits<int64_t>::max())-256 ||
       options.elements>(std::numeric_limits<size_t>::max()/sizeof(float))-256)
        throw std::runtime_error("element count is too large");
    return options;
}
void put_half(uint8_t* dst,float value) {
    uint16_t half;f32_to_f16(&value,&half,1);std::memcpy(dst,&half,2);
}
struct Input { uint64_t elements;std::vector<uint8_t> raw;std::vector<float> reference; };
Input synthetic(const Format& format,uint64_t requested) {
    const bool plain=format.type==0 || format.type==1 || format.type==30;
    const uint64_t block=plain?1:uint64_t(quant_block_elements(format.type));
    const uint64_t bytes=plain?(format.type==0?4:2):uint64_t(quant_type_size(format.type));
    if(!block || !bytes)throw std::runtime_error("missing format geometry");
    Input input;input.elements=((requested+block-1)/block)*block;
    input.raw.resize(input.elements/block*bytes);input.reference.resize(input.elements);
    if(plain) {
        for(uint64_t i=0;i<input.elements;++i) {
            float value=float(int(i%1021)-510)/256.0f;
            if(format.type==0)std::memcpy(input.raw.data()+4*i,&value,4);
            else if(format.type==1) {
                uint16_t half;f32_to_f16(&value,&half,1);
                std::memcpy(input.raw.data()+2*i,&half,2);f16_to_f32(&half,&value,1);
            } else {
                uint32_t bits;std::memcpy(&bits,&value,4);const uint16_t upper=uint16_t(bits>>16);
                std::memcpy(input.raw.data()+2*i,&upper,2);bits&=0xffff0000u;std::memcpy(&value,&bits,4);
            }
            input.reference[i]=value;
        }
    } else {
        // Fixed xorshift32, independent of standard-library RNG implementations.
        uint32_t state=0x19660dU+uint32_t(format.type);
        for(auto& byte:input.raw) {state^=state<<13;state^=state>>17;state^=state<<5;byte=uint8_t(state);}
        const float scales[]={0.0123f,-0.0141f,0x1p-24f,0.5f};
        for(uint64_t b=0;b<input.elements/block;++b) {
            auto* dst=input.raw.data()+b*bytes;
            put_half(dst+(format.type==14?208:0),scales[b%4]);
            if(format.type==12 || format.type==13)put_half(dst+2,0.0071f);
        }
        if(!dequantize(format.type,input.raw.data(),input.reference.data(),int64_t(input.elements)))
            throw std::runtime_error("CPU verification decoder rejected "+std::string(format.name));
    }
    return input;
}
struct Check { double max_abs=0,sum=0,sum_abs=0;uint64_t hash=14695981039346656037ULL;bool exact=true; };
Check verify(const std::vector<float>& result,const std::vector<float>& reference) {
    if(result.size()!=reference.size())throw std::runtime_error("result size mismatch");
    Check check;
    for(size_t i=0;i<result.size();++i) {
        if(!std::isfinite(result[i]) || !std::isfinite(reference[i]))
            throw std::runtime_error("nonfinite decoded value at "+std::to_string(i));
        const double error=std::abs(double(result[i])-reference[i]);
        check.max_abs=std::max(check.max_abs,error);
        if(error>1e-6+2e-6*std::abs(double(reference[i]))) {
            std::fprintf(stderr,"mismatch[%zu] NPU=%.9g CPU=%.9g\n",i,result[i],reference[i]);
            throw std::runtime_error("decoded data failed CPU comparison");
        }
        uint32_t bits,expected;std::memcpy(&bits,&result[i],4);std::memcpy(&expected,&reference[i],4);
        check.exact=check.exact && bits==expected;
        for(int shift=0;shift<32;shift+=8) {check.hash^=uint8_t(bits>>shift);check.hash*=1099511628211ULL;}
        check.sum+=result[i];check.sum_abs+=std::abs(double(result[i]));
    }
    return check;
}
void bench(Stream& stream,const Format& format,const Options& options) {
    const auto input=synthetic(format,options.elements);
    Buffer raw(input.raw.size()),decoded(input.elements*sizeof(float));
    raw.copy_from_host(input.raw.data(),input.raw.size());
    std::vector<double> milliseconds;
    for(int trial=-options.warmup;trial<options.repeats;++trial) {
        // Poison out of the timed region so a skipped kernel cannot reuse a
        // successful warmup's output and accidentally report a valid fast result.
        const auto status=aclrtMemset(decoded.data(),decoded.size(),0xff,decoded.size());
        if(status!=ACL_SUCCESS)throw Error("benchmark output poison failed",status);
        stream.synchronize();
        const auto start=std::chrono::steady_clock::now();
        dequantize_device(stream,format.type,raw.data(),input.elements,decoded.data());
        stream.synchronize();
        const auto stop=std::chrono::steady_clock::now();
        if(trial>=0)milliseconds.push_back(std::chrono::duration<double,std::milli>(stop-start).count());
    }
    // Readback and validation deliberately happen after all timed repetitions.
    std::vector<float> result(input.elements);decoded.copy_to_host(result.data(),decoded.size());
    const auto check=verify(result,input.reference);
    auto ordered=milliseconds;std::sort(ordered.begin(),ordered.end());
    const size_t middle=ordered.size()/2;
    const double median=ordered.size()%2?ordered[middle]:(ordered[middle-1]+ordered[middle])*0.5;
    double total=0;for(double value:milliseconds)total+=value;
    const double input_gbps=double(input.raw.size())/(median*1e6);
    const double io_gbps=double(input.raw.size()+decoded.size())/(median*1e6);
    std::printf("BENCH {\"label\":\"%s\",\"type\":%d,\"name\":\"%s\",\"elements\":%llu,\"input_bytes\":%zu,\"output_bytes\":%zu,"
                "\"warmup\":%d,\"repeats\":%d,\"min_ms\":%.6f,\"median_ms\":%.6f,\"mean_ms\":%.6f,"
                "\"input_GBps\":%.9f,\"read_write_GBps\":%.9f,\"max_abs_error\":%.9g,\"bit_exact\":%s,"
                "\"sum\":%.17g,\"sum_abs\":%.17g,\"fnv1a64\":\"%016llx\",\"samples_ms\":[",
                options.label.c_str(),format.type,format.name,(unsigned long long)input.elements,input.raw.size(),decoded.size(),
                options.warmup,options.repeats,ordered.front(),median,total/milliseconds.size(),input_gbps,io_gbps,
                check.max_abs,check.exact?"true":"false",check.sum,check.sum_abs,(unsigned long long)check.hash);
    for(size_t i=0;i<milliseconds.size();++i)std::printf("%s%.6f",i?",":"",milliseconds[i]);
    std::puts("],\"verified\":true}");
}
}

int main(int argc,char** argv) {
    setvbuf(stdout,nullptr,_IONBF,0);
    try {
        const auto options=parse(argc,argv);
        Device device(std::getenv("STRATA_CANN_DEVICE")?std::stoi(std::getenv("STRATA_CANN_DEVICE")):0);
        Stream stream(device);
        const auto info=device_info(device.ordinal());
        std::printf("Device: %s ordinal=%d; scope=dequantize_device API + synchronization; decimal GB/s\n",info.soc.c_str(),device.ordinal());
        std::puts("Excluded from timing: synthetic generation, input H2D, output poison, output D2H, full CPU verification. API setup/codebook uploads remain included.");
        for(const auto& format:options.selected)bench(stream,format,options);
        std::puts("All decode benchmarks verified against full CPU outputs.");
        return 0;
    } catch(const std::exception& error) {
        std::fprintf(stderr,"FAILED: %s\n",error.what());return 1;
    }
}

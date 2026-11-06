// Same-stream reuse, metadata lifetime, host fences and dual-device regression.
#include "strata/cann/device.hpp"
#include "strata/cann/npu_tensor.hpp"
#include "strata/cann/dequant_npu.hpp"
#include <acl/acl.h>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

using namespace strata::cann;
namespace npu = strata::cann::npu;
namespace {
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
void check(const std::vector<float>& got, const std::vector<float>& expected) {
    require(got.size() == expected.size(), "result size mismatch");
    for (size_t i=0;i<got.size();++i)
        if (!std::isfinite(got[i]) || std::abs(got[i]-expected[i]) > 1e-5f)
            throw std::runtime_error("result mismatch at " + std::to_string(i));
}
void single_device() {
    Device device(0); Stream stream(device), other(device);
    std::vector<float> host(256); for (size_t i=0;i<host.size();++i) host[i]=float(i%13);
    auto input=npu::from_host(host.data(),{256});
    auto tables=make_quant_tables(stream);
    {
        DeferredScope scope(stream);
        bool denied=false;try {other.activate();} catch(const Error&) {denied=true;}
        require(denied,"second stream was allowed inside deferred scope");
        denied=false;try {DeferredScope nested(stream);} catch(const Error&) {denied=true;}
        require(denied,"nested deferred scope was allowed");
        auto value=input;
        for(int i=0;i<512;++i) value=npu::add_scalar(stream,value,0.25f);
        require(stream.pending(),"512 ACLNN ops were unexpectedly drained");
        bool released=false;
        defer_device_cleanup(0,[&released]{released=true;});
        require(!released,"cleanup ran before completion");
        auto expected=host;for(auto& v:expected)v+=128;
        // Raw readback (not read_f32) must fence outstanding work itself.
        std::vector<float> actual(256);copy_to_host(value.data(),actual.data(),1024);
        check(actual,expected);
        require(released && !stream.pending(),"D2H did not drain queued metadata");

        auto old=npu::scale(stream,input,2);
        std::vector<float> replacement(256,7);
        input.storage->copy_from_host(replacement.data(),1024);
        require(!stream.pending(),"H2D did not drain prior input readers");
        expected=host;for(auto& v:expected)v*=2;
        check(npu::read_f32(stream,old),expected);

        auto temporary=npu::add_scalar(stream,input,1);
        auto result=npu::scale(stream,temporary,3);
        const void* retired=temporary.data();temporary={};
        auto zero=npu::zeros({256});
        require(zero.data()==retired,"test did not exercise same-size pool reuse");
        require(stream.pending(),"async memset unexpectedly drained stream");
        check(npu::read_f32(stream,result),std::vector<float>(256,24));
        check(npu::read_f32(stream,zero),std::vector<float>(256,0));

        // TensorList owns its input descriptors; all survive the queued Cat.
        for(int i=0;i<16;++i) {
            auto joined=npu::cat(stream,{npu::slice(input,0,0,16),npu::slice(input,0,16,16)},0);
            auto rows=npu::reshape(stream,joined,{2,16});
            auto summed=npu::sum(stream,rows,1,false);
            auto [values,indices]=npu::topk(stream,summed,2);
            auto selected=npu::index_select(stream,summed,0,indices);
            auto cast=npu::cast(stream,npu::cast(stream,selected,npu::Type::F16),npu::Type::F32);
            check(npu::read_f32(stream,cast),{112,112});
        }
        auto final=npu::scale(stream,input,3);
        scope.finish();
        require(!stream.deferred() && !stream.pending(),"explicit finish did not close scope");
        check(npu::read_f32(stream,final),std::vector<float>(256,21));
    }
    std::puts("Deferred ACLNN, pooled reuse, async zero, metadata and host fences: PASS");

    // Borrowed tables permit custom kernel queueing, owned tables force a drain.
    std::vector<float> matrix(32*32,0),x(32,1);
    for(int i=0;i<32;++i)matrix[i*32+i]=2;
    Buffer weights(matrix.size()*4);weights.copy_from_host(matrix.data(),weights.size());
    auto in=npu::from_host(x.data(),{32});
    {
        DeferredScope scope(stream);
        auto out=npu::empty({32});
        quant_matvec_device(stream,0,weights.data(),32,32,static_cast<float*>(in.data()),
            static_cast<float*>(out.data()),tables->data());
        require(stream.pending(),"borrowed-table matvec was not queued");
        auto scaled=npu::scale(stream,out,3);
        check(npu::read_f32(stream,scaled),std::vector<float>(32,6));
        quant_matvec_device(stream,0,weights.data(),32,32,static_cast<float*>(in.data()),
            static_cast<float*>(out.data()));
        require(!stream.pending(),"owned tables were freed before completion");
        check(npu::read_f32(stream,out),std::vector<float>(32,2));
        scope.finish();
    }
    std::puts("Borrowed/owned table matvec completion boundaries: PASS");

    bool released=false;
    try {
        DeferredScope scope(stream);
        auto unused=npu::scale(stream,input,2);
        defer_device_cleanup(0,[&released]{released=true;});
        throw std::runtime_error("intentional unwind");
    } catch(const std::runtime_error&) {}
    require(released && !stream.pending() && !stream.deferred(),"unwind did not drain/restore ownership");
    other.activate();
    std::puts("Exception unwind and second stream restoration: PASS");
}
void dual_device() {
    Device zero(0),one(1);Stream s0(zero),s1(one);
    zero.activate();auto x0=npu::zeros({256});
    one.activate();auto x1=npu::zeros({256});
    DeferredScope d0(s0),d1(s1);
    s0.activate();x0=npu::add_scalar(s0,x0,7);
    s1.activate();x1=npu::add_scalar(s1,x1,3);
    require(s0.pending() && s1.pending(),"independent streams were not queued");
    copy_peer(x1.data(),1,x0.data(),0,1024);
    require(!s0.pending() && !s1.pending(),"P2P did not drain both owners");
    check(npu::read_f32(s1,x1),std::vector<float>(256,7));
    s0.activate();x0=npu::scale(s0,x0,2);
    s1.activate();
    d0.finish();
    int32_t ordinal=-1;
    require(aclrtGetDevice(&ordinal)==ACL_SUCCESS && ordinal==1,"finish changed caller's current device");
    d1.finish();
    check(npu::read_f32(s0,x0),std::vector<float>(256,14));
    std::puts("Dual-device ownership, P2P fences and context restoration: PASS");
}
}
int main() {
    std::setvbuf(stdout,nullptr,_IONBF,0);
    try {
        single_device();
        if(device_count()>1)dual_device();
        else std::puts("Dual-device check skipped: only one NPU");
        shutdown();
        std::puts("ALL deferred stream parity checks PASS");
        return 0;
    } catch(const std::exception& error) {
        std::fprintf(stderr,"FAILED: %s\n",error.what());return 1;
    }
}

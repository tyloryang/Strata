// Sequential two-NPU ownership, workspace and direct P2P regression checks.
#include "strata/cann/device.hpp"
#include "strata/cann/ops.hpp"

#include <acl/acl.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <vector>

using namespace strata::cann;

namespace {
void require(bool condition,const char* message) {if(!condition)throw std::runtime_error(message);}
void check_owner(const Buffer& buffer) {
    aclrtPtrAttributes attributes{};
    require(aclrtPointerGetAttributes(buffer.data(),&attributes)==ACL_SUCCESS,"cannot query allocation owner");
    require(attributes.location.type==ACL_MEM_LOCATION_TYPE_DEVICE,"allocation is not HBM");
    require(static_cast<int>(attributes.location.id)==buffer.device_ordinal(),"pooled allocation crossed device boundary");
}
void check_values(const Buffer& buffer,const std::vector<float>& expected) {
    std::vector<float> actual(expected.size());buffer.copy_to_host(actual.data(),actual.size()*sizeof(float));
    for(size_t i=0;i<actual.size();++i)
        if(!std::isfinite(actual[i]) || std::abs(actual[i]-expected[i])>1e-5f)
            throw std::runtime_error("device result mismatch at "+std::to_string(i));
}
void check_matmul(Stream& stream,Stream& other) {
    stream.activate();
    constexpr int n=32;
    std::vector<float> identity(n*n,0),input(n*n);
    for(int i=0;i<n;++i)identity[i*n+i]=2;
    for(int i=0;i<n*n;++i)input[i]=float(i%17-8);
    Buffer a(identity.size()*4),b(input.size()*4),out(input.size()*4);
    a.copy_from_host(identity.data(),a.size());b.copy_from_host(input.data(),b.size());
    Tensor ta{a.data(),DType::F32,n,n},tb{b.data(),DType::F32,n,n},tc{out.data(),DType::F32,n,n};
    other.activate(); // Execute must reactivate its own context/workspace.
    matmul(stream,ta,tb,tc);
    for(float& value:input)value*=2;
    other.activate(); // Readback must activate the allocation's owner too.
    check_values(out,input);check_owner(out);
}
void two_devices() {
    Device zero(0),one(1);Stream s0(zero),s1(one);
    require(s0.device_ordinal()==0 && s1.device_ordinal()==1,"stream ordinals incorrect");
    // Equal-size frees must go to different per-device pools.
    for(int round=0;round<4;++round) {
        zero.activate();{Buffer cached(4096);check_owner(cached);}
        one.activate();{Buffer cached(4096);check_owner(cached);}
        check_matmul(s0,s1);check_matmul(s1,s0);
    }
    std::puts("Per-device allocation pools and alternating matmul workspaces: PASS");
    {Device second_zero(0);second_zero.activate();}
    check_matmul(s0,s1);
    std::puts("Destroying a duplicate Device owner retains live context: PASS");

    std::vector<float> host(1024);for(size_t i=0;i<host.size();++i)host[i]=float(int(i%97)-48);
    zero.activate();Buffer source(host.size()*4),same(host.size()*4),returned(host.size()*4);
    source.copy_from_host(host.data(),source.size());
    one.activate();Buffer destination(host.size()*4);
    const bool forward=peer_access_supported(1,0),reverse=peer_access_supported(0,1);
    std::printf("P2P support: device 1 accesses 0=%d; device 0 accesses 1=%d\n",int(forward),int(reverse));
    require(forward && reverse,"both directions of direct P2P are required; host fallback disabled");
    copy_peer(same,source,source.size());check_values(same,host);
    copy_peer(destination,source,source.size());check_values(destination,host);
    std::puts("Aligned full-buffer P2P: PASS");
    std::vector<float> expected(host.size(),0);
    destination.copy_from_host(expected.data(),destination.size());
    copy_peer(destination,source,1024,128,64);
    std::copy(host.begin()+16,host.begin()+272,expected.begin()+32);
    check_values(destination,expected);
    std::puts("Offset-view P2P via device staging: PASS");
    copy_peer(returned,destination,destination.size());check_values(returned,expected);
    copy_peer(destination.data(),1,source.data(),0,source.size());check_values(destination,host);
    int32_t current=-1;require(aclrtGetDevice(&current)==ACL_SUCCESS && current==1,"P2P did not leave destination active");
    bool rejected=false;
    try {copy_peer(destination,source,source.size()+1);} catch(const Error&) {rejected=true;}
    require(rejected,"out-of-range P2P copy was not rejected");
    std::puts("Same-device D2D, bidirectional P2P, offset copy and bounds rejection: PASS");
}
void survivor_lifetimes() {
    std::unique_ptr<Stream> stream;
    std::unique_ptr<Buffer> buffer;
    const std::vector<float> expected{1,2,3,4};
    {
        Device temporary(0);stream=std::make_unique<Stream>(temporary);
        buffer=std::make_unique<Buffer>(expected.size()*sizeof(float));
        buffer->copy_from_host(expected.data(),buffer->size());
    }
    stream->synchronize();check_values(*buffer,expected);
    stream.reset();check_values(*buffer,expected);buffer.reset();
    // The last Buffer teardown has now cleared the old pool and reset device 0.
    Device reopened(0);Buffer clean(4096);check_owner(clean);
    std::puts("Stream/Buffer outlive Device and reset/reopen without stale cache: PASS");
}
void destruction_restores_device() {
    Device one(1);Stream s1(one);
    {
        Device zero(0);Stream s0(zero);Buffer old(1024);
        one.activate();
        // Destruction order is old buffer, stream 0, then last device-0 owner.
        // None may change the caller's selected device 1.
    }
    int32_t current=-1;
    require(aclrtGetDevice(&current)==ACL_SUCCESS && current==1,"cross-device destruction changed active device");
    Buffer subsequent(1024);require(subsequent.device_ordinal()==1,"allocation after cross-device teardown used wrong NPU");
    check_owner(subsequent);s1.synchronize();
    std::puts("Cross-device Buffer/Stream/Device teardown restores current context: PASS");
}
}
int main() {
    setvbuf(stdout,nullptr,_IONBF,0);
    try {
        require(device_count()>=2,"multi_device_parity requires two NPUs");
        two_devices();survivor_lifetimes();destruction_restores_device();shutdown();
        std::puts("All multi-device ownership and P2P checks passed.");return 0;
    } catch(const std::exception& error) {
        std::fprintf(stderr,"FAILED: %s\n",error.what());return 1;
    }
}

#include "strata/cann/device.hpp"

#include <acl/acl.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <set>
#include <unordered_map>
#include <vector>

namespace strata::cann {

void print_xfer_stats();
void release_workspace_for_device(int ordinal);
void clear_buffer_pool_for_device(int ordinal);

struct DeviceLifetime {
    explicit DeviceLifetime(int device);
    ~DeviceLifetime();
    void activate() const;
    int ordinal;
    void * context = nullptr;
};

namespace {

std::mutex g_init_mutex;
bool g_initialized = false;
std::mutex g_lifetimes_mutex;
std::unordered_map<int, std::weak_ptr<DeviceLifetime>> g_lifetimes;
std::set<std::pair<int, int>> g_peer_enabled;
std::mutex g_deferred_mutex;
std::unordered_map<int, Stream*> g_deferred_streams;

Stream* deferred_stream(int ordinal) {
    std::lock_guard<std::mutex> lock(g_deferred_mutex);
    const auto found = g_deferred_streams.find(ordinal);
    return found == g_deferred_streams.end() ? nullptr : found->second;
}

[[noreturn]] void fail(const char * what, int code) {
    std::string message = std::string("cann: ") + what + " failed (aclError " + std::to_string(code) + ")";
    if (const char * recent = aclGetRecentErrMsg(); recent && *recent) message += ": " + std::string(recent);
    throw Error(message, code);
}

void check(aclError err, const char * what) {
    if (err != ACL_SUCCESS) {
        fail(what, (int) err);
    }
}

// Cleanup and read-only inspection may need a different device, but must not
// change which NPU receives the caller's next allocation. Never restore a
// context that this cleanup is about to destroy.
class ContextRestore {
public:
    explicit ContextRestore(void * retiring = nullptr) {
        aclrtContext previous = nullptr;
        if (aclrtGetCurrentContext(&previous) == ACL_SUCCESS && previous != retiring) previous_ = previous;
    }
    ~ContextRestore() {
        if (previous_ && g_initialized) {
            const auto status = aclrtSetCurrentContext(previous_);
            if (status != ACL_SUCCESS)
                std::fprintf(stderr, "cann: restoring context failed (%d)\n", int(status));
        }
    }
private:
    aclrtContext previous_ = nullptr;
};

std::shared_ptr<DeviceLifetime> lifetime_for(int ordinal) {
    std::lock_guard<std::mutex> lock(g_lifetimes_mutex);
    auto result = g_lifetimes[ordinal].lock();
    if (!result) {
        result = std::make_shared<DeviceLifetime>(ordinal);
        g_lifetimes[ordinal] = result;
    }
    return result;
}

std::shared_ptr<DeviceLifetime> current_lifetime() {
    int32_t ordinal = -1;
    check(aclrtGetDevice(&ordinal), "aclrtGetDevice");
    return lifetime_for(ordinal);
}

int pointer_device(const void * pointer) {
    aclrtPtrAttributes attributes{};
    check(aclrtPointerGetAttributes(pointer, &attributes), "aclrtPointerGetAttributes");
    if (attributes.location.type != ACL_MEM_LOCATION_TYPE_DEVICE)
        throw Error("cann: device copy received a non-device pointer", -1);
    return static_cast<int>(attributes.location.id);
}

} // namespace

void init() {
    std::lock_guard<std::mutex> lock(g_init_mutex);
    if (!g_initialized) {
        aclError err = aclInit(nullptr);
        if (err != ACL_SUCCESS) {
            fail("aclInit", (int) err);
        }
        g_initialized = true;
    }
}

DeviceLifetime::DeviceLifetime(int device) : ordinal(device) {
    init();
    check(aclrtSetDevice(ordinal), "aclrtSetDevice");
    check(aclrtGetCurrentContext(reinterpret_cast<aclrtContext *>(&context)), "aclrtGetCurrentContext");
}

void DeviceLifetime::activate() const {
    if (!g_initialized) throw Error("cann: activating a device after runtime shutdown", -1);
    check(aclrtSetCurrentContext(static_cast<aclrtContext>(context)), "aclrtSetCurrentContext");
}

DeviceLifetime::~DeviceLifetime() {
    if (!g_initialized) return;
    ContextRestore restore(context);
    try {
        activate();
        check(aclrtSynchronizeDevice(), "aclrtSynchronizeDevice(teardown)");
        release_workspace_for_device(ordinal);
        clear_buffer_pool_for_device(ordinal);
    } catch (const std::exception & error) {
        // Destructors cannot throw; ResetDevice below still reclaims this device.
        std::fprintf(stderr, "cann device %d teardown: %s\n", ordinal, error.what());
    }
    for (auto it = g_peer_enabled.begin(); it != g_peer_enabled.end();) {
        if (it->first == ordinal || it->second == ordinal) it = g_peer_enabled.erase(it);
        else ++it;
    }
    const auto status = aclrtResetDevice(ordinal);
    if (status != ACL_SUCCESS)
        std::fprintf(stderr, "cann: aclrtResetDevice(%d) failed (%d)\n", ordinal, int(status));
}

void shutdown() {
    std::lock_guard<std::mutex> init_lock(g_init_mutex);
    if (g_initialized) {
        {
            std::lock_guard<std::mutex> lock(g_lifetimes_mutex);
            for (const auto & entry : g_lifetimes)
                if (!entry.second.expired())
                    throw Error("cann: shutdown requires all Device, Stream and Buffer owners to be destroyed", -1);
            g_lifetimes.clear();
        }
        // Last-owner teardown normally emptied the per-device caches already.
        extern void release_workspace();
        release_workspace();
        extern void clear_buffer_pool();
        clear_buffer_pool();
        print_xfer_stats();
        check(aclFinalize(), "aclFinalize");
        g_initialized = false;
    }
}

int device_count() {
    init();
    uint32_t count = 0;
    if (aclrtGetDeviceCount(&count) != ACL_SUCCESS) {
        return 0;
    }
    return (int) count;
}

DeviceInfo device_info(int ordinal) {
    init();
    ContextRestore restore;
    DeviceInfo info;
    info.ordinal = ordinal;

    const auto lifetime = lifetime_for(ordinal);
    lifetime->activate();

    const char * soc = aclrtGetSocName();
    info.soc = soc ? soc : "";

    size_t free_bytes = 0, total_bytes = 0;
    check(aclrtGetMemInfo(ACL_HBM_MEM, &free_bytes, &total_bytes), "aclrtGetMemInfo");
    info.free_bytes = free_bytes;
    info.total_bytes = total_bytes;
    return info;
}

Device::Device(int ordinal) : ordinal_(ordinal) {
    lifetime_ = lifetime_for(ordinal);
    activate();
}

Device::~Device() = default;

void Device::activate() const { lifetime_->activate(); }

Stream::Stream(const Device & device) : ordinal_(device.ordinal()), lifetime_(device.lifetime_) {
    activate();
    check(aclrtCreateStream((aclrtStream *) &stream_), "aclrtCreateStream");
}

Stream::~Stream() {
    if (stream_) {
        ContextRestore restore;
        try {
            if (deferred_) end_deferred();
            else fence_device(ordinal_);
            lifetime_->activate();
            check(aclrtSynchronizeStream((aclrtStream) stream_), "aclrtSynchronizeStream(teardown)");
            check(aclrtDestroyStream((aclrtStream) stream_), "aclrtDestroyStream");
        } catch (const std::exception & error) {
            std::fprintf(stderr, "cann stream teardown: %s\n", error.what());
            if (deferred_) std::terminate();
        }
        if (deferred_submissions_ && std::getenv("STRATA_CANN_STATS"))
            std::fprintf(stderr, "DEFERRED_STATS device=%d submitted=%llu drains=%llu wait_s=%.6f metadata=%llu\n",
                ordinal_, (unsigned long long)deferred_submissions_, (unsigned long long)deferred_drains_,
                deferred_wait_seconds_, (unsigned long long)deferred_cleanups_);
    }
}

void Stream::synchronize() const {
    activate();
    const auto begin = std::chrono::steady_clock::now();
    check(aclrtSynchronizeStream((aclrtStream) stream_), "aclrtSynchronizeStream");
    if (deferred_ && pending_) {
        ++deferred_drains_;
        deferred_wait_seconds_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
    }
    pending_ = false;
    // Move before invoking cleanup so callbacks cannot invalidate iteration.
    auto cleanup = std::move(cleanups_);
    cleanups_.clear();
    for (auto& release : cleanup) release();
}

void Stream::activate() const {
    if (auto* owner = deferred_stream(ordinal_); owner && owner != this)
        throw Error("cann: another stream owns this device's deferred scope", -1);
    if (deferred_ && deferred_thread_ != std::this_thread::get_id())
        throw Error("cann: deferred stream may only be used by its owning host thread", -1);
    lifetime_->activate();
}

void Stream::begin_deferred() {
    ContextRestore restore;
    if (deferred_) throw Error("cann: nested deferred scope on the same stream", -1);
    activate();
    // The pool/workspace may previously have served another synchronous stream.
    check(aclrtSynchronizeDevice(), "aclrtSynchronizeDevice(begin deferred)");
    std::lock_guard<std::mutex> lock(g_deferred_mutex);
    if (g_deferred_streams.count(ordinal_)) throw Error("cann: device already has a deferred stream", -1);
    g_deferred_streams[ordinal_] = this;
    deferred_thread_ = std::this_thread::get_id();
    deferred_ = true;
}

void Stream::end_deferred() {
    ContextRestore restore;
    if (!deferred_) return;
    // On failure retain the ownership guard: no other stream may consume pool
    // addresses whose outstanding use could not be proven complete.
    synchronize();
    std::lock_guard<std::mutex> lock(g_deferred_mutex);
    g_deferred_streams.erase(ordinal_);
    deferred_ = false;
}

void Stream::submitted() const {
    pending_ = true; // Mark before launch, including a partially failed launch.
    if (deferred_) ++deferred_submissions_;
}

void Stream::defer_cleanup(std::function<void()> cleanup) const {
    if (deferred_ && pending_) {
        cleanups_.push_back(std::move(cleanup));
        ++deferred_cleanups_;
    } else cleanup();
}

DeferredScope::DeferredScope(Stream& stream, bool enabled) {
    if (enabled) { stream.begin_deferred(); stream_ = &stream; }
}
DeferredScope::~DeferredScope() {
    if (stream_) try { finish(); }
    catch (const std::exception& error) {
        std::fprintf(stderr, "cann deferred scope teardown: %s\n", error.what());
        // Continuing after a failed completion fence would permit unsafe reuse.
        std::terminate();
    }
}
void DeferredScope::finish() {
    if (stream_) { stream_->end_deferred(); stream_ = nullptr; }
}

void fence_device(int ordinal) {
    ContextRestore restore;
    if (auto* stream = deferred_stream(ordinal); stream && stream->pending()) stream->synchronize();
}
void defer_device_cleanup(int ordinal, std::function<void()> cleanup) {
    if (auto* stream = deferred_stream(ordinal)) stream->defer_cleanup(std::move(cleanup));
    else cleanup();
}
void zero_device(void* pointer, size_t bytes, int ordinal) {
    ContextRestore restore;
    if (auto* stream = deferred_stream(ordinal)) {
        stream->activate();
        stream->submitted();
        check(aclrtMemsetAsync(pointer, bytes, 0, bytes, (aclrtStream)stream->handle()), "aclrtMemsetAsync");
    } else {
        lifetime_for(ordinal)->activate();
        check(aclrtMemset(pointer, bytes, 0, bytes), "aclrtMemset");
    }
}

// Device memory is pooled. A single token allocates thousands of small temporaries, and one
// aclrtMalloc per temporary was the real cost of the whole model: 12.8 s per token on ops that
// each measure 0.043 ms.
namespace {

std::mutex g_pool_mutex;
using DevicePool = std::unordered_map<size_t, std::vector<void *>>;
std::unordered_map<int, DevicePool> g_pool;
struct PeerStaging { void * data = nullptr; size_t capacity = 0; };
std::unordered_map<int, PeerStaging> g_peer_staging;

void * peer_staging(int ordinal, size_t bytes) {
    // One staging allocation per device is enough for sequential layer handoff.
    // It is freed by the same last-owner cleanup as that device's regular pool.
    std::lock_guard<std::mutex> lock(g_pool_mutex);
    auto & staging = g_peer_staging[ordinal];
    if (staging.capacity >= bytes) return staging.data;
    check(aclrtSetDevice(ordinal), "aclrtSetDevice(P2P staging)");
    check(aclrtSynchronizeDevice(), "aclrtSynchronizeDevice(P2P staging grow)");
    if (staging.data) {
        check(aclrtFree(staging.data), "aclrtFree(P2P staging)");
        staging.data = nullptr; staging.capacity = 0;
    }
    const size_t capacity = std::max<size_t>(bytes, 64u << 10);
    check(aclrtMalloc(&staging.data, capacity, ACL_MEM_MALLOC_HUGE_FIRST_P2P),
          "aclrtMalloc(P2P staging)");
    staging.capacity = capacity;
    return staging.data;
}

void * pool_take(int ordinal, size_t bytes) {
    {
        std::lock_guard<std::mutex> lock(g_pool_mutex);
        auto & pool = g_pool[ordinal];
        auto it = pool.find(bytes);
        if (it != pool.end() && !it->second.empty()) {
            void * p = it->second.back();
            it->second.pop_back();
            return p;
        }
    }
    void * p = nullptr;
    check(aclrtMalloc(&p, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc");
    return p;
}

void pool_give(int ordinal, void * p, size_t bytes) {
    std::lock_guard<std::mutex> lock(g_pool_mutex);
    g_pool[ordinal][bytes].push_back(p);
}

} // namespace

// Profiling: the model spends its time outside the ops, so transfers are counted too.
long long g_xfer_count = 0;
long long g_xfer_bytes = 0;
double g_xfer_seconds = 0.0;

void print_xfer_stats() {
    if (g_xfer_count > 0 && getenv("STRATA_CANN_STATS") != nullptr) {
        printf("xfer: %lld copies, %.1f MB, %.3f s total, %.3f ms each\n", g_xfer_count,
               (double) g_xfer_bytes / 1e6, g_xfer_seconds, 1000.0 * g_xfer_seconds / (double) g_xfer_count);
    }
}

void clear_buffer_pool_for_device(int ordinal) {
    ContextRestore restore;
    fence_device(ordinal);
    std::lock_guard<std::mutex> lock(g_pool_mutex);
    auto device = g_pool.find(ordinal);
    auto staging = g_peer_staging.find(ordinal);
    if (device == g_pool.end() && staging == g_peer_staging.end()) return;
    check(aclrtSetDevice(ordinal), "aclrtSetDevice(pool cleanup)");
    if (device != g_pool.end()) {
        for (auto & entry : device->second) {
            for (void * p : entry.second) {
                check(aclrtFree(p), "aclrtFree(pool)");
            }
        }
        g_pool.erase(device);
    }
    if (staging != g_peer_staging.end()) {
        if (staging->second.data) check(aclrtFree(staging->second.data), "aclrtFree(P2P pool)");
        g_peer_staging.erase(staging);
    }
}

void clear_buffer_pool() {
    std::vector<int> devices;
    {
        std::lock_guard<std::mutex> lock(g_pool_mutex);
        for (const auto & entry : g_pool) devices.push_back(entry.first);
        for (const auto & entry : g_peer_staging)
            if (std::find(devices.begin(), devices.end(), entry.first) == devices.end()) devices.push_back(entry.first);
    }
    for (int ordinal : devices) clear_buffer_pool_for_device(ordinal);
}

Buffer::Buffer(size_t bytes) : bytes_(bytes) {
    if (!bytes) throw Error("cann: cannot allocate an empty Buffer", -1);
    lifetime_ = current_lifetime();
    ordinal_ = lifetime_->ordinal;
    activate();
    data_ = pool_take(ordinal_, bytes);
}

Buffer::~Buffer() {
    if (data_) {
        pool_give(ordinal_, data_, bytes_);
    }
}

void Buffer::activate() const { lifetime_->activate(); }

void copy_to_host(const void * device_ptr, void * host, size_t bytes) {
    ContextRestore restore;
    const auto lifetime = lifetime_for(pointer_device(device_ptr));
    lifetime->activate();
    fence_device(lifetime->ordinal);
    check(aclrtMemcpy(host, bytes, device_ptr, bytes, ACL_MEMCPY_DEVICE_TO_HOST), "aclrtMemcpy(D2H raw)");
}

void copy_to_device(void * device_ptr, const void * host, size_t bytes) {
    ContextRestore restore;
    const auto lifetime = lifetime_for(pointer_device(device_ptr));
    lifetime->activate();
    fence_device(lifetime->ordinal);
    check(aclrtMemcpy(device_ptr, bytes, host, bytes, ACL_MEMCPY_HOST_TO_DEVICE), "aclrtMemcpy(H2D raw)");
}

void Buffer::copy_from_host(const void * host, size_t bytes) {
    ContextRestore restore;
    activate();
    fence_device(ordinal_);
    const auto t0 = std::chrono::steady_clock::now();
    struct Timer {
        std::chrono::steady_clock::time_point t0;
        size_t bytes;
        ~Timer() {
            g_xfer_count += 1;
            g_xfer_bytes += (long long) bytes;
            g_xfer_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        }
    } timer{ t0, bytes };
    if (bytes > bytes_) {
        throw Error("cann: copy_from_host larger than the buffer", -1);
    }
    check(aclrtMemcpy(data_, bytes_, host, bytes, ACL_MEMCPY_HOST_TO_DEVICE), "aclrtMemcpy(H2D)");
}

void Buffer::copy_to_host(void * host, size_t bytes) const {
    ContextRestore restore;
    activate();
    fence_device(ordinal_);
    const auto t0 = std::chrono::steady_clock::now();
    struct Timer {
        std::chrono::steady_clock::time_point t0;
        size_t bytes;
        ~Timer() {
            g_xfer_count += 1;
            g_xfer_bytes += (long long) bytes;
            g_xfer_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        }
    } timer{ t0, bytes };
    if (bytes > bytes_) {
        throw Error("cann: copy_to_host larger than the buffer", -1);
    }
    check(aclrtMemcpy(host, bytes, data_, bytes, ACL_MEMCPY_DEVICE_TO_HOST), "aclrtMemcpy(D2H)");
}

bool peer_access_supported(int dst_device, int src_device) {
    init();
    if (dst_device < 0 || src_device < 0) throw Error("cann: negative peer device ordinal", -1);
    if (dst_device == src_device) return true;
    int32_t supported = 0;
    check(aclrtDeviceCanAccessPeer(&supported, dst_device, src_device), "aclrtDeviceCanAccessPeer");
    return supported != 0;
}

void enable_peer_access(int dst_device, int src_device) {
    if (dst_device == src_device) return;
    const auto key = std::make_pair(dst_device, src_device);
    if (g_peer_enabled.count(key)) return;
    if (!peer_access_supported(dst_device, src_device))
        throw Error("cann: P2P unavailable from device " + std::to_string(dst_device) +
                    " to device " + std::to_string(src_device) + "; host staging is disabled", -1);
    const auto destination = lifetime_for(dst_device);
    destination->activate();
    check(aclrtDeviceEnablePeerAccess(src_device, 0), "aclrtDeviceEnablePeerAccess");
    g_peer_enabled.insert(key);
}

void copy_peer(void * dst, int dst_device, const void * src, int src_device, size_t bytes) {
    fence_device(src_device);
    if (dst_device != src_device) fence_device(dst_device);
    if (!bytes) return;
    if (!dst || !src) throw Error("cann: null pointer in device copy", -1);
    const int dst_owner = pointer_device(dst), src_owner = pointer_device(src);
    if (dst_owner != dst_device || src_owner != src_device)
        throw Error("cann: copy_peer pointer owners dst=" + std::to_string(dst_owner) +
                    " src=" + std::to_string(src_owner) + " differ from requested dst=" +
                    std::to_string(dst_device) + " src=" + std::to_string(src_device), -1);
    const auto source = lifetime_for(src_device), destination = lifetime_for(dst_device);
    source->activate();
    check(aclrtSynchronizeDevice(), "aclrtSynchronizeDevice(peer source)");
    destination->activate();
    check(aclrtSynchronizeDevice(), "aclrtSynchronizeDevice(peer destination)");
    if (dst_device == src_device) {
        check(aclrtMemcpy(dst, bytes, src, bytes, ACL_MEMCPY_DEVICE_TO_DEVICE), "aclrtMemcpy(D2D)");
        check(aclrtSynchronizeDevice(), "aclrtSynchronizeDevice(D2D complete)");
        return;
    }
    // CANN's inter-device example enables both directions and allocates both
    // endpoints with *_P2P. Using aligned P2P bases also avoids driver failures
    // when the model's source/destination are differently-offset tensor views.
    enable_peer_access(src_device, dst_device);
    enable_peer_access(dst_device, src_device);
    source->activate();
    void * src_staging = peer_staging(src_device, bytes);
    check(aclrtMemcpy(src_staging, bytes, src, bytes, ACL_MEMCPY_DEVICE_TO_DEVICE), "aclrtMemcpy(P2P source staging)");
    check(aclrtSynchronizeDevice(), "aclrtSynchronizeDevice(P2P source staging)");
    destination->activate();
    void * dst_staging = peer_staging(dst_device, bytes);
    check(aclrtMemcpy(dst_staging, bytes, src_staging, bytes, ACL_MEMCPY_DEVICE_TO_DEVICE), "aclrtMemcpy(P2P aligned bases)");
    check(aclrtSynchronizeDevice(), "aclrtSynchronizeDevice(P2P aligned bases)");
    destination->activate();
    check(aclrtMemcpy(dst, bytes, dst_staging, bytes, ACL_MEMCPY_DEVICE_TO_DEVICE),
          "aclrtMemcpy(P2P destination staging)");
    check(aclrtSynchronizeDevice(), "aclrtSynchronizeDevice(peer complete)");
}

void copy_peer(Buffer & dst, const Buffer & src, size_t bytes, size_t dst_offset, size_t src_offset) {
    if (dst_offset > dst.size() || bytes > dst.size() - dst_offset ||
        src_offset > src.size() || bytes > src.size() - src_offset)
        throw Error("cann: copy_peer range exceeds Buffer", -1);
    copy_peer(static_cast<char *>(dst.data()) + dst_offset, dst.device_ordinal(),
              static_cast<const char *>(src.data()) + src_offset, src.device_ordinal(), bytes);
}

} // namespace strata::cann

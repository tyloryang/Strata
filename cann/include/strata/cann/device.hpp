// cann/include/strata/cann/device.hpp - the CANN device layer of the Ascend port.
//
// This is the smallest piece the rest of the port needs: create a device context, make a
// stream, allocate HBM, move bytes, and turn aclError into an exception that names the call.
// It deliberately does not touch Strata's engine types yet - the engine's kernels are ported
// on top of this layer, one op at a time (see cann/README.md).
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace strata::cann {
struct DeviceLifetime;

// Any ACL failure, with the numeric code and the call that produced it.
class Error : public std::runtime_error {
public:
    Error(const std::string & what, int code) : std::runtime_error(what), code_(code) {}
    int code() const { return code_; }

private:
    int code_;
};

// aclInit/aclFinalize once per process. Safe to call repeatedly.
void init();
void shutdown();

int device_count();

struct DeviceInfo {
    int ordinal = -1;
    std::string soc;          // "Ascend910B2C"
    uint64_t total_bytes = 0; // HBM total, from aclrtGetMemInfo
    uint64_t free_bytes = 0;
};

// Throws Error when the ordinal does not exist.
DeviceInfo device_info(int ordinal = 0);

// Shares the device context lifetime with its streams and allocations. The last
// owner releases device-local caches before aclrtResetDevice.
class Device {
public:
    explicit Device(int ordinal = 0);
    ~Device();
    Device(const Device &) = delete;
    Device & operator=(const Device &) = delete;

    int ordinal() const { return ordinal_; }
    void activate() const;

private:
    friend class Stream;
    int ordinal_;
    std::shared_ptr<DeviceLifetime> lifetime_;
};

// Owns one aclrtStream.
class Stream {
public:
    explicit Stream(const Device & device);
    ~Stream();
    Stream(const Stream &) = delete;
    Stream & operator=(const Stream &) = delete;

    void synchronize() const;
    // Explicit opt-in batching. Only one stream on a device may enqueue while
    // this mode is active. Host copies and memset drain the owning stream.
    void begin_deferred();
    void end_deferred();
    bool deferred() const { return deferred_; }
    void submitted() const;
    bool pending() const { return pending_; }
    void defer_cleanup(std::function<void()> cleanup) const;
    int device_ordinal() const { return ordinal_; }
    void activate() const;
    void * handle() const { activate(); return stream_; }

private:
    void * stream_ = nullptr;
    int ordinal_ = -1;
    std::shared_ptr<DeviceLifetime> lifetime_;
    bool deferred_ = false;
    std::thread::id deferred_thread_;
    mutable bool pending_ = false;
    mutable std::vector<std::function<void()>> cleanups_;
    mutable uint64_t deferred_submissions_ = 0, deferred_drains_ = 0, deferred_cleanups_ = 0;
    mutable double deferred_wait_seconds_ = 0;
};

class DeferredScope {
public:
    explicit DeferredScope(Stream& stream, bool enabled = true);
    ~DeferredScope();
    DeferredScope(const DeferredScope&) = delete;
    DeferredScope& operator=(const DeferredScope&) = delete;
    void finish();
private:
    Stream* stream_ = nullptr;
};

// Fence only an active deferred stream, preserving the caller's device context.
void fence_device(int ordinal);
// Metadata stays alive until the corresponding queued work has completed.
void defer_device_cleanup(int ordinal, std::function<void()> cleanup);
void zero_device(void* pointer, size_t bytes, int ordinal);

// Raw copies for a pointer inside a larger tensor (a row, a scalar).
void copy_to_host(const void * device_ptr, void * host, size_t bytes);
void copy_to_device(void * device_ptr, const void * host, size_t bytes);

// Pure device-to-device copies. Cross-device copies require supported P2P and
// throw if access cannot be enabled; there is no host staging fallback. Sources
// and destinations are synchronized, and the destination remains current.
bool peer_access_supported(int dst_device, int src_device);
void enable_peer_access(int dst_device, int src_device);
void copy_peer(void * dst, int dst_device, const void * src, int src_device, size_t bytes);

// Owns one aclrtMalloc block.
class Buffer {
public:
    explicit Buffer(size_t bytes);
    ~Buffer();
    Buffer(const Buffer &) = delete;
    Buffer & operator=(const Buffer &) = delete;

    void * data() const { return data_; }
    size_t size() const { return bytes_; }
    int device_ordinal() const { return ordinal_; }
    void activate() const;

    void copy_from_host(const void * host, size_t bytes);
    void copy_to_host(void * host, size_t bytes) const;

private:
    void * data_ = nullptr;
    size_t bytes_ = 0;
    int ordinal_ = -1;
    std::shared_ptr<DeviceLifetime> lifetime_;
};

void copy_peer(Buffer & dst, const Buffer & src, size_t bytes, size_t dst_offset = 0, size_t src_offset = 0);

} // namespace strata::cann

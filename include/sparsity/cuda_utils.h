#pragma once

#ifdef WITH_CUDA

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace sparsity {

// Abort-on-error macro for CUDA calls.
// Throws std::runtime_error so errors propagate through C++ cleanly.
#define SPARSITY_CUDA_CHECK(call)                                                            \
    do {                                                                                     \
        cudaError_t _err = (call);                                                           \
        if (_err != cudaSuccess) {                                                           \
            throw std::runtime_error(std::string("CUDA error in " __FILE__ ":")             \
                                     + std::to_string(__LINE__) + " — "                     \
                                     + cudaGetErrorString(_err));                            \
        }                                                                                    \
    } while (0)

// Returns the number of CUDA-capable devices visible to the process.
int cuda_device_count();

// RAII wrapper for device memory allocated with cudaMalloc.
// Not copyable; movable.
template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;

    explicit DeviceBuffer(size_t n) { alloc(n); }

    ~DeviceBuffer() { free(); }

    DeviceBuffer(const DeviceBuffer&)            = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept : ptr_(other.ptr_), n_(other.n_) {
        other.ptr_ = nullptr;
        other.n_   = 0;
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            free();
            ptr_       = other.ptr_;
            n_         = other.n_;
            other.ptr_ = nullptr;
            other.n_   = 0;
        }
        return *this;
    }

    void alloc(size_t n) {
        free();
        SPARSITY_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr_), n * sizeof(T)));
        n_ = n;
    }

    void free() {
        if (ptr_) {
            cudaFree(ptr_);
            ptr_ = nullptr;
            n_   = 0;
        }
    }

    void copy_from_host(const T* src, size_t n) {
        SPARSITY_CUDA_CHECK(cudaMemcpy(ptr_, src, n * sizeof(T), cudaMemcpyHostToDevice));
    }

    void copy_to_host(T* dst, size_t n) const {
        SPARSITY_CUDA_CHECK(cudaMemcpy(dst, ptr_, n * sizeof(T), cudaMemcpyDeviceToHost));
    }

    T*       get()        { return ptr_; }
    const T* get() const  { return ptr_; }
    size_t   size() const { return n_; }
    bool     empty() const { return n_ == 0; }

private:
    T*     ptr_ = nullptr;
    size_t n_   = 0;
};

} // namespace sparsity

#endif // WITH_CUDA

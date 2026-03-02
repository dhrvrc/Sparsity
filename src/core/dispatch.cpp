#include "sparsity/cuda_utils.h"

#ifdef WITH_CUDA
#include <cuda_runtime.h>

namespace sparsity {

int cuda_device_count() {
    int count = 0;
    cudaGetDeviceCount(&count);
    return count;
}

} // namespace sparsity

#endif // WITH_CUDA

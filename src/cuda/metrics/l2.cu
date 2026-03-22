#include "sparsity/cuda_dense.h"
#include "sparsity/cuda_utils.h"

#include <cuda_runtime.h>
#include <cstdint>

namespace sparsity {

// ---------------------------------------------------------------------------
// l2_batch_kernel
//
// Computes squared-then-rooted L2 distances for n_queries query vectors
// against n_db database vectors, writing results to distances[n_queries * n_db].
//
// Grid  : (ceil(n_db / CUDA_BLOCK_ROWS), n_queries)
// Block : (CUDA_BLOCK_DIM=32, CUDA_BLOCK_ROWS=4)
//
// n_db tiles use gridDim.x (limit 2^31-1); n_queries uses gridDim.y (limit
// 65535).  This avoids the gridDim.y overflow that occurs when n_db > 65535*4.
//
// Each block handles CUDA_BLOCK_ROWS (query, db_row) pairs.
// Each pair is owned by one warp (threadIdx.y selects the pair; threadIdx.x
// is the lane).  Threads stride over the dimension in steps of CUDA_BLOCK_DIM,
// accumulate the squared difference, then reduce within the warp via
// __shfl_down_sync.  Thread 0 of each warp writes sqrtf(acc) to output.
// ---------------------------------------------------------------------------
__global__ void l2_batch_kernel(const float* __restrict__ queries,
                                const float* __restrict__ db,
                                float*       __restrict__ distances,
                                uint32_t n_db,
                                uint32_t dim)
{
    const uint32_t q_row  = blockIdx.y;
    const uint32_t db_row = blockIdx.x * CUDA_BLOCK_ROWS + threadIdx.y;

    if (db_row >= n_db) return;

    const float* q   = queries + (size_t)q_row  * dim;
    const float* row = db      + (size_t)db_row * dim;

    // Accumulate squared differences for this thread's slice of dim.
    float acc = 0.0f;
    for (uint32_t d = threadIdx.x; d < dim; d += CUDA_BLOCK_DIM) {
        const float diff = q[d] - row[d];
        acc += diff * diff;
    }

    // Warp-level reduction.  CUDA_BLOCK_DIM == 32 == warp size, so all lanes
    // in threadIdx.y-th warp participate.  Mask 0xffffffff is safe because
    // the block is always launched with a full warp per row.
    acc += __shfl_down_sync(0xffffffff, acc, 16);
    acc += __shfl_down_sync(0xffffffff, acc, 8);
    acc += __shfl_down_sync(0xffffffff, acc, 4);
    acc += __shfl_down_sync(0xffffffff, acc, 2);
    acc += __shfl_down_sync(0xffffffff, acc, 1);

    if (threadIdx.x == 0) {
        distances[(size_t)q_row * n_db + db_row] = sqrtf(acc);
    }
}

// ---------------------------------------------------------------------------
// Host launcher
// ---------------------------------------------------------------------------
void cuda_l2_batch(const float* d_queries,
                   const float* d_db,
                   float*       d_distances,
                   uint32_t     n_queries,
                   uint32_t     n_db,
                   uint32_t     dim)
{
    if (n_queries == 0 || n_db == 0) return;

    const dim3 block(CUDA_BLOCK_DIM, CUDA_BLOCK_ROWS);
    const dim3 grid((n_db + CUDA_BLOCK_ROWS - 1) / CUDA_BLOCK_ROWS, n_queries);

    l2_batch_kernel<<<grid, block>>>(d_queries, d_db, d_distances, n_db, dim);
    SPARSITY_CUDA_CHECK(cudaGetLastError());
}

} // namespace sparsity

#include "sparsity/cuda_dense.h"
#include "sparsity/cuda_utils.h"

#include <cuda_runtime.h>
#include <cstdint>

namespace sparsity {

// ---------------------------------------------------------------------------
// cosine_batch_kernel
//
// Computes cosine distances for n_queries query vectors against n_db database
// vectors, writing distance = 1 - dot(q, db_row) to distances[n_queries*n_db].
//
// IMPORTANT: both queries and db must be unit-normalised before calling.
//   - db vectors: normalised at DenseIndex::add time (same as CPU path).
//   - query vectors: normalised on the host inside dispatch_dense_search before
//     the device transfer.
//
// With pre-normalised vectors, cosine distance reduces to 1 - dot product,
// which has the same kernel structure as L2 but with a dot accumulator.
//
// Grid / block / reduction: identical to l2_batch_kernel (n_db tiles in x,
// n_queries in y to avoid the 65535 gridDim.y limit for large databases).
// ---------------------------------------------------------------------------
__global__ void cosine_batch_kernel(const float* __restrict__ queries,
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

    // Accumulate dot product for this thread's slice of dim.
    float acc = 0.0f;
    for (uint32_t d = threadIdx.x; d < dim; d += CUDA_BLOCK_DIM) {
        acc += q[d] * row[d];
    }

    // Warp-level reduction (same as L2 kernel).
    acc += __shfl_down_sync(0xffffffff, acc, 16);
    acc += __shfl_down_sync(0xffffffff, acc, 8);
    acc += __shfl_down_sync(0xffffffff, acc, 4);
    acc += __shfl_down_sync(0xffffffff, acc, 2);
    acc += __shfl_down_sync(0xffffffff, acc, 1);

    // acc is now the full dot product (cosine similarity).
    // Convert to distance: closer vectors → smaller distance.
    if (threadIdx.x == 0) {
        distances[(size_t)q_row * n_db + db_row] = 1.0f - acc;
    }
}

// ---------------------------------------------------------------------------
// Host launcher
// ---------------------------------------------------------------------------
void cuda_cosine_batch(const float* d_queries,
                       const float* d_db,
                       float*       d_distances,
                       uint32_t     n_queries,
                       uint32_t     n_db,
                       uint32_t     dim)
{
    if (n_queries == 0 || n_db == 0) return;

    const dim3 block(CUDA_BLOCK_DIM, CUDA_BLOCK_ROWS);
    const dim3 grid((n_db + CUDA_BLOCK_ROWS - 1) / CUDA_BLOCK_ROWS, n_queries);

    cosine_batch_kernel<<<grid, block>>>(d_queries, d_db, d_distances, n_db, dim);
    SPARSITY_CUDA_CHECK(cudaGetLastError());
}

} // namespace sparsity

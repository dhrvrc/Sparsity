#include "sparsity/cuda_sparse.h"
#include "sparsity/cuda_utils.h"

#include <cuda_runtime.h>
#include <cstdint>

namespace sparsity {

// ---------------------------------------------------------------------------
// tanimoto_batch_kernel
//
// Computes Tanimoto similarity for n_queries binary query vectors against n_db
// binary database vectors, writing NEGATIVE similarity to distances[n_queries*n_db].
//
// We store -sim so that ascending top-k (shared with L2/cosine) naturally
// picks the highest-similarity (most similar) vectors first.  The caller
// un-negates the output before returning to the user.
//
// Grid : (ceil(n_db / CUDA_BLOCK_ROWS), n_queries)  — same convention as L2/cosine
// Block: (CUDA_BLOCK_DIM=32, CUDA_BLOCK_ROWS=4)     — one warp per (query, db_row)
//
// Each warp strides over the wpr (words per row) dimension, accumulating
// intersection and union popcount via warp-shuffle reduction — the same
// pattern used in the dense L2 and cosine kernels.
// ---------------------------------------------------------------------------
__global__ void tanimoto_batch_kernel(const uint64_t* __restrict__ queries,
                                       const uint64_t* __restrict__ db,
                                       float*          __restrict__ distances,
                                       uint32_t n_db,
                                       uint32_t wpr)
{
    const uint32_t q_row  = blockIdx.y;
    const uint32_t db_row = blockIdx.x * CUDA_BLOCK_ROWS + threadIdx.y;

    if (db_row >= n_db) return;

    const uint64_t* q   = queries + (size_t)q_row  * wpr;
    const uint64_t* row = db      + (size_t)db_row * wpr;

    // Each thread accumulates popcount for its slice of the wpr dimension.
    uint32_t inter = 0;
    uint32_t uni   = 0;
    for (uint32_t w = threadIdx.x; w < wpr; w += CUDA_BLOCK_DIM) {
        inter += __popcll(q[w] & row[w]);
        uni   += __popcll(q[w] | row[w]);
    }

    // Warp-level reduction over intersection and union counts.
    // CUDA_BLOCK_DIM == 32 == warp size, so this covers the full warp.
    inter += __shfl_down_sync(0xffffffff, inter, 16);
    inter += __shfl_down_sync(0xffffffff, inter,  8);
    inter += __shfl_down_sync(0xffffffff, inter,  4);
    inter += __shfl_down_sync(0xffffffff, inter,  2);
    inter += __shfl_down_sync(0xffffffff, inter,  1);

    uni += __shfl_down_sync(0xffffffff, uni, 16);
    uni += __shfl_down_sync(0xffffffff, uni,  8);
    uni += __shfl_down_sync(0xffffffff, uni,  4);
    uni += __shfl_down_sync(0xffffffff, uni,  2);
    uni += __shfl_down_sync(0xffffffff, uni,  1);

    if (threadIdx.x == 0) {
        const float sim = (uni == 0) ? 1.0f
                                     : static_cast<float>(inter) / static_cast<float>(uni);
        // Store negative similarity: ascending top-k gives most-similar first.
        distances[(size_t)q_row * n_db + db_row] = -sim;
    }
}

// ---------------------------------------------------------------------------
// Host launcher
// ---------------------------------------------------------------------------
void cuda_tanimoto_batch(const uint64_t* d_queries,
                          const uint64_t* d_db,
                          float*          d_distances,
                          uint32_t        n_queries,
                          uint32_t        n_db,
                          uint32_t        wpr)
{
    if (n_queries == 0 || n_db == 0) return;

    const dim3 block(CUDA_BLOCK_DIM, CUDA_BLOCK_ROWS);
    const dim3 grid((n_db + CUDA_BLOCK_ROWS - 1) / CUDA_BLOCK_ROWS, n_queries);

    tanimoto_batch_kernel<<<grid, block>>>(d_queries, d_db, d_distances, n_db, wpr);
    SPARSITY_CUDA_CHECK(cudaGetLastError());
}

} // namespace sparsity

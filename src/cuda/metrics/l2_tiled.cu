#include "sparsity/cuda_dense.h"
#include "sparsity/cuda_utils.h"

#include <cuda_runtime.h>
#include <cstdint>

namespace sparsity {

// ---------------------------------------------------------------------------
// l2_tiled_kernel
//
// Shared-memory tiled L2 distance computation.  Unlike the warp-per-row
// l2_batch_kernel, one block here computes an (L2_TILE × L2_TILE) sub-matrix
// of the full [n_queries × n_db] distance matrix simultaneously.
//
// Thread (tx, ty) in the block is responsible for:
//   query row  : blockIdx.y * L2_TILE + ty
//   db    row  : blockIdx.x * L2_TILE + tx
//
// Dimension is traversed in L2_TILE-wide chunks.  Per chunk, all threads
// cooperatively load two shared-memory tiles:
//   smem_q [ty][tx]  ← queries[(q_base+ty)*dim + d+tx]
//   smem_db[ty][tx]  ← db    [(db_base+ty)*dim + d+tx]
//
// After a __syncthreads(), thread (tx, ty) accumulates:
//   for k in 0..L2_TILE-1:
//     diff = smem_q[ty][k] - smem_db[tx][k]
//     acc += diff * diff
//
// Key: smem_q[ty][k] is the k-th dim element of query (q_base+ty),
//      smem_db[tx][k] is the k-th dim element of db row (db_base+tx).
//      Each thread reads its own row from smem_db[tx][*] while broadcasting
//      smem_q[ty][*] across all threads in the same ty group.
//
// Out-of-bounds tiles are padded with 0 (safe for L2: (0−0)²=0).
//
// Shared memory: 2 × L2_TILE × (L2_TILE+1) × 4 bytes
//                = 2 × 16 × 17 × 4 = 2176 bytes
//                (+1 column pads stride to 17, making the stride coprime to
//                the 32-bank count, eliminating the 8-way bank conflicts that
//                would arise with stride 16.)
//
// Grid : (ceil(n_db/L2_TILE), ceil(n_queries/L2_TILE))
// Block: (L2_TILE, L2_TILE) = 256 threads
// ---------------------------------------------------------------------------

constexpr int L2_TILE = 16;

__global__ void l2_tiled_kernel(const float* __restrict__ queries,
                                 const float* __restrict__ db,
                                 float*       __restrict__ distances,
                                 uint32_t n_queries,
                                 uint32_t n_db,
                                 uint32_t dim)
{
    __shared__ float smem_q [L2_TILE][L2_TILE + 1];   // +1: bank-conflict padding
    __shared__ float smem_db[L2_TILE][L2_TILE + 1];

    const uint32_t tx = threadIdx.x;   // db-tile axis (0 .. L2_TILE-1)
    const uint32_t ty = threadIdx.y;   // query-tile axis (0 .. L2_TILE-1)

    const uint32_t q_base  = blockIdx.y * L2_TILE;
    const uint32_t db_base = blockIdx.x * L2_TILE;

    const uint32_t q_row  = q_base  + ty;
    const uint32_t db_row = db_base + tx;

    float acc = 0.0f;

    // ── Outer loop: stride over the dimension in chunks of L2_TILE ──────────
    for (uint32_t d = 0; d < dim; d += L2_TILE) {

        // Each thread loads exactly one float into each shared-memory tile.
        //   smem_q [ty][tx] = query   row (q_base+ty),  dim element (d+tx)
        //   smem_db[ty][tx] = db      row (db_base+ty), dim element (d+tx)
        smem_q[ty][tx] = (q_base + ty < n_queries && d + tx < dim)
                         ? queries[static_cast<size_t>(q_base  + ty) * dim + d + tx]
                         : 0.0f;

        smem_db[ty][tx] = (db_base + ty < n_db && d + tx < dim)
                          ? db[static_cast<size_t>(db_base + ty) * dim + d + tx]
                          : 0.0f;

        __syncthreads();

        // ── Inner loop: accumulate squared differences ───────────────────────
        // Thread (tx, ty) computes the L2² contribution of this dim chunk for
        // the pair (query q_row, db row db_row).
        //
        //   smem_q [ty][k]  = query q_row's   k-th element in chunk
        //   smem_db[tx][k]  = db    db_row's  k-th element in chunk
        //                     (loaded earlier as smem_db[row=tx][col=k]
        //                      by the thread whose ty==tx in the load phase)
        #pragma unroll
        for (int k = 0; k < L2_TILE; ++k) {
            const float diff = smem_q[ty][k] - smem_db[tx][k];
            acc += diff * diff;
        }

        __syncthreads();
    }

    if (q_row < n_queries && db_row < n_db) {
        distances[static_cast<size_t>(q_row) * n_db + db_row] = sqrtf(acc);
    }
}

// ---------------------------------------------------------------------------
// Host launcher
// ---------------------------------------------------------------------------
void cuda_l2_tiled_batch(const float* d_queries,
                          const float* d_db,
                          float*       d_distances,
                          uint32_t     n_queries,
                          uint32_t     n_db,
                          uint32_t     dim)
{
    if (n_queries == 0 || n_db == 0) return;

    const dim3 block(L2_TILE, L2_TILE);
    const dim3 grid((n_db      + L2_TILE - 1) / L2_TILE,
                    (n_queries + L2_TILE - 1) / L2_TILE);

    l2_tiled_kernel<<<grid, block>>>(d_queries, d_db, d_distances, n_queries, n_db, dim);
    SPARSITY_CUDA_CHECK(cudaGetLastError());
}

} // namespace sparsity

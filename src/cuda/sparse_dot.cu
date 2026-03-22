#include "sparsity/cuda_sparse.h"
#include "sparsity/cuda_utils.h"

#include <cuda_runtime.h>
#include <cstdint>

namespace sparsity {

// Block size for the sparse dot kernel: one warp per (query, db_block).
static constexpr uint32_t BELL_B = 32;

// ---------------------------------------------------------------------------
// sparse_dot_kernel
//
// Computes dot(query_q, db_row_d) for all (q, d) pairs using Blocked-ELL
// layout for the db vectors and shared memory binary search for query lookup.
//
// Grid : (n_db_blocks, n_queries)   — one block per (db_block, query)
// Block: (BELL_B=32, 1)             — one warp; thread lane = db_row within block
//
// Shared memory layout (allocated by launcher):
//   uint32_t s_q_idx[max_q_nnz]  — sorted query column indices
//   float    s_q_val[max_q_nnz]  — corresponding query values
//
// Algorithm per thread (lane):
//   1. All 32 lanes cooperatively load the query non-zeros into shared memory.
//   2. Each lane owns one db row: iterates over k = 0..block_max_nnz-1.
//      For each Blocked-ELL entry (col, val):
//        - Binary search s_q_idx for col.
//        - If found, accumulate val * s_q_val[match] into dot.
//   3. Thread writes dot to out_dots[q * n_db + db_row].
//
// Coalesced access: all lanes read bell_col[base + k*BELL_B + 0..31] for the
// same k simultaneously — 32 consecutive uint32_t → one cache line.
// No warp divergence: all lanes iterate max_nnz_b times (padded sentinels
// have val=0.0f so they are skipped cheaply).
// ---------------------------------------------------------------------------
__global__ void sparse_dot_kernel(
    const uint32_t* __restrict__ d_q_idx,     // [n_queries * max_q_nnz]
    const float*    __restrict__ d_q_val,     // [n_queries * max_q_nnz]
    const uint32_t* __restrict__ d_q_nnz,    // [n_queries]
    const uint32_t* __restrict__ d_bell_col,  // Blocked-ELL column indices
    const float*    __restrict__ d_bell_val,  // Blocked-ELL values
    const uint32_t* __restrict__ d_bell_off,  // [n_blocks] block start offsets
    const uint32_t* __restrict__ d_bell_mnnz, // [n_blocks] max nnz per block
    float*          __restrict__ d_dots,      // [n_queries * n_db] output
    uint32_t n_db,
    uint32_t max_q_nnz)
{
    const uint32_t q_row   = blockIdx.y;
    const uint32_t block_b = blockIdx.x;
    const uint32_t lane    = threadIdx.x;   // in [0, BELL_B)
    const uint32_t db_row  = block_b * BELL_B + lane;

    // Shared memory: query indices then query values.
    extern __shared__ char shmem[];
    uint32_t* s_q_idx = reinterpret_cast<uint32_t*>(shmem);
    float*    s_q_val = reinterpret_cast<float*>(s_q_idx + max_q_nnz);

    const uint32_t this_q_nnz = d_q_nnz[q_row];

    // Cooperatively load query non-zeros into shared memory.
    for (uint32_t i = lane; i < this_q_nnz; i += BELL_B) {
        s_q_idx[i] = d_q_idx[q_row * max_q_nnz + i];
        s_q_val[i] = d_q_val[q_row * max_q_nnz + i];
    }
    __syncthreads();

    // Ghost lanes (db_row >= n_db) still participated in the query load above.
    if (db_row >= n_db) return;

    const uint32_t b_offset  = d_bell_off[block_b];
    const uint32_t b_max_nnz = d_bell_mnnz[block_b];

    float dot = 0.0f;

    for (uint32_t k = 0; k < b_max_nnz; ++k) {
        const float val = d_bell_val[b_offset + k * BELL_B + lane];
        // Skip padding sentinels (val == 0.0f means this slot is padded).
        if (val == 0.0f) continue;

        const uint32_t col = d_bell_col[b_offset + k * BELL_B + lane];

        // Binary search for col in s_q_idx[0..this_q_nnz).
        // Query indices are sorted ascending (CSR invariant).
        uint32_t lo = 0, hi = this_q_nnz;
        while (lo < hi) {
            const uint32_t mid = (lo + hi) >> 1;
            if (s_q_idx[mid] < col) lo = mid + 1;
            else                    hi = mid;
        }
        if (lo < this_q_nnz && s_q_idx[lo] == col) {
            dot += val * s_q_val[lo];
        }
    }

    d_dots[(size_t)q_row * n_db + db_row] = dot;
}

// ---------------------------------------------------------------------------
// sparse_cosine_finalize_kernel
//
// In-place: dots[q,d] = 1 - dots[q,d] / (q_norms[q] * db_norms[d])
// Grid / Block: simple 2D covering (n_queries, n_db).
// ---------------------------------------------------------------------------
__global__ void sparse_cosine_finalize_kernel(
    float*       __restrict__ d_dots,
    const float* __restrict__ d_q_norms,
    const float* __restrict__ d_db_norms,
    uint32_t n_queries,
    uint32_t n_db)
{
    const uint32_t q  = blockIdx.y * blockDim.y + threadIdx.y;
    const uint32_t db = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= n_queries || db >= n_db) return;

    const float qn   = d_q_norms[q];
    const float dn   = d_db_norms[db];
    const float dot  = d_dots[(size_t)q * n_db + db];
    const float dist = (qn == 0.0f || dn == 0.0f) ? 1.0f : 1.0f - dot / (qn * dn);
    d_dots[(size_t)q * n_db + db] = dist;
}

// ---------------------------------------------------------------------------
// sparse_l2_finalize_kernel
//
// In-place: dots[q,d] = sqrt(max(0, nq2[q] + nd2[d] - 2*dots[q,d]))
// ---------------------------------------------------------------------------
__global__ void sparse_l2_finalize_kernel(
    float*       __restrict__ d_dots,
    const float* __restrict__ d_q_norms_sq,
    const float* __restrict__ d_db_norms_sq,
    uint32_t n_queries,
    uint32_t n_db)
{
    const uint32_t q  = blockIdx.y * blockDim.y + threadIdx.y;
    const uint32_t db = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= n_queries || db >= n_db) return;

    const float dot    = d_dots[(size_t)q * n_db + db];
    float       dist_sq = d_q_norms_sq[q] + d_db_norms_sq[db] - 2.0f * dot;
    if (dist_sq < 0.0f) dist_sq = 0.0f;
    d_dots[(size_t)q * n_db + db] = sqrtf(dist_sq);
}

// ---------------------------------------------------------------------------
// Host launchers
// ---------------------------------------------------------------------------

void cuda_sparse_dot_batch(const uint32_t* d_q_idx,
                            const float*    d_q_val,
                            const uint32_t* d_q_nnz,
                            const uint32_t* d_bell_col,
                            const float*    d_bell_val,
                            const uint32_t* d_bell_off,
                            const uint32_t* d_bell_mnnz,
                            float*          d_dots,
                            uint32_t        n_queries,
                            uint32_t        n_db,
                            uint32_t        max_q_nnz,
                            uint32_t        n_blocks)
{
    if (n_queries == 0 || n_db == 0 || n_blocks == 0) return;

    const dim3 block(BELL_B, 1);
    const dim3 grid(n_blocks, n_queries);

    // Shared memory: max_q_nnz * (uint32_t + float)
    const size_t shmem = static_cast<size_t>(max_q_nnz) * (sizeof(uint32_t) + sizeof(float));

    sparse_dot_kernel<<<grid, block, shmem>>>(
        d_q_idx, d_q_val, d_q_nnz,
        d_bell_col, d_bell_val, d_bell_off, d_bell_mnnz,
        d_dots, n_db, max_q_nnz);
    SPARSITY_CUDA_CHECK(cudaGetLastError());
}

void cuda_sparse_cosine_finalize(float*       d_dots,
                                  const float* d_q_norms,
                                  const float* d_db_norms,
                                  uint32_t     n_queries,
                                  uint32_t     n_db)
{
    const dim3 block(32, 8);
    const dim3 grid((n_db     + block.x - 1) / block.x,
                    (n_queries + block.y - 1) / block.y);

    sparse_cosine_finalize_kernel<<<grid, block>>>(
        d_dots, d_q_norms, d_db_norms, n_queries, n_db);
    SPARSITY_CUDA_CHECK(cudaGetLastError());
}

void cuda_sparse_l2_finalize(float*       d_dots,
                              const float* d_q_norms_sq,
                              const float* d_db_norms_sq,
                              uint32_t     n_queries,
                              uint32_t     n_db)
{
    const dim3 block(32, 8);
    const dim3 grid((n_db     + block.x - 1) / block.x,
                    (n_queries + block.y - 1) / block.y);

    sparse_l2_finalize_kernel<<<grid, block>>>(
        d_dots, d_q_norms_sq, d_db_norms_sq, n_queries, n_db);
    SPARSITY_CUDA_CHECK(cudaGetLastError());
}

} // namespace sparsity

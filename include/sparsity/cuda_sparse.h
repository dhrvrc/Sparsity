#pragma once

#ifdef WITH_CUDA

// GPU launcher declarations for sparse vector search.
// This header is safe to include from .cu files — it uses only C types and
// the cuda_dense.h constants, with no std::vector or other C++ STL headers.
// For the host-side BlockedELLHost struct, see blocked_ell.h (CPU only).

#include "sparsity/cuda_dense.h"  // CUDA_TOPK_BLOCK, CUDA_BLOCK_DIM, CUDA_BLOCK_ROWS

#include <cstdint>

namespace sparsity {

// ---------------------------------------------------------------------------
// Sparse dot product (Blocked-ELL layout for db, padded array for queries)
//
// Grid : (n_blocks, n_queries) — one block per (db_block, query)
// Block: (BELL_B=32, 1)        — one warp; thread lane = db_row in block
//
// d_q_idx     : device uint32_t[n_queries * max_q_nnz]   padded query indices
// d_q_val     : device float[n_queries * max_q_nnz]      padded query values
// d_q_nnz     : device uint32_t[n_queries]                actual nnz per query
// d_bell_col  : device uint32_t[total_bell_entries]       Blocked-ELL col indices
// d_bell_val  : device float[total_bell_entries]          Blocked-ELL values
// d_bell_off  : device uint32_t[n_blocks]                 block start offsets
// d_bell_mnnz : device uint32_t[n_blocks]                 max nnz per block
// d_dots      : device float[n_queries * n_db]            output dot products
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
                            uint32_t        n_blocks);

// ---------------------------------------------------------------------------
// Finalize kernels — in-place update of the dot product matrix
// ---------------------------------------------------------------------------

// Cosine: dots[q,d] = 1 - dots[q,d] / (q_norms[q] * db_norms[d])
void cuda_sparse_cosine_finalize(float*       d_dots,
                                  const float* d_q_norms,
                                  const float* d_db_norms,
                                  uint32_t     n_queries,
                                  uint32_t     n_db);

// L2: dots[q,d] = sqrt(max(0, q_norms_sq[q] + db_norms_sq[d] - 2*dots[q,d]))
void cuda_sparse_l2_finalize(float*       d_dots,
                              const float* d_q_norms_sq,
                              const float* d_db_norms_sq,
                              uint32_t     n_queries,
                              uint32_t     n_db);

// ---------------------------------------------------------------------------
// Tanimoto similarity for binary packed vectors
//
// Stores NEGATIVE similarity (-sim) so ascending top-k gives most-similar first.
// Caller un-negates the output before returning to the user.
//
// Grid : (ceil(n_db / CUDA_BLOCK_ROWS), n_queries)
// Block: (CUDA_BLOCK_DIM=32, CUDA_BLOCK_ROWS=4)
//
// d_queries  : device uint64_t[n_queries * wpr]
// d_db       : device uint64_t[n_db * wpr]
// d_distances: device float[n_queries * n_db]   output: -tanimoto_similarity
// ---------------------------------------------------------------------------
void cuda_tanimoto_batch(const uint64_t* d_queries,
                          const uint64_t* d_db,
                          float*          d_distances,
                          uint32_t        n_queries,
                          uint32_t        n_db,
                          uint32_t        wpr);

} // namespace sparsity

#endif // WITH_CUDA

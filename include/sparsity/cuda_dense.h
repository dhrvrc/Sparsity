#pragma once

#ifdef WITH_CUDA

#include "sparsity/cuda_utils.h"

#include <cstdint>

namespace sparsity {

// ---------------------------------------------------------------------------
// Thread-block constants (shared between kernels and the dispatch layer so the
// caller can compute shared-memory requirements without re-hardcoding them).
// ---------------------------------------------------------------------------

// Distance kernels: BLOCK_DIM threads collaborate on one (query, db_row) pair
// via warp-shuffle reduction; BLOCK_ROWS such pairs run in parallel per block.
constexpr int CUDA_BLOCK_DIM  = 32;
constexpr int CUDA_BLOCK_ROWS = 4;

// Top-k kernel: one block per query, TOPK_BLOCK threads each maintain a local
// max-heap in shared memory.
constexpr int CUDA_TOPK_BLOCK = 32;

// ---------------------------------------------------------------------------
// Distance kernels
// All pointers must be device pointers.  Caller is responsible for H<->D
// transfer and synchronisation.  Both kernels call cudaGetLastError() and
// throw std::runtime_error on failure.
// ---------------------------------------------------------------------------

// Batched L2 distances.
// d_queries : device float[n_queries * dim]
// d_db      : device float[n_db * dim]
// d_distances: device float[n_queries * n_db]  (output, row-major)
void cuda_l2_batch(const float* d_queries,
                   const float* d_db,
                   float*       d_distances,
                   uint32_t     n_queries,
                   uint32_t     n_db,
                   uint32_t     dim);

// Batched cosine distances.
// IMPORTANT: both d_queries and d_db must already be unit-normalised.
// The kernel computes distance = 1 - dot(q_normed, db_normed).
void cuda_cosine_batch(const float* d_queries,
                       const float* d_db,
                       float*       d_distances,
                       uint32_t     n_queries,
                       uint32_t     n_db,
                       uint32_t     dim);

// Batched L2 distances — shared-memory tiled variant.
// Processes (L2_TILE × L2_TILE) output cells per block by staging dimension
// chunks in shared memory, amortising global memory reads across L2_TILE
// queries simultaneously.  Same interface as cuda_l2_batch.
void cuda_l2_tiled_batch(const float* d_queries,
                          const float* d_db,
                          float*       d_distances,
                          uint32_t     n_queries,
                          uint32_t     n_db,
                          uint32_t     dim);

// ---------------------------------------------------------------------------
// Top-k kernel
// Reduces the n_queries × n_db distance matrix to n_queries × k_actual
// (k_actual = min(k, n_db)) on device, keeping only the smallest k distances
// per query row.
//
// Shared memory per block: CUDA_TOPK_BLOCK * k_actual * (sizeof(float) + sizeof(int64_t))
// The caller should verify this fits within the device limit before calling.
//
// d_distances : device float[n_queries * n_db]   (input, row-major)
// d_out_dist  : device float[n_queries * k]       (output)
// d_out_idx   : device int64_t[n_queries * k]     (output, 0-based index into db)
// ---------------------------------------------------------------------------
void cuda_topk_batch(const float* d_distances,
                     uint32_t     n_db,
                     uint32_t     k,
                     float*       d_out_dist,
                     int64_t*     d_out_idx,
                     uint32_t     n_queries);

} // namespace sparsity

#endif // WITH_CUDA

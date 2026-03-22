#pragma once

// Host-side Blocked-ELL data structure and CSR conversion.
// This header must NOT be included from .cu files — it uses std::vector which
// triggers nvcc 11.x C++17 compatibility bugs.  Include only from .cpp files.

#include "sparsity/sparse.h"

#include <cstdint>
#include <vector>

namespace sparsity {

// ---------------------------------------------------------------------------
// Blocked-ELL format (host side)
//
// CSR rows have variable nnz, causing warp divergence on GPU.  Blocked-ELL
// pads rows to a uniform length within blocks of BELL_B rows (= 32 = warp
// size), enabling coalesced, divergence-free access patterns.
//
// For block b covering db rows [b*BELL_B, (b+1)*BELL_B):
//   max_nnz_b = max nnz across those rows
//   Storage  : max_nnz_b * BELL_B entries in column-major order:
//     col_idx[block_offsets[b] + k * BELL_B + lane]  k=0..max_nnz_b-1
//     values [block_offsets[b] + k * BELL_B + lane]
//   Padding  : col_idx=0, value=0.0f  (contribute 0 to dot product)
//
// Column-major ensures all BELL_B threads (one warp) access contiguous
// addresses when iterating over position k → fully coalesced global reads.
// ---------------------------------------------------------------------------
constexpr uint32_t BELL_B = 32;  // block size == warp size

struct BlockedELLHost {
    std::vector<uint32_t> col_idx;       // all blocks concatenated, column-major within block
    std::vector<float>    values;        // same layout
    std::vector<uint32_t> block_offsets; // [n_blocks] start index in col_idx/values per block
    std::vector<uint32_t> block_max_nnz; // [n_blocks] max nnz per block
    uint32_t              n_rows   = 0;
    uint32_t              n_blocks = 0;
};

// Convert a CSR SparseMatrix to Blocked-ELL format on the host.
// Called once per search call (before GPU transfer).
BlockedELLHost csr_to_blocked_ell(const SparseMatrix& csr);

} // namespace sparsity

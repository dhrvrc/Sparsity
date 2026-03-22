#include "sparsity/cuda_sparse.h"
#include "sparsity/sparse.h"

#include <algorithm>
#include <cstdint>

namespace sparsity {

// ---------------------------------------------------------------------------
// csr_to_blocked_ell
//
// Converts a CSR SparseMatrix to Blocked-ELL format on the host.
//
// For each block of BELL_B rows:
//   1. Find max_nnz_b = max nnz across rows in the block.
//   2. For each position k in [0, max_nnz_b): store all BELL_B rows' k-th
//      non-zero (col, val) in contiguous memory (column-major order).
//   3. Rows with fewer than max_nnz_b non-zeros are padded with (0, 0.0f).
//
// Column-major within each block means all BELL_B threads access contiguous
// addresses when they all read position k of their respective rows, giving
// coalesced global memory loads on GPU.
// ---------------------------------------------------------------------------
BlockedELLHost csr_to_blocked_ell(const SparseMatrix& csr) {
    BlockedELLHost bell;
    bell.n_rows   = csr.n_rows;
    bell.n_blocks = (csr.n_rows + BELL_B - 1) / BELL_B;

    bell.block_offsets.resize(bell.n_blocks);
    bell.block_max_nnz.resize(bell.n_blocks);

    uint32_t total_entries = 0;

    // First pass: compute max_nnz per block and total storage needed.
    for (uint32_t b = 0; b < bell.n_blocks; ++b) {
        const uint32_t row_start = b * BELL_B;
        const uint32_t row_end   = std::min(row_start + BELL_B, csr.n_rows);

        uint32_t max_nnz_b = 0;
        for (uint32_t r = row_start; r < row_end; ++r) {
            const uint32_t nnz_r = csr.indptr[r + 1] - csr.indptr[r];
            if (nnz_r > max_nnz_b) max_nnz_b = nnz_r;
        }

        bell.block_offsets[b] = total_entries;
        bell.block_max_nnz[b] = max_nnz_b;
        total_entries += max_nnz_b * BELL_B;
    }

    bell.col_idx.assign(total_entries, 0);
    bell.values.assign(total_entries, 0.0f);

    // Second pass: fill col_idx and values in column-major order within each block.
    for (uint32_t b = 0; b < bell.n_blocks; ++b) {
        const uint32_t row_start  = b * BELL_B;
        const uint32_t row_end    = std::min(row_start + BELL_B, csr.n_rows);
        const uint32_t max_nnz_b  = bell.block_max_nnz[b];
        const uint32_t base       = bell.block_offsets[b];

        for (uint32_t lane = 0; lane < (row_end - row_start); ++lane) {
            const uint32_t r     = row_start + lane;
            const uint32_t start = csr.indptr[r];
            const uint32_t nnz_r = csr.indptr[r + 1] - start;

            for (uint32_t k = 0; k < max_nnz_b; ++k) {
                const uint32_t pos = base + k * BELL_B + lane;
                if (k < nnz_r) {
                    bell.col_idx[pos] = csr.indices[start + k];
                    bell.values[pos]  = csr.values[start + k];
                }
                // else: already zero-initialised (padding sentinel)
            }
        }
        // Ghost lanes (when row_end < row_start + BELL_B) remain zero.
    }

    return bell;
}

} // namespace sparsity

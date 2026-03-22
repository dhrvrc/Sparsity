#include "sparsity/blocked_ell.h"
#include "sparsity/sparse.h"

#include <algorithm>

namespace sparsity {

// ---------------------------------------------------------------------------
// csr_to_blocked_ell
//
// Converts a CSR SparseMatrix to Blocked-ELL format entirely on the host.
// The result is transferred to the GPU by dispatch_sparse_search.
//
// Two-pass algorithm:
//   Pass 1 — compute max_nnz per block and total storage needed.
//   Pass 2 — fill col_idx and values in column-major order within each block.
//
// Ghost lanes (when n_rows is not a multiple of BELL_B) remain zero-padded.
// Padding sentinels (col_idx=0, value=0.0f) contribute 0 to every dot product
// and are cheaply skipped by the GPU kernel.
// ---------------------------------------------------------------------------
BlockedELLHost csr_to_blocked_ell(const SparseMatrix& csr) {
    BlockedELLHost bell;
    bell.n_rows   = csr.n_rows;
    bell.n_blocks = (csr.n_rows + BELL_B - 1) / BELL_B;

    bell.block_offsets.resize(bell.n_blocks);
    bell.block_max_nnz.resize(bell.n_blocks);

    uint32_t total_entries = 0;

    // Pass 1: compute per-block max_nnz and accumulate total storage.
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

    // Zero-init: sentinel entries are already (col=0, val=0.0f).
    bell.col_idx.assign(total_entries, 0u);
    bell.values.assign(total_entries,  0.0f);

    // Pass 2: fill non-padding entries in column-major order within each block.
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
                if (k < nnz_r) {
                    bell.col_idx[base + k * BELL_B + lane] = csr.indices[start + k];
                    bell.values [base + k * BELL_B + lane] = csr.values[start + k];
                }
                // k >= nnz_r: already zero (padding sentinel)
            }
        }
        // Ghost lanes (lane >= row_end-row_start) remain zero.
    }

    return bell;
}

} // namespace sparsity

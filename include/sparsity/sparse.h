#pragma once

#include <cstdint>

namespace sparsity {

// Single sparse vector in CSR format.
// INVARIANT: indices[0..nnz) are sorted strictly ascending.
struct SparseVector {
    const uint32_t* indices = nullptr; // non-zero dimension indices (sorted asc)
    const float*    values  = nullptr; // corresponding non-zero values
    uint32_t        nnz     = 0;       // number of non-zero entries
    uint32_t        dim     = 0;       // total dimensionality (including zeros)
};

// Collection of N sparse vectors stored in CSR format.
// Row i spans columns indices[indptr[i] .. indptr[i+1]).
//
// INVARIANTS:
//   indptr[0] == 0
//   indptr[n_rows] == nnz
//   indices within each row are sorted ascending
//   dim is consistent across all rows (callers enforce this)
//
// This struct is non-owning — see SparseIndex for owned storage.
struct SparseMatrix {
    uint32_t* indptr  = nullptr; // length n_rows + 1
    uint32_t* indices = nullptr; // all column indices concatenated
    float*    values  = nullptr; // all non-zero values concatenated
    uint32_t  n_rows  = 0;
    uint32_t  n_cols  = 0; // total dimensionality
    uint32_t  nnz     = 0; // total non-zeros across all rows

    // View of row i as a SparseVector.
    SparseVector row(uint32_t i) const {
        return {indices + indptr[i], values + indptr[i], indptr[i + 1] - indptr[i], n_cols};
    }
};

} // namespace sparsity

#pragma once

#include <cstdint>

namespace sparsity {

// Row-major dense matrix of float32 values.
// Element at row i, column j: data[i * n_cols + j]
// This struct is non-owning — see DenseIndex for owned storage.
struct DenseMatrix {
    float*   data   = nullptr;
    uint32_t n_rows = 0;
    uint32_t n_cols = 0;

    float*       row(uint32_t i) { return data + static_cast<size_t>(i) * n_cols; }
    const float* row(uint32_t i) const { return data + static_cast<size_t>(i) * n_cols; }

    size_t total_floats() const { return static_cast<size_t>(n_rows) * n_cols; }
};

} // namespace sparsity

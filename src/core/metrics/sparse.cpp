#include "sparsity/metrics.h"
#include "sparsity/sparse.h"

#include <cmath>

namespace sparsity {

// ---------------------------------------------------------------------------
// sparse_dot
//
// Merge-walk over two sorted CSR index arrays.  Accumulates a[i] * b[i] for
// every dimension index i that appears in both vectors.  Dimensions present
// in only one vector contribute zero (their value in the other is 0).
// O(nnz_a + nnz_b).
// ---------------------------------------------------------------------------
float sparse_dot(SparseVector a, SparseVector b) {
    float    dot = 0.0f;
    uint32_t i   = 0;
    uint32_t j   = 0;

    while (i < a.nnz && j < b.nnz) {
        if (a.indices[i] == b.indices[j]) {
            dot += a.values[i] * b.values[j];
            ++i;
            ++j;
        } else if (a.indices[i] < b.indices[j]) {
            ++i;
        } else {
            ++j;
        }
    }

    return dot;
}

// ---------------------------------------------------------------------------
// sparse_cosine_distance
//
// 1 - dot(a, b) / (norm_a * norm_b).
// norm_a and norm_b must be the L2 norms of a and b, pre-computed at index
// build time and passed in by the caller to avoid recomputation.
// Returns 1.0 (maximum distance) when either vector has zero norm.
// ---------------------------------------------------------------------------
float sparse_cosine_distance(SparseVector a, SparseVector b,
                              float norm_a, float norm_b) {
    if (norm_a == 0.0f || norm_b == 0.0f) return 1.0f;
    return 1.0f - sparse_dot(a, b) / (norm_a * norm_b);
}

// ---------------------------------------------------------------------------
// sparse_l2_distance
//
// Uses the identity: ||a - b||^2 = ||a||^2 + ||b||^2 - 2 * dot(a, b).
// This avoids iterating over all dimensions — only the non-zeros are touched
// via sparse_dot, which is O(nnz_a + nnz_b) instead of O(dim).
//
// norm_sq_a / norm_sq_b are the squared norms (norm^2), pre-computed at
// index build time.  A small negative guard handles floating-point rounding
// that can produce dist_sq slightly below zero for identical vectors.
// ---------------------------------------------------------------------------
float sparse_l2_distance(SparseVector a, SparseVector b,
                          float norm_sq_a, float norm_sq_b) {
    const float dot     = sparse_dot(a, b);
    const float dist_sq = norm_sq_a + norm_sq_b - 2.0f * dot;
    return std::sqrt(dist_sq < 0.0f ? 0.0f : dist_sq);
}

} // namespace sparsity

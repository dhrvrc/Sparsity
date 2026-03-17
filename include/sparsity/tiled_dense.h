#pragma once

#include "sparsity/index.h"
#include "sparsity/types.h"

#include <cstdint>
#include <vector>

namespace sparsity {

// ---------------------------------------------------------------------------
// TiledDenseIndex — cache-tiled brute-force search for dense float vectors
//
// Implements the same interface as DenseIndex but uses loop tiling to reduce
// memory bandwidth: instead of making a full DB pass per query, it processes
// TILE_Q queries in a single DB pass so each database vector is loaded from
// memory once per tile (amortised across TILE_Q queries).
//
// For L2 the norm decomposition identity is exploited:
//   ||qi - dj||² = ||qi||² + ||dj||² - 2·dot(qi, dj)
// Database norms ||dj||² are pre-computed at add() time. This turns batch L2
// into the same dot-product kernel used by Cosine, giving a uniform tiled
// inner loop for both metrics.
//
// Supported metrics: L2, COSINE
// Coexists with DenseIndex for benchmarking comparison.
// ---------------------------------------------------------------------------
class TiledDenseIndex : public IndexBase {
public:
    // Number of queries processed per DB pass.
    // Each DB vector is loaded once and its contribution accumulated into
    // TILE_Q independent dot-product accumulators simultaneously.
    static constexpr uint32_t TILE_Q = 4;

    explicit TiledDenseIndex(Metric metric);
    ~TiledDenseIndex() override = default;

    // Add n vectors of dimension dim.  All subsequent adds must use same dim.
    // For L2: stores raw vectors and pre-computes ||di||² per vector.
    // For Cosine: pre-normalises to unit length (same as DenseIndex).
    void add(const float* data, uint32_t n, uint32_t dim);

    // Approximate-exact k-NN search using tiled distance computation.
    // queries must be float[n_queries * dim()].
    SearchResult search(const float* queries, uint32_t n_queries, uint32_t k) const;

    uint32_t size()   const override { return mat_.n_rows; }
    uint32_t dim()    const override { return mat_.n_cols; }
    Metric   metric() const override { return metric_; }

    void validate() const override;

private:
    Metric             metric_;
    DenseMatrix        mat_;          // non-owning view into storage_
    std::vector<float> storage_;      // owns the vector data (raw or normalised)
    std::vector<float> db_norms_sq_;  // ||di||² per indexed vector; L2 only
};

} // namespace sparsity

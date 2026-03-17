#pragma once

#include "sparsity/index.h"
#include "sparsity/types.h"

#include <cstdint>
#include <vector>

namespace sparsity {

// ---------------------------------------------------------------------------
// IVFIndex — approximate nearest-neighbour search for dense float vectors
//
// Based on the Inverted File Index (IVF) algorithm:
//   1. train()  — cluster the dataset into n_lists Voronoi cells (k-means)
//   2. add()    — assign each vector to its nearest centroid's inverted list
//   3. search() — probe the n_probe nearest centroids per query, brute-force
//                 only those lists, return top-k candidates
//
// Supported metrics: L2, COSINE
//
// n_probe controls the recall/speed trade-off:
//   n_probe == n_lists → exact (same results as DenseIndex brute-force)
//   n_probe == 1       → fastest, lowest recall
//
// Typical usage:
//   IVFIndex idx(Metric::L2, /*n_lists=*/256, /*n_probe=*/16);
//   idx.train(train_data, n_train, dim);
//   idx.add(data, n, dim);
//   auto result = idx.search(queries, n_queries, k);
// ---------------------------------------------------------------------------
class IVFIndex : public IndexBase {
public:
    // Construct an IVF index.
    // n_lists: number of Voronoi cells (clusters). Rule of thumb: sqrt(N).
    // n_probe: cells to search per query. Higher = better recall, slower search.
    // Throws std::invalid_argument if metric is not L2 or COSINE,
    // or if n_lists == 0 or n_probe == 0.
    IVFIndex(Metric metric, uint32_t n_lists = 100, uint32_t n_probe = 10);
    ~IVFIndex() override = default;

    // Phase 1: run k-means on training data to establish centroids.
    // Training data need not be the same as the indexed data, but should be
    // drawn from the same distribution.
    // Throws std::invalid_argument if already trained, dim == 0, or n < n_lists.
    void train(const float* data, uint32_t n, uint32_t dim);

    // Phase 2: insert vectors into their nearest centroid's inverted list.
    // train() must have been called first.
    // Throws std::runtime_error  if not yet trained.
    // Throws std::invalid_argument on dimension mismatch.
    void add(const float* data, uint32_t n, uint32_t dim);

    // Approximate k-NN search over all indexed vectors.
    // Probes the n_probe nearest centroids per query.
    // Throws std::runtime_error if not trained or index is empty.
    SearchResult search(const float* queries, uint32_t n_queries, uint32_t k) const;

    // --- Accessors ---
    bool     is_trained() const noexcept { return trained_; }
    uint32_t n_lists()    const noexcept { return n_lists_; }
    uint32_t n_probe()    const noexcept { return n_probe_; }

    // Adjust n_probe without rebuilding the index (takes effect on the next search).
    // Clamps the value to [1, n_lists].
    void set_n_probe(uint32_t n_probe) noexcept;

    uint32_t size()   const override { return total_; }
    uint32_t dim()    const override { return dim_; }
    Metric   metric() const override { return metric_; }

    void validate() const override;

private:
    // Per-centroid storage.
    struct InvertedList {
        std::vector<float>   vectors;  // row-major; pre-normalised for COSINE, raw for L2
        std::vector<int64_t> ids;      // global (insertion-order) index of each stored vector
    };

    Metric   metric_;
    uint32_t n_lists_;
    uint32_t n_probe_;
    uint32_t dim_    = 0;
    uint32_t total_  = 0;    // total vectors added across all lists
    bool     trained_ = false;

    std::vector<float>        centroids_;  // [n_lists × dim], row-major; unit-norm for COSINE
    std::vector<InvertedList> lists_;      // one InvertedList per centroid

    // Index of the single nearest centroid to vec.
    // vec must already be normalised for COSINE.
    uint32_t nearest_centroid(const float* vec) const;

    // Indices of the n_probe nearest centroids to vec, in ascending distance order.
    // vec must already be normalised for COSINE.
    std::vector<uint32_t> nearest_centroids(const float* vec, uint32_t n_probe) const;
};

} // namespace sparsity

#include "sparsity/ivf.h"
#include "sparsity/kmeans.h"
#include "sparsity/metrics.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace sparsity {

namespace {

// Normalise a vector to unit L2 norm in-place. No-op if norm is zero.
void normalise(float* vec, uint32_t dim) {
    float norm_sq = 0.0f;
    for (uint32_t i = 0; i < dim; ++i) norm_sq += vec[i] * vec[i];
    if (norm_sq > 0.0f) {
        const float inv = 1.0f / std::sqrt(norm_sq);
        for (uint32_t i = 0; i < dim; ++i) vec[i] *= inv;
    }
}

// Scalar distance from vec to a centroid, used for centroid ranking.
// Both inputs must be normalised for COSINE.
// Uses squared L2 for L2 (monotone for ranking; avoids sqrt).
float centroid_dist(const float* vec, const float* centroid, uint32_t dim, Metric metric) {
    if (metric == Metric::L2) {
        return l2_distance_sq(vec, centroid, dim);
    }
    float dot = 0.0f;
    for (uint32_t i = 0; i < dim; ++i) dot += vec[i] * centroid[i];
    return 1.0f - dot;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// IVFIndex — construction
// ---------------------------------------------------------------------------

IVFIndex::IVFIndex(Metric metric, uint32_t n_lists, uint32_t n_probe)
    : metric_(metric), n_lists_(n_lists), n_probe_(n_probe) {
    if (metric != Metric::L2 && metric != Metric::COSINE) {
        throw std::invalid_argument("IVFIndex: only L2 and COSINE metrics are supported");
    }
    if (n_lists == 0) throw std::invalid_argument("IVFIndex: n_lists must be > 0");
    if (n_probe == 0) throw std::invalid_argument("IVFIndex: n_probe must be > 0");
}

void IVFIndex::set_n_probe(uint32_t n_probe) noexcept {
    n_probe_ = std::max(1u, std::min(n_probe, n_lists_));
}

// ---------------------------------------------------------------------------
// train
// ---------------------------------------------------------------------------

void IVFIndex::train(const float* data, uint32_t n, uint32_t dim) {
    if (trained_) {
        throw std::invalid_argument("IVFIndex::train — already trained");
    }
    if (dim == 0) {
        throw std::invalid_argument("IVFIndex::train — dim must be > 0");
    }
    // kmeans() validates n >= n_lists internally.
    KMeansResult km = kmeans(data, n, dim, n_lists_, metric_);
    centroids_ = std::move(km.centroids);
    dim_       = dim;
    lists_.resize(n_lists_);
    trained_   = true;
}

// ---------------------------------------------------------------------------
// add
// ---------------------------------------------------------------------------

void IVFIndex::add(const float* data, uint32_t n, uint32_t dim) {
    if (!trained_) {
        throw std::runtime_error(
            "IVFIndex::add — index must be trained before adding vectors");
    }
    if (n == 0) return;
    if (dim != dim_) {
        throw std::invalid_argument(
            "IVFIndex::add — dimension mismatch: index has dim=" +
            std::to_string(dim_) + ", got dim=" + std::to_string(dim));
    }

    std::vector<float> vec(dim);

    for (uint32_t i = 0; i < n; ++i) {
        std::memcpy(vec.data(), data + static_cast<size_t>(i) * dim, dim * sizeof(float));

        // Pre-normalise for COSINE (same convention as DenseIndex).
        if (metric_ == Metric::COSINE) normalise(vec.data(), dim);

        const uint32_t c = nearest_centroid(vec.data());
        InvertedList&  list = lists_[c];
        list.ids.push_back(static_cast<int64_t>(total_));
        list.vectors.insert(list.vectors.end(), vec.begin(), vec.end());
        ++total_;
    }
}

// ---------------------------------------------------------------------------
// search
// ---------------------------------------------------------------------------

SearchResult IVFIndex::search(const float* queries, uint32_t n_queries, uint32_t k) const {
    if (!trained_) {
        throw std::runtime_error("IVFIndex::search — index must be trained first");
    }
    if (total_ == 0) {
        throw std::runtime_error("IVFIndex::search — index is empty");
    }

    const uint32_t k_actual = std::min(k, total_);
    const uint32_t probe    = std::min(n_probe_, n_lists_);

    SearchResult result;
    result.n_queries = n_queries;
    result.k         = k_actual;
    result.distances.resize(static_cast<size_t>(n_queries) * k_actual);
    result.indices.resize(static_cast<size_t>(n_queries) * k_actual, -1);

    const bool cosine = (metric_ == Metric::COSINE);

    std::vector<float>                     q_norm(dim_);
    std::vector<std::pair<float, int64_t>> candidates;

    for (uint32_t q = 0; q < n_queries; ++q) {
        const float* query = queries + static_cast<size_t>(q) * dim_;

        // Optionally normalise the query.
        std::memcpy(q_norm.data(), query, dim_ * sizeof(float));
        if (cosine) normalise(q_norm.data(), dim_);
        const float* qvec = q_norm.data();

        // Find the n_probe nearest centroids.
        auto probed = nearest_centroids(qvec, probe);

        // Collect (distance, global_id) pairs from those lists.
        candidates.clear();
        for (const uint32_t ci : probed) {
            const InvertedList& list   = lists_[ci];
            const uint32_t      list_n = static_cast<uint32_t>(list.ids.size());

            for (uint32_t j = 0; j < list_n; ++j) {
                const float* vec = list.vectors.data() + static_cast<size_t>(j) * dim_;
                float dist;
                if (cosine) {
                    // Both qvec and stored vec are pre-normalised.
                    float dot = 0.0f;
                    for (uint32_t d = 0; d < dim_; ++d) dot += qvec[d] * vec[d];
                    dist = 1.0f - dot;
                } else {
                    dist = l2_distance(qvec, vec, dim_);
                }
                candidates.emplace_back(dist, list.ids[j]);
            }
        }

        // Select the top-k candidates.
        const uint32_t nc     = static_cast<uint32_t>(candidates.size());
        const uint32_t k_take = std::min(k_actual, nc);
        std::partial_sort(candidates.begin(), candidates.begin() + k_take, candidates.end());

        float*   out_dist = result.distances.data() + static_cast<size_t>(q) * k_actual;
        int64_t* out_idx  = result.indices.data()   + static_cast<size_t>(q) * k_actual;
        for (uint32_t j = 0; j < k_take; ++j) {
            out_dist[j] = candidates[j].first;
            out_idx[j]  = candidates[j].second;
        }
    }

    return result;
}

// ---------------------------------------------------------------------------
// validate
// ---------------------------------------------------------------------------

void IVFIndex::validate() const {
    assert(trained_ || (dim_ == 0 && centroids_.empty() && lists_.empty()));
    if (!trained_) return;
    assert(centroids_.size() == static_cast<size_t>(n_lists_) * dim_);
    assert(lists_.size() == n_lists_);
    uint32_t counted = 0;
    for (const auto& list : lists_) {
        assert(list.ids.size() * dim_ == list.vectors.size());
        counted += static_cast<uint32_t>(list.ids.size());
    }
    assert(counted == total_);
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

uint32_t IVFIndex::nearest_centroid(const float* vec) const {
    float    best_d = std::numeric_limits<float>::max();
    uint32_t best_c = 0;
    for (uint32_t c = 0; c < n_lists_; ++c) {
        const float d = centroid_dist(
            vec, centroids_.data() + static_cast<size_t>(c) * dim_, dim_, metric_);
        if (d < best_d) { best_d = d; best_c = c; }
    }
    return best_c;
}

std::vector<uint32_t> IVFIndex::nearest_centroids(const float* vec, uint32_t n_probe) const {
    std::vector<std::pair<float, uint32_t>> dist_idx(n_lists_);
    for (uint32_t c = 0; c < n_lists_; ++c) {
        dist_idx[c] = {centroid_dist(vec,
                                     centroids_.data() + static_cast<size_t>(c) * dim_,
                                     dim_, metric_),
                       c};
    }
    std::partial_sort(dist_idx.begin(), dist_idx.begin() + n_probe, dist_idx.end());
    std::vector<uint32_t> result(n_probe);
    for (uint32_t i = 0; i < n_probe; ++i) result[i] = dist_idx[i].second;
    return result;
}

} // namespace sparsity

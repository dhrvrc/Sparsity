#include "sparsity/kmeans.h"
#include "sparsity/metrics.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace sparsity {

namespace {

// Normalise a single vector to unit L2 norm in-place.
// No-op if the norm is zero.
void normalise(float* vec, uint32_t dim) {
    float norm_sq = 0.0f;
    for (uint32_t i = 0; i < dim; ++i) norm_sq += vec[i] * vec[i];
    if (norm_sq > 0.0f) {
        const float inv = 1.0f / std::sqrt(norm_sq);
        for (uint32_t i = 0; i < dim; ++i) vec[i] *= inv;
    }
}

// Distance from vec to centroid used for centroid ranking.
// Uses squared L2 for L2 metric (avoids sqrt; monotone for ranking).
// Uses 1 - dot for COSINE (both inputs must already be unit-norm).
float centroid_dist(const float* vec, const float* centroid, uint32_t dim, Metric metric) {
    if (metric == Metric::L2) {
        return l2_distance_sq(vec, centroid, dim);
    }
    // COSINE: both normalised → distance = 1 - dot
    float dot = 0.0f;
    for (uint32_t i = 0; i < dim; ++i) dot += vec[i] * centroid[i];
    return 1.0f - dot;
}

// k-means++ initialisation: selects n_lists seed indices from [0, n).
// Each subsequent seed is chosen with probability proportional to its squared
// distance to the nearest already-chosen seed (D² weighting).
std::vector<uint32_t> kmeans_pp_init(const float* pts, uint32_t n, uint32_t dim,
                                      uint32_t n_lists, Metric metric, std::mt19937& rng) {
    std::vector<uint32_t> chosen;
    chosen.reserve(n_lists);

    std::uniform_int_distribution<uint32_t> uniform(0, n - 1);
    chosen.push_back(uniform(rng));

    std::vector<float> min_dist(n, std::numeric_limits<float>::max());

    for (uint32_t c = 1; c < n_lists; ++c) {
        // Update min distances using the most recently chosen centroid.
        const float* last = pts + static_cast<size_t>(chosen.back()) * dim;
        for (uint32_t i = 0; i < n; ++i) {
            float d = centroid_dist(pts + static_cast<size_t>(i) * dim, last, dim, metric);
            if (d < min_dist[i]) min_dist[i] = d;
        }
        // Sample next centroid proportional to min_dist (D² weighting).
        std::discrete_distribution<uint32_t> weighted(min_dist.begin(), min_dist.end());
        chosen.push_back(weighted(rng));
    }
    return chosen;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// kmeans
// ---------------------------------------------------------------------------

KMeansResult kmeans(const float* data, uint32_t n, uint32_t dim,
                    uint32_t n_lists, Metric metric,
                    uint32_t max_iter, uint32_t seed) {
    if (dim == 0) {
        throw std::invalid_argument("kmeans: dim must be > 0");
    }
    if (n < n_lists) {
        throw std::invalid_argument(
            "kmeans: n (" + std::to_string(n) + ") must be >= n_lists (" +
            std::to_string(n_lists) + ")");
    }

    // For COSINE, work on a normalised copy of the data (spherical k-means).
    std::vector<float> work;
    const float*       pts = data;
    if (metric == Metric::COSINE) {
        work.assign(data, data + static_cast<size_t>(n) * dim);
        for (uint32_t i = 0; i < n; ++i)
            normalise(work.data() + static_cast<size_t>(i) * dim, dim);
        pts = work.data();
    }

    std::mt19937 rng(seed);

    // Seed centroids with k-means++.
    auto init_idx = kmeans_pp_init(pts, n, dim, n_lists, metric, rng);
    std::vector<float> centroids(static_cast<size_t>(n_lists) * dim);
    for (uint32_t c = 0; c < n_lists; ++c) {
        std::memcpy(centroids.data() + static_cast<size_t>(c) * dim,
                    pts + static_cast<size_t>(init_idx[c]) * dim,
                    dim * sizeof(float));
    }

    std::vector<uint32_t> assignments(n, 0);
    std::vector<uint32_t> counts(n_lists);
    std::vector<float>    new_centroids(static_cast<size_t>(n_lists) * dim);

    for (uint32_t iter = 0; iter < max_iter; ++iter) {
        bool changed = false;

        // --- Assignment step ---
        for (uint32_t i = 0; i < n; ++i) {
            const float* vec    = pts + static_cast<size_t>(i) * dim;
            float        best_d = std::numeric_limits<float>::max();
            uint32_t     best_c = 0;
            for (uint32_t c = 0; c < n_lists; ++c) {
                float d = centroid_dist(vec,
                                        centroids.data() + static_cast<size_t>(c) * dim,
                                        dim, metric);
                if (d < best_d) { best_d = d; best_c = c; }
            }
            if (assignments[i] != best_c) { assignments[i] = best_c; changed = true; }
        }

        if (!changed) break;

        // --- Update step ---
        std::fill(new_centroids.begin(), new_centroids.end(), 0.0f);
        std::fill(counts.begin(), counts.end(), 0u);

        for (uint32_t i = 0; i < n; ++i) {
            const uint32_t c   = assignments[i];
            float*         dst = new_centroids.data() + static_cast<size_t>(c) * dim;
            const float*   src = pts + static_cast<size_t>(i) * dim;
            counts[c]++;
            for (uint32_t j = 0; j < dim; ++j) dst[j] += src[j];
        }

        for (uint32_t c = 0; c < n_lists; ++c) {
            float* cen = new_centroids.data() + static_cast<size_t>(c) * dim;
            if (counts[c] > 0) {
                const float inv = 1.0f / static_cast<float>(counts[c]);
                for (uint32_t j = 0; j < dim; ++j) cen[j] *= inv;
                // Spherical k-means: re-normalise after averaging.
                if (metric == Metric::COSINE) normalise(cen, dim);
            } else {
                // Empty cluster: reinitialise to a random data point.
                std::uniform_int_distribution<uint32_t> u(0, n - 1);
                std::memcpy(cen, pts + static_cast<size_t>(u(rng)) * dim,
                            dim * sizeof(float));
            }
        }

        centroids.swap(new_centroids);
    }

    return KMeansResult{std::move(centroids), std::move(assignments)};
}

} // namespace sparsity

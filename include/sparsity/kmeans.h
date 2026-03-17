#pragma once

#include "sparsity/types.h"

#include <cstdint>
#include <vector>

namespace sparsity {

// Result of a k-means run.
struct KMeansResult {
    std::vector<float>    centroids;    // [n_lists × dim], row-major
    std::vector<uint32_t> assignments;  // [n] — centroid index for each training vector
};

// Run Lloyd's k-means with k-means++ initialisation.
//
// metric controls the distance used during assignment and centroid update:
//   L2     → standard Euclidean k-means; centroids are arithmetic means
//   COSINE → spherical k-means; centroids are normalised to unit length after
//             each update step; training data is normalised internally (do NOT
//             pre-normalise before calling)
//
// max_iter: maximum Lloyd iterations before stopping (default 25)
// seed:     RNG seed for reproducible k-means++ initialisation
//
// Throws std::invalid_argument if n < n_lists or dim == 0.
KMeansResult kmeans(const float* data, uint32_t n, uint32_t dim,
                    uint32_t n_lists, Metric metric,
                    uint32_t max_iter = 25, uint32_t seed = 42);

} // namespace sparsity

#pragma once

#include <cstdint>

// CPU reference implementations of all distance / similarity metrics.
// These are the ground truth — CUDA kernels must match within 1e-5.
//
// Convention: lower = more similar for L2 and Cosine.
//             HIGHER = more similar for Tanimoto (it is a similarity, not a distance).

namespace sparsity {

// ---------------------------------------------------------------------------
// L2 (Euclidean) distance — dense float vectors
// ---------------------------------------------------------------------------

// Full L2 distance: sqrt( sum((a_i - b_i)^2) ).
// Use for final results returned to callers.
float l2_distance(const float* a, const float* b, uint32_t dim);

// Squared L2: sum((a_i - b_i)^2).
// Use internally for ranking — avoids the sqrt.
float l2_distance_sq(const float* a, const float* b, uint32_t dim);

// ---------------------------------------------------------------------------
// Cosine distance — dense float vectors
// ---------------------------------------------------------------------------

// Cosine distance: 1 - dot(a,b) / (||a|| * ||b||).
// Returns 1.0 if either vector has zero norm.
float cosine_distance(const float* a, const float* b, uint32_t dim);

// ---------------------------------------------------------------------------
// Tanimoto similarity — binary packed vectors (uint64_t words)
// ---------------------------------------------------------------------------

// Binary Tanimoto: popcount(a AND b) / popcount(a OR b).
// Returns 1.0 if both vectors are all-zero (identical empty sets).
// n_words = ceil(dim / 64); caller must pass the correct word count.
float tanimoto_similarity(const uint64_t* a, const uint64_t* b, uint32_t n_words);

} // namespace sparsity

#pragma once

#include "sparsity/index.h"
#include "sparsity/types.h"

#include <cstdint>

namespace sparsity {

// ---------------------------------------------------------------------------
// dispatch_dense_search
//
// Unified entry point for brute-force dense search.  Automatically uses the
// GPU pipeline (distance kernel + top-k kernel) when a CUDA device is present
// and the shared-memory budget allows GPU top-k.  Falls back to a per-query
// CPU path otherwise.
//
// queries : float[n_queries * dim]  — pre-processed by caller:
//             L2     → raw vectors
//             COSINE → unit-normalised vectors
// db      : float[n_db * dim]       — pre-processed at add time:
//             L2     → raw vectors
//             COSINE → unit-normalised vectors
//
// Returns a SearchResult with k_actual = min(k, n_db) results per query,
// sorted ascending by distance (lower = closer for both L2 and COSINE).
// ---------------------------------------------------------------------------
SearchResult dispatch_dense_search(const float* queries,
                                   const float* db,
                                   uint32_t     n_queries,
                                   uint32_t     n_db,
                                   uint32_t     dim,
                                   uint32_t     k,
                                   Metric       metric);

} // namespace sparsity

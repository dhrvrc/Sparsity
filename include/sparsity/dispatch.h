#pragma once

#include "sparsity/binary.h"
#include "sparsity/index.h"
#include "sparsity/sparse.h"
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

// ---------------------------------------------------------------------------
// dispatch_sparse_search
//
// Brute-force search over float sparse vectors (CSR format).
// Supports Metric::COSINE and Metric::L2.
//
// db_norms : float[n_db] — L2 norm of each indexed vector, pre-computed at
//            add() time.  Used for cosine (norm) and L2 (norm^2 = norm*norm).
//
// Queries are supplied in CSR: q_indptr[n_queries+1], q_indices[q_indptr[n]],
// q_values[q_indptr[n]].
//
// Returns a SearchResult sorted ascending by distance (lower = closer).
// ---------------------------------------------------------------------------
SearchResult dispatch_sparse_search(const SparseMatrix& db,
                                    const float*        db_norms,
                                    const uint32_t*     q_indptr,
                                    const uint32_t*     q_indices,
                                    const float*        q_values,
                                    uint32_t            n_queries,
                                    uint32_t            k,
                                    Metric              metric);

// ---------------------------------------------------------------------------
// dispatch_binary_search
//
// Brute-force Tanimoto similarity search over packed-bit binary vectors.
//
// queries : uint64_t[n_queries * db.words_per_row()] — row-major packed bits.
//
// Returns a SearchResult where distances are Tanimoto similarities in [0, 1].
// Results are sorted DESCENDING by similarity (higher = more similar),
// so result.distances[0] is the most similar vector.
// ---------------------------------------------------------------------------
SearchResult dispatch_binary_search(const BinaryMatrix& db,
                                    const uint64_t*     queries,
                                    uint32_t            n_queries,
                                    uint32_t            k);

} // namespace sparsity

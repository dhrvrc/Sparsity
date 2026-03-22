#include "sparsity/dispatch.h"
#include "sparsity/metrics.h"
#include "sparsity/sparse.h"
#include "sparsity/binary.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <utility>
#include <vector>

#ifdef WITH_CUDA
#include "sparsity/blocked_ell.h"
#include "sparsity/cuda_dense.h"
#include "sparsity/cuda_sparse.h"
#include "sparsity/cuda_utils.h"
#include <cuda_runtime.h>
#endif

namespace sparsity {

#ifdef WITH_CUDA
// Returns the number of CUDA-capable devices visible to the process.
int cuda_device_count() {
    int count = 0;
    cudaGetDeviceCount(&count);
    return count;
}
#endif

#ifdef WITH_CUDA
// Used only in the hybrid GPU-distances + CPU-top-k fallback path.
static void cpu_topk_row(const float* dist_row,
                          uint32_t     n_db,
                          uint32_t     k_actual,
                          float*       out_dist,
                          int64_t*     out_idx)
{
    std::vector<std::pair<float, int64_t>> pairs(n_db);
    for (uint32_t i = 0; i < n_db; ++i) {
        pairs[i] = {dist_row[i], static_cast<int64_t>(i)};
    }
    std::partial_sort(pairs.begin(), pairs.begin() + k_actual, pairs.end());
    for (uint32_t j = 0; j < k_actual; ++j) {
        out_dist[j] = pairs[j].first;
        out_idx[j]  = pairs[j].second;
    }
}
#endif

// ---------------------------------------------------------------------------
// dispatch_dense_search
// ---------------------------------------------------------------------------
SearchResult dispatch_dense_search(const float* queries,
                                   const float* db,
                                   uint32_t     n_queries,
                                   uint32_t     n_db,
                                   uint32_t     dim,
                                   uint32_t     k,
                                   Metric       metric)
{
    const uint32_t k_actual = (k <= n_db) ? k : n_db;

    SearchResult result;
    result.n_queries = n_queries;
    result.k         = k_actual;
    result.distances.resize(static_cast<size_t>(n_queries) * k_actual);
    result.indices.resize(static_cast<size_t>(n_queries) * k_actual, -1);

    if (n_queries == 0 || n_db == 0) return result;

#ifdef WITH_CUDA
    if (cuda_device_count() > 0) {
        // ----------------------------------------------------------------
        // Determine whether GPU top-k fits in shared memory on this device.
        // ----------------------------------------------------------------
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceProp prop{};
        cudaGetDeviceProperties(&prop, dev);

        const size_t topk_shmem = static_cast<size_t>(CUDA_TOPK_BLOCK) * k_actual
                                  * (sizeof(float) + sizeof(int64_t));
        const bool   gpu_topk_fits = topk_shmem <= prop.sharedMemPerBlock;

        // ----------------------------------------------------------------
        // Allocate device buffers.
        // ----------------------------------------------------------------
        DeviceBuffer<float>   d_queries(static_cast<size_t>(n_queries) * dim);
        DeviceBuffer<float>   d_db(static_cast<size_t>(n_db) * dim);
        DeviceBuffer<float>   d_dist(static_cast<size_t>(n_queries) * n_db);

        d_queries.copy_from_host(queries, static_cast<size_t>(n_queries) * dim);
        d_db.copy_from_host(db, static_cast<size_t>(n_db) * dim);

        // ----------------------------------------------------------------
        // Launch the appropriate distance kernel.
        // ----------------------------------------------------------------
        if (metric == Metric::L2) {
            cuda_l2_batch(d_queries.get(), d_db.get(), d_dist.get(),
                          n_queries, n_db, dim);
        } else {
            // COSINE — queries already unit-normalised by the caller.
            cuda_cosine_batch(d_queries.get(), d_db.get(), d_dist.get(),
                              n_queries, n_db, dim);
        }

        if (gpu_topk_fits) {
            // ------------------------------------------------------------
            // Full GPU path: top-k on device, copy only winners to host.
            // ------------------------------------------------------------
            DeviceBuffer<float>   d_out_dist(static_cast<size_t>(n_queries) * k_actual);
            DeviceBuffer<int64_t> d_out_idx(static_cast<size_t>(n_queries) * k_actual);

            cuda_topk_batch(d_dist.get(), n_db, k,
                            d_out_dist.get(), d_out_idx.get(), n_queries);

            SPARSITY_CUDA_CHECK(cudaDeviceSynchronize());

            d_out_dist.copy_to_host(result.distances.data(),
                                    static_cast<size_t>(n_queries) * k_actual);
            d_out_idx.copy_to_host(result.indices.data(),
                                   static_cast<size_t>(n_queries) * k_actual);
        } else {
            // ------------------------------------------------------------
            // Hybrid path: k too large for GPU top-k shared memory budget.
            // Copy full distance matrix to host and run CPU partial_sort.
            // ------------------------------------------------------------
            SPARSITY_CUDA_CHECK(cudaDeviceSynchronize());

            std::vector<float> host_dist(static_cast<size_t>(n_queries) * n_db);
            d_dist.copy_to_host(host_dist.data(),
                                 static_cast<size_t>(n_queries) * n_db);

            for (uint32_t q = 0; q < n_queries; ++q) {
                cpu_topk_row(host_dist.data() + static_cast<size_t>(q) * n_db,
                             n_db, k_actual,
                             result.distances.data() + static_cast<size_t>(q) * k_actual,
                             result.indices.data()   + static_cast<size_t>(q) * k_actual);
            }
        }

        return result;
    }
#endif // WITH_CUDA

    // -----------------------------------------------------------------------
    // CPU-only fallback: per-query to avoid a large n_queries*n_db allocation.
    // -----------------------------------------------------------------------
    std::vector<std::pair<float, int64_t>> pairs(n_db);

    for (uint32_t q = 0; q < n_queries; ++q) {
        const float* qv      = queries + static_cast<size_t>(q) * dim;
        float*       out_d   = result.distances.data() + static_cast<size_t>(q) * k_actual;
        int64_t*     out_i   = result.indices.data()   + static_cast<size_t>(q) * k_actual;

        if (metric == Metric::L2) {
            for (uint32_t i = 0; i < n_db; ++i) {
                pairs[i] = {l2_distance(qv, db + static_cast<size_t>(i) * dim, dim),
                            static_cast<int64_t>(i)};
            }
        } else {
            // COSINE — queries pre-normalised; db pre-normalised.
            for (uint32_t i = 0; i < n_db; ++i) {
                const float* dv  = db + static_cast<size_t>(i) * dim;
                float        dot = 0.0f;
                for (uint32_t d = 0; d < dim; ++d) dot += qv[d] * dv[d];
                pairs[i] = {1.0f - dot, static_cast<int64_t>(i)};
            }
        }

        std::partial_sort(pairs.begin(), pairs.begin() + k_actual, pairs.end());
        for (uint32_t j = 0; j < k_actual; ++j) {
            out_d[j] = pairs[j].first;
            out_i[j] = pairs[j].second;
        }
    }

    return result;
}

// ---------------------------------------------------------------------------
// dispatch_sparse_search  (CPU + optional CUDA path)
// ---------------------------------------------------------------------------
SearchResult dispatch_sparse_search(const SparseMatrix& db,
                                    const float*        db_norms,
                                    const uint32_t*     q_indptr,
                                    const uint32_t*     q_indices,
                                    const float*        q_values,
                                    uint32_t            n_queries,
                                    uint32_t            k,
                                    Metric              metric)
{
    const uint32_t n_db     = db.n_rows;
    const uint32_t k_actual = (k <= n_db) ? k : n_db;

    SearchResult result;
    result.n_queries = n_queries;
    result.k         = k_actual;
    result.distances.resize(static_cast<size_t>(n_queries) * k_actual);
    result.indices.resize(static_cast<size_t>(n_queries) * k_actual, -1);

    if (n_queries == 0 || n_db == 0) return result;

#ifdef WITH_CUDA
    if (cuda_device_count() > 0) {
        // ----------------------------------------------------------------
        // Prepare query data: padded arrays for the GPU kernel.
        // ----------------------------------------------------------------

        // Find max query nnz across all queries.
        uint32_t max_q_nnz = 0;
        for (uint32_t q = 0; q < n_queries; ++q) {
            const uint32_t nnz = q_indptr[q + 1] - q_indptr[q];
            if (nnz > max_q_nnz) max_q_nnz = nnz;
        }
        if (max_q_nnz == 0) {
            // All queries are zero-vectors: distance = 1.0 (cosine) or norm_db (L2).
            for (uint32_t q = 0; q < n_queries; ++q) {
                std::vector<std::pair<float, int64_t>> pairs(n_db);
                for (uint32_t d = 0; d < n_db; ++d) {
                    float dist = (metric == Metric::L2)
                        ? db_norms[d]  // ||0 - db|| = ||db||
                        : 1.0f;        // cosine of zero-vector = 1.0
                    pairs[d] = {dist, static_cast<int64_t>(d)};
                }
                std::partial_sort(pairs.begin(), pairs.begin() + k_actual, pairs.end());
                float*   od = result.distances.data() + static_cast<size_t>(q) * k_actual;
                int64_t* oi = result.indices.data()   + static_cast<size_t>(q) * k_actual;
                for (uint32_t j = 0; j < k_actual; ++j) { od[j] = pairs[j].first; oi[j] = pairs[j].second; }
            }
            return result;
        }

        // Build padded host arrays for queries.
        const size_t q_pad_size = static_cast<size_t>(n_queries) * max_q_nnz;
        std::vector<uint32_t> h_q_idx(q_pad_size, 0);
        std::vector<float>    h_q_val(q_pad_size, 0.0f);
        std::vector<uint32_t> h_q_nnz(n_queries);
        std::vector<float>    h_q_norm(n_queries);

        for (uint32_t q = 0; q < n_queries; ++q) {
            const uint32_t start = q_indptr[q];
            const uint32_t nnz   = q_indptr[q + 1] - start;
            h_q_nnz[q] = nnz;

            float norm_sq = 0.0f;
            for (uint32_t i = 0; i < nnz; ++i) {
                h_q_idx[q * max_q_nnz + i] = q_indices[start + i];
                h_q_val[q * max_q_nnz + i] = q_values[start + i];
                norm_sq += q_values[start + i] * q_values[start + i];
            }
            h_q_norm[q] = std::sqrt(norm_sq);
        }

        // Build Blocked-ELL from db CSR on host.
        BlockedELLHost bell = csr_to_blocked_ell(db);

        // ----------------------------------------------------------------
        // Transfer to device.
        // ----------------------------------------------------------------
        DeviceBuffer<uint32_t> d_q_idx(q_pad_size);
        DeviceBuffer<float>    d_q_val(q_pad_size);
        DeviceBuffer<uint32_t> d_q_nnz(n_queries);
        DeviceBuffer<float>    d_q_norm(n_queries);
        DeviceBuffer<float>    d_db_norm(n_db);
        DeviceBuffer<uint32_t> d_bell_col(bell.col_idx.size());
        DeviceBuffer<float>    d_bell_val(bell.values.size());
        DeviceBuffer<uint32_t> d_bell_offsets(bell.block_offsets.size());
        DeviceBuffer<uint32_t> d_bell_max_nnz(bell.block_max_nnz.size());
        DeviceBuffer<float>    d_dots(static_cast<size_t>(n_queries) * n_db);

        d_q_idx.copy_from_host(h_q_idx.data(), q_pad_size);
        d_q_val.copy_from_host(h_q_val.data(), q_pad_size);
        d_q_nnz.copy_from_host(h_q_nnz.data(), n_queries);
        d_q_norm.copy_from_host(h_q_norm.data(), n_queries);
        d_db_norm.copy_from_host(db_norms, n_db);
        d_bell_col.copy_from_host(bell.col_idx.data(), bell.col_idx.size());
        d_bell_val.copy_from_host(bell.values.data(), bell.values.size());
        d_bell_offsets.copy_from_host(bell.block_offsets.data(), bell.block_offsets.size());
        d_bell_max_nnz.copy_from_host(bell.block_max_nnz.data(), bell.block_max_nnz.size());

        // ----------------------------------------------------------------
        // Launch sparse dot-product kernel, then finalize kernel.
        // ----------------------------------------------------------------
        cuda_sparse_dot_batch(d_q_idx.get(), d_q_val.get(), d_q_nnz.get(),
                              d_bell_col.get(), d_bell_val.get(),
                              d_bell_offsets.get(), d_bell_max_nnz.get(),
                              d_dots.get(),
                              n_queries, n_db, max_q_nnz, bell.n_blocks);

        if (metric == Metric::COSINE) {
            cuda_sparse_cosine_finalize(d_dots.get(), d_q_norm.get(), d_db_norm.get(),
                                        n_queries, n_db);
        } else {
            // L2: need squared norms.
            std::vector<float> h_q_norm_sq(n_queries);
            std::vector<float> h_db_norm_sq(n_db);
            for (uint32_t i = 0; i < n_queries; ++i) h_q_norm_sq[i] = h_q_norm[i] * h_q_norm[i];
            for (uint32_t i = 0; i < n_db; ++i)      h_db_norm_sq[i] = db_norms[i] * db_norms[i];

            DeviceBuffer<float> d_q_norm_sq(n_queries);
            DeviceBuffer<float> d_db_norm_sq(n_db);
            d_q_norm_sq.copy_from_host(h_q_norm_sq.data(), n_queries);
            d_db_norm_sq.copy_from_host(h_db_norm_sq.data(), n_db);

            cuda_sparse_l2_finalize(d_dots.get(), d_q_norm_sq.get(), d_db_norm_sq.get(),
                                    n_queries, n_db);
        }

        // ----------------------------------------------------------------
        // Top-k selection.
        // ----------------------------------------------------------------
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceProp prop{};
        cudaGetDeviceProperties(&prop, dev);

        const size_t topk_shmem = static_cast<size_t>(CUDA_TOPK_BLOCK) * k_actual
                                  * (sizeof(float) + sizeof(int64_t));
        const bool gpu_topk_fits = topk_shmem <= prop.sharedMemPerBlock;

        if (gpu_topk_fits) {
            DeviceBuffer<float>   d_out_dist(static_cast<size_t>(n_queries) * k_actual);
            DeviceBuffer<int64_t> d_out_idx(static_cast<size_t>(n_queries) * k_actual);

            cuda_topk_batch(d_dots.get(), n_db, k,
                            d_out_dist.get(), d_out_idx.get(), n_queries);

            SPARSITY_CUDA_CHECK(cudaDeviceSynchronize());

            d_out_dist.copy_to_host(result.distances.data(),
                                    static_cast<size_t>(n_queries) * k_actual);
            d_out_idx.copy_to_host(result.indices.data(),
                                   static_cast<size_t>(n_queries) * k_actual);
        } else {
            SPARSITY_CUDA_CHECK(cudaDeviceSynchronize());

            std::vector<float> host_dist(static_cast<size_t>(n_queries) * n_db);
            d_dots.copy_to_host(host_dist.data(), static_cast<size_t>(n_queries) * n_db);

            for (uint32_t q = 0; q < n_queries; ++q) {
                cpu_topk_row(host_dist.data() + static_cast<size_t>(q) * n_db,
                             n_db, k_actual,
                             result.distances.data() + static_cast<size_t>(q) * k_actual,
                             result.indices.data()   + static_cast<size_t>(q) * k_actual);
            }
        }

        return result;
    }
#endif // WITH_CUDA

    // -----------------------------------------------------------------------
    // CPU path: merge-walk dot product per (query, db) pair.
    // -----------------------------------------------------------------------
    std::vector<std::pair<float, int64_t>> pairs(n_db);

    for (uint32_t q = 0; q < n_queries; ++q) {
        const uint32_t q_start = q_indptr[q];
        const uint32_t q_nnz   = q_indptr[q + 1] - q_start;

        SparseVector qv;
        qv.indices = q_indices + q_start;
        qv.values  = q_values  + q_start;
        qv.nnz     = q_nnz;
        qv.dim     = db.n_cols;

        // Compute query norm once per query.
        float q_norm_sq = 0.0f;
        for (uint32_t i = 0; i < q_nnz; ++i) q_norm_sq += qv.values[i] * qv.values[i];
        const float q_norm = std::sqrt(q_norm_sq);

        float*   out_d = result.distances.data() + static_cast<size_t>(q) * k_actual;
        int64_t* out_i = result.indices.data()   + static_cast<size_t>(q) * k_actual;

        for (uint32_t d = 0; d < n_db; ++d) {
            SparseVector dv = db.row(d);
            float dist;
            if (metric == Metric::COSINE) {
                dist = sparse_cosine_distance(qv, dv, q_norm, db_norms[d]);
            } else {
                // L2: norm_sq_db = db_norms[d]^2
                dist = sparse_l2_distance(qv, dv, q_norm_sq, db_norms[d] * db_norms[d]);
            }
            pairs[d] = {dist, static_cast<int64_t>(d)};
        }

        std::partial_sort(pairs.begin(), pairs.begin() + k_actual, pairs.end());
        for (uint32_t j = 0; j < k_actual; ++j) {
            out_d[j] = pairs[j].first;
            out_i[j] = pairs[j].second;
        }
    }

    return result;
}

// ---------------------------------------------------------------------------
// dispatch_binary_search  (CPU + optional CUDA path)
//
// Tanimoto similarity is a SIMILARITY (higher = more similar), so we negate
// it internally and sort ascending — this puts the most similar vectors at
// index 0.  The negation is undone before writing to SearchResult.distances
// so the caller receives actual similarity values in [0, 1].
// ---------------------------------------------------------------------------
SearchResult dispatch_binary_search(const BinaryMatrix& db,
                                    const uint64_t*     queries,
                                    uint32_t            n_queries,
                                    uint32_t            k)
{
    const uint32_t n_db     = db.n_rows;
    const uint32_t k_actual = (k <= n_db) ? k : n_db;
    const uint32_t wpr      = db.words_per_row();

    SearchResult result;
    result.n_queries = n_queries;
    result.k         = k_actual;
    result.distances.resize(static_cast<size_t>(n_queries) * k_actual);
    result.indices.resize(static_cast<size_t>(n_queries) * k_actual, -1);

    if (n_queries == 0 || n_db == 0) return result;

#ifdef WITH_CUDA
    if (cuda_device_count() > 0) {
        // Transfer db and queries to device.
        const size_t db_words    = static_cast<size_t>(n_db)      * wpr;
        const size_t query_words = static_cast<size_t>(n_queries)  * wpr;

        DeviceBuffer<uint64_t> d_db(db_words);
        DeviceBuffer<uint64_t> d_queries(query_words);
        DeviceBuffer<float>    d_dist(static_cast<size_t>(n_queries) * n_db);

        d_db.copy_from_host(db.data, db_words);
        d_queries.copy_from_host(queries, query_words);

        // Kernel stores -similarity so ascending top-k gives most-similar first.
        cuda_tanimoto_batch(d_queries.get(), d_db.get(), d_dist.get(),
                            n_queries, n_db, wpr);

        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceProp prop{};
        cudaGetDeviceProperties(&prop, dev);

        const size_t topk_shmem = static_cast<size_t>(CUDA_TOPK_BLOCK) * k_actual
                                  * (sizeof(float) + sizeof(int64_t));
        const bool gpu_topk_fits = topk_shmem <= prop.sharedMemPerBlock;

        if (gpu_topk_fits) {
            DeviceBuffer<float>   d_out_dist(static_cast<size_t>(n_queries) * k_actual);
            DeviceBuffer<int64_t> d_out_idx(static_cast<size_t>(n_queries) * k_actual);

            cuda_topk_batch(d_dist.get(), n_db, k,
                            d_out_dist.get(), d_out_idx.get(), n_queries);

            SPARSITY_CUDA_CHECK(cudaDeviceSynchronize());

            d_out_dist.copy_to_host(result.distances.data(),
                                    static_cast<size_t>(n_queries) * k_actual);
            d_out_idx.copy_to_host(result.indices.data(),
                                   static_cast<size_t>(n_queries) * k_actual);
        } else {
            SPARSITY_CUDA_CHECK(cudaDeviceSynchronize());

            std::vector<float> host_dist(static_cast<size_t>(n_queries) * n_db);
            d_dist.copy_to_host(host_dist.data(), static_cast<size_t>(n_queries) * n_db);

            for (uint32_t q = 0; q < n_queries; ++q) {
                cpu_topk_row(host_dist.data() + static_cast<size_t>(q) * n_db,
                             n_db, k_actual,
                             result.distances.data() + static_cast<size_t>(q) * k_actual,
                             result.indices.data()   + static_cast<size_t>(q) * k_actual);
            }
        }

        // Un-negate: stored as -sim, return actual similarity.
        for (float& d : result.distances) d = -d;

        return result;
    }
#endif // WITH_CUDA

    // -----------------------------------------------------------------------
    // CPU path.
    // -----------------------------------------------------------------------
    std::vector<std::pair<float, int64_t>> pairs(n_db);

    for (uint32_t q = 0; q < n_queries; ++q) {
        const uint64_t* qv = queries + static_cast<size_t>(q) * wpr;

        for (uint32_t d = 0; d < n_db; ++d) {
            const uint64_t* dv  = db.data + static_cast<size_t>(d) * wpr;
            const float     sim = tanimoto_similarity(qv, dv, wpr);
            // Negate so partial_sort (ascending) picks highest similarity first.
            pairs[d] = {-sim, static_cast<int64_t>(d)};
        }

        std::partial_sort(pairs.begin(), pairs.begin() + k_actual, pairs.end());

        float*   out_d = result.distances.data() + static_cast<size_t>(q) * k_actual;
        int64_t* out_i = result.indices.data()   + static_cast<size_t>(q) * k_actual;
        for (uint32_t j = 0; j < k_actual; ++j) {
            out_d[j] = -pairs[j].first;  // un-negate: return actual similarity
            out_i[j] = pairs[j].second;
        }
    }

    return result;
}

} // namespace sparsity

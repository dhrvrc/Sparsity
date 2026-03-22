#include "sparsity/dispatch.h"
#include "sparsity/cuda_utils.h"
#include "sparsity/metrics.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <utility>
#include <vector>

#ifdef WITH_CUDA
#include "sparsity/cuda_dense.h"
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

} // namespace sparsity

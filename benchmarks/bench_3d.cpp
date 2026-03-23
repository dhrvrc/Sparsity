// bench_3d.cpp
// 3-D sweep: n_queries × n_db × batch_time_ms for four backends.
//
// Backends
//   cpu_nontiled  — per-query scalar loop (mirrors DenseIndex CPU fallback)
//   cpu_tiled     — TiledDenseIndex (cache-blocked over TILE_Q=4 queries)
//   gpu_nontiled  — cuda_l2_batch (warp-per-row kernel)          [WITH_CUDA]
//   gpu_tiled     — cuda_l2_tiled_batch (shared-mem tiled kernel) [WITH_CUDA]
//
// Both GPU paths include host↔device transfer time (real wall-clock cost).
// Metric: L2.  Fixed dim=128 so one plot captures the algorithmic comparison.
//
// Usage:  ./bench_3d [output.json]

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "sparsity/index.h"          // Metric, SearchResult
#include "sparsity/metrics.h"        // l2_distance
#include "sparsity/tiled_dense.h"    // TiledDenseIndex

#ifdef WITH_CUDA
#include "sparsity/cuda_dense.h"
#include "sparsity/cuda_utils.h"
#include <cuda_runtime.h>
#endif

using namespace sparsity;
using Clock = std::chrono::high_resolution_clock;

// ============================================================
// Parameters
// ============================================================

static constexpr uint32_t DIM      = 128;
static constexpr uint32_t K        = 10;
static constexpr int      N_WARMUP = 2;   // throw-away runs before timing

// n_queries and n_db grids for the 3-D sweep
static const std::vector<uint32_t> N_QUERIES_LIST = {1, 4, 16, 64, 256, 512};
static const std::vector<uint32_t> N_DB_LIST      = {10000, 50000, 100000, 500000};

// Maximum GPU distance-matrix size (bytes).  Cells exceeding this are skipped.
static constexpr size_t GPU_MAX_DIST_BYTES = static_cast<size_t>(512) * 1024 * 1024;

// ============================================================
// Timing
// ============================================================

template <typename F>
static double elapsed_ms(F&& fn) {
    auto t0 = Clock::now();
    fn();
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// Run fn for n_reps iterations and return the median elapsed time (ms).
template <typename F>
static double median_ms(F&& fn, int n_reps) {
    std::vector<double> ts;
    ts.reserve(n_reps);
    for (int i = 0; i < n_reps; ++i)
        ts.push_back(elapsed_ms(fn));
    std::sort(ts.begin(), ts.end());
    return ts[n_reps / 2];
}

static int n_reps_for(uint32_t n_queries, uint32_t n_db) {
    const size_t work = static_cast<size_t>(n_queries) * n_db;
    if (work >= 50000000ULL) return 3;
    if (work >=  5000000ULL) return 5;
    return 10;
}

// ============================================================
// Data generation
// ============================================================

static std::vector<float> gen_gaussian(uint32_t n, uint32_t dim, uint32_t seed = 42) {
    std::mt19937                    rng(seed);
    std::normal_distribution<float> dist(0.f, 1.f);
    std::vector<float>              v(static_cast<size_t>(n) * dim);
    for (auto& x : v) x = dist(rng);
    return v;
}

// ============================================================
// CPU non-tiled  (per-query scalar loop)
// ============================================================

static void cpu_nontiled_search(const float* queries,
                                 const float* db,
                                 uint32_t     n_queries,
                                 uint32_t     n_db,
                                 uint32_t     dim,
                                 uint32_t     k)
{
    const uint32_t k_actual = std::min(k, n_db);
    std::vector<std::pair<float, int64_t>> pairs(n_db);

    for (uint32_t q = 0; q < n_queries; ++q) {
        const float* qv = queries + static_cast<size_t>(q) * dim;
        for (uint32_t i = 0; i < n_db; ++i)
            pairs[i] = {l2_distance(qv, db + static_cast<size_t>(i) * dim, dim),
                        static_cast<int64_t>(i)};
        std::partial_sort(pairs.begin(), pairs.begin() + k_actual, pairs.end());
    }
}

// ============================================================
// GPU helpers  (compiled only with -DWITH_CUDA=ON)
// ============================================================

#ifdef WITH_CUDA

static bool cuda_available() {
    int n = 0;
    cudaGetDeviceCount(&n);
    return n > 0;
}

static std::string cuda_device_name() {
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    return prop.name;
}

// Returns false if the distance matrix won't fit in GPU memory.
static bool gpu_mem_ok(uint32_t n_queries, uint32_t n_db, uint32_t dim) {
    const size_t dist_bytes = static_cast<size_t>(n_queries) * n_db * sizeof(float);
    if (dist_bytes > GPU_MAX_DIST_BYTES) return false;

    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);
    const size_t total_needed = dist_bytes
                              + static_cast<size_t>(n_queries) * dim * sizeof(float)
                              + static_cast<size_t>(n_db) * dim * sizeof(float);
    return total_needed < free_mem * 8 / 10;
}

// Run one full GPU search (H→D, kernel, top-k, D→H) using the given distance
// kernel.  The distance_fn signature:
//   void(d_queries, d_db, d_distances, n_queries, n_db, dim)
template <typename DistFn>
static void gpu_search(const float* queries,
                        const float* db,
                        uint32_t     n_queries,
                        uint32_t     n_db,
                        uint32_t     dim,
                        uint32_t     k,
                        DistFn&&     distance_fn)
{
    const uint32_t k_actual = std::min(k, n_db);

    DeviceBuffer<float> d_queries(static_cast<size_t>(n_queries) * dim);
    DeviceBuffer<float> d_db(static_cast<size_t>(n_db) * dim);
    DeviceBuffer<float> d_dist(static_cast<size_t>(n_queries) * n_db);

    d_queries.copy_from_host(queries, static_cast<size_t>(n_queries) * dim);
    d_db.copy_from_host(db,           static_cast<size_t>(n_db) * dim);

    distance_fn(d_queries.get(), d_db.get(), d_dist.get(), n_queries, n_db, dim);

    int            dev = 0;
    cudaDeviceProp prop{};
    cudaGetDevice(&dev);
    cudaGetDeviceProperties(&prop, dev);

    const size_t topk_shmem = static_cast<size_t>(CUDA_TOPK_BLOCK) * k_actual
                               * (sizeof(float) + sizeof(int64_t));

    if (topk_shmem <= prop.sharedMemPerBlock) {
        DeviceBuffer<float>   d_out_dist(static_cast<size_t>(n_queries) * k_actual);
        DeviceBuffer<int64_t> d_out_idx(static_cast<size_t>(n_queries) * k_actual);

        cuda_topk_batch(d_dist.get(), n_db, k,
                        d_out_dist.get(), d_out_idx.get(), n_queries);
        SPARSITY_CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<float>   h_dist(static_cast<size_t>(n_queries) * k_actual);
        std::vector<int64_t> h_idx(static_cast<size_t>(n_queries) * k_actual);
        d_out_dist.copy_to_host(h_dist.data(), h_dist.size());
        d_out_idx.copy_to_host(h_idx.data(),   h_idx.size());
    } else {
        SPARSITY_CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<float> h_dist(static_cast<size_t>(n_queries) * n_db);
        d_dist.copy_to_host(h_dist.data(), h_dist.size());

        std::vector<std::pair<float, int64_t>> pairs(n_db);
        for (uint32_t q = 0; q < n_queries; ++q) {
            const float* row = h_dist.data() + static_cast<size_t>(q) * n_db;
            for (uint32_t i = 0; i < n_db; ++i) pairs[i] = {row[i], static_cast<int64_t>(i)};
            std::partial_sort(pairs.begin(), pairs.begin() + k_actual, pairs.end());
        }
    }
}

#endif // WITH_CUDA

// ============================================================
// Record and JSON
// ============================================================

struct Record {
    std::string backend;       // "cpu_nontiled" | "cpu_tiled" | "gpu_nontiled" | "gpu_tiled"
    uint32_t    n_queries = 0;
    uint32_t    n_db      = 0;
    double      batch_time_ms = -1.0;  // -1 = skipped (OOM or no CUDA)
};

static std::string to_json(const Record& r) {
    std::ostringstream o;
    o << "    {\n";
    o << "      \"backend\": \""    << r.backend    << "\",\n";
    o << "      \"n_queries\": "    << r.n_queries   << ",\n";
    o << "      \"n_db\": "         << r.n_db        << ",\n";
    if (r.batch_time_ms < 0.0)
        o << "      \"batch_time_ms\": null\n";
    else
        o << "      \"batch_time_ms\": " << std::fixed << std::setprecision(4)
          << r.batch_time_ms << "\n";
    o << "    }";
    return o.str();
}

// ============================================================
// main
// ============================================================

int main(int argc, char** argv) {
    std::string out_path = "bench_3d_results.json";
    if (argc > 1) out_path = argv[1];

#ifdef WITH_CUDA
    const bool        gpu_present = cuda_available();
    const std::string gpu_name    = gpu_present ? cuda_device_name() : "";
    if (gpu_present)
        fprintf(stderr, "GPU: %s\n\n", gpu_name.c_str());
    else
        fprintf(stderr, "GPU: none found — GPU backends will be skipped.\n\n");
#else
    const bool        gpu_present = false;
    const std::string gpu_name    = "";
    fprintf(stderr, "Built without -DWITH_CUDA=ON — GPU backends will be skipped.\n\n");
#endif

    std::vector<Record> records;

    for (uint32_t n_db : N_DB_LIST) {
        fprintf(stderr, "=== n_db = %u ===\n", n_db);

        // Generate database and build CPU tiled index (shared across n_queries)
        auto db = gen_gaussian(n_db, DIM, 42);

        TiledDenseIndex tiled_idx(Metric::L2);
        tiled_idx.add(db.data(), n_db, DIM);

        for (uint32_t n_queries : N_QUERIES_LIST) {
            const int reps = n_reps_for(n_queries, n_db);

            auto queries = gen_gaussian(n_queries, DIM, 100 + n_queries);

            fprintf(stderr, "  n_queries=%-4u  (reps=%d)\n", n_queries, reps);

            // ── CPU non-tiled ──────────────────────────────────────────────
            {
                // warmup
                for (int w = 0; w < N_WARMUP; ++w)
                    cpu_nontiled_search(queries.data(), db.data(), n_queries, n_db, DIM, K);

                Record r;
                r.backend  = "cpu_nontiled";
                r.n_queries = n_queries;
                r.n_db      = n_db;
                r.batch_time_ms = median_ms([&] {
                    cpu_nontiled_search(queries.data(), db.data(), n_queries, n_db, DIM, K);
                }, reps);
                records.push_back(r);
                fprintf(stderr, "    cpu_nontiled  %.3f ms\n", r.batch_time_ms);
            }

            // ── CPU tiled ─────────────────────────────────────────────────
            {
                for (int w = 0; w < N_WARMUP; ++w)
                    tiled_idx.search(queries.data(), n_queries, K);

                Record r;
                r.backend   = "cpu_tiled";
                r.n_queries = n_queries;
                r.n_db      = n_db;
                r.batch_time_ms = median_ms([&] {
                    tiled_idx.search(queries.data(), n_queries, K);
                }, reps);
                records.push_back(r);
                fprintf(stderr, "    cpu_tiled     %.3f ms\n", r.batch_time_ms);
            }

#ifdef WITH_CUDA
            if (!gpu_present) {
                // Insert skipped records so the JSON is complete
                records.push_back({"gpu_nontiled", n_queries, n_db, -1.0});
                records.push_back({"gpu_tiled",    n_queries, n_db, -1.0});
                fprintf(stderr, "    gpu_*         [skipped — no CUDA device]\n");
                continue;
            }

            if (!gpu_mem_ok(n_queries, n_db, DIM)) {
                records.push_back({"gpu_nontiled", n_queries, n_db, -1.0});
                records.push_back({"gpu_tiled",    n_queries, n_db, -1.0});
                const size_t mb = static_cast<size_t>(n_queries) * n_db * 4 / (1024*1024);
                fprintf(stderr, "    gpu_*         [skipped — dist matrix ~%zu MB]\n", mb);
                continue;
            }

            // ── GPU non-tiled ──────────────────────────────────────────────
            {
                for (int w = 0; w < N_WARMUP; ++w)
                    gpu_search(queries.data(), db.data(), n_queries, n_db, DIM, K,
                               [](const float* dq, const float* ddb, float* dd,
                                  uint32_t nq, uint32_t nd, uint32_t d) {
                                   cuda_l2_batch(dq, ddb, dd, nq, nd, d);
                               });

                Record r;
                r.backend   = "gpu_nontiled";
                r.n_queries = n_queries;
                r.n_db      = n_db;
                r.batch_time_ms = median_ms([&] {
                    gpu_search(queries.data(), db.data(), n_queries, n_db, DIM, K,
                               [](const float* dq, const float* ddb, float* dd,
                                  uint32_t nq, uint32_t nd, uint32_t d) {
                                   cuda_l2_batch(dq, ddb, dd, nq, nd, d);
                               });
                }, reps);
                records.push_back(r);
                fprintf(stderr, "    gpu_nontiled  %.3f ms\n", r.batch_time_ms);
            }

            // ── GPU tiled ──────────────────────────────────────────────────
            {
                for (int w = 0; w < N_WARMUP; ++w)
                    gpu_search(queries.data(), db.data(), n_queries, n_db, DIM, K,
                               [](const float* dq, const float* ddb, float* dd,
                                  uint32_t nq, uint32_t nd, uint32_t d) {
                                   cuda_l2_tiled_batch(dq, ddb, dd, nq, nd, d);
                               });

                Record r;
                r.backend   = "gpu_tiled";
                r.n_queries = n_queries;
                r.n_db      = n_db;
                r.batch_time_ms = median_ms([&] {
                    gpu_search(queries.data(), db.data(), n_queries, n_db, DIM, K,
                               [](const float* dq, const float* ddb, float* dd,
                                  uint32_t nq, uint32_t nd, uint32_t d) {
                                   cuda_l2_tiled_batch(dq, ddb, dd, nq, nd, d);
                               });
                }, reps);
                records.push_back(r);
                fprintf(stderr, "    gpu_tiled     %.3f ms\n", r.batch_time_ms);
            }
#else
            records.push_back({"gpu_nontiled", n_queries, n_db, -1.0});
            records.push_back({"gpu_tiled",    n_queries, n_db, -1.0});
#endif // WITH_CUDA
        }
        fprintf(stderr, "\n");
    }

    // ── Write JSON ──────────────────────────────────────────────────────────
    std::ofstream f(out_path);
    if (!f) {
        fprintf(stderr, "ERROR: cannot open output file: %s\n", out_path.c_str());
        return 1;
    }

    char        tbuf[64];
    std::time_t now = std::time(nullptr);
    std::strftime(tbuf, sizeof(tbuf), "%Y-%m-%dT%H:%M:%S", std::localtime(&now));

    f << "{\n";
    f << "  \"version\": \"0.1.0\",\n";
    f << "  \"timestamp\": \"" << tbuf << "\",\n";
    f << "  \"gpu_available\": " << (gpu_present ? "true" : "false") << ",\n";
    f << "  \"gpu_name\": \""    << gpu_name << "\",\n";
    f << "  \"dim\": "           << DIM      << ",\n";
    f << "  \"k\": "             << K        << ",\n";
    f << "  \"metric\": \"L2\",\n";

    // Emit the grid dimensions so the plotter can read them without inferring
    f << "  \"n_queries_list\": [";
    for (size_t i = 0; i < N_QUERIES_LIST.size(); ++i) {
        if (i) f << ", ";
        f << N_QUERIES_LIST[i];
    }
    f << "],\n";

    f << "  \"n_db_list\": [";
    for (size_t i = 0; i < N_DB_LIST.size(); ++i) {
        if (i) f << ", ";
        f << N_DB_LIST[i];
    }
    f << "],\n";

    f << "  \"n_records\": " << records.size() << ",\n";
    f << "  \"results\": [\n";
    for (size_t i = 0; i < records.size(); ++i) {
        f << to_json(records[i]);
        if (i + 1 < records.size()) f << ",";
        f << "\n";
    }
    f << "  ]\n}\n";

    fprintf(stderr, "Wrote %zu records to %s\n", records.size(), out_path.c_str());
    return 0;
}

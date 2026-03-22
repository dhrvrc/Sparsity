// bench_gpu.cpp
// Explicit CPU vs GPU head-to-head benchmark for dense brute-force search.
//
// CPU path  : per-query loop with l2_distance / cosine dot-product + partial_sort
//             (mirrors the fallback inside dispatch_dense_search)
// GPU path  : cuda_l2_batch / cuda_cosine_batch + cuda_topk_batch called directly,
//             including host<->device transfer overhead
//
// When compiled with    -DWITH_CUDA=ON  : both paths are benchmarked.
// When compiled without -DWITH_CUDA=ON  : only CPU numbers are recorded and the
//                                         JSON marks gpu_available=false.
//
// Usage:  ./bench_gpu [output.json]
// Output: JSON (default: bench_gpu_results.json), progress to stderr.

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
#include <unordered_set>
#include <vector>

#include "sparsity/index.h"    // SearchResult, Metric
#include "sparsity/metrics.h"  // l2_distance

#ifdef WITH_CUDA
#include "sparsity/cuda_dense.h"
#include "sparsity/cuda_utils.h"
#include <cuda_runtime.h>
#endif

using namespace sparsity;
using Clock = std::chrono::high_resolution_clock;

// ============================================================
// Benchmark parameters
// ============================================================

static constexpr uint32_t K              = 10;
static constexpr uint32_t WARMUP_QUERIES = 20;
static constexpr uint32_t TIMING_QUERIES = 200;
static constexpr uint32_t BATCH_QPS_SIZE = 500;

// ============================================================
// Timing helpers
// ============================================================

template <typename F>
static double elapsed_ms(F&& fn) {
    auto t0 = Clock::now();
    fn();
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

template <typename F>
static double elapsed_us(F&& fn) {
    auto t0 = Clock::now();
    fn();
    return std::chrono::duration<double, std::micro>(Clock::now() - t0).count();
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

static void normalise_rows(float* data, uint32_t n, uint32_t dim) {
    for (uint32_t i = 0; i < n; ++i) {
        float* row = data + static_cast<size_t>(i) * dim;
        float  sq  = 0.f;
        for (uint32_t d = 0; d < dim; ++d) sq += row[d] * row[d];
        if (sq > 0.f) {
            const float inv = 1.f / std::sqrt(sq);
            for (uint32_t d = 0; d < dim; ++d) row[d] *= inv;
        }
    }
}

// ============================================================
// Latency and throughput measurement
// ============================================================

struct LatencyStats { double p50, p95, p99; };

static double sorted_pct(std::vector<double>& v, double pct) {
    if (v.empty()) return 0.0;
    size_t i = static_cast<size_t>(pct / 100.0 * static_cast<double>(v.size() - 1));
    return v[i];
}

// fn: (const float* query_ptr, uint32_t n_queries) -> void
template <typename SearchFn>
static LatencyStats measure_latency(SearchFn&&   fn,
                                     const float* queries,
                                     uint32_t     n_queries,
                                     uint32_t     dim) {
    for (uint32_t i = 0; i < WARMUP_QUERIES; i++)
        fn(queries + static_cast<size_t>(i % n_queries) * dim, 1);

    std::vector<double> times;
    times.reserve(n_queries);
    for (uint32_t i = 0; i < n_queries; i++) {
        times.push_back(elapsed_us([&] {
            fn(queries + static_cast<size_t>(i) * dim, 1);
        }));
    }
    std::sort(times.begin(), times.end());
    return {sorted_pct(times, 50.0), sorted_pct(times, 95.0), sorted_pct(times, 99.0)};
}

template <typename SearchFn>
static double measure_qps(SearchFn&&   fn,
                            const float* queries,
                            uint32_t     n_queries) {
    fn(queries, std::min(n_queries, 10u));  // warmup
    double ms = elapsed_ms([&] { fn(queries, n_queries); });
    return static_cast<double>(n_queries) / (ms / 1000.0);
}

// ============================================================
// CPU brute-force search (mirrors dispatch_dense_search CPU fallback)
// ============================================================

static SearchResult cpu_dense_search(const float* queries,
                                      const float* db,
                                      uint32_t     n_queries,
                                      uint32_t     n_db,
                                      uint32_t     dim,
                                      uint32_t     k,
                                      Metric       metric) {
    const uint32_t k_actual = (k <= n_db) ? k : n_db;

    SearchResult result;
    result.n_queries = n_queries;
    result.k         = k_actual;
    result.distances.resize(static_cast<size_t>(n_queries) * k_actual);
    result.indices.resize(static_cast<size_t>(n_queries) * k_actual, -1);

    std::vector<std::pair<float, int64_t>> pairs(n_db);

    for (uint32_t q = 0; q < n_queries; ++q) {
        const float* qv    = queries + static_cast<size_t>(q) * dim;
        float*       out_d = result.distances.data() + static_cast<size_t>(q) * k_actual;
        int64_t*     out_i = result.indices.data()   + static_cast<size_t>(q) * k_actual;

        if (metric == Metric::L2) {
            for (uint32_t i = 0; i < n_db; ++i)
                pairs[i] = {l2_distance(qv, db + static_cast<size_t>(i) * dim, dim),
                            static_cast<int64_t>(i)};
        } else {
            // COSINE — both qv and db rows are pre-normalised.
            for (uint32_t i = 0; i < n_db; ++i) {
                const float* dv  = db + static_cast<size_t>(i) * dim;
                float        dot = 0.f;
                for (uint32_t d = 0; d < dim; ++d) dot += qv[d] * dv[d];
                pairs[i] = {1.f - dot, static_cast<int64_t>(i)};
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

// ============================================================
// GPU pipeline (only compiled with -DWITH_CUDA=ON)
// Mirrors the GPU branch of dispatch_dense_search, including H<->D transfer.
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

static SearchResult gpu_dense_search(const float* queries,
                                      const float* db,
                                      uint32_t     n_queries,
                                      uint32_t     n_db,
                                      uint32_t     dim,
                                      uint32_t     k,
                                      Metric       metric) {
    const uint32_t k_actual = (k <= n_db) ? k : n_db;

    SearchResult result;
    result.n_queries = n_queries;
    result.k         = k_actual;
    result.distances.resize(static_cast<size_t>(n_queries) * k_actual);
    result.indices.resize(static_cast<size_t>(n_queries) * k_actual, -1);

    // H → D (included in timing: this is the real wall-clock cost)
    DeviceBuffer<float> d_queries(static_cast<size_t>(n_queries) * dim);
    DeviceBuffer<float> d_db(static_cast<size_t>(n_db) * dim);
    DeviceBuffer<float> d_dist(static_cast<size_t>(n_queries) * n_db);

    d_queries.copy_from_host(queries, static_cast<size_t>(n_queries) * dim);
    d_db.copy_from_host(db,           static_cast<size_t>(n_db) * dim);

    // Distance kernel
    if (metric == Metric::L2) {
        cuda_l2_batch(d_queries.get(), d_db.get(), d_dist.get(), n_queries, n_db, dim);
    } else {
        cuda_cosine_batch(d_queries.get(), d_db.get(), d_dist.get(), n_queries, n_db, dim);
    }

    // Top-k: GPU path if k fits in shared memory, hybrid CPU partial_sort otherwise
    int            dev = 0;
    cudaDeviceProp prop{};
    cudaGetDevice(&dev);
    cudaGetDeviceProperties(&prop, dev);

    const size_t topk_shmem    = static_cast<size_t>(CUDA_TOPK_BLOCK) * k_actual
                                  * (sizeof(float) + sizeof(int64_t));
    const bool   gpu_topk_fits = topk_shmem <= prop.sharedMemPerBlock;

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
        // k too large for GPU top-k shared memory — copy distances to host
        SPARSITY_CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<float> host_dist(static_cast<size_t>(n_queries) * n_db);
        d_dist.copy_to_host(host_dist.data(), static_cast<size_t>(n_queries) * n_db);

        std::vector<std::pair<float, int64_t>> pairs(n_db);
        for (uint32_t q = 0; q < n_queries; ++q) {
            const float* row   = host_dist.data() + static_cast<size_t>(q) * n_db;
            float*       out_d = result.distances.data() + static_cast<size_t>(q) * k_actual;
            int64_t*     out_i = result.indices.data()   + static_cast<size_t>(q) * k_actual;

            for (uint32_t i = 0; i < n_db; ++i) pairs[i] = {row[i], static_cast<int64_t>(i)};
            std::partial_sort(pairs.begin(), pairs.begin() + k_actual, pairs.end());
            for (uint32_t j = 0; j < k_actual; ++j) {
                out_d[j] = pairs[j].first;
                out_i[j] = pairs[j].second;
            }
        }
    }

    return result;
}

#endif // WITH_CUDA

// ============================================================
// Recall helper
// ============================================================

static float compute_recall(const std::vector<int64_t>& result_indices,
                             const std::vector<int64_t>& gt,
                             uint32_t                    n_queries) {
    uint32_t found = 0;
    for (uint32_t q = 0; q < n_queries; q++) {
        std::unordered_set<int64_t> gt_set(
            gt.begin() + static_cast<ptrdiff_t>(q) * K,
            gt.begin() + static_cast<ptrdiff_t>(q) * K + K);
        for (uint32_t j = 0; j < K; j++) {
            int64_t id = result_indices[static_cast<size_t>(q) * K + j];
            if (id >= 0 && gt_set.count(id)) found++;
        }
    }
    return static_cast<float>(found) / (n_queries * K);
}

// ============================================================
// Record and JSON serialisation
// ============================================================

struct Record {
    std::string backend;   // "cpu" or "gpu"
    std::string metric;
    uint32_t    dim   = 0;
    uint32_t    n_db  = 0;
    double      p50_us = 0.0;
    double      p95_us = 0.0;
    double      p99_us = 0.0;
    double      qps    = 0.0;
    double      recall = -1.0;  // GPU recall vs CPU ground truth (should be ~1.0)
};

static std::string to_json(const Record& r) {
    auto opt_double = [](double v) -> std::string {
        if (v < 0.0) return "null";
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(4) << v;
        return ss.str();
    };
    std::ostringstream o;
    o << "    {\n";
    o << "      \"backend\": \""  << r.backend << "\",\n";
    o << "      \"metric\": \""   << r.metric  << "\",\n";
    o << "      \"dim\": "        << r.dim     << ",\n";
    o << "      \"n_db\": "       << r.n_db    << ",\n";
    o << "      \"k\": "          << K         << ",\n";
    o << "      \"p50_us\": "     << std::fixed << std::setprecision(1) << r.p50_us  << ",\n";
    o << "      \"p95_us\": "     << std::fixed << std::setprecision(1) << r.p95_us  << ",\n";
    o << "      \"p99_us\": "     << std::fixed << std::setprecision(1) << r.p99_us  << ",\n";
    o << "      \"qps\": "        << std::fixed << std::setprecision(1) << r.qps     << ",\n";
    o << "      \"recall\": "     << opt_double(r.recall) << "\n";
    o << "    }";
    return o.str();
}

// ============================================================
// Benchmark: CPU vs GPU for one (dim, n_db, metric) combination
// ============================================================

static void bench_cpu_vs_gpu(std::vector<Record>& out,
                               uint32_t             dim,
                               uint32_t             n_db,
                               Metric               metric,
                               const std::string&   mstr) {
    const uint32_t n_lat   = (n_db >= 500000) ? 30  : TIMING_QUERIES;
    const uint32_t n_batch = (n_db >= 500000) ? 50  : BATCH_QPS_SIZE;

    auto db            = gen_gaussian(n_db,    dim, 42);
    auto queries_lat   = gen_gaussian(n_lat,   dim, 100);
    auto queries_batch = gen_gaussian(n_batch, dim, 200);

    // For cosine, both db and queries must be unit-normalised (mirrors DenseIndex).
    if (metric == Metric::COSINE) {
        normalise_rows(db.data(),            n_db,    dim);
        normalise_rows(queries_lat.data(),   n_lat,   dim);
        normalise_rows(queries_batch.data(), n_batch, dim);
    }

    // Ground truth: CPU exact search over the latency query set.
    auto gt = cpu_dense_search(queries_lat.data(), db.data(),
                               n_lat, n_db, dim, K, metric).indices;

    // ----------------------------------------------------------------
    // CPU
    // ----------------------------------------------------------------
    {
        Record r;
        r.backend = "cpu";
        r.metric  = mstr;
        r.dim     = dim;
        r.n_db    = n_db;

        auto lat = measure_latency(
            [&](const float* q, uint32_t nq) {
                cpu_dense_search(q, db.data(), nq, n_db, dim, K, metric);
            },
            queries_lat.data(), n_lat, dim);
        r.p50_us = lat.p50;
        r.p95_us = lat.p95;
        r.p99_us = lat.p99;

        r.qps = measure_qps(
            [&](const float* q, uint32_t nq) {
                cpu_dense_search(q, db.data(), nq, n_db, dim, K, metric);
            },
            queries_batch.data(), n_batch);

        // recall vs self should be 1.0 — confirms ground truth is consistent.
        auto res = cpu_dense_search(queries_lat.data(), db.data(),
                                    n_lat, n_db, dim, K, metric);
        r.recall = compute_recall(res.indices, gt, n_lat);

        out.push_back(r);
        fprintf(stderr,
                "  %-6s  dim=%-5u n_db=%-7u  p50=%8.1f us  qps=%9.0f  recall=%.3f\n",
                "CPU", dim, n_db, r.p50_us, r.qps, r.recall);
    }

#ifdef WITH_CUDA
    // ----------------------------------------------------------------
    // GPU (skipped at runtime if no device, skipped at compile time if no CUDA)
    // ----------------------------------------------------------------
    if (!cuda_available()) {
        fprintf(stderr,
                "  %-6s  dim=%-5u n_db=%-7u  [skipped — no CUDA device]\n",
                "GPU", dim, n_db);
        return;
    }

    {
        Record r;
        r.backend = "gpu";
        r.metric  = mstr;
        r.dim     = dim;
        r.n_db    = n_db;

        auto lat = measure_latency(
            [&](const float* q, uint32_t nq) {
                gpu_dense_search(q, db.data(), nq, n_db, dim, K, metric);
            },
            queries_lat.data(), n_lat, dim);
        r.p50_us = lat.p50;
        r.p95_us = lat.p95;
        r.p99_us = lat.p99;

        r.qps = measure_qps(
            [&](const float* q, uint32_t nq) {
                gpu_dense_search(q, db.data(), nq, n_db, dim, K, metric);
            },
            queries_batch.data(), n_batch);

        // GPU recall vs CPU ground truth: sanity-check correctness.
        auto res = gpu_dense_search(queries_lat.data(), db.data(),
                                     n_lat, n_db, dim, K, metric);
        r.recall = compute_recall(res.indices, gt, n_lat);

        out.push_back(r);
        fprintf(stderr,
                "  %-6s  dim=%-5u n_db=%-7u  p50=%8.1f us  qps=%9.0f  recall=%.3f\n",
                "GPU", dim, n_db, r.p50_us, r.qps, r.recall);
    }
#endif // WITH_CUDA
}

// ============================================================
// main
// ============================================================

int main(int argc, char** argv) {
    std::string out_path = "bench_gpu_results.json";
    if (argc > 1) out_path = argv[1];

#ifdef WITH_CUDA
    const bool        gpu_present = cuda_available();
    const std::string gpu_name    = gpu_present ? cuda_device_name() : "";
    if (gpu_present)
        fprintf(stderr, "GPU: %s\n", gpu_name.c_str());
    else
        fprintf(stderr,
                "GPU: none found — only CPU numbers will be recorded.\n"
                "     Run on a CUDA-capable machine for a full comparison.\n");
#else
    const bool        gpu_present = false;
    const std::string gpu_name    = "";
    fprintf(stderr,
            "Built without -DWITH_CUDA=ON — only CPU numbers will be recorded.\n"
            "Reconfigure with -DWITH_CUDA=ON for GPU vs CPU comparison.\n");
#endif

    const std::vector<uint32_t> DIMS  = {32, 128, 512, 1024};
    const std::vector<uint32_t> N_DBS = {10000, 100000, 500000};

    const std::vector<std::pair<Metric, std::string>> METRICS = {
        {Metric::L2,     "L2"},
        {Metric::COSINE, "Cosine"},
    };

    std::vector<Record> records;

    for (const auto& mp : METRICS) {
        Metric             metric = mp.first;
        const std::string& mstr   = mp.second;
        fprintf(stderr, "\n=== CPU vs GPU [%s] ===\n", mstr.c_str());

        for (uint32_t dim : DIMS) {
            for (uint32_t n_db : N_DBS) {
                if (n_db == 500000 && dim >= 512) {
                    fprintf(stderr,
                            "  [skip: dim=%-4u n_db=%-7u — too slow for full grid]\n",
                            dim, n_db);
                    continue;
                }
                bench_cpu_vs_gpu(records, dim, n_db, metric, mstr);
            }
        }
    }

    // ----------------------------------------------------------------
    // Write JSON
    // ----------------------------------------------------------------
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
    f << "  \"gpu_name\": \""    << gpu_name  << "\",\n";
    f << "  \"k\": "             << K         << ",\n";
    f << "  \"n_records\": "     << records.size() << ",\n";
    f << "  \"results\": [\n";
    for (size_t i = 0; i < records.size(); i++) {
        f << to_json(records[i]);
        if (i + 1 < records.size()) f << ",";
        f << "\n";
    }
    f << "  ]\n}\n";

    fprintf(stderr, "\nWrote %zu records to %s\n", records.size(), out_path.c_str());
    return 0;
}

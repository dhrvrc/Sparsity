// bench_dense.cpp
// Benchmarks DenseIndex, TiledDenseIndex, and IVFIndex across:
//   1. Scaling sweep       — latency / QPS / memory vs (dim, n_db, metric)
//   2. IVF n_probe sweep   — recall vs QPS trade-off curve
//   3. Batch size sweep    — tiling benefit: DenseIndex vs TiledDenseIndex
//
// Outputs a JSON file (default: bench_results.json).
// Pass an output path as the first argument to override.
//
// Usage:
//   ./bench_dense [output.json]
//
// Progress is printed to stderr; only the JSON goes to the output file.

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

#include "sparsity/index.h"
#include "sparsity/ivf.h"
#include "sparsity/tiled_dense.h"

using namespace sparsity;
using Clock = std::chrono::high_resolution_clock;

// ============================================================
// Benchmark parameters
// ============================================================

static constexpr uint32_t K               = 10;   // neighbours to retrieve
static constexpr uint32_t WARMUP_QUERIES  = 20;   // discarded before latency timing
static constexpr uint32_t TIMING_QUERIES  = 200;  // individual queries timed for p50/p95/p99
static constexpr uint32_t BATCH_QPS_SIZE  = 500;  // single batch for QPS measurement
static constexpr int      BATCH_REPS      = 30;   // repetitions for batch-size sweep

// ============================================================
// System utilities
// ============================================================

// Read resident set size from /proc/self/status (Linux only).
// Returns 0 if unavailable.
static long rss_kb() {
    std::ifstream f("/proc/self/status");
    std::string   line;
    while (std::getline(f, line)) {
        if (line.rfind("VmRSS:", 0) == 0) {
            long kb = 0;
            sscanf(line.c_str(), "VmRSS: %ld kB", &kb);
            return kb;
        }
    }
    return 0;
}

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

// IID Gaussian vectors.
static std::vector<float> gen_gaussian(uint32_t n, uint32_t dim, uint32_t seed = 42) {
    std::mt19937                    rng(seed);
    std::normal_distribution<float> dist(0.f, 1.f);
    std::vector<float>              v(static_cast<size_t>(n) * dim);
    for (auto& x : v) x = dist(rng);
    return v;
}

// Clustered Gaussian vectors (n_clusters centres, tight noise around each).
// Produces data with locality structure that IVF exploits for high recall.
static std::vector<float>
gen_clustered(uint32_t n, uint32_t dim, uint32_t n_clusters = 10, uint32_t seed = 42) {
    std::mt19937                    rng(seed);
    std::normal_distribution<float> center_d(0.f, 4.f);
    std::normal_distribution<float> noise_d(0.f, 0.5f);
    std::uniform_int_distribution<uint32_t> pick(0, n_clusters - 1);

    std::vector<float> centers(static_cast<size_t>(n_clusters) * dim);
    for (auto& x : centers) x = center_d(rng);

    std::vector<float> v(static_cast<size_t>(n) * dim);
    for (uint32_t i = 0; i < n; i++) {
        uint32_t c = pick(rng);
        for (uint32_t d = 0; d < dim; d++)
            v[static_cast<size_t>(i) * dim + d] =
                centers[static_cast<size_t>(c) * dim + d] + noise_d(rng);
    }
    return v;
}

// ============================================================
// Ground truth and recall
// ============================================================

// Exact top-K ground truth via DenseIndex brute-force.
// Returns indices shaped [n_queries * K].
static std::vector<int64_t> compute_ground_truth(const float* db,
                                                  uint32_t     n_db,
                                                  const float* queries,
                                                  uint32_t     n_queries,
                                                  uint32_t     dim,
                                                  Metric       metric) {
    DenseIndex idx(metric);
    idx.add(db, n_db, dim);
    auto r = idx.search(queries, n_queries, K);
    return r.indices;
}

// Recall@K: fraction of true top-K neighbours present in result.
static float compute_recall(const SearchResult&        result,
                             const std::vector<int64_t>& gt) {
    uint32_t found = 0;
    for (uint32_t q = 0; q < result.n_queries; q++) {
        std::unordered_set<int64_t> gt_set(gt.begin() + static_cast<ptrdiff_t>(q) * K,
                                           gt.begin() + static_cast<ptrdiff_t>(q) * K + K);
        for (uint32_t j = 0; j < result.k; j++) {
            int64_t id = result.indices[static_cast<size_t>(q) * result.k + j];
            if (id >= 0 && gt_set.count(id)) found++;
        }
    }
    return static_cast<float>(found) / (result.n_queries * K);
}

// ============================================================
// Latency and throughput measurement
// ============================================================

struct LatencyStats {
    double p50, p95, p99;
};

static double sorted_pct(std::vector<double>& v, double pct) {
    if (v.empty()) return 0.0;
    size_t i = static_cast<size_t>(pct / 100.0 * static_cast<double>(v.size() - 1));
    return v[i];
}

// SearchFn: void(const float* q, uint32_t n_queries)
// Measures single-query latency over TIMING_QUERIES individual calls.
template <typename SearchFn>
static LatencyStats measure_latency(SearchFn&&   fn,
                                     const float* queries,
                                     uint32_t     n_queries,
                                     uint32_t     dim) {
    // Warmup
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

// Measures batch throughput (queries per second).
template <typename SearchFn>
static double measure_qps(SearchFn&&   fn,
                            const float* queries,
                            uint32_t     n_queries) {
    fn(queries, std::min(n_queries, 10u));  // warmup
    double ms = elapsed_ms([&] { fn(queries, n_queries); });
    return static_cast<double>(n_queries) / (ms / 1000.0);
}

// ============================================================
// Result record and JSON serialisation
// ============================================================

struct Record {
    std::string index_name;
    std::string metric;
    std::string distribution;
    uint32_t    dim        = 0;
    uint32_t    n_db       = 0;
    int         n_lists    = -1;   // -1 = not applicable
    int         n_probe    = -1;
    int         batch_size = -1;   // -1 = not a batch-sweep record

    double build_ms            = 0.0;
    double p50_us              = 0.0;
    double p95_us              = 0.0;
    double p99_us              = 0.0;
    double qps                 = 0.0;
    double recall              = -1.0;  // -1 = not measured
    long   build_rss_delta_kb  = 0;
    long   theoretical_bytes   = 0;
};

static std::string to_json(const Record& r) {
    auto opt_int = [](int v) -> std::string {
        return v < 0 ? "null" : std::to_string(v);
    };
    auto opt_double = [](double v) -> std::string {
        if (v < 0.0) return "null";
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(4) << v;
        return ss.str();
    };
    std::ostringstream o;
    o << "    {\n";
    o << "      \"index\": \""        << r.index_name      << "\",\n";
    o << "      \"metric\": \""       << r.metric          << "\",\n";
    o << "      \"distribution\": \"" << r.distribution    << "\",\n";
    o << "      \"dim\": "            << r.dim              << ",\n";
    o << "      \"n_db\": "           << r.n_db             << ",\n";
    o << "      \"k\": "              << K                  << ",\n";
    o << "      \"n_lists\": "        << opt_int(r.n_lists) << ",\n";
    o << "      \"n_probe\": "        << opt_int(r.n_probe) << ",\n";
    o << "      \"batch_size\": "     << opt_int(r.batch_size) << ",\n";
    o << "      \"build_ms\": "       << std::fixed << std::setprecision(2) << r.build_ms << ",\n";
    o << "      \"p50_us\": "         << std::fixed << std::setprecision(1) << r.p50_us   << ",\n";
    o << "      \"p95_us\": "         << std::fixed << std::setprecision(1) << r.p95_us   << ",\n";
    o << "      \"p99_us\": "         << std::fixed << std::setprecision(1) << r.p99_us   << ",\n";
    o << "      \"qps\": "            << std::fixed << std::setprecision(1) << r.qps      << ",\n";
    o << "      \"recall\": "         << opt_double(r.recall) << ",\n";
    o << "      \"build_rss_delta_kb\": " << r.build_rss_delta_kb << ",\n";
    o << "      \"theoretical_bytes\": "  << r.theoretical_bytes  << "\n";
    o << "    }";
    return o.str();
}

// ============================================================
// Benchmark 1: Scaling sweep — DenseIndex vs TiledDenseIndex
// ============================================================

static void bench_scaling(std::vector<Record>& out,
                           uint32_t             dim,
                           uint32_t             n_db,
                           Metric               metric,
                           const std::string&   mstr) {
    // Scale down query counts for large DBs so the benchmark stays interactive.
    // Each distance computation at n_db=500K, dim=128 takes ~30 ms; 200 queries
    // would take ~6 s per index just for latency timing.
    const uint32_t n_lat   = (n_db >= 500000) ? 30  : TIMING_QUERIES;
    const uint32_t n_batch = (n_db >= 500000) ? 50  : BATCH_QPS_SIZE;

    auto db            = gen_gaussian(n_db,    dim, 42);
    auto queries_lat   = gen_gaussian(n_lat,   dim, 100);
    auto queries_batch = gen_gaussian(n_batch, dim, 200);
    auto gt = compute_ground_truth(db.data(), n_db, queries_lat.data(), n_lat, dim, metric);

    // ---- DenseIndex ----
    {
        Record r;
        r.index_name   = "DenseIndex";
        r.metric       = mstr;
        r.distribution = "gaussian";
        r.dim          = dim;
        r.n_db         = n_db;
        r.theoretical_bytes = static_cast<long>(n_db) * dim * 4;

        long       rss0 = rss_kb();
        DenseIndex idx(metric);
        r.build_ms           = elapsed_ms([&] { idx.add(db.data(), n_db, dim); });
        r.build_rss_delta_kb = rss_kb() - rss0;

        auto lat = measure_latency(
            [&](const float* q, uint32_t nq) { idx.search(q, nq, K); },
            queries_lat.data(), n_lat, dim);
        r.p50_us = lat.p50;
        r.p95_us = lat.p95;
        r.p99_us = lat.p99;

        r.qps = measure_qps(
            [&](const float* q, uint32_t nq) { idx.search(q, nq, K); },
            queries_batch.data(), n_batch);

        auto res = idx.search(queries_lat.data(), n_lat, K);
        r.recall = compute_recall(res, gt);
        out.push_back(r);

        fprintf(stderr, "  %-18s dim=%-5u n_db=%-7u  p50=%8.1f us  qps=%9.0f  recall=%.3f\n",
                "DenseIndex", dim, n_db, r.p50_us, r.qps, r.recall);
    }

    // ---- TiledDenseIndex ----
    {
        Record r;
        r.index_name   = "TiledDenseIndex";
        r.metric       = mstr;
        r.distribution = "gaussian";
        r.dim          = dim;
        r.n_db         = n_db;
        // L2 also stores one float norm per vector
        r.theoretical_bytes = static_cast<long>(n_db) * dim * 4
                              + (metric == Metric::L2 ? static_cast<long>(n_db) * 4 : 0);

        long            rss0 = rss_kb();
        TiledDenseIndex idx(metric);
        r.build_ms           = elapsed_ms([&] { idx.add(db.data(), n_db, dim); });
        r.build_rss_delta_kb = rss_kb() - rss0;

        auto lat = measure_latency(
            [&](const float* q, uint32_t nq) { idx.search(q, nq, K); },
            queries_lat.data(), n_lat, dim);
        r.p50_us = lat.p50;
        r.p95_us = lat.p95;
        r.p99_us = lat.p99;

        r.qps = measure_qps(
            [&](const float* q, uint32_t nq) { idx.search(q, nq, K); },
            queries_batch.data(), n_batch);

        auto res = idx.search(queries_lat.data(), n_lat, K);
        r.recall = compute_recall(res, gt);
        out.push_back(r);

        fprintf(stderr, "  %-18s dim=%-5u n_db=%-7u  p50=%8.1f us  qps=%9.0f  recall=%.3f\n",
                "TiledDenseIndex", dim, n_db, r.p50_us, r.qps, r.recall);
    }
}


// ============================================================
// Benchmark 2: IVF n_probe sweep
// ============================================================

static void bench_ivf_nprobe(std::vector<Record>& out,
                               uint32_t             dim,
                               uint32_t             n_db,
                               uint32_t             n_lists,
                               Metric               metric,
                               const std::string&   mstr,
                               const std::string&   dist) {
    auto db = (dist == "clustered") ? gen_clustered(n_db, dim, 10, 42)
                                    : gen_gaussian(n_db, dim, 42);
    auto queries_lat   = gen_gaussian(TIMING_QUERIES, dim, 100);
    auto queries_batch = gen_gaussian(BATCH_QPS_SIZE, dim, 200);
    auto gt = compute_ground_truth(db.data(), n_db, queries_lat.data(), TIMING_QUERIES, dim, metric);

    // Build IVF index once — shared across all n_probe values
    long     rss0 = rss_kb();
    IVFIndex idx(metric, n_lists, 1);

    fprintf(stderr, "    training (n_lists=%u)...", n_lists);
    fflush(stderr);
    double train_ms = elapsed_ms([&] { idx.train(db.data(), n_db, dim); });

    fprintf(stderr, " %.0f ms  |  adding...", train_ms);
    fflush(stderr);
    double add_ms = elapsed_ms([&] { idx.add(db.data(), n_db, dim); });
    fprintf(stderr, " %.0f ms\n", add_ms);

    double build_ms   = train_ms + add_ms;
    long   build_rss  = rss_kb() - rss0;
    // vectors + centroids + ids
    long   theory = static_cast<long>(n_db) * dim * 4
                  + static_cast<long>(n_lists) * dim * 4
                  + static_cast<long>(n_db) * 8;

    // n_probe values: powers of 2 from 1 up to n_lists
    std::vector<uint32_t> probes;
    for (uint32_t p = 1; p < n_lists; p *= 2) probes.push_back(p);
    probes.push_back(n_lists);

    for (uint32_t np : probes) {
        idx.set_n_probe(np);

        Record r;
        r.index_name       = "IVFIndex";
        r.metric           = mstr;
        r.distribution     = dist;
        r.dim              = dim;
        r.n_db             = n_db;
        r.n_lists          = static_cast<int>(n_lists);
        r.n_probe          = static_cast<int>(np);
        r.build_ms         = build_ms;
        r.build_rss_delta_kb = build_rss;
        r.theoretical_bytes  = theory;

        auto lat = measure_latency(
            [&](const float* q, uint32_t nq) { idx.search(q, nq, K); },
            queries_lat.data(), TIMING_QUERIES, dim);
        r.p50_us = lat.p50;
        r.p95_us = lat.p95;
        r.p99_us = lat.p99;

        r.qps = measure_qps(
            [&](const float* q, uint32_t nq) { idx.search(q, nq, K); },
            queries_batch.data(), BATCH_QPS_SIZE);

        auto res = idx.search(queries_lat.data(), TIMING_QUERIES, K);
        r.recall = compute_recall(res, gt);
        out.push_back(r);

        fprintf(stderr,
                "  IVFIndex[%-9s] n_probe=%-4u  p50=%8.1f us  qps=%9.0f  recall=%.3f\n",
                dist.c_str(), np, r.p50_us, r.qps, r.recall);
    }
}

// ============================================================
// Benchmark 3: Batch size sweep — tiling benefit
// ============================================================

static void bench_batch_sizes(std::vector<Record>& out,
                                uint32_t             dim,
                                uint32_t             n_db,
                                Metric               metric,
                                const std::string&   mstr) {
    static const std::vector<uint32_t> BATCH_SIZES = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512};
    uint32_t max_bs = BATCH_SIZES.back();

    auto db      = gen_gaussian(n_db, dim, 42);
    auto queries = gen_gaussian(max_bs, dim, 100);

    DenseIndex      dense(metric);
    TiledDenseIndex tiled(metric);
    dense.add(db.data(), n_db, dim);
    tiled.add(db.data(), n_db, dim);

    for (uint32_t bs : BATCH_SIZES) {
        // DenseIndex
        {
            dense.search(queries.data(), bs, K);  // warmup
            double ms = elapsed_ms([&] {
                for (int rep = 0; rep < BATCH_REPS; rep++)
                    dense.search(queries.data(), bs, K);
            });
            Record r;
            r.index_name   = "DenseIndex";
            r.metric       = mstr;
            r.distribution = "gaussian";
            r.dim          = dim;
            r.n_db         = n_db;
            r.batch_size   = static_cast<int>(bs);
            r.qps          = static_cast<double>(BATCH_REPS * bs) / (ms / 1000.0);
            out.push_back(r);
        }
        // TiledDenseIndex
        {
            tiled.search(queries.data(), bs, K);  // warmup
            double ms = elapsed_ms([&] {
                for (int rep = 0; rep < BATCH_REPS; rep++)
                    tiled.search(queries.data(), bs, K);
            });
            Record r;
            r.index_name   = "TiledDenseIndex";
            r.metric       = mstr;
            r.distribution = "gaussian";
            r.dim          = dim;
            r.n_db         = n_db;
            r.batch_size   = static_cast<int>(bs);
            r.qps          = static_cast<double>(BATCH_REPS * bs) / (ms / 1000.0);
            out.push_back(r);
        }
        fprintf(stderr, "  batch=%-4u  Dense=%9.0f qps   Tiled=%9.0f qps\n",
                bs, out[out.size() - 2].qps, out.back().qps);
    }
}

// ============================================================
// main
// ============================================================

int main(int argc, char** argv) {
    std::string out_path = "bench_results.json";
    if (argc > 1) out_path = argv[1];

    std::vector<Record> records;

    // ----------------------------------------------------------------
    // 1. Scaling sweep: DenseIndex vs TiledDenseIndex
    //    dims × n_dbs × metrics
    //    Skip (n_db=500K, dim>=512) to keep runtime manageable.
    // ----------------------------------------------------------------
    const std::vector<uint32_t> DIMS  = {32, 128, 512, 1024};
    const std::vector<uint32_t> N_DBS = {10000, 100000, 500000};

    const std::vector<std::pair<Metric, std::string>> METRICS = {
        {Metric::L2,     "L2"},
        {Metric::COSINE, "Cosine"},
    };

    for (const auto& mp : METRICS) {
        Metric             metric = mp.first;
        const std::string& mstr   = mp.second;
        fprintf(stderr, "\n=== Scaling sweep [%s] ===\n", mstr.c_str());

        for (uint32_t dim : DIMS) {
            for (uint32_t n_db : N_DBS) {
                if (n_db == 500000 && dim >= 512) {
                    fprintf(stderr, "  [skip: dim=%-4u n_db=%-7u — too slow for full grid]\n",
                            dim, n_db);
                    continue;
                }
                bench_scaling(records, dim, n_db, metric, mstr);
            }
        }
    }

    // ----------------------------------------------------------------
    // 2. IVF n_probe sweep
    //    Fixed dim=128; two n_db sizes; both gaussian and clustered data
    // ----------------------------------------------------------------
    struct IvfCfg {
        uint32_t dim, n_db, n_lists;
    };
    const std::vector<IvfCfg> IVF_CFGS = {
        {128, 10000,  64},
        {128, 100000, 128},
    };

    for (const auto& mp : METRICS) {
        Metric             metric = mp.first;
        const std::string& mstr   = mp.second;
        fprintf(stderr, "\n=== IVF n_probe sweep [%s] ===\n", mstr.c_str());

        for (const auto& cfg : IVF_CFGS) {
            fprintf(stderr, "\n  dim=%u  n_db=%u  n_lists=%u  [gaussian]\n",
                    cfg.dim, cfg.n_db, cfg.n_lists);
            bench_ivf_nprobe(records, cfg.dim, cfg.n_db, cfg.n_lists, metric, mstr, "gaussian");

            fprintf(stderr, "\n  dim=%u  n_db=%u  n_lists=%u  [clustered]\n",
                    cfg.dim, cfg.n_db, cfg.n_lists);
            bench_ivf_nprobe(records, cfg.dim, cfg.n_db, cfg.n_lists, metric, mstr, "clustered");
        }
    }

    // ----------------------------------------------------------------
    // 3. Batch size sweep — tiling benefit
    //    Two dims to show how the benefit grows with vector size
    // ----------------------------------------------------------------
    fprintf(stderr, "\n=== Batch size sweep [L2] ===\n");

    fprintf(stderr, "\n  dim=128  n_db=100000\n");
    bench_batch_sizes(records, 128, 100000, Metric::L2, "L2");

    fprintf(stderr, "\n  dim=512  n_db=100000\n");
    bench_batch_sizes(records, 512, 100000, Metric::L2, "L2");

    // ----------------------------------------------------------------
    // Write JSON
    // ----------------------------------------------------------------
    std::ofstream f(out_path);
    if (!f) {
        fprintf(stderr, "ERROR: cannot open output file: %s\n", out_path.c_str());
        return 1;
    }

    char     tbuf[64];
    std::time_t now = std::time(nullptr);
    std::strftime(tbuf, sizeof(tbuf), "%Y-%m-%dT%H:%M:%S", std::localtime(&now));

    f << "{\n";
    f << "  \"version\": \"0.1.0\",\n";
    f << "  \"timestamp\": \"" << tbuf << "\",\n";
    f << "  \"k\": " << K << ",\n";
    f << "  \"n_records\": " << records.size() << ",\n";
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

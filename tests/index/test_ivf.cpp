#include "sparsity/ivf.h"
#include "sparsity/index.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <set>
#include <vector>

using namespace sparsity;

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

// recall@k: fraction of ground-truth top-k indices present in candidate top-k.
static float recall_at_k(const int64_t* gt, const int64_t* cand, uint32_t k) {
    std::set<int64_t> gt_set(gt, gt + k);
    uint32_t hits = 0;
    for (uint32_t i = 0; i < k; ++i) {
        if (gt_set.count(cand[i])) ++hits;
    }
    return static_cast<float>(hits) / static_cast<float>(k);
}

// Generate n random vectors of dimension dim (not normalised).
static std::vector<float> random_vecs(uint32_t n, uint32_t dim, uint32_t seed = 1234) {
    std::mt19937                    rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float>              v(static_cast<size_t>(n) * dim);
    for (auto& x : v) x = nd(rng);
    return v;
}

// Generate n clustered vectors: n_clusters centres spread with std=5, points
// perturbed with std=0.3. IVF recall is high on this kind of data because
// nearest neighbours tend to live in the same Voronoi cell.
static std::vector<float> clustered_vecs(uint32_t n, uint32_t dim,
                                          uint32_t n_clusters, uint32_t seed = 1234) {
    std::mt19937                    rng(seed);
    std::normal_distribution<float> centre_nd(0.0f, 5.0f);
    std::normal_distribution<float> point_nd(0.0f, 0.3f);

    std::vector<float> centres(static_cast<size_t>(n_clusters) * dim);
    for (auto& x : centres) x = centre_nd(rng);

    std::uniform_int_distribution<uint32_t> pick(0, n_clusters - 1);
    std::vector<float> vecs(static_cast<size_t>(n) * dim);
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t c = pick(rng);
        for (uint32_t d = 0; d < dim; ++d)
            vecs[static_cast<size_t>(i) * dim + d] =
                centres[static_cast<size_t>(c) * dim + d] + point_nd(rng);
    }
    return vecs;
}

// ---------------------------------------------------------------------------
// Error handling
// ---------------------------------------------------------------------------

TEST(IVFIndexTest, SearchBeforeTrainThrows) {
    IVFIndex idx(Metric::L2, 4, 2);
    float q[4] = {1, 0, 0, 0};
    EXPECT_THROW(idx.search(q, 1, 1), std::runtime_error);
}

TEST(IVFIndexTest, AddBeforeTrainThrows) {
    IVFIndex idx(Metric::L2, 4, 2);
    float d[4] = {1, 0, 0, 0};
    EXPECT_THROW(idx.add(d, 1, 4), std::runtime_error);
}

TEST(IVFIndexTest, AlreadyTrainedThrows) {
    IVFIndex idx(Metric::L2, 2, 1);
    float data[8] = {1, 0, 0, 0, 0, 1, 0, 0};
    idx.train(data, 2, 4);
    EXPECT_THROW(idx.train(data, 2, 4), std::invalid_argument);
}

TEST(IVFIndexTest, InvalidMetricThrows) {
    EXPECT_THROW(IVFIndex(Metric::TANIMOTO, 4, 2), std::invalid_argument);
}

TEST(IVFIndexTest, DimMismatchOnAddThrows) {
    IVFIndex idx(Metric::L2, 2, 1);
    float data[8] = {1, 0, 0, 0, 0, 1, 0, 0};
    idx.train(data, 2, 4);
    idx.add(data, 2, 4);
    float bad[3] = {1, 0, 0};
    EXPECT_THROW(idx.add(bad, 1, 3), std::invalid_argument);
}

TEST(IVFIndexTest, TooFewTrainingVectorsThrows) {
    IVFIndex idx(Metric::L2, 10, 2);
    float data[8] = {1, 0, 0, 0, 0, 1, 0, 0};
    EXPECT_THROW(idx.train(data, 2, 4), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Exact recall: n_probe == n_lists probes every cell → same results as brute-force
// ---------------------------------------------------------------------------

TEST(IVFIndexTest, ExactRecallFullProbeL2) {
    const uint32_t n = 500, dim = 32, n_lists = 8, k = 5;
    auto vecs = random_vecs(n, dim, 42);

    // Brute-force ground truth.
    DenseIndex bf(Metric::L2);
    bf.add(vecs.data(), n, dim);

    // IVF probing all cells.
    IVFIndex idx(Metric::L2, n_lists, n_lists);
    idx.train(vecs.data(), n, dim);
    idx.add(vecs.data(), n, dim);

    auto queries = random_vecs(10, dim, 99);
    auto bf_res  = bf.search(queries.data(), 10, k);
    auto ivf_res = idx.search(queries.data(), 10, k);

    for (uint32_t q = 0; q < 10; ++q) {
        float r = recall_at_k(bf_res.indices.data()  + q * k,
                              ivf_res.indices.data() + q * k, k);
        EXPECT_FLOAT_EQ(r, 1.0f) << "query " << q << " — L2 exact recall failed";
    }
}

TEST(IVFIndexTest, ExactRecallFullProbeCosine) {
    const uint32_t n = 500, dim = 32, n_lists = 8, k = 5;
    auto vecs = random_vecs(n, dim, 42);

    DenseIndex bf(Metric::COSINE);
    bf.add(vecs.data(), n, dim);

    IVFIndex idx(Metric::COSINE, n_lists, n_lists);
    idx.train(vecs.data(), n, dim);
    idx.add(vecs.data(), n, dim);

    auto queries = random_vecs(10, dim, 99);
    auto bf_res  = bf.search(queries.data(), 10, k);
    auto ivf_res = idx.search(queries.data(), 10, k);

    for (uint32_t q = 0; q < 10; ++q) {
        float r = recall_at_k(bf_res.indices.data()  + q * k,
                              ivf_res.indices.data() + q * k, k);
        EXPECT_FLOAT_EQ(r, 1.0f) << "query " << q << " — Cosine exact recall failed";
    }
}

// ---------------------------------------------------------------------------
// Partial probe — acceptable recall on random data
// ---------------------------------------------------------------------------

TEST(IVFIndexTest, HighRecallPartialProbeL2) {
    // Use clustered data: IVF is designed for data with locality structure.
    // Points near each other share a Voronoi cell, so partial probing finds
    // most true neighbours.
    const uint32_t n = 2000, dim = 32, n_lists = 16, n_probe = 4, k = 10;
    auto vecs    = clustered_vecs(n, dim, n_lists, 7);
    auto queries = clustered_vecs(50, dim, n_lists, 17);

    DenseIndex bf(Metric::L2);
    bf.add(vecs.data(), n, dim);

    IVFIndex idx(Metric::L2, n_lists, n_probe);
    idx.train(vecs.data(), n, dim);
    idx.add(vecs.data(), n, dim);

    const uint32_t n_queries = 50;
    auto bf_res  = bf.search(queries.data(), n_queries, k);
    auto ivf_res = idx.search(queries.data(), n_queries, k);

    float total = 0.0f;
    for (uint32_t q = 0; q < n_queries; ++q) {
        total += recall_at_k(bf_res.indices.data()  + q * k,
                             ivf_res.indices.data() + q * k, k);
    }
    const float avg = total / static_cast<float>(n_queries);
    EXPECT_GE(avg, 0.80f) << "L2 average recall@10 too low: " << avg;
}

TEST(IVFIndexTest, HighRecallPartialProbeCosine) {
    const uint32_t n = 2000, dim = 32, n_lists = 16, n_probe = 4, k = 10;
    auto vecs    = clustered_vecs(n, dim, n_lists, 7);
    auto queries = clustered_vecs(50, dim, n_lists, 17);

    DenseIndex bf(Metric::COSINE);
    bf.add(vecs.data(), n, dim);

    IVFIndex idx(Metric::COSINE, n_lists, n_probe);
    idx.train(vecs.data(), n, dim);
    idx.add(vecs.data(), n, dim);

    const uint32_t n_queries = 50;
    auto bf_res  = bf.search(queries.data(), n_queries, k);
    auto ivf_res = idx.search(queries.data(), n_queries, k);

    float total = 0.0f;
    for (uint32_t q = 0; q < n_queries; ++q) {
        total += recall_at_k(bf_res.indices.data()  + q * k,
                             ivf_res.indices.data() + q * k, k);
    }
    const float avg = total / static_cast<float>(n_queries);
    EXPECT_GE(avg, 0.80f) << "Cosine average recall@10 too low: " << avg;
}

// ---------------------------------------------------------------------------
// Edge cases
// ---------------------------------------------------------------------------

TEST(IVFIndexTest, KGreaterThanN) {
    IVFIndex idx(Metric::L2, 2, 2);
    float data[8] = {1, 0, 0, 0, 0, 1, 0, 0};
    idx.train(data, 2, 4);
    idx.add(data, 2, 4);

    float q[4] = {1, 0, 0, 0};
    auto res = idx.search(q, 1, 100); // k=100, only 2 vectors indexed
    EXPECT_EQ(res.k, 2u);
    EXPECT_EQ(res.indices.size(), 2u);
}

TEST(IVFIndexTest, SetNProbeAdjustsRecall) {
    const uint32_t n = 1000, dim = 32, n_lists = 8, k = 5;
    auto vecs = random_vecs(n, dim, 55);

    DenseIndex bf(Metric::L2);
    bf.add(vecs.data(), n, dim);

    // Start with minimum probe.
    IVFIndex idx(Metric::L2, n_lists, 1);
    idx.train(vecs.data(), n, dim);
    idx.add(vecs.data(), n, dim);

    const uint32_t n_queries = 20;
    auto queries = random_vecs(n_queries, dim, 66);
    auto bf_res  = bf.search(queries.data(), n_queries, k);

    auto res_low = idx.search(queries.data(), n_queries, k);

    // Raise n_probe to n_lists → exact results.
    idx.set_n_probe(n_lists);
    auto res_high = idx.search(queries.data(), n_queries, k);

    float recall_low = 0.0f, recall_high = 0.0f;
    for (uint32_t q = 0; q < n_queries; ++q) {
        recall_low  += recall_at_k(bf_res.indices.data()   + q * k,
                                   res_low.indices.data()  + q * k, k);
        recall_high += recall_at_k(bf_res.indices.data()   + q * k,
                                   res_high.indices.data() + q * k, k);
    }
    recall_low  /= static_cast<float>(n_queries);
    recall_high /= static_cast<float>(n_queries);

    EXPECT_GE(recall_high, recall_low);
    EXPECT_FLOAT_EQ(recall_high, 1.0f); // n_probe == n_lists → exact
}

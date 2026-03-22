#include "sparsity/index.h"
#include "sparsity/metrics.h"

#include <gtest/gtest.h>
#include <cmath>
#include <numeric>
#include <vector>

using namespace sparsity;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Build a CSR batch from a list of dense rows (non-zeros only; zeros elided).
// Each element of `rows` is a dense float vector; we convert to CSR.
struct CSRBatch {
    std::vector<uint32_t> indptr;
    std::vector<uint32_t> indices;
    std::vector<float>    values;
    uint32_t              dim;

    CSRBatch(const std::vector<std::vector<float>>& rows, uint32_t d)
        : dim(d) {
        indptr.push_back(0);
        for (const auto& row : rows) {
            for (uint32_t j = 0; j < static_cast<uint32_t>(row.size()); ++j) {
                if (row[j] != 0.0f) {
                    indices.push_back(j);
                    values.push_back(row[j]);
                }
            }
            indptr.push_back(static_cast<uint32_t>(indices.size()));
        }
    }
};

// Compute brute-force L2 distances between one dense query and dense rows.
static std::vector<float> ref_l2(const std::vector<float>& q,
                                  const std::vector<std::vector<float>>& db) {
    std::vector<float> dists;
    for (const auto& row : db) {
        float s = 0.f;
        for (uint32_t i = 0; i < static_cast<uint32_t>(row.size()); ++i) {
            const float d = q[i] - row[i];
            s += d * d;
        }
        dists.push_back(std::sqrt(s));
    }
    return dists;
}

// Compute brute-force cosine distances between one dense query and dense rows.
static std::vector<float> ref_cosine(const std::vector<float>& q,
                                     const std::vector<std::vector<float>>& db) {
    float qnorm = 0.f;
    for (float v : q) qnorm += v * v;
    qnorm = std::sqrt(qnorm);

    std::vector<float> dists;
    for (const auto& row : db) {
        float dot = 0.f, rnorm = 0.f;
        for (uint32_t i = 0; i < static_cast<uint32_t>(row.size()); ++i) {
            dot   += q[i] * row[i];
            rnorm += row[i] * row[i];
        }
        rnorm = std::sqrt(rnorm);
        dists.push_back((qnorm == 0.f || rnorm == 0.f) ? 1.f : 1.f - dot / (qnorm * rnorm));
    }
    return dists;
}

// Return sorted (distance, original_index) pairs, ascending by distance.
static std::vector<std::pair<float, int64_t>> topk(const std::vector<float>& dists, uint32_t k) {
    std::vector<std::pair<float, int64_t>> pairs;
    for (uint32_t i = 0; i < static_cast<uint32_t>(dists.size()); ++i)
        pairs.push_back({dists[i], static_cast<int64_t>(i)});
    std::sort(pairs.begin(), pairs.end());
    pairs.resize(k);
    return pairs;
}

// ---------------------------------------------------------------------------
// COSINE search tests
// ---------------------------------------------------------------------------

TEST(SparseIndexCosine, BasicTopK) {
    // 4 sparse vectors in a 5-dim space:
    //   v0 = [1, 0, 0, 0, 0]
    //   v1 = [0, 1, 0, 0, 0]
    //   v2 = [1, 1, 0, 0, 0]  (same direction as (1,1,0,0,0))
    //   v3 = [0, 0, 1, 0, 0]
    const uint32_t dim = 5;
    std::vector<std::vector<float>> db_rows = {
        {1, 0, 0, 0, 0},
        {0, 1, 0, 0, 0},
        {1, 1, 0, 0, 0},
        {0, 0, 1, 0, 0},
    };
    CSRBatch db(db_rows, dim);

    Index idx(DataType::SparseFloat, Metric::COSINE, dim);
    idx.add(db.indptr.data(), db.indices.data(), db.values.data(),
            static_cast<uint32_t>(db_rows.size()));

    // Query: [1, 1, 0, 0, 0] — should be closest to v2 (cos dist = 0), then v0/v1.
    std::vector<float> q_dense = {1, 1, 0, 0, 0};
    CSRBatch           q({{q_dense}}, dim);

    auto res = idx.search(q.indptr.data(), q.indices.data(), q.values.data(), 1, 3);

    ASSERT_EQ(res.n_queries, 1u);
    ASSERT_EQ(res.k, 3u);

    // v2 is identical direction → cosine distance ≈ 0.
    EXPECT_EQ(res.indices[0], 2);
    EXPECT_NEAR(res.distances[0], 0.0f, 1e-5f);
}

TEST(SparseIndexCosine, MatchesBruteForce) {
    const uint32_t dim = 8;
    std::vector<std::vector<float>> db_rows = {
        {2, 0, 0, 3, 0, 0, 0, 1},
        {0, 1, 0, 0, 2, 0, 3, 0},
        {1, 1, 0, 0, 0, 0, 1, 1},
        {0, 0, 4, 0, 0, 2, 0, 0},
        {3, 0, 0, 0, 1, 0, 0, 2},
    };
    CSRBatch db(db_rows, dim);

    Index idx(DataType::SparseFloat, Metric::COSINE, dim);
    idx.add(db.indptr.data(), db.indices.data(), db.values.data(),
            static_cast<uint32_t>(db_rows.size()));

    std::vector<float> q_dense = {1, 0, 0, 2, 0, 0, 0, 1};
    CSRBatch q({{q_dense}}, dim);

    const uint32_t k = 3;
    auto res = idx.search(q.indptr.data(), q.indices.data(), q.values.data(), 1, k);

    // Brute-force reference.
    auto ref_dists = ref_cosine(q_dense, db_rows);
    auto ref_topk  = topk(ref_dists, k);

    for (uint32_t j = 0; j < k; ++j) {
        EXPECT_EQ(res.indices[j],   ref_topk[j].second) << "rank " << j;
        EXPECT_NEAR(res.distances[j], ref_topk[j].first, 1e-5f) << "rank " << j;
    }
}

TEST(SparseIndexCosine, MultipleQueries) {
    const uint32_t dim = 6;
    std::vector<std::vector<float>> db_rows = {
        {1, 0, 0, 0, 0, 0},
        {0, 1, 0, 0, 0, 0},
        {0, 0, 1, 0, 0, 0},
        {1, 1, 0, 0, 0, 0},
    };
    CSRBatch db(db_rows, dim);

    Index idx(DataType::SparseFloat, Metric::COSINE, dim);
    idx.add(db.indptr.data(), db.indices.data(), db.values.data(),
            static_cast<uint32_t>(db_rows.size()));

    std::vector<std::vector<float>> q_rows = {
        {1, 0, 0, 0, 0, 0},
        {0, 0, 1, 0, 0, 0},
    };
    CSRBatch q(q_rows, dim);

    auto res = idx.search(q.indptr.data(), q.indices.data(), q.values.data(), 2, 1);

    ASSERT_EQ(res.n_queries, 2u);
    // Query 0 is v0 → best match is index 0 (distance 0)
    EXPECT_EQ(res.indices[0], 0);
    EXPECT_NEAR(res.distances[0], 0.0f, 1e-5f);
    // Query 1 is v2 → best match is index 2 (distance 0)
    EXPECT_EQ(res.indices[1], 2);
    EXPECT_NEAR(res.distances[1], 0.0f, 1e-5f);
}

TEST(SparseIndexCosine, IncrementalAdd) {
    const uint32_t dim = 4;
    // Add two batches and verify index sees all vectors.
    std::vector<std::vector<float>> batch1 = {{1, 0, 0, 0}, {0, 1, 0, 0}};
    std::vector<std::vector<float>> batch2 = {{0, 0, 1, 0}, {0, 0, 0, 1}};
    CSRBatch b1(batch1, dim), b2(batch2, dim);

    Index idx(DataType::SparseFloat, Metric::COSINE, dim);
    idx.add(b1.indptr.data(), b1.indices.data(), b1.values.data(), 2);
    idx.add(b2.indptr.data(), b2.indices.data(), b2.values.data(), 2);

    EXPECT_EQ(idx.size(), 4u);

    // Query closest to dim 2 → should find index 2 (added in batch 2)
    std::vector<float> q_dense = {0, 0, 1, 0};
    CSRBatch q({{q_dense}}, dim);
    auto res = idx.search(q.indptr.data(), q.indices.data(), q.values.data(), 1, 1);
    EXPECT_EQ(res.indices[0], 2);
    EXPECT_NEAR(res.distances[0], 0.0f, 1e-5f);
}

TEST(SparseIndexCosine, KLargerThanN) {
    const uint32_t dim = 3;
    std::vector<std::vector<float>> db_rows = {{1, 0, 0}, {0, 1, 0}};
    CSRBatch db(db_rows, dim);

    Index idx(DataType::SparseFloat, Metric::COSINE, dim);
    idx.add(db.indptr.data(), db.indices.data(), db.values.data(), 2);

    CSRBatch q({{{1, 0, 0}}}, dim);
    auto res = idx.search(q.indptr.data(), q.indices.data(), q.values.data(), 1, 10);

    // k clamped to n_db = 2
    EXPECT_EQ(res.k, 2u);
}

TEST(SparseIndexCosine, EmptyIndexThrows) {
    Index idx(DataType::SparseFloat, Metric::COSINE, 4);
    CSRBatch q({{{1, 0, 0, 0}}}, 4);
    EXPECT_THROW(
        idx.search(q.indptr.data(), q.indices.data(), q.values.data(), 1, 1),
        std::runtime_error);
}

// ---------------------------------------------------------------------------
// L2 search tests
// ---------------------------------------------------------------------------

TEST(SparseIndexL2, MatchesBruteForce) {
    const uint32_t dim = 6;
    std::vector<std::vector<float>> db_rows = {
        {1, 0, 0, 2, 0, 0},
        {0, 3, 0, 0, 1, 0},
        {2, 2, 0, 0, 0, 2},
        {0, 0, 5, 0, 0, 0},
        {1, 1, 1, 1, 0, 0},
    };
    CSRBatch db(db_rows, dim);

    Index idx(DataType::SparseFloat, Metric::L2, dim);
    idx.add(db.indptr.data(), db.indices.data(), db.values.data(),
            static_cast<uint32_t>(db_rows.size()));

    std::vector<float> q_dense = {1, 0, 0, 2, 0, 0};
    CSRBatch q({{q_dense}}, dim);

    const uint32_t k = 3;
    auto res = idx.search(q.indptr.data(), q.indices.data(), q.values.data(), 1, k);

    auto ref_dists = ref_l2(q_dense, db_rows);
    auto ref_top   = topk(ref_dists, k);

    for (uint32_t j = 0; j < k; ++j) {
        EXPECT_EQ(res.indices[j],   ref_top[j].second)  << "rank " << j;
        EXPECT_NEAR(res.distances[j], ref_top[j].first, 1e-4f) << "rank " << j;
    }
}

TEST(SparseIndexL2, IdenticalVectorDistanceZero) {
    const uint32_t dim = 5;
    std::vector<std::vector<float>> db_rows = {
        {3, 0, 4, 0, 0},
        {0, 1, 0, 0, 2},
    };
    CSRBatch db(db_rows, dim);

    Index idx(DataType::SparseFloat, Metric::L2, dim);
    idx.add(db.indptr.data(), db.indices.data(), db.values.data(), 2);

    // Query identical to first row → distance should be 0.
    CSRBatch q({{{3, 0, 4, 0, 0}}}, dim);
    auto res = idx.search(q.indptr.data(), q.indices.data(), q.values.data(), 1, 1);
    EXPECT_EQ(res.indices[0], 0);
    EXPECT_NEAR(res.distances[0], 0.0f, 1e-5f);
}

TEST(SparseIndexL2, MultipleQueriesMatchBruteForce) {
    const uint32_t dim = 4;
    std::vector<std::vector<float>> db_rows = {
        {1, 0, 0, 0},
        {0, 2, 0, 0},
        {0, 0, 3, 0},
        {0, 0, 0, 4},
    };
    CSRBatch db(db_rows, dim);

    Index idx(DataType::SparseFloat, Metric::L2, dim);
    idx.add(db.indptr.data(), db.indices.data(), db.values.data(), 4);

    std::vector<std::vector<float>> q_rows = {
        {0.9f, 0, 0, 0},
        {0,    0, 3.1f, 0},
    };
    CSRBatch q(q_rows, dim);

    auto res = idx.search(q.indptr.data(), q.indices.data(), q.values.data(), 2, 2);

    // Query 0 → closest to db[0] (L2=0.1), then db[1] (L2=2.0)
    EXPECT_EQ(res.indices[0], 0);
    EXPECT_EQ(res.indices[1], 1);

    // Query 1 → closest to db[2] (L2=0.1), then db[1] (L2=sqrt(4+9.61))
    EXPECT_EQ(res.indices[2], 2);
}


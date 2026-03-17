#include "sparsity/tiled_dense.h"
#include "sparsity/index.h"

#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

using namespace sparsity;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static std::vector<float> random_vecs(uint32_t n, uint32_t dim, uint32_t seed = 1) {
    std::mt19937                    rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float>              v(static_cast<size_t>(n) * dim);
    for (auto& x : v) x = nd(rng);
    return v;
}

// Check that two SearchResults agree: same indices and distances within tol.
// Only checks the first k results per query.
static void expect_results_match(const SearchResult& ref, const SearchResult& got,
                                  float dist_tol = 1e-4f) {
    ASSERT_EQ(ref.n_queries, got.n_queries);
    ASSERT_EQ(ref.k, got.k);
    const uint32_t nq = ref.n_queries;
    const uint32_t k  = ref.k;
    for (uint32_t q = 0; q < nq; ++q) {
        for (uint32_t j = 0; j < k; ++j) {
            const size_t idx = static_cast<size_t>(q) * k + j;
            EXPECT_EQ(ref.indices[idx], got.indices[idx])
                << "query=" << q << " rank=" << j << " index mismatch";
            EXPECT_NEAR(ref.distances[idx], got.distances[idx], dist_tol)
                << "query=" << q << " rank=" << j << " distance mismatch";
        }
    }
}

// ---------------------------------------------------------------------------
// L2 correctness — compare TiledDenseIndex against DenseIndex (ground truth)
// ---------------------------------------------------------------------------

TEST(TiledDenseIndexTest, TiledMatchesBruteForceL2) {
    const uint32_t n = 200, dim = 64, k = 5, n_queries = 12;
    auto db      = random_vecs(n, dim, 10);
    auto queries = random_vecs(n_queries, dim, 20);

    DenseIndex     bf(Metric::L2);
    TiledDenseIndex tiled(Metric::L2);
    bf.add(db.data(), n, dim);
    tiled.add(db.data(), n, dim);

    auto ref = bf.search(queries.data(), n_queries, k);
    auto got = tiled.search(queries.data(), n_queries, k);

    expect_results_match(ref, got);
}

// n_queries == 1: partial tile (only 1 of TILE_Q slots used).
TEST(TiledDenseIndexTest, TiledMatchesBruteForceL2_SingleQuery) {
    const uint32_t n = 150, dim = 32, k = 3;
    auto db    = random_vecs(n, dim, 11);
    auto query = random_vecs(1, dim, 21);

    DenseIndex     bf(Metric::L2);
    TiledDenseIndex tiled(Metric::L2);
    bf.add(db.data(), n, dim);
    tiled.add(db.data(), n, dim);

    auto ref = bf.search(query.data(), 1, k);
    auto got = tiled.search(query.data(), 1, k);

    expect_results_match(ref, got);
}

// n_queries not a multiple of TILE_Q: last tile is partial.
TEST(TiledDenseIndexTest, TiledMatchesBruteForceL2_NonMultipleOfTile) {
    const uint32_t n = 300, dim = 48, k = 4;
    // TILE_Q = 4; use 7 queries → tiles of {4, 3}
    const uint32_t n_queries = 7;
    auto db      = random_vecs(n, dim, 12);
    auto queries = random_vecs(n_queries, dim, 22);

    DenseIndex     bf(Metric::L2);
    TiledDenseIndex tiled(Metric::L2);
    bf.add(db.data(), n, dim);
    tiled.add(db.data(), n, dim);

    auto ref = bf.search(queries.data(), n_queries, k);
    auto got = tiled.search(queries.data(), n_queries, k);

    expect_results_match(ref, got);
}

// Multiple add() calls (same as DenseIndex incremental adds).
TEST(TiledDenseIndexTest, TiledMatchesBruteForceL2_MultipleAdds) {
    const uint32_t n1 = 80, n2 = 120, dim = 32, k = 5, n_queries = 8;
    auto db1     = random_vecs(n1, dim, 13);
    auto db2     = random_vecs(n2, dim, 14);
    auto queries = random_vecs(n_queries, dim, 23);

    DenseIndex     bf(Metric::L2);
    TiledDenseIndex tiled(Metric::L2);
    bf.add(db1.data(), n1, dim);
    bf.add(db2.data(), n2, dim);
    tiled.add(db1.data(), n1, dim);
    tiled.add(db2.data(), n2, dim);

    auto ref = bf.search(queries.data(), n_queries, k);
    auto got = tiled.search(queries.data(), n_queries, k);

    expect_results_match(ref, got);
}

// ---------------------------------------------------------------------------
// Cosine correctness
// ---------------------------------------------------------------------------

TEST(TiledDenseIndexTest, TiledMatchesBruteForceCosine) {
    const uint32_t n = 200, dim = 64, k = 5, n_queries = 12;
    auto db      = random_vecs(n, dim, 30);
    auto queries = random_vecs(n_queries, dim, 40);

    DenseIndex     bf(Metric::COSINE);
    TiledDenseIndex tiled(Metric::COSINE);
    bf.add(db.data(), n, dim);
    tiled.add(db.data(), n, dim);

    auto ref = bf.search(queries.data(), n_queries, k);
    auto got = tiled.search(queries.data(), n_queries, k);

    expect_results_match(ref, got);
}

TEST(TiledDenseIndexTest, TiledMatchesBruteForceCosine_SingleQuery) {
    const uint32_t n = 150, dim = 32, k = 3;
    auto db    = random_vecs(n, dim, 31);
    auto query = random_vecs(1, dim, 41);

    DenseIndex     bf(Metric::COSINE);
    TiledDenseIndex tiled(Metric::COSINE);
    bf.add(db.data(), n, dim);
    tiled.add(db.data(), n, dim);

    auto ref = bf.search(query.data(), 1, k);
    auto got = tiled.search(query.data(), 1, k);

    expect_results_match(ref, got);
}

TEST(TiledDenseIndexTest, TiledMatchesBruteForceCosine_NonMultipleOfTile) {
    const uint32_t n = 300, dim = 48, k = 4, n_queries = 7;
    auto db      = random_vecs(n, dim, 32);
    auto queries = random_vecs(n_queries, dim, 42);

    DenseIndex     bf(Metric::COSINE);
    TiledDenseIndex tiled(Metric::COSINE);
    bf.add(db.data(), n, dim);
    tiled.add(db.data(), n, dim);

    auto ref = bf.search(queries.data(), n_queries, k);
    auto got = tiled.search(queries.data(), n_queries, k);

    expect_results_match(ref, got);
}

// ---------------------------------------------------------------------------
// Error handling
// ---------------------------------------------------------------------------

TEST(TiledDenseIndexTest, EmptyIndexThrows) {
    TiledDenseIndex idx(Metric::L2);
    float q[4] = {1, 0, 0, 0};
    EXPECT_THROW(idx.search(q, 1, 1), std::runtime_error);
}

TEST(TiledDenseIndexTest, DimMismatchThrows) {
    TiledDenseIndex idx(Metric::L2);
    float data[8] = {1, 0, 0, 0, 0, 1, 0, 0};
    idx.add(data, 2, 4);
    float bad[3] = {1, 0, 0};
    EXPECT_THROW(idx.add(bad, 1, 3), std::invalid_argument);
}

TEST(TiledDenseIndexTest, InvalidMetricThrows) {
    EXPECT_THROW(TiledDenseIndex(Metric::TANIMOTO), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Edge cases
// ---------------------------------------------------------------------------

TEST(TiledDenseIndexTest, KGreaterThanN) {
    TiledDenseIndex idx(Metric::L2);
    float data[8] = {1, 0, 0, 0, 0, 1, 0, 0};
    idx.add(data, 2, 4);
    float q[4] = {1, 0, 0, 0};
    auto res = idx.search(q, 1, 100);
    EXPECT_EQ(res.k, 2u);
    EXPECT_EQ(res.indices.size(), 2u);
}

// Identical vectors: tiled L2 distance should be 0.
TEST(TiledDenseIndexTest, SelfDistanceIsZeroL2) {
    const uint32_t dim = 16;
    auto vecs = random_vecs(10, dim, 99);

    TiledDenseIndex idx(Metric::L2);
    idx.add(vecs.data(), 10, dim);

    // Query with the first vector — nearest neighbour must be itself (index 0, dist 0).
    auto res = idx.search(vecs.data(), 1, 1);
    EXPECT_EQ(res.indices[0], 0);
    EXPECT_NEAR(res.distances[0], 0.0f, 1e-5f);
}

TEST(TiledDenseIndexTest, SelfDistanceIsZeroCosine) {
    const uint32_t dim = 16;
    auto vecs = random_vecs(10, dim, 100);

    TiledDenseIndex idx(Metric::COSINE);
    idx.add(vecs.data(), 10, dim);

    auto res = idx.search(vecs.data(), 1, 1);
    EXPECT_EQ(res.indices[0], 0);
    EXPECT_NEAR(res.distances[0], 0.0f, 1e-5f);
}

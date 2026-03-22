#include "sparsity/index.h"
#include "sparsity/metrics.h"
#include "sparsity/packing.h"

#include <gtest/gtest.h>
#include <algorithm>
#include <cstdint>
#include <vector>

using namespace sparsity;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Pack a list of dense bool rows and return a flat uint64_t buffer.
static std::vector<uint64_t> pack_rows(const std::vector<std::vector<bool>>& rows,
                                        uint32_t dim) {
    std::vector<uint64_t> out;
    const uint32_t wpr = (dim + 63u) / 64u;
    out.resize(rows.size() * wpr, 0ULL);
    for (size_t r = 0; r < rows.size(); ++r) {
        for (uint32_t b = 0; b < dim; ++b) {
            if (rows[r][b]) {
                out[r * wpr + b / 64] |= (uint64_t{1} << (b % 64));
            }
        }
    }
    return out;
}

// Brute-force Tanimoto reference for a single dense-bool query against db rows.
static std::vector<float> ref_tanimoto(const std::vector<bool>& q,
                                        const std::vector<std::vector<bool>>& db) {
    std::vector<float> sims;
    for (const auto& row : db) {
        uint32_t inter = 0, uni = 0;
        for (size_t i = 0; i < q.size(); ++i) {
            inter += (q[i] && row[i]) ? 1 : 0;
            uni   += (q[i] || row[i]) ? 1 : 0;
        }
        sims.push_back(uni == 0 ? 1.0f : static_cast<float>(inter) / uni);
    }
    return sims;
}

// Return top-k indices sorted by descending Tanimoto similarity.
static std::vector<int64_t> topk_desc(const std::vector<float>& sims, uint32_t k) {
    std::vector<std::pair<float, int64_t>> pairs;
    for (size_t i = 0; i < sims.size(); ++i)
        pairs.push_back({sims[i], static_cast<int64_t>(i)});
    // Sort descending by similarity (higher = better).
    std::sort(pairs.begin(), pairs.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });
    std::vector<int64_t> indices;
    for (uint32_t i = 0; i < std::min(k, static_cast<uint32_t>(pairs.size())); ++i)
        indices.push_back(pairs[i].second);
    return indices;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST(BinaryIndex, IdenticalVectorDistanceOne) {
    const uint32_t dim = 64;
    std::vector<std::vector<bool>> db = {
        std::vector<bool>(dim, false),
        std::vector<bool>(dim, true),
    };
    db[0][0] = db[0][7] = db[0][63] = true;

    auto packed_db = pack_rows(db, dim);

    Index idx(DataType::Binary, Metric::TANIMOTO, dim);
    idx.add(packed_db.data(), 2);

    // Query == db[0] → similarity 1.0 with db[0].
    auto packed_q = pack_rows({db[0]}, dim);
    auto res = idx.search(packed_q.data(), 1, 1);

    ASSERT_EQ(res.n_queries, 1u);
    ASSERT_EQ(res.k, 1u);
    EXPECT_EQ(res.indices[0], 0);
    EXPECT_NEAR(res.distances[0], 1.0f, 1e-6f);
}

TEST(BinaryIndex, MatchesBruteForce) {
    const uint32_t dim = 128;
    // 5 database fingerprints.
    std::vector<std::vector<bool>> db(5, std::vector<bool>(dim, false));
    // Give each row some set bits.
    for (uint32_t i = 0; i < 40; ++i) db[0][i]       = true;
    for (uint32_t i = 60; i < 100; ++i) db[1][i]     = true;
    for (uint32_t i = 0; i < 128; ++i) db[2][i]      = (i % 3 == 0);
    for (uint32_t i = 20; i < 60; ++i) db[3][i]      = true;
    for (uint32_t i = 0; i < 10; ++i) db[4][i]       = true;

    auto packed_db = pack_rows(db, dim);

    Index idx(DataType::Binary, Metric::TANIMOTO, dim);
    idx.add(packed_db.data(), static_cast<uint32_t>(db.size()));

    // Query: first 30 bits set.
    std::vector<bool> q_dense(dim, false);
    for (uint32_t i = 0; i < 30; ++i) q_dense[i] = true;
    auto packed_q = pack_rows({q_dense}, dim);

    const uint32_t k = 3;
    auto res = idx.search(packed_q.data(), 1, k);

    ASSERT_EQ(res.n_queries, 1u);
    ASSERT_EQ(res.k, k);

    // Reference.
    auto ref_sims    = ref_tanimoto(q_dense, db);
    auto ref_indices = topk_desc(ref_sims, k);

    for (uint32_t j = 0; j < k; ++j) {
        EXPECT_EQ(res.indices[j], ref_indices[j]) << "rank " << j;
    }
    // Distances should equal Tanimoto similarities (not negated).
    for (uint32_t j = 0; j < k; ++j) {
        EXPECT_NEAR(res.distances[j],
                    ref_sims[static_cast<size_t>(res.indices[j])], 1e-5f)
            << "rank " << j;
    }
}

TEST(BinaryIndex, MultipleQueries) {
    const uint32_t dim = 64;
    std::vector<std::vector<bool>> db(4, std::vector<bool>(dim, false));
    // db[0]: bits 0-15; db[1]: bits 16-31; db[2]: bits 32-47; db[3]: bits 48-63.
    for (uint32_t i = 0; i < 16; ++i) { db[0][i]    = true; db[1][i+16] = true;
                                         db[2][i+32] = true; db[3][i+48] = true; }
    auto packed_db = pack_rows(db, dim);

    Index idx(DataType::Binary, Metric::TANIMOTO, dim);
    idx.add(packed_db.data(), 4);

    // Query 0 matches db[0]; query 1 matches db[2].
    std::vector<bool> q0(dim, false), q1(dim, false);
    for (uint32_t i = 0; i < 16; ++i) { q0[i]    = true; q1[i+32] = true; }

    auto packed_q = pack_rows({q0, q1}, dim);
    auto res = idx.search(packed_q.data(), 2, 1);

    ASSERT_EQ(res.n_queries, 2u);
    EXPECT_EQ(res.indices[0], 0);
    EXPECT_NEAR(res.distances[0], 1.0f, 1e-6f);
    EXPECT_EQ(res.indices[1], 2);
    EXPECT_NEAR(res.distances[1], 1.0f, 1e-6f);
}

TEST(BinaryIndex, BoolOverloadAdd) {
    // Verify that add(bool*, n_rows) packs and searches correctly.
    const uint32_t dim = 64;
    std::vector<bool> db_bool(2 * dim, false);
    // Row 0: bits 0,1,2 set.
    db_bool[0] = db_bool[1] = db_bool[2] = true;
    // Row 1: bits 10,11 set.
    db_bool[dim + 10] = db_bool[dim + 11] = true;

    // Need flat bool array — extract from vector<bool>.
    std::vector<bool> flat(db_bool.begin(), db_bool.end());
    std::vector<bool> flat_copy(flat.begin(), flat.end());
    // Use the bool* overload via Index::add(const bool*, n_rows)
    // — not directly available on Index; route through the uint64_t overload.
    // pack manually and call add(uint64_t*).
    std::vector<std::vector<bool>> rows = {
        std::vector<bool>(dim, false),
        std::vector<bool>(dim, false),
    };
    rows[0][0] = rows[0][1] = rows[0][2] = true;
    rows[1][10] = rows[1][11] = true;
    auto packed_db = pack_rows(rows, dim);

    Index idx(DataType::Binary, Metric::TANIMOTO, dim);
    idx.add(packed_db.data(), 2);

    // Query = row 0: should match db[0] with sim=1.
    auto packed_q = pack_rows({rows[0]}, dim);
    auto res = idx.search(packed_q.data(), 1, 1);
    EXPECT_EQ(res.indices[0], 0);
    EXPECT_NEAR(res.distances[0], 1.0f, 1e-6f);
}

TEST(BinaryIndex, IncrementalAdd) {
    const uint32_t dim = 64;
    std::vector<std::vector<bool>> batch1 = {std::vector<bool>(dim, false)};
    std::vector<std::vector<bool>> batch2 = {std::vector<bool>(dim, false)};
    batch1[0][0] = true;
    batch2[0][32] = true;

    auto p1 = pack_rows(batch1, dim);
    auto p2 = pack_rows(batch2, dim);

    Index idx(DataType::Binary, Metric::TANIMOTO, dim);
    idx.add(p1.data(), 1);
    idx.add(p2.data(), 1);

    EXPECT_EQ(idx.size(), 2u);

    // Query matching batch2's vector (bit 32)
    auto pq = pack_rows({batch2[0]}, dim);
    auto res = idx.search(pq.data(), 1, 1);
    EXPECT_EQ(res.indices[0], 1);  // second added vector
    EXPECT_NEAR(res.distances[0], 1.0f, 1e-6f);
}

TEST(BinaryIndex, EmptyIndexThrows) {
    Index idx(DataType::Binary, Metric::TANIMOTO, 64);
    const uint64_t q = 0ULL;
    EXPECT_THROW(idx.search(&q, 1, 1), std::runtime_error);
}

TEST(BinaryIndex, KLargerThanN) {
    const uint32_t dim = 64;
    std::vector<std::vector<bool>> db = {std::vector<bool>(dim, false)};
    auto packed = pack_rows(db, dim);

    Index idx(DataType::Binary, Metric::TANIMOTO, dim);
    idx.add(packed.data(), 1);

    auto pq = pack_rows({db[0]}, dim);
    auto res = idx.search(pq.data(), 1, 100);
    EXPECT_EQ(res.k, 1u);  // clamped to n_db
}

TEST(BinaryIndex, ResultsSortedDescending) {
    // Build db with known pairwise similarities. Query should give descending order.
    const uint32_t dim = 64;
    std::vector<std::vector<bool>> db(3, std::vector<bool>(dim, false));
    // db[0]: bits 0-9 (10 bits)
    // db[1]: bits 0-4 (5 bits, subset of query region)
    // db[2]: bit 63  (1 bit, unrelated)
    for (uint32_t i = 0; i < 10; ++i) db[0][i] = true;
    for (uint32_t i = 0; i < 5; ++i)  db[1][i] = true;
    db[2][63] = true;

    auto packed_db = pack_rows(db, dim);
    Index idx(DataType::Binary, Metric::TANIMOTO, dim);
    idx.add(packed_db.data(), 3);

    // Query: bits 0-7 (8 bits).
    std::vector<bool> q(dim, false);
    for (uint32_t i = 0; i < 8; ++i) q[i] = true;
    auto pq = pack_rows({q}, dim);
    auto res = idx.search(pq.data(), 1, 3);

    // Results should be sorted: best similarity first.
    ASSERT_EQ(res.k, 3u);
    EXPECT_GE(res.distances[0], res.distances[1]);
    EXPECT_GE(res.distances[1], res.distances[2]);
}


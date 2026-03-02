#include "sparsity/metrics.h"
#include "sparsity/index.h"

#include <gtest/gtest.h>
#include <cmath>
#include <vector>

namespace sparsity {
namespace {

// ---------------------------------------------------------------------------
// l2_distance / l2_distance_sq
// ---------------------------------------------------------------------------

TEST(L2Distance, IdenticalVectors) {
    const std::vector<float> a = {1.f, 2.f, 3.f, 4.f};
    EXPECT_FLOAT_EQ(l2_distance_sq(a.data(), a.data(), 4), 0.f);
    EXPECT_FLOAT_EQ(l2_distance(a.data(), a.data(), 4), 0.f);
}

TEST(L2Distance, KnownValue) {
    // ||[0,0,0] - [3,4,0]|| = 5
    const std::vector<float> a = {0.f, 0.f, 0.f};
    const std::vector<float> b = {3.f, 4.f, 0.f};
    EXPECT_FLOAT_EQ(l2_distance_sq(a.data(), b.data(), 3), 25.f);
    EXPECT_FLOAT_EQ(l2_distance(a.data(), b.data(), 3), 5.f);
}

TEST(L2Distance, Symmetry) {
    const std::vector<float> a = {1.f, 0.f, -1.f, 2.f};
    const std::vector<float> b = {0.f, 3.f,  2.f, 1.f};
    EXPECT_FLOAT_EQ(l2_distance(a.data(), b.data(), 4),
                    l2_distance(b.data(), a.data(), 4));
}

TEST(L2Distance, SingleElement) {
    const float a = 3.f, b = 7.f;
    EXPECT_FLOAT_EQ(l2_distance(&a, &b, 1), 4.f);
}

TEST(L2Distance, ZeroVectors) {
    const std::vector<float> z(64, 0.f);
    EXPECT_FLOAT_EQ(l2_distance(z.data(), z.data(), 64), 0.f);
}

TEST(L2Distance, HighDimensional) {
    // 1024-d vectors with a known squared distance
    constexpr uint32_t D = 1024;
    std::vector<float> a(D, 1.f), b(D, 0.f);
    // each element differs by 1, so sq_dist = D, dist = sqrt(D)
    EXPECT_NEAR(l2_distance_sq(a.data(), b.data(), D), static_cast<float>(D), 1e-3f);
    EXPECT_NEAR(l2_distance(a.data(), b.data(), D), std::sqrt(static_cast<float>(D)), 1e-3f);
}

// ---------------------------------------------------------------------------
// DenseIndex — L2 search
// ---------------------------------------------------------------------------

TEST(DenseIndexL2, SearchReturnsClosest) {
    DenseIndex idx(Metric::L2);

    // Index: 4 vectors in 3-D
    // clang-format off
    const std::vector<float> vecs = {
        0.f, 0.f, 0.f,   // 0 — origin
        1.f, 0.f, 0.f,   // 1
        0.f, 1.f, 0.f,   // 2
        0.f, 0.f, 1.f,   // 3
    };
    // clang-format on
    idx.add(vecs.data(), 4, 3);

    // Query: very close to vector 0
    const std::vector<float> q = {0.01f, 0.01f, 0.01f};
    auto result = idx.search(q.data(), 1, 1);

    ASSERT_EQ(result.indices.size(), 1u);
    EXPECT_EQ(result.indices[0], 0);
    EXPECT_NEAR(result.distances[0], std::sqrt(3.f) * 0.01f, 1e-4f);
}

TEST(DenseIndexL2, TopKOrdering) {
    DenseIndex idx(Metric::L2);

    // 5 vectors along the x-axis at x = 1..5
    std::vector<float> vecs(5 * 1);
    for (int i = 0; i < 5; ++i) vecs[i] = static_cast<float>(i + 1);
    idx.add(vecs.data(), 5, 1);

    // Query at x=0: nearest should be x=1 (idx 0), then x=2 (idx 1), ...
    const float q = 0.f;
    auto result = idx.search(&q, 1, 3);

    ASSERT_EQ(result.indices.size(), 3u);
    EXPECT_EQ(result.indices[0], 0);
    EXPECT_EQ(result.indices[1], 1);
    EXPECT_EQ(result.indices[2], 2);
    // distances should be non-decreasing
    EXPECT_LE(result.distances[0], result.distances[1]);
    EXPECT_LE(result.distances[1], result.distances[2]);
}

TEST(DenseIndexL2, KLargerThanN) {
    DenseIndex idx(Metric::L2);
    const std::vector<float> v = {1.f, 2.f, 3.f};
    idx.add(v.data(), 1, 3);

    const std::vector<float> q = {0.f, 0.f, 0.f};
    // k=10 but only 1 vector indexed — should return 1 result, not crash
    auto result = idx.search(q.data(), 1, 10);
    EXPECT_EQ(result.k, 1u);
    EXPECT_EQ(result.indices.size(), 1u);
}

TEST(DenseIndexL2, MultipleAdds) {
    DenseIndex idx(Metric::L2);

    const std::vector<float> batch1 = {0.f, 0.f};
    const std::vector<float> batch2 = {10.f, 10.f};
    idx.add(batch1.data(), 1, 2);
    idx.add(batch2.data(), 1, 2);
    EXPECT_EQ(idx.size(), 2u);

    const std::vector<float> q = {0.1f, 0.1f};
    auto result = idx.search(q.data(), 1, 1);
    EXPECT_EQ(result.indices[0], 0); // closer to origin
}

TEST(DenseIndexL2, DimMismatchThrows) {
    DenseIndex idx(Metric::L2);
    const std::vector<float> v(4, 1.f);
    idx.add(v.data(), 1, 4);

    const std::vector<float> bad(3, 1.f);
    EXPECT_THROW(idx.add(bad.data(), 1, 3), std::invalid_argument);
}

TEST(DenseIndexL2, EmptyIndexThrows) {
    DenseIndex idx(Metric::L2);
    const float q = 0.f;
    EXPECT_THROW(idx.search(&q, 1, 1), std::runtime_error);
}

} // namespace
} // namespace sparsity

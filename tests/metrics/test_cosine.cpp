#include "sparsity/metrics.h"
#include "sparsity/index.h"

#include <gtest/gtest.h>
#include <cmath>
#include <vector>

namespace sparsity {
namespace {

// ---------------------------------------------------------------------------
// cosine_distance — metric function tests
// ---------------------------------------------------------------------------

TEST(CosineDistance, IdenticalVectors) {
    const std::vector<float> a = {1.f, 2.f, 3.f};
    EXPECT_NEAR(cosine_distance(a.data(), a.data(), 3), 0.f, 1e-6f);
}

TEST(CosineDistance, OrthogonalVectors) {
    const std::vector<float> a = {1.f, 0.f};
    const std::vector<float> b = {0.f, 1.f};
    EXPECT_NEAR(cosine_distance(a.data(), b.data(), 2), 1.f, 1e-6f);
}

TEST(CosineDistance, OppositeVectors) {
    const std::vector<float> a = {1.f, 0.f};
    const std::vector<float> b = {-1.f, 0.f};
    EXPECT_NEAR(cosine_distance(a.data(), b.data(), 2), 2.f, 1e-6f);
}

TEST(CosineDistance, ZeroNormReturnsOne) {
    const std::vector<float> z = {0.f, 0.f, 0.f};
    const std::vector<float> a = {1.f, 2.f, 3.f};
    EXPECT_FLOAT_EQ(cosine_distance(z.data(), a.data(), 3), 1.f);
    EXPECT_FLOAT_EQ(cosine_distance(a.data(), z.data(), 3), 1.f);
}

TEST(CosineDistance, Symmetry) {
    const std::vector<float> a = {1.f, 2.f, 3.f};
    const std::vector<float> b = {4.f, -1.f, 0.5f};
    EXPECT_FLOAT_EQ(cosine_distance(a.data(), b.data(), 3),
                    cosine_distance(b.data(), a.data(), 3));
}

TEST(CosineDistance, HighDimensional) {
    // Vectors pointing in the same direction — distance should be ~0.
    constexpr uint32_t D = 1024;
    std::vector<float> a(D, 1.f), b(D, 2.f);
    EXPECT_NEAR(cosine_distance(a.data(), b.data(), D), 0.f, 1e-5f);
}

// ---------------------------------------------------------------------------
// Index API — DenseFloat + COSINE integration tests
// ---------------------------------------------------------------------------

TEST(IndexCosine, NearestNeighbour) {
    // Three vectors: x-axis, y-axis, diagonal.
    // Query along x-axis should return x-axis vector first.
    // clang-format off
    const std::vector<float> vecs = {
        1.f, 0.f,   // idx 0 — along x
        0.f, 1.f,   // idx 1 — along y
        1.f, 1.f,   // idx 2 — diagonal (45°)
    };
    // clang-format on
    Index idx(DataType::DenseFloat, Metric::COSINE, 2);
    idx.add(vecs.data(), 3, 2);

    const std::vector<float> q = {1.f, 0.f};
    auto result = idx.search(q.data(), 1, 1);

    ASSERT_EQ(result.indices.size(), 1u);
    EXPECT_EQ(result.indices[0], 0);
    EXPECT_NEAR(result.distances[0], 0.f, 1e-6f);
}

TEST(IndexCosine, TopKOrdering) {
    // Four unit vectors at 0°, 30°, 60°, 90°.
    const float angles[] = {0.f, 30.f, 60.f, 90.f};
    const float pi       = std::acos(-1.f);
    std::vector<float> vecs(4 * 2);
    for (int i = 0; i < 4; ++i) {
        vecs[2 * i + 0] = std::cos(angles[i] * pi / 180.f);
        vecs[2 * i + 1] = std::sin(angles[i] * pi / 180.f);
    }

    Index idx(DataType::DenseFloat, Metric::COSINE, 2);
    idx.add(vecs.data(), 4, 2);

    // Query at 0° — nearest is idx 0 (0°), then 1 (30°), 2 (60°), 3 (90°).
    const std::vector<float> q = {1.f, 0.f};
    auto result = idx.search(q.data(), 1, 4);

    ASSERT_EQ(result.indices.size(), 4u);
    EXPECT_EQ(result.indices[0], 0);
    EXPECT_EQ(result.indices[1], 1);
    EXPECT_EQ(result.indices[2], 2);
    EXPECT_EQ(result.indices[3], 3);
    // Distances must be non-decreasing.
    for (uint32_t j = 1; j < 4; ++j) {
        EXPECT_LE(result.distances[j - 1], result.distances[j]);
    }
}

TEST(IndexCosine, KLargerThanN) {
    const std::vector<float> v = {1.f, 0.f};
    Index idx(DataType::DenseFloat, Metric::COSINE, 2);
    idx.add(v.data(), 1, 2);

    const std::vector<float> q = {1.f, 0.f};
    auto result = idx.search(q.data(), 1, 10);
    EXPECT_EQ(result.k, 1u);
    EXPECT_EQ(result.indices.size(), 1u);
}

TEST(IndexCosine, MultipleAdds) {
    Index idx(DataType::DenseFloat, Metric::COSINE, 2);

    const std::vector<float> batch1 = {1.f, 0.f};   // idx 0 — along x
    const std::vector<float> batch2 = {0.f, 1.f};   // idx 1 — along y
    idx.add(batch1.data(), 1, 2);
    idx.add(batch2.data(), 1, 2);
    EXPECT_EQ(idx.size(), 2u);

    // Query along x: should pick idx 0.
    const std::vector<float> q = {1.f, 0.f};
    auto result = idx.search(q.data(), 1, 1);
    EXPECT_EQ(result.indices[0], 0);
}

TEST(IndexCosine, EmptyIndexThrows) {
    Index idx(DataType::DenseFloat, Metric::COSINE, 2);
    const std::vector<float> q = {1.f, 0.f};
    EXPECT_THROW(idx.search(q.data(), 1, 1), std::runtime_error);
}

} // namespace
} // namespace sparsity

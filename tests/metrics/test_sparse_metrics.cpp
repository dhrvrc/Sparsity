#include "sparsity/metrics.h"
#include "sparsity/sparse.h"

#include <gtest/gtest.h>
#include <cmath>
#include <vector>

namespace sparsity {
namespace {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Build a SparseVector from parallel index/value arrays (must be sorted asc).
static SparseVector make_sv(const std::vector<uint32_t>& idx,
                             const std::vector<float>& val,
                             uint32_t dim) {
    SparseVector sv;
    sv.indices = idx.data();
    sv.values  = val.data();
    sv.nnz     = static_cast<uint32_t>(idx.size());
    sv.dim     = dim;
    return sv;
}

static float norm(const std::vector<float>& vals) {
    float s = 0.0f;
    for (float v : vals) s += v * v;
    return std::sqrt(s);
}

// ---------------------------------------------------------------------------
// sparse_dot tests
// ---------------------------------------------------------------------------

TEST(SparseDot, DisjointVectors) {
    // a has dim 0,2; b has dim 1,3 — no overlap → dot = 0
    std::vector<uint32_t> ia = {0, 2}, ib = {1, 3};
    std::vector<float>    va = {1.f, 1.f}, vb = {1.f, 1.f};
    auto a = make_sv(ia, va, 4);
    auto b = make_sv(ib, vb, 4);
    EXPECT_FLOAT_EQ(sparse_dot(a, b), 0.0f);
}

TEST(SparseDot, FullOverlap) {
    std::vector<uint32_t> idx = {0, 1, 2};
    std::vector<float>    va  = {1.f, 2.f, 3.f};
    std::vector<float>    vb  = {4.f, 5.f, 6.f};
    auto a = make_sv(idx, va, 5);
    auto b = make_sv(idx, vb, 5);
    // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    EXPECT_FLOAT_EQ(sparse_dot(a, b), 32.0f);
}

TEST(SparseDot, PartialOverlap) {
    std::vector<uint32_t> ia = {0, 1, 3};
    std::vector<uint32_t> ib = {1, 2, 3};
    std::vector<float>    va = {1.f, 2.f, 4.f};
    std::vector<float>    vb = {3.f, 5.f, 6.f};
    auto a = make_sv(ia, va, 5);
    auto b = make_sv(ib, vb, 5);
    // overlap at dim 1: 2*3=6, dim 3: 4*6=24 → dot = 30
    EXPECT_FLOAT_EQ(sparse_dot(a, b), 30.0f);
}

TEST(SparseDot, OneEmpty) {
    std::vector<uint32_t> ia = {0, 2};
    std::vector<float>    va = {1.f, 2.f};
    std::vector<uint32_t> ib = {};
    std::vector<float>    vb = {};
    auto a = make_sv(ia, va, 5);
    auto b = make_sv(ib, vb, 5);
    EXPECT_FLOAT_EQ(sparse_dot(a, b), 0.0f);
}

TEST(SparseDot, BothEmpty) {
    std::vector<uint32_t> ie = {};
    std::vector<float>    ve = {};
    auto a = make_sv(ie, ve, 10);
    auto b = make_sv(ie, ve, 10);
    EXPECT_FLOAT_EQ(sparse_dot(a, b), 0.0f);
}

TEST(SparseDot, Symmetry) {
    std::vector<uint32_t> ia = {0, 5, 9};
    std::vector<uint32_t> ib = {2, 5, 7};
    std::vector<float>    va = {1.f, 3.f, 2.f};
    std::vector<float>    vb = {4.f, 6.f, 5.f};
    auto a = make_sv(ia, va, 10);
    auto b = make_sv(ib, vb, 10);
    EXPECT_FLOAT_EQ(sparse_dot(a, b), sparse_dot(b, a));
}

// ---------------------------------------------------------------------------
// sparse_cosine_distance tests
// ---------------------------------------------------------------------------

TEST(SparseCosine, IdenticalVectors) {
    std::vector<uint32_t> idx = {0, 1, 2};
    std::vector<float>    val = {3.f, 4.f, 0.f};
    auto v = make_sv(idx, val, 5);
    const float n = norm(val);
    EXPECT_NEAR(sparse_cosine_distance(v, v, n, n), 0.0f, 1e-6f);
}

TEST(SparseCosine, OrthogonalVectors) {
    std::vector<uint32_t> ia = {0}, ib = {1};
    std::vector<float>    va = {1.f}, vb = {1.f};
    auto a = make_sv(ia, va, 2);
    auto b = make_sv(ib, vb, 2);
    // dot = 0 → cosine distance = 1
    EXPECT_FLOAT_EQ(sparse_cosine_distance(a, b, norm(va), norm(vb)), 1.0f);
}

TEST(SparseCosine, ZeroNormVector) {
    std::vector<uint32_t> ia = {0}, ib = {};
    std::vector<float>    va = {1.f}, vb = {};
    auto a = make_sv(ia, va, 2);
    auto b = make_sv(ib, vb, 2);
    EXPECT_FLOAT_EQ(sparse_cosine_distance(a, b, norm(va), 0.0f), 1.0f);
    EXPECT_FLOAT_EQ(sparse_cosine_distance(b, a, 0.0f, norm(va)), 1.0f);
}

TEST(SparseCosine, KnownValue) {
    // a = (3, 0, 4) at dims 0,2; b = (0, 5, 0) at dim 1 → orthogonal
    std::vector<uint32_t> ia = {0, 2}, ib = {1};
    std::vector<float>    va = {3.f, 4.f}, vb = {5.f};
    auto a = make_sv(ia, va, 3);
    auto b = make_sv(ib, vb, 3);
    EXPECT_NEAR(sparse_cosine_distance(a, b, norm(va), norm(vb)), 1.0f, 1e-6f);
}

TEST(SparseCosine, Symmetry) {
    std::vector<uint32_t> ia = {1, 3}, ib = {1, 2};
    std::vector<float>    va = {2.f, 5.f}, vb = {3.f, 4.f};
    auto a = make_sv(ia, va, 5);
    auto b = make_sv(ib, vb, 5);
    const float na = norm(va), nb = norm(vb);
    EXPECT_NEAR(sparse_cosine_distance(a, b, na, nb),
                sparse_cosine_distance(b, a, nb, na), 1e-6f);
}

// ---------------------------------------------------------------------------
// sparse_l2_distance tests
// ---------------------------------------------------------------------------

TEST(SparseL2, IdenticalVectors) {
    std::vector<uint32_t> idx = {0, 2, 4};
    std::vector<float>    val = {1.f, 2.f, 3.f};
    auto v = make_sv(idx, val, 5);
    // Compute norm_sq directly (not via sqrt round-trip) to avoid fp rounding.
    const float n2 = 1.f * 1.f + 2.f * 2.f + 3.f * 3.f;  // = 14
    EXPECT_NEAR(sparse_l2_distance(v, v, n2, n2), 0.0f, 1e-6f);
}

TEST(SparseL2, DisjointVectors) {
    // a = [1, 0] (dim 0), b = [0, 1] (dim 1)
    // ||a-b||^2 = 1^2 + 1^2 = 2 → ||a-b|| = sqrt(2)
    std::vector<uint32_t> ia = {0}, ib = {1};
    std::vector<float>    va = {1.f}, vb = {1.f};
    auto a = make_sv(ia, va, 2);
    auto b = make_sv(ib, vb, 2);
    const float na2 = 1.f, nb2 = 1.f;
    EXPECT_NEAR(sparse_l2_distance(a, b, na2, nb2), std::sqrt(2.0f), 1e-6f);
}

TEST(SparseL2, KnownValue) {
    // a = (1, 2) at dims 0,1; b = (3, 5) at dims 0,1
    // ||a-b||^2 = (1-3)^2 + (2-5)^2 = 4 + 9 = 13
    std::vector<uint32_t> ia = {0, 1}, ib = {0, 1};
    std::vector<float>    va = {1.f, 2.f}, vb = {3.f, 5.f};
    auto a = make_sv(ia, va, 2);
    auto b = make_sv(ib, vb, 2);
    const float na2 = 1.f + 4.f, nb2 = 9.f + 25.f;
    EXPECT_NEAR(sparse_l2_distance(a, b, na2, nb2), std::sqrt(13.0f), 1e-5f);
}

TEST(SparseL2, Symmetry) {
    std::vector<uint32_t> ia = {0, 3}, ib = {1, 3};
    std::vector<float>    va = {2.f, 4.f}, vb = {3.f, 1.f};
    auto a = make_sv(ia, va, 5);
    auto b = make_sv(ib, vb, 5);
    const float na = norm(va), nb = norm(vb);
    EXPECT_NEAR(sparse_l2_distance(a, b, na*na, nb*nb),
                sparse_l2_distance(b, a, nb*nb, na*na), 1e-6f);
}

TEST(SparseL2, AgainstDenseReference) {
    // Build sparse and compute dense L2 reference manually.
    // a = [0, 3, 0, 4, 0] (dims 1,3)
    // b = [2, 0, 0, 4, 1] (dims 0,3,4)
    // diff = [-2, 3, 0, 0, -1] → dist = sqrt(4+9+1) = sqrt(14)
    std::vector<uint32_t> ia = {1, 3}, ib = {0, 3, 4};
    std::vector<float>    va = {3.f, 4.f}, vb = {2.f, 4.f, 1.f};
    auto a = make_sv(ia, va, 5);
    auto b = make_sv(ib, vb, 5);
    const float na = norm(va), nb = norm(vb);
    EXPECT_NEAR(sparse_l2_distance(a, b, na*na, nb*nb), std::sqrt(14.0f), 1e-5f);
}

} // namespace
} // namespace sparsity

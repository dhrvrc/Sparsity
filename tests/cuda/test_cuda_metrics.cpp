#include "sparsity/dispatch.h"
#include "sparsity/metrics.h"
#include "sparsity/index.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

#ifdef WITH_CUDA
#include "sparsity/cuda_dense.h"
#include "sparsity/cuda_utils.h"
#include <cuda_runtime.h>
#endif

namespace sparsity {
namespace {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static std::vector<float> random_floats(size_t n, float lo = -1.f, float hi = 1.f,
                                         unsigned seed = 42)
{
    std::mt19937                    rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    std::vector<float> v(n);
    for (auto& x : v) x = dist(rng);
    return v;
}

// Unit-normalise each row of a row-major matrix in-place.
static void normalise_rows(float* data, uint32_t n_rows, uint32_t dim)
{
    for (uint32_t i = 0; i < n_rows; ++i) {
        float* row = data + static_cast<size_t>(i) * dim;
        float  sq  = 0.f;
        for (uint32_t d = 0; d < dim; ++d) sq += row[d] * row[d];
        if (sq > 0.f) {
            const float inv = 1.f / std::sqrt(sq);
            for (uint32_t d = 0; d < dim; ++d) row[d] *= inv;
        }
    }
}

// CPU reference: L2 distance matrix [n_q × n_db].
static std::vector<float> ref_l2_matrix(const float* q, uint32_t n_q,
                                         const float* db, uint32_t n_db,
                                         uint32_t dim)
{
    std::vector<float> out(static_cast<size_t>(n_q) * n_db);
    for (uint32_t i = 0; i < n_q; ++i)
        for (uint32_t j = 0; j < n_db; ++j)
            out[static_cast<size_t>(i) * n_db + j] =
                l2_distance(q + static_cast<size_t>(i) * dim,
                            db + static_cast<size_t>(j) * dim, dim);
    return out;
}

// CPU reference: cosine distance matrix (pre-normalised vectors).
static std::vector<float> ref_cosine_matrix(const float* q, uint32_t n_q,
                                             const float* db, uint32_t n_db,
                                             uint32_t dim)
{
    std::vector<float> out(static_cast<size_t>(n_q) * n_db);
    for (uint32_t i = 0; i < n_q; ++i) {
        const float* qv = q + static_cast<size_t>(i) * dim;
        for (uint32_t j = 0; j < n_db; ++j) {
            const float* dv = db + static_cast<size_t>(j) * dim;
            float dot = 0.f;
            for (uint32_t d = 0; d < dim; ++d) dot += qv[d] * dv[d];
            out[static_cast<size_t>(i) * n_db + j] = 1.f - dot;
        }
    }
    return out;
}

// CPU top-k for a single row (returns sorted indices).
static std::vector<int64_t> ref_topk_indices(const float* dist_row, uint32_t n_db, uint32_t k)
{
    const uint32_t k_actual = std::min(k, n_db);
    std::vector<std::pair<float, int64_t>> pairs(n_db);
    for (uint32_t i = 0; i < n_db; ++i) pairs[i] = {dist_row[i], static_cast<int64_t>(i)};
    std::partial_sort(pairs.begin(), pairs.begin() + k_actual, pairs.end());
    std::vector<int64_t> idx(k_actual);
    for (uint32_t j = 0; j < k_actual; ++j) idx[j] = pairs[j].second;
    return idx;
}

// Check that two sorted index sets are identical (order matters — partial_sort
// is deterministic for distinct distances).
static bool indices_match(const int64_t* a, const int64_t* b, uint32_t k)
{
    for (uint32_t i = 0; i < k; ++i)
        if (a[i] != b[i]) return false;
    return true;
}

// ---------------------------------------------------------------------------
// Test: dispatch_dense_search L2 — correctness vs CPU reference
// ---------------------------------------------------------------------------

TEST(DispatchL2, MatchesCpuReference)
{
    constexpr uint32_t N_Q = 8, N_DB = 200, DIM = 64, K = 10;

    auto q_data  = random_floats(N_Q  * DIM, -1.f, 1.f, 1);
    auto db_data = random_floats(N_DB * DIM, -1.f, 1.f, 2);

    // CPU reference: full distance matrix then top-k.
    auto ref_dist = ref_l2_matrix(q_data.data(), N_Q, db_data.data(), N_DB, DIM);

    // dispatch (uses GPU if available, CPU otherwise).
    auto result = dispatch_dense_search(q_data.data(), db_data.data(),
                                        N_Q, N_DB, DIM, K, Metric::L2);

    ASSERT_EQ(result.n_queries, N_Q);
    ASSERT_EQ(result.k, K);

    for (uint32_t q = 0; q < N_Q; ++q) {
        const float*   rd = result.distances.data() + static_cast<size_t>(q) * K;
        const int64_t* ri = result.indices.data()   + static_cast<size_t>(q) * K;

        // Distances must match CPU reference for the returned indices.
        for (uint32_t j = 0; j < K; ++j) {
            const float ref_d = ref_dist[static_cast<size_t>(q) * N_DB + ri[j]];
            EXPECT_NEAR(rd[j], ref_d, 1e-4f)
                << "query=" << q << " rank=" << j;
        }

        // Results must be sorted ascending.
        for (uint32_t j = 1; j < K; ++j)
            EXPECT_LE(rd[j - 1], rd[j]) << "query=" << q;

        // Top-k indices must match CPU partial_sort.
        auto ref_idx = ref_topk_indices(ref_dist.data() + static_cast<size_t>(q) * N_DB,
                                         N_DB, K);
        EXPECT_TRUE(indices_match(ri, ref_idx.data(), K))
            << "query=" << q << ": top-k index mismatch";
    }
}

TEST(DispatchL2, SingleQuery)
{
    constexpr uint32_t N_Q = 1, N_DB = 50, DIM = 16, K = 5;
    auto q_data  = random_floats(N_Q  * DIM, -2.f, 2.f, 10);
    auto db_data = random_floats(N_DB * DIM, -2.f, 2.f, 20);

    auto ref_dist = ref_l2_matrix(q_data.data(), N_Q, db_data.data(), N_DB, DIM);
    auto result   = dispatch_dense_search(q_data.data(), db_data.data(),
                                          N_Q, N_DB, DIM, K, Metric::L2);

    ASSERT_EQ(result.k, K);
    auto ref_idx = ref_topk_indices(ref_dist.data(), N_DB, K);
    EXPECT_TRUE(indices_match(result.indices.data(), ref_idx.data(), K));
}

TEST(DispatchL2, KLargerThanNDb)
{
    constexpr uint32_t N_Q = 3, N_DB = 7, DIM = 8, K = 20;
    auto q_data  = random_floats(N_Q  * DIM, -1.f, 1.f, 30);
    auto db_data = random_floats(N_DB * DIM, -1.f, 1.f, 40);

    auto result = dispatch_dense_search(q_data.data(), db_data.data(),
                                        N_Q, N_DB, DIM, K, Metric::L2);

    EXPECT_EQ(result.k, N_DB);  // clamped to n_db
    EXPECT_EQ(result.distances.size(), static_cast<size_t>(N_Q) * N_DB);
}

TEST(DispatchL2, HighDimensional)
{
    constexpr uint32_t N_Q = 4, N_DB = 64, DIM = 1024, K = 5;
    auto q_data  = random_floats(N_Q  * DIM, -1.f, 1.f, 50);
    auto db_data = random_floats(N_DB * DIM, -1.f, 1.f, 60);

    auto ref_dist = ref_l2_matrix(q_data.data(), N_Q, db_data.data(), N_DB, DIM);
    auto result   = dispatch_dense_search(q_data.data(), db_data.data(),
                                          N_Q, N_DB, DIM, K, Metric::L2);

    for (uint32_t q = 0; q < N_Q; ++q) {
        auto ref_idx = ref_topk_indices(ref_dist.data() + static_cast<size_t>(q) * N_DB,
                                         N_DB, K);
        EXPECT_TRUE(indices_match(result.indices.data() + static_cast<size_t>(q) * K,
                                  ref_idx.data(), K))
            << "query=" << q;
    }
}

// ---------------------------------------------------------------------------
// Test: dispatch_dense_search COSINE — correctness vs CPU reference
// ---------------------------------------------------------------------------

TEST(DispatchCosine, MatchesCpuReference)
{
    constexpr uint32_t N_Q = 8, N_DB = 200, DIM = 64, K = 10;

    auto q_data  = random_floats(N_Q  * DIM, -1.f, 1.f, 3);
    auto db_data = random_floats(N_DB * DIM, -1.f, 1.f, 4);

    // DenseIndex normalises db at add time and queries at search time.
    // Replicate that here to build the reference.
    std::vector<float> q_normed  = q_data;
    std::vector<float> db_normed = db_data;
    normalise_rows(q_normed.data(),  N_Q,  DIM);
    normalise_rows(db_normed.data(), N_DB, DIM);

    auto ref_dist = ref_cosine_matrix(q_normed.data(), N_Q,
                                       db_normed.data(), N_DB, DIM);

    // dispatch expects pre-normalised queries and db.
    auto result = dispatch_dense_search(q_normed.data(), db_normed.data(),
                                        N_Q, N_DB, DIM, K, Metric::COSINE);

    ASSERT_EQ(result.n_queries, N_Q);
    ASSERT_EQ(result.k, K);

    for (uint32_t q = 0; q < N_Q; ++q) {
        const float*   rd = result.distances.data() + static_cast<size_t>(q) * K;
        const int64_t* ri = result.indices.data()   + static_cast<size_t>(q) * K;

        for (uint32_t j = 0; j < K; ++j) {
            const float ref_d = ref_dist[static_cast<size_t>(q) * N_DB + ri[j]];
            EXPECT_NEAR(rd[j], ref_d, 1e-4f)
                << "query=" << q << " rank=" << j;
        }

        for (uint32_t j = 1; j < K; ++j)
            EXPECT_LE(rd[j - 1], rd[j]) << "query=" << q;

        auto ref_idx = ref_topk_indices(ref_dist.data() + static_cast<size_t>(q) * N_DB,
                                         N_DB, K);
        EXPECT_TRUE(indices_match(ri, ref_idx.data(), K))
            << "query=" << q;
    }
}

TEST(DispatchCosine, KLargerThanNDb)
{
    constexpr uint32_t N_Q = 2, N_DB = 5, DIM = 8, K = 20;
    auto q_data  = random_floats(N_Q  * DIM, -1.f, 1.f, 70);
    auto db_data = random_floats(N_DB * DIM, -1.f, 1.f, 80);
    normalise_rows(q_data.data(),  N_Q,  DIM);
    normalise_rows(db_data.data(), N_DB, DIM);

    auto result = dispatch_dense_search(q_data.data(), db_data.data(),
                                        N_Q, N_DB, DIM, K, Metric::COSINE);
    EXPECT_EQ(result.k, N_DB);
}

// ---------------------------------------------------------------------------
// Test: DenseIndex end-to-end — results identical before and after wiring
// to dispatch (GPU path or CPU fallback should give same answers as the old
// per-query loop, which is tested exhaustively in test_l2.cpp / test_cosine.cpp).
// ---------------------------------------------------------------------------

TEST(DenseIndexDispatch, L2ResultsUnchanged)
{
    DenseIndex idx(Metric::L2);

    const std::vector<float> vecs = {
        0.f, 0.f, 0.f,
        1.f, 0.f, 0.f,
        0.f, 1.f, 0.f,
        0.f, 0.f, 1.f,
    };
    idx.add(vecs.data(), 4, 3);

    const std::vector<float> q = {0.01f, 0.01f, 0.01f};
    auto result = idx.search(q.data(), 1, 1);

    ASSERT_EQ(result.indices.size(), 1u);
    EXPECT_EQ(result.indices[0], 0);
    EXPECT_NEAR(result.distances[0], std::sqrt(3.f) * 0.01f, 1e-4f);
}

TEST(DenseIndexDispatch, CosineResultsUnchanged)
{
    DenseIndex idx(Metric::COSINE);

    // Four vectors pointing along axes — cosine distances are all 1.0 except
    // the query's own axis which is 0.0.
    const std::vector<float> vecs = {
        1.f, 0.f, 0.f,
        0.f, 1.f, 0.f,
        0.f, 0.f, 1.f,
        -1.f, 0.f, 0.f,
    };
    idx.add(vecs.data(), 4, 3);

    const std::vector<float> q = {1.f, 0.f, 0.f};
    auto result = idx.search(q.data(), 1, 4);

    ASSERT_EQ(result.k, 4u);
    EXPECT_EQ(result.indices[0], 0);        // same direction → distance 0
    EXPECT_NEAR(result.distances[0], 0.f, 1e-5f);
    EXPECT_EQ(result.indices[3], 3);        // opposite → distance 2
    EXPECT_NEAR(result.distances[3], 2.f, 1e-5f);
}

#ifdef WITH_CUDA
// ---------------------------------------------------------------------------
// Tests that exercise the GPU kernels directly (skipped if no device found).
// ---------------------------------------------------------------------------

// Helper: skip the test if no GPU is available.
#define SKIP_IF_NO_GPU()                                    \
    do {                                                    \
        if (cuda_device_count() == 0) {                    \
            GTEST_SKIP() << "No CUDA device — skipping";   \
        }                                                   \
    } while (0)

// Verify that cuda_l2_batch output matches CPU reference within 1e-5.
TEST(CudaL2Kernel, MatchesCpuReference)
{
    SKIP_IF_NO_GPU();

    constexpr uint32_t N_Q = 5, N_DB = 100, DIM = 64;

    auto q_data  = random_floats(N_Q  * DIM, -1.f, 1.f, 101);
    auto db_data = random_floats(N_DB * DIM, -1.f, 1.f, 102);
    auto ref     = ref_l2_matrix(q_data.data(), N_Q, db_data.data(), N_DB, DIM);

    DeviceBuffer<float> d_q(N_Q  * DIM);
    DeviceBuffer<float> d_db(N_DB * DIM);
    DeviceBuffer<float> d_out(N_Q * N_DB);
    d_q.copy_from_host(q_data.data(),  N_Q  * DIM);
    d_db.copy_from_host(db_data.data(), N_DB * DIM);

    cuda_l2_batch(d_q.get(), d_db.get(), d_out.get(), N_Q, N_DB, DIM);
    SPARSITY_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> out(N_Q * N_DB);
    d_out.copy_to_host(out.data(), N_Q * N_DB);

    for (size_t i = 0; i < out.size(); ++i) {
        EXPECT_NEAR(out[i], ref[i], 1e-4f) << "element " << i;
    }
}

TEST(CudaL2Kernel, ZeroVectors)
{
    SKIP_IF_NO_GPU();

    constexpr uint32_t N_Q = 2, N_DB = 3, DIM = 8;
    std::vector<float> zeros(N_Q * DIM, 0.f);
    std::vector<float> db(N_DB * DIM, 0.f);

    DeviceBuffer<float> d_q(N_Q * DIM), d_db(N_DB * DIM), d_out(N_Q * N_DB);
    d_q.copy_from_host(zeros.data(), N_Q * DIM);
    d_db.copy_from_host(db.data(), N_DB * DIM);
    cuda_l2_batch(d_q.get(), d_db.get(), d_out.get(), N_Q, N_DB, DIM);
    SPARSITY_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> out(N_Q * N_DB);
    d_out.copy_to_host(out.data(), N_Q * N_DB);
    for (float v : out) EXPECT_FLOAT_EQ(v, 0.f);
}

TEST(CudaL2Kernel, HighDimensional)
{
    SKIP_IF_NO_GPU();

    constexpr uint32_t N_Q = 2, N_DB = 32, DIM = 1024;
    auto q_data  = random_floats(N_Q  * DIM, -1.f, 1.f, 200);
    auto db_data = random_floats(N_DB * DIM, -1.f, 1.f, 201);
    auto ref     = ref_l2_matrix(q_data.data(), N_Q, db_data.data(), N_DB, DIM);

    DeviceBuffer<float> d_q(N_Q * DIM), d_db(N_DB * DIM), d_out(N_Q * N_DB);
    d_q.copy_from_host(q_data.data(),  N_Q  * DIM);
    d_db.copy_from_host(db_data.data(), N_DB * DIM);
    cuda_l2_batch(d_q.get(), d_db.get(), d_out.get(), N_Q, N_DB, DIM);
    SPARSITY_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> out(N_Q * N_DB);
    d_out.copy_to_host(out.data(), N_Q * N_DB);
    for (size_t i = 0; i < out.size(); ++i)
        EXPECT_NEAR(out[i], ref[i], 1e-3f) << "element " << i;
}

TEST(CudaCosineKernel, MatchesCpuReference)
{
    SKIP_IF_NO_GPU();

    constexpr uint32_t N_Q = 5, N_DB = 100, DIM = 64;
    auto q_data  = random_floats(N_Q  * DIM, -1.f, 1.f, 300);
    auto db_data = random_floats(N_DB * DIM, -1.f, 1.f, 301);
    normalise_rows(q_data.data(),  N_Q,  DIM);
    normalise_rows(db_data.data(), N_DB, DIM);
    auto ref = ref_cosine_matrix(q_data.data(), N_Q, db_data.data(), N_DB, DIM);

    DeviceBuffer<float> d_q(N_Q * DIM), d_db(N_DB * DIM), d_out(N_Q * N_DB);
    d_q.copy_from_host(q_data.data(),  N_Q  * DIM);
    d_db.copy_from_host(db_data.data(), N_DB * DIM);
    cuda_cosine_batch(d_q.get(), d_db.get(), d_out.get(), N_Q, N_DB, DIM);
    SPARSITY_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> out(N_Q * N_DB);
    d_out.copy_to_host(out.data(), N_Q * N_DB);
    for (size_t i = 0; i < out.size(); ++i)
        EXPECT_NEAR(out[i], ref[i], 1e-4f) << "element " << i;
}

TEST(CudaTopKKernel, MatchesCpuTopK)
{
    SKIP_IF_NO_GPU();

    constexpr uint32_t N_Q = 6, N_DB = 200, K = 10;
    // Use a flat distance matrix (no need for real vectors here).
    auto dist_mat = random_floats(N_Q * N_DB, 0.f, 10.f, 400);

    DeviceBuffer<float>   d_dist(N_Q * N_DB);
    DeviceBuffer<float>   d_out_d(N_Q * K);
    DeviceBuffer<int64_t> d_out_i(N_Q * K);
    d_dist.copy_from_host(dist_mat.data(), N_Q * N_DB);

    cuda_topk_batch(d_dist.get(), N_DB, K, d_out_d.get(), d_out_i.get(), N_Q);
    SPARSITY_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float>   out_d(N_Q * K);
    std::vector<int64_t> out_i(N_Q * K);
    d_out_d.copy_to_host(out_d.data(), N_Q * K);
    d_out_i.copy_to_host(out_i.data(), N_Q * K);

    for (uint32_t q = 0; q < N_Q; ++q) {
        const float*   rd = out_d.data() + static_cast<size_t>(q) * K;
        const int64_t* ri = out_i.data() + static_cast<size_t>(q) * K;

        auto ref_idx = ref_topk_indices(dist_mat.data() + static_cast<size_t>(q) * N_DB,
                                         N_DB, K);
        EXPECT_TRUE(indices_match(ri, ref_idx.data(), K)) << "query=" << q;

        for (uint32_t j = 1; j < K; ++j)
            EXPECT_LE(rd[j - 1], rd[j]) << "query=" << q;
    }
}

TEST(CudaTopKKernel, KLargerThanNDb)
{
    SKIP_IF_NO_GPU();

    constexpr uint32_t N_Q = 2, N_DB = 5, K = 20;
    auto dist_mat = random_floats(N_Q * N_DB, 0.f, 5.f, 500);

    DeviceBuffer<float>   d_dist(N_Q * N_DB);
    DeviceBuffer<float>   d_out_d(N_Q * K);
    DeviceBuffer<int64_t> d_out_i(N_Q * K);
    d_dist.copy_from_host(dist_mat.data(), N_Q * N_DB);

    cuda_topk_batch(d_dist.get(), N_DB, K, d_out_d.get(), d_out_i.get(), N_Q);
    SPARSITY_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float>   out_d(N_Q * K);
    std::vector<int64_t> out_i(N_Q * K);
    d_out_d.copy_to_host(out_d.data(), N_Q * K);
    d_out_i.copy_to_host(out_i.data(), N_Q * K);

    // Only first N_DB slots should have valid indices; rest should be -1.
    for (uint32_t q = 0; q < N_Q; ++q) {
        for (uint32_t j = 0; j < N_DB; ++j)
            EXPECT_GE(out_i[static_cast<size_t>(q) * K + j], 0)
                << "query=" << q << " rank=" << j;
        for (uint32_t j = N_DB; j < K; ++j)
            EXPECT_EQ(out_i[static_cast<size_t>(q) * K + j], -1)
                << "query=" << q << " rank=" << j;
    }
}
#endif // WITH_CUDA

} // namespace
} // namespace sparsity

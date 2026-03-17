#include "sparsity/metrics.h"
#include "sparsity/packing.h"

#include <gtest/gtest.h>
#include <cstdint>
#include <vector>

namespace sparsity {
namespace {

TEST(TanimotoSimilarity, IdenticalVectors) {
    const std::vector<uint64_t> a = {0xDEADBEEFCAFEBABEULL, 0x0123456789ABCDEFULL};
    EXPECT_FLOAT_EQ(tanimoto_similarity(a.data(), a.data(), 2), 1.0f);
}

TEST(TanimotoSimilarity, DisjointVectors) {
    // a and b have no bits in common
    const std::vector<uint64_t> a = {0xAAAAAAAAAAAAAAAAULL};  // alternating 1010...
    const std::vector<uint64_t> b = {0x5555555555555555ULL};  // alternating 0101...
    EXPECT_FLOAT_EQ(tanimoto_similarity(a.data(), b.data(), 1), 0.0f);
}

TEST(TanimotoSimilarity, BothZero) {
    const std::vector<uint64_t> a = {0ULL, 0ULL};
    EXPECT_FLOAT_EQ(tanimoto_similarity(a.data(), a.data(), 2), 1.0f);
}

TEST(TanimotoSimilarity, OneZero) {
    const std::vector<uint64_t> a = {0ULL};
    const std::vector<uint64_t> b = {0xFFFFFFFFFFFFFFFFULL};
    // intersection = 0, union = 64 bits set → similarity = 0
    EXPECT_FLOAT_EQ(tanimoto_similarity(a.data(), b.data(), 1), 0.0f);
}

TEST(TanimotoSimilarity, KnownValue) {
    // a = 0b1100, b = 0b1010
    // intersection = 0b1000 → popcount = 1
    // union        = 0b1110 → popcount = 3
    // similarity   = 1/3
    const std::vector<uint64_t> a = {0b1100ULL};
    const std::vector<uint64_t> b = {0b1010ULL};
    EXPECT_NEAR(tanimoto_similarity(a.data(), b.data(), 1), 1.0f / 3.0f, 1e-6f);
}

TEST(TanimotoSimilarity, Symmetry) {
    const std::vector<uint64_t> a = {0xDEADBEEF00000000ULL};
    const std::vector<uint64_t> b = {uint64_t{0x00000000CAFEBABE}};
    EXPECT_FLOAT_EQ(tanimoto_similarity(a.data(), b.data(), 1),
                    tanimoto_similarity(b.data(), a.data(), 1));
}

TEST(TanimotoSimilarity, MultiWord) {
    // 4 words; each word is half-overlapping
    // a = 0b1100 repeated, b = 0b1010 repeated → each word gives 1/3
    const std::vector<uint64_t> a(4, 0b1100ULL);
    const std::vector<uint64_t> b(4, 0b1010ULL);
    EXPECT_NEAR(tanimoto_similarity(a.data(), b.data(), 4), 1.0f / 3.0f, 1e-6f);
}

// ---------------------------------------------------------------------------
// pack_bits / pack_bits_batch
// ---------------------------------------------------------------------------

TEST(PackBits, SingleWordKnownBits) {
    // bits 0, 2 set → word should be 0b0101 = 5
    const bool src[] = {true, false, true, false};
    uint64_t dst = 0;
    pack_bits(src, 4, &dst);
    EXPECT_EQ(dst, uint64_t{0b0101});
}

TEST(PackBits, PaddingZeroed) {
    // dim=3, only bits 0 and 1 set; bit 63 must stay zero
    const bool src[] = {true, true, false};
    uint64_t dst = ~uint64_t{0};  // all ones to start
    pack_bits(src, 3, &dst);
    EXPECT_EQ(dst, uint64_t{0b011});
}

TEST(PackBits, SpansMultipleWords) {
    // 65 bits: bit 0 and bit 64 set
    std::vector<bool> src(65, false);
    src[0]  = true;
    src[64] = true;
    std::vector<uint64_t> dst(2, 0);
    pack_bits(src.data(), 65, dst.data());
    EXPECT_EQ(dst[0], uint64_t{1});   // bit 0 in word 0
    EXPECT_EQ(dst[1], uint64_t{1});   // bit 64 → bit 0 in word 1
}

TEST(PackBits, RoundTripTanimoto) {
    // Pack two identical bool vectors and verify Tanimoto = 1.0
    std::vector<bool> src(128, false);
    src[0] = src[7] = src[63] = src[127] = true;
    auto packed = pack_bits_batch(src.data(), 1, 128);
    EXPECT_FLOAT_EQ(tanimoto_similarity(packed.data(), packed.data(), 2), 1.0f);
}

} // namespace
} // namespace sparsity

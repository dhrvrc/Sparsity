#pragma once

#include <cstdint>

namespace sparsity {

// Packed-bit binary vector for Tanimoto similarity.
// Bit i is stored at words[i / 64], position (i % 64)  (LSB = bit 0).
// INVARIANT: padding bits beyond dim must be zero.
struct BinaryVector {
    const uint64_t* words = nullptr; // packed bits
    uint32_t        dim   = 0;       // total number of bits

    uint32_t n_words() const { return (dim + 63u) / 64u; }
};

// Collection of N binary vectors stored row-major.
// Row i: data[i * words_per_row .. i * words_per_row + words_per_row)
// This struct is non-owning — see SparseIndex for owned storage.
struct BinaryMatrix {
    uint64_t* data   = nullptr;
    uint32_t  n_rows = 0;
    uint32_t  dim    = 0; // bits per vector

    uint32_t words_per_row() const { return (dim + 63u) / 64u; }

    BinaryVector row(uint32_t i) const {
        return {data + static_cast<size_t>(i) * words_per_row(), dim};
    }
};

} // namespace sparsity

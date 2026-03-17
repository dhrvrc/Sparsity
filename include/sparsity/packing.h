#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

// Helpers for packing raw boolean arrays into the packed uint64_t format
// expected by BinaryMatrix / tanimoto_similarity.
//
// Storage format: bit i of a vector lives at words[i / 64], LSB = bit 0.
// Padding bits beyond dim are always zeroed.

namespace sparsity {

// Pack a single boolean vector of length dim into dst.
// dst must point to at least ceil(dim / 64) uint64_t words.
void pack_bits(const bool* src, uint32_t dim, uint64_t* dst);

// Pack a batch of n_rows boolean vectors, each of length dim, into dst.
// src is row-major: src[row * dim + col].
// dst must point to at least n_rows * ceil(dim / 64) uint64_t words.
void pack_bits_batch(const bool* src, uint32_t n_rows, uint32_t dim, uint64_t* dst);

// Convenience: pack and return an owned buffer.
std::vector<uint64_t> pack_bits_batch(const bool* src, uint32_t n_rows, uint32_t dim);

} // namespace sparsity

#include "sparsity/packing.h"

#include <cstring>

namespace sparsity {

void pack_bits(const bool* src, uint32_t dim, uint64_t* dst) {
    const uint32_t n_words = (dim + 63u) / 64u;
    std::memset(dst, 0, n_words * sizeof(uint64_t));

    for (uint32_t i = 0; i < dim; ++i) {
        if (src[i]) {
            dst[i / 64u] |= uint64_t{1} << (i % 64u);
        }
    }
}

void pack_bits_batch(const bool* src, uint32_t n_rows, uint32_t dim, uint64_t* dst) {
    const uint32_t words_per_row = (dim + 63u) / 64u;
    for (uint32_t r = 0; r < n_rows; ++r) {
        pack_bits(src + static_cast<size_t>(r) * dim,
                  dim,
                  dst  + static_cast<size_t>(r) * words_per_row);
    }
}

std::vector<uint64_t> pack_bits_batch(const bool* src, uint32_t n_rows, uint32_t dim) {
    const uint32_t words_per_row = (dim + 63u) / 64u;
    std::vector<uint64_t> buf(static_cast<size_t>(n_rows) * words_per_row);
    pack_bits_batch(src, n_rows, dim, buf.data());
    return buf;
}

} // namespace sparsity

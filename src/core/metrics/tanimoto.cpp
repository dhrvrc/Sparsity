#include "sparsity/metrics.h"

namespace sparsity {

float tanimoto_similarity(const uint64_t* a, const uint64_t* b, uint32_t n_words) {
    uint32_t intersection = 0;
    uint32_t union_       = 0;

    for (uint32_t i = 0; i < n_words; ++i) {
        intersection += __builtin_popcountll(a[i] & b[i]);
        union_       += __builtin_popcountll(a[i] | b[i]);
    }

    if (union_ == 0) return 1.0f;  // both all-zero — identical empty sets
    return static_cast<float>(intersection) / static_cast<float>(union_);
}

} // namespace sparsity

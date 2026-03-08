#include "sparsity/metrics.h"

#include <cmath>

namespace sparsity {

float cosine_distance(const float* a, const float* b, uint32_t dim) {
    float dot    = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;

    for (uint32_t i = 0; i < dim; ++i) {
        dot    += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    if (norm_a == 0.0f || norm_b == 0.0f) return 1.0f;
    return 1.0f - dot / std::sqrt(norm_a * norm_b);
}

} // namespace sparsity

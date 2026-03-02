#include "sparsity/metrics.h"

#include <cmath>
#include <cstdint>

namespace sparsity {

float l2_distance_sq(const float* a, const float* b, uint32_t dim) {
    float    sum = 0.0f;
    uint32_t i   = 0;

    // 4-wide unroll: lets the compiler emit SIMD and hides FP latency.
    for (; i + 4 <= dim; i += 4) {
        const float d0 = a[i + 0] - b[i + 0];
        const float d1 = a[i + 1] - b[i + 1];
        const float d2 = a[i + 2] - b[i + 2];
        const float d3 = a[i + 3] - b[i + 3];
        sum += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
    }
    for (; i < dim; ++i) {
        const float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

float l2_distance(const float* a, const float* b, uint32_t dim) {
    return std::sqrt(l2_distance_sq(a, b, dim));
}

} // namespace sparsity

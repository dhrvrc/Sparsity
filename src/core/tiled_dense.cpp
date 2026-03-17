#include "sparsity/tiled_dense.h"
#include "sparsity/metrics.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

namespace sparsity {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

TiledDenseIndex::TiledDenseIndex(Metric metric) : metric_(metric) {
    if (metric != Metric::L2 && metric != Metric::COSINE) {
        throw std::invalid_argument(
            "TiledDenseIndex: only L2 and COSINE metrics are supported");
    }
}

// ---------------------------------------------------------------------------
// add
// ---------------------------------------------------------------------------

void TiledDenseIndex::add(const float* data, uint32_t n, uint32_t dim) {
    if (n == 0) return;

    if (mat_.n_cols != 0 && mat_.n_cols != dim) {
        throw std::invalid_argument(
            "TiledDenseIndex::add — dimension mismatch: index has dim=" +
            std::to_string(mat_.n_cols) + ", got dim=" + std::to_string(dim));
    }

    const size_t n_new_floats = static_cast<size_t>(n) * dim;
    const size_t old_size     = storage_.size();
    storage_.resize(old_size + n_new_floats);
    std::memcpy(storage_.data() + old_size, data, n_new_floats * sizeof(float));

    // Refresh the non-owning view (storage_ may have reallocated).
    mat_.data    = storage_.data();
    mat_.n_rows += n;
    mat_.n_cols  = dim;

    float* new_base = storage_.data() + old_size;

    if (metric_ == Metric::COSINE) {
        // Pre-normalise to unit length — same convention as DenseIndex.
        for (uint32_t i = 0; i < n; ++i) {
            float* vec     = new_base + static_cast<size_t>(i) * dim;
            float  norm_sq = 0.0f;
            for (uint32_t j = 0; j < dim; ++j) norm_sq += vec[j] * vec[j];
            if (norm_sq > 0.0f) {
                const float inv = 1.0f / std::sqrt(norm_sq);
                for (uint32_t j = 0; j < dim; ++j) vec[j] *= inv;
            }
        }
    } else {
        // L2: pre-compute ||di||² for the norm decomposition trick.
        db_norms_sq_.reserve(db_norms_sq_.size() + n);
        for (uint32_t i = 0; i < n; ++i) {
            const float* vec     = new_base + static_cast<size_t>(i) * dim;
            float        norm_sq = 0.0f;
            for (uint32_t j = 0; j < dim; ++j) norm_sq += vec[j] * vec[j];
            db_norms_sq_.push_back(norm_sq);
        }
    }
}

// ---------------------------------------------------------------------------
// search
// ---------------------------------------------------------------------------

SearchResult TiledDenseIndex::search(const float* queries, uint32_t n_queries,
                                     uint32_t k) const {
    if (mat_.n_rows == 0) {
        throw std::runtime_error("TiledDenseIndex::search — index is empty");
    }

    const uint32_t n        = mat_.n_rows;
    const uint32_t d        = mat_.n_cols;
    const uint32_t k_actual = std::min(k, n);
    const bool     cosine   = (metric_ == Metric::COSINE);

    SearchResult result;
    result.n_queries = n_queries;
    result.k         = k_actual;
    result.distances.resize(static_cast<size_t>(n_queries) * k_actual);
    result.indices.resize(static_cast<size_t>(n_queries) * k_actual, -1);

    // -----------------------------------------------------------------------
    // Pre-process queries
    //   L2:    copy as-is; query norms computed per tile below.
    //   Cosine: normalise each query to unit length once (not per DB vector).
    // -----------------------------------------------------------------------
    std::vector<float> q_storage(static_cast<size_t>(n_queries) * d);
    std::memcpy(q_storage.data(), queries, q_storage.size() * sizeof(float));

    if (cosine) {
        for (uint32_t q = 0; q < n_queries; ++q) {
            float* qv      = q_storage.data() + static_cast<size_t>(q) * d;
            float  norm_sq = 0.0f;
            for (uint32_t j = 0; j < d; ++j) norm_sq += qv[j] * qv[j];
            if (norm_sq > 0.0f) {
                const float inv = 1.0f / std::sqrt(norm_sq);
                for (uint32_t j = 0; j < d; ++j) qv[j] *= inv;
            }
        }
    }

    // -----------------------------------------------------------------------
    // Scratch buffer for one tile: TILE_Q rows of n (distance, index) pairs.
    // Reused across tiles to avoid repeated allocation.
    // -----------------------------------------------------------------------
    std::vector<std::pair<float, int64_t>> tile_buf(
        static_cast<size_t>(TILE_Q) * n);

    // -----------------------------------------------------------------------
    // Tiled loop: one DB pass per tile of TILE_Q queries.
    //
    // Inner kernel (for each DB vector `db_vec`):
    //   Load db_vec element-wise in 4-wide chunks.
    //   Each chunk (4 floats) is reused across all qn query accumulators —
    //   this is the key cache amortisation.
    // -----------------------------------------------------------------------
    for (uint32_t q0 = 0; q0 < n_queries; q0 += TILE_Q) {
        const uint32_t qn = std::min(TILE_Q, n_queries - q0);

        // Pointers into (normalised) query storage for this tile.
        const float* qptrs[TILE_Q];
        for (uint32_t qi = 0; qi < qn; ++qi)
            qptrs[qi] = q_storage.data() + static_cast<size_t>(q0 + qi) * d;

        // Row pointers into the tile scratch buffer.
        std::pair<float, int64_t>* rows[TILE_Q];
        for (uint32_t qi = 0; qi < qn; ++qi)
            rows[qi] = tile_buf.data() + static_cast<size_t>(qi) * n;

        // For L2: compute ||qi||² once per tile (cheap, avoids repeating).
        float q_norms_sq[TILE_Q] = {};
        if (!cosine) {
            for (uint32_t qi = 0; qi < qn; ++qi) {
                for (uint32_t j = 0; j < d; ++j)
                    q_norms_sq[qi] += qptrs[qi][j] * qptrs[qi][j];
            }
        }

        // Single pass over all DB vectors for this query tile.
        for (uint32_t i = 0; i < n; ++i) {
            const float* db_vec = mat_.row(i);

            // Accumulate dot products: load each db element once, multiply
            // into all qn query accumulators.  4-wide unroll matches l2.cpp.
            float dots[TILE_Q] = {};
            uint32_t j = 0;
            for (; j + 4 <= d; j += 4) {
                const float dv0 = db_vec[j + 0];
                const float dv1 = db_vec[j + 1];
                const float dv2 = db_vec[j + 2];
                const float dv3 = db_vec[j + 3];
                for (uint32_t qi = 0; qi < qn; ++qi) {
                    dots[qi] += qptrs[qi][j + 0] * dv0
                              + qptrs[qi][j + 1] * dv1
                              + qptrs[qi][j + 2] * dv2
                              + qptrs[qi][j + 3] * dv3;
                }
            }
            // Scalar tail for dimensions not divisible by 4.
            for (; j < d; ++j) {
                const float dv = db_vec[j];
                for (uint32_t qi = 0; qi < qn; ++qi)
                    dots[qi] += qptrs[qi][j] * dv;
            }

            // Convert dot products to distances and store in tile buffer.
            if (cosine) {
                // Both query and DB vectors are pre-normalised.
                // Cosine distance = 1 - dot.
                for (uint32_t qi = 0; qi < qn; ++qi)
                    rows[qi][i] = {1.0f - dots[qi], static_cast<int64_t>(i)};
            } else {
                // L2 decomposition:
                //   ||qi - dj||² = ||qi||² + ||dj||² - 2·dot(qi, dj)
                // Clamp to 0 to absorb floating-point rounding error.
                const float db_norm_sq = db_norms_sq_[i];
                for (uint32_t qi = 0; qi < qn; ++qi) {
                    const float dsq = std::max(
                        0.0f,
                        q_norms_sq[qi] + db_norm_sq - 2.0f * dots[qi]);
                    rows[qi][i] = {dsq, static_cast<int64_t>(i)};
                }
            }
        }

        // Top-k selection for each query in this tile.
        for (uint32_t qi = 0; qi < qn; ++qi) {
            std::partial_sort(rows[qi], rows[qi] + k_actual, rows[qi] + n);

            float*   out_dist = result.distances.data() +
                                static_cast<size_t>(q0 + qi) * k_actual;
            int64_t* out_idx  = result.indices.data() +
                                static_cast<size_t>(q0 + qi) * k_actual;

            for (uint32_t jj = 0; jj < k_actual; ++jj) {
                // For L2: ranked by squared distance; convert to true L2 now.
                out_dist[jj] = cosine ? rows[qi][jj].first
                                      : std::sqrt(rows[qi][jj].first);
                out_idx[jj]  = rows[qi][jj].second;
            }
        }
    }

    return result;
}

// ---------------------------------------------------------------------------
// validate
// ---------------------------------------------------------------------------

void TiledDenseIndex::validate() const {
    assert(storage_.size() == static_cast<size_t>(mat_.n_rows) * mat_.n_cols);
    assert(mat_.data == storage_.data() || storage_.empty());
    assert(mat_.n_rows == 0 || mat_.n_cols > 0);
    if (metric_ == Metric::L2) {
        assert(db_norms_sq_.size() == mat_.n_rows);
    }
}

} // namespace sparsity

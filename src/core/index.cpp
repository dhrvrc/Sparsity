#include "sparsity/index.h"
#include "sparsity/dispatch.h"
#include "sparsity/metrics.h"
#include "sparsity/packing.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

namespace sparsity {

// ---------------------------------------------------------------------------
// DenseIndex
// ---------------------------------------------------------------------------

DenseIndex::DenseIndex(Metric metric) : metric_(metric) {}

DenseIndex::~DenseIndex() = default;

void DenseIndex::add(const float* data, uint32_t n, uint32_t dim) {
    if (n == 0) return;

    if (mat_.n_cols != 0 && mat_.n_cols != dim) {
        throw std::invalid_argument(
            "DenseIndex::add — dimension mismatch: index has dim=" +
            std::to_string(mat_.n_cols) + ", got dim=" + std::to_string(dim));
    }

    // Append the new rows to backing storage.
    // After resize the old mat_.data pointer may be invalid so we fix it below.
    const size_t n_new_floats = static_cast<size_t>(n) * dim;
    const size_t old_size     = storage_.size();
    storage_.resize(old_size + n_new_floats);
    std::memcpy(storage_.data() + old_size, data, n_new_floats * sizeof(float));

    // Keep mat_ pointing at storage_ (stable after we stop resizing this call).
    mat_.data    = storage_.data();
    mat_.n_rows += n;
    mat_.n_cols  = dim;

    // For cosine, pre-normalise new rows to unit length so that search-time
    // inner products equal cosine similarity directly.
    if (metric_ == Metric::COSINE) {
        float* base = storage_.data() + old_size;
        for (uint32_t i = 0; i < n; ++i) {
            float* vec     = base + static_cast<size_t>(i) * dim;
            float  norm_sq = 0.0f;
            for (uint32_t j = 0; j < dim; ++j) norm_sq += vec[j] * vec[j];
            if (norm_sq > 0.0f) {
                const float inv = 1.0f / std::sqrt(norm_sq);
                for (uint32_t j = 0; j < dim; ++j) vec[j] *= inv;
            }
            // zero-norm row: leave as zero — distance to any valid query = 1.0
        }
    }
}

SearchResult DenseIndex::search(const float* queries, uint32_t n_queries, uint32_t k) const {
    if (mat_.n_rows == 0) {
        throw std::runtime_error("DenseIndex::search — index is empty");
    }
    if (metric_ != Metric::L2 && metric_ != Metric::COSINE) {
        throw std::logic_error("DenseIndex::search — unsupported metric");
    }

    const uint32_t d = mat_.n_cols;

    if (metric_ == Metric::L2) {
        // L2: pass raw queries and raw db directly to dispatch.
        return dispatch_dense_search(queries, mat_.data,
                                     n_queries, mat_.n_rows, d, k, Metric::L2);
    }

    // COSINE: db vectors are already unit-normalised (done at add time).
    // Normalise all queries on the host before dispatch so the GPU kernel
    // only needs to compute dot products.
    std::vector<float> q_normed(static_cast<size_t>(n_queries) * d);
    for (uint32_t q = 0; q < n_queries; ++q) {
        const float* src = queries + static_cast<size_t>(q) * d;
        float*       dst = q_normed.data() + static_cast<size_t>(q) * d;

        float norm_sq = 0.0f;
        for (uint32_t j = 0; j < d; ++j) norm_sq += src[j] * src[j];

        if (norm_sq > 0.0f) {
            const float inv = 1.0f / std::sqrt(norm_sq);
            for (uint32_t j = 0; j < d; ++j) dst[j] = src[j] * inv;
        } else {
            // Zero-norm query: distance to every db vector will be 1 - 0 = 1.
            std::fill(dst, dst + d, 0.0f);
        }
    }

    return dispatch_dense_search(q_normed.data(), mat_.data,
                                  n_queries, mat_.n_rows, d, k, Metric::COSINE);
}

void DenseIndex::validate() const {
    assert(storage_.size() == static_cast<size_t>(mat_.n_rows) * mat_.n_cols);
    assert(mat_.data == storage_.data() || storage_.empty());
    assert(mat_.n_rows == 0 || mat_.n_cols > 0);
}

// ---------------------------------------------------------------------------
// SparseIndex stubs
// ---------------------------------------------------------------------------

SparseIndex::SparseIndex(uint32_t dim, Metric metric) : dim_(dim), metric_(metric) {}

void SparseIndex::add(const uint32_t*, const uint32_t*, const float*, uint32_t) {
    throw std::logic_error("SparseIndex::add — not yet implemented");
}

void SparseIndex::add_binary(const uint64_t*, uint32_t) {
    throw std::logic_error("SparseIndex::add_binary — not yet implemented");
}

void SparseIndex::add_binary(const bool* data, uint32_t n_rows) {
    auto packed = pack_bits_batch(data, n_rows, dim_);
    add_binary(packed.data(), n_rows);
}

SearchResult SparseIndex::search(const uint32_t*, const uint32_t*, const float*,
                                 uint32_t, uint32_t) const {
    throw std::logic_error("SparseIndex::search — not yet implemented");
}

SearchResult SparseIndex::search_binary(const uint64_t*, uint32_t, uint32_t) const {
    throw std::logic_error("SparseIndex::search_binary — not yet implemented");
}

void SparseIndex::validate() const {
    throw std::logic_error("SparseIndex::validate — not yet implemented");
}

// ---------------------------------------------------------------------------
// Index — unified entry point
// The goal of this is to make the creation of indices easy for the caller
// they can essentially specify type of metric, datatype they want to use for storage of the vectors and  whether vectors are expected to be sparse or dense
// ---------------------------------------------------------------------------

Index::Index(Metric metric)
    : dtype_(DataType::DenseFloat), metric_(metric),
      dense_idx_(std::make_unique<DenseIndex>(metric)) {}

Index::Index(DataType dtype, Metric metric, uint32_t dim)
    : dtype_(dtype), metric_(metric) {
    if (!is_valid_combination(dtype, metric)) {
        throw std::invalid_argument(
            std::string("Index: unsupported dtype+metric combination — dtype=") +
            data_type_name(dtype) + ", metric=" + metric_name(metric));
    }
    switch (dtype) {
        case DataType::DenseFloat:
        case DataType::DenseInt32:
            dense_idx_ = std::make_unique<DenseIndex>(metric);
            break;
        case DataType::SparseFloat:
        case DataType::SparseInt32:
        case DataType::Binary:
            if (dim == 0) {
                throw std::invalid_argument(
                    "Index: dim must be > 0 for sparse/binary types");
            }
            sparse_idx_ = std::make_unique<SparseIndex>(dim, metric);
            break;
    }
}

uint32_t Index::dim() const noexcept {
    if (dense_idx_)  return dense_idx_->dim();
    if (sparse_idx_) return sparse_idx_->dim();
    return 0;
}

uint32_t Index::size() const noexcept {
    if (dense_idx_)  return dense_idx_->size();
    if (sparse_idx_) return sparse_idx_->size();
    return 0;
}

// --- add() ---

void Index::add(const float* data, uint32_t n, uint32_t dim) {
    if (dtype_ != DataType::DenseFloat) {
        throw std::invalid_argument(
            std::string("Index::add(float*) — index dtype is ") +
            data_type_name(dtype_) + ", expected dense_float");
    }
    dense_idx_->add(data, n, dim);
}

void Index::add(const int32_t* /*data*/, uint32_t /*n*/, uint32_t /*dim*/) {
    if (dtype_ != DataType::DenseInt32) {
        throw std::invalid_argument(
            std::string("Index::add(int32_t*) — index dtype is ") +
            data_type_name(dtype_) + ", expected dense_int32");
    }
    throw std::logic_error("Index::add — DenseInt32 not yet implemented");
}

void Index::add(const uint64_t* data, uint32_t n_rows) {
    if (dtype_ != DataType::Binary) {
        throw std::invalid_argument(
            std::string("Index::add(uint64_t*) — index dtype is ") +
            data_type_name(dtype_) + ", expected binary");
    }
    sparse_idx_->add_binary(data, n_rows);
}

void Index::add(const bool* data, uint32_t n_rows) {
    if (dtype_ != DataType::Binary) {
        throw std::invalid_argument(
            std::string("Index::add(bool*) — index dtype is ") +
            data_type_name(dtype_) + ", expected binary");
    }
    sparse_idx_->add_binary(data, n_rows);
}

void Index::add(const uint32_t* indptr, const uint32_t* indices,
                const float* values, uint32_t n_rows) {
    if (dtype_ != DataType::SparseFloat) {
        throw std::invalid_argument(
            std::string("Index::add(CSR float32) — index dtype is ") +
            data_type_name(dtype_) + ", expected sparse_float");
    }
    sparse_idx_->add(indptr, indices, values, n_rows);
}

void Index::add(const uint32_t* /*indptr*/, const uint32_t* /*indices*/,
                const int32_t* /*values*/, uint32_t /*n_rows*/) {
    if (dtype_ != DataType::SparseInt32) {
        throw std::invalid_argument(
            std::string("Index::add(CSR int32) — index dtype is ") +
            data_type_name(dtype_) + ", expected sparse_int32");
    }
    throw std::logic_error("Index::add — SparseInt32 not yet implemented");
}

// --- search() stubs ---

SearchResult Index::search(const float* queries, uint32_t n, uint32_t k) const {
    if (dtype_ != DataType::DenseFloat) {
        throw std::invalid_argument(
            std::string("Index::search(float*) — index dtype is ") +
            data_type_name(dtype_) + ", expected dense_float");
    }
    return dense_idx_->search(queries, n, k);
}

SearchResult Index::search(const int32_t*, uint32_t, uint32_t) const {
    throw std::logic_error("Index::search — not yet implemented");
}

SearchResult Index::search(const uint64_t*, uint32_t, uint32_t) const {
    throw std::logic_error("Index::search — not yet implemented");
}

SearchResult Index::search(const uint32_t*, const uint32_t*,
                           const float*, uint32_t, uint32_t) const {
    throw std::logic_error("Index::search — not yet implemented");
}

SearchResult Index::search(const uint32_t*, const uint32_t*,
                           const int32_t*, uint32_t, uint32_t) const {
    throw std::logic_error("Index::search — not yet implemented");
}

} // namespace sparsity

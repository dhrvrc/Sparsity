#pragma once

#include "sparsity/types.h"
#include "sparsity/dense.h"
#include "sparsity/sparse.h"
#include "sparsity/binary.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace sparsity {

// Result of a k-nearest-neighbour search.
// distances and indices are laid out row-major: (n_queries, k).
struct SearchResult {
    std::vector<float>   distances; // shape [n_queries * k]
    std::vector<int64_t> indices;   // shape [n_queries * k]; -1 = not found
    uint32_t             n_queries = 0;
    uint32_t             k         = 0;
};

// ---------------------------------------------------------------------------
// IndexBase — thin abstract base class (internal implementation detail)
//
// Provides the shared identity of every index type (size, dim, metric) and
// enforces a validate() contract.  Prefer the unified Index class in user code.
// ---------------------------------------------------------------------------
class IndexBase {
public:
    virtual ~IndexBase() = default;

    virtual uint32_t size()   const = 0;
    virtual uint32_t dim()    const = 0;
    virtual Metric   metric() const = 0;

    // Assert internal invariants.  Should be a no-op in Release builds;
    // implementations may throw or abort on violation.
    virtual void validate() const = 0;

protected:
    IndexBase() = default;
    // Non-copyable / non-movable at the base level; subclasses decide.
    IndexBase(const IndexBase&)            = delete;
    IndexBase& operator=(const IndexBase&) = delete;
};

// ---------------------------------------------------------------------------
// DenseIndex — brute-force search over dense float vectors
// Supported metrics: L2, COSINE
// ---------------------------------------------------------------------------
class DenseIndex : public IndexBase {
public:
    explicit DenseIndex(Metric metric);
    ~DenseIndex() override;

    // Add n vectors of dimension dim.  All subsequent adds must use same dim.
    void add(const float* data, uint32_t n, uint32_t dim);

    // Search for the k nearest neighbours of n_queries query vectors.
    // queries must be float[n_queries * dim()].
    SearchResult search(const float* queries, uint32_t n_queries, uint32_t k) const;

    uint32_t size()   const override { return mat_.n_rows; }
    uint32_t dim()    const override { return mat_.n_cols; }
    Metric   metric() const override { return metric_; }

    void validate() const override;

private:
    Metric              metric_;
    DenseMatrix         mat_;          // non-owning view into storage_
    std::vector<float>  storage_;      // owns the vector data
};

// ---------------------------------------------------------------------------
// SparseIndex — brute-force search over sparse vectors
// Supported metrics: TANIMOTO (binary), COSINE (float sparse — TODO)
// ---------------------------------------------------------------------------
class SparseIndex : public IndexBase {
public:
    SparseIndex(uint32_t dim, Metric metric);
    ~SparseIndex() override = default;

    // Add n float sparse vectors in CSR format.
    // indptr: uint32_t[n+1], indices: uint32_t[indptr[n]], values: float[indptr[n]]
    void add(const uint32_t* indptr, const uint32_t* indices, const float* values,
             uint32_t n_rows);

    // Add n binary vectors as packed uint64_t rows.
    // data: uint64_t[n * ceil(dim/64)], row-major.
    void add_binary(const uint64_t* data, uint32_t n_rows);

    // Add n binary vectors as a flat bool array (one bool per bit).
    // data: bool[n * dim], row-major. Packs internally.
    void add_binary(const bool* data, uint32_t n_rows);

    // Search float sparse queries.
    SearchResult search(const uint32_t* q_indptr, const uint32_t* q_indices,
                        const float* q_values, uint32_t n_queries, uint32_t k) const;

    // Search binary queries (Tanimoto).
    // queries: uint64_t[n_queries * ceil(dim/64)], row-major.
    SearchResult search_binary(const uint64_t* queries, uint32_t n_queries, uint32_t k) const;

    uint32_t size()   const override { return binary_mode_ ? bin_.n_rows : mat_.n_rows; }
    uint32_t dim()    const override { return dim_; }
    Metric   metric() const override { return metric_; }

    void validate() const override;

private:
    uint32_t dim_;
    Metric   metric_;
    bool     binary_mode_ = false;

    // Float sparse storage (CSR)
    SparseMatrix          mat_;
    std::vector<uint32_t> indptr_storage_;
    std::vector<uint32_t> indices_storage_;
    std::vector<float>    values_storage_;

    // Binary storage
    BinaryMatrix          bin_;
    std::vector<uint64_t> binary_storage_;
};

// ---------------------------------------------------------------------------
// Index — unified, user-facing entry point
//
// Accepts any supported (DataType, Metric) combination (validated at
// construction) and dispatches add()/search() to the appropriate backend.
//
// Quick start:
//   sparsity::Index idx;                                       // DenseFloat + L2
//   sparsity::Index idx(Metric::COSINE);                      // DenseFloat + Cosine
//   sparsity::Index idx(DataType::Binary, Metric::TANIMOTO, 2048);
//
//   idx.add(data, n, dim);                   // float* for DenseFloat
//   idx.add(fps,  n_rows);                   // uint64_t* for Binary
//   auto result = idx.search(query, n, k);   // (once implemented)
// ---------------------------------------------------------------------------
class Index {
public:
    // Convenience constructor: defaults to DataType::DenseFloat.
    explicit Index(Metric metric = Metric::L2);

    // Full constructor.
    // dim is required for sparse/binary types (ignored for dense types).
    // Throws std::invalid_argument if the dtype+metric combination is not
    // supported (see is_valid_combination() in types.h).
    Index(DataType dtype, Metric metric, uint32_t dim = 0);

    ~Index() = default;

    Index(const Index&)            = delete;
    Index& operator=(const Index&) = delete;
    Index(Index&&)                 = default;
    Index& operator=(Index&&)      = default;

    // --- Accessors ---
    DataType dtype()  const noexcept { return dtype_; }
    Metric   metric() const noexcept { return metric_; }
    uint32_t dim()    const noexcept;
    uint32_t size()   const noexcept;

    // --- add() overloads — one per DataType ---
    // Each overload validates that the index DataType matches; throws
    // std::invalid_argument if the caller passes data of the wrong type.

    // DenseFloat: data is row-major float[n × dim]
    void add(const float*    data,   uint32_t n, uint32_t dim);
    // DenseInt32: data is row-major int32_t[n × dim]
    void add(const int32_t*  data,   uint32_t n, uint32_t dim);
    // Binary: data is row-major uint64_t[n × ceil(dim_/64)]
    void add(const uint64_t* data,   uint32_t n_rows);
    // Binary: data is row-major bool[n × dim] — packs internally
    void add(const bool*     data,   uint32_t n_rows);
    // SparseFloat: CSR — indptr[n+1], indices[nnz], values[nnz] (float32)
    void add(const uint32_t* indptr, const uint32_t* indices,
             const float*    values, uint32_t n_rows);
    // SparseInt32: CSR — indptr[n+1], indices[nnz], values[nnz] (int32)
    void add(const uint32_t* indptr, const uint32_t* indices,
             const int32_t*  values, uint32_t n_rows);

    // --- search() stubs (not yet implemented, will throw std::logic_error) ---
    SearchResult search(const float*    queries, uint32_t n, uint32_t k) const;
    SearchResult search(const int32_t*  queries, uint32_t n, uint32_t k) const;
    SearchResult search(const uint64_t* queries, uint32_t n, uint32_t k) const;
    SearchResult search(const uint32_t* q_indptr, const uint32_t* q_indices,
                        const float*    q_values,  uint32_t n, uint32_t k) const;
    SearchResult search(const uint32_t* q_indptr, const uint32_t* q_indices,
                        const int32_t*  q_values,  uint32_t n, uint32_t k) const;

private:
    DataType dtype_;
    Metric   metric_;
    std::unique_ptr<DenseIndex>  dense_idx_;   // non-null for DenseFloat / DenseInt32
    std::unique_ptr<SparseIndex> sparse_idx_;  // non-null for SparseFloat / SparseInt32 / Binary
};

} // namespace sparsity

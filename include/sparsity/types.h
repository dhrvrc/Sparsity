#pragma once

namespace sparsity {

constexpr char VERSION[] = "0.1.0";

// Supported distance / similarity metrics.
// Convention: lower return value = more similar, EXCEPT Tanimoto which is a
// similarity (higher = more similar).  Call sites must document which they use.
//
//   Metric::L2       — DenseFloat, DenseInt32, SparseFloat
//   Metric::COSINE   — DenseFloat, DenseInt32, SparseFloat, SparseInt32
//   Metric::TANIMOTO — Binary (packed bit vectors)
enum class Metric {
    L2,       // Euclidean distance
    COSINE,   // 1 - cosine_similarity
    TANIMOTO, // Jaccard / Tanimoto similarity
};

inline const char* metric_name(Metric m) {
    switch (m) {
        case Metric::L2:       return "l2";
        case Metric::COSINE:   return "cosine";
        case Metric::TANIMOTO: return "tanimoto";
    }
    return "unknown";
}

// Supported vector data types.
// Determines memory layout expected by add()/search() and which algorithmic
// optimisations are enabled (see agent_docs/integer_optimisations.md).
enum class DataType {
    DenseFloat,  // float32, row-major array  [n × dim]
    DenseInt32,  // int32,   row-major array  [n × dim]  (integer-optimised path)
    SparseFloat, // CSR format: uint32 indptr/indices + float32 values
    SparseInt32, // CSR format: uint32 indptr/indices + int32  values
    Binary,      // packed bit vectors — uint64_t words, LSB-first [n × ceil(dim/64)]
};

inline const char* data_type_name(DataType dt) {
    switch (dt) {
        case DataType::DenseFloat:  return "dense_float";
        case DataType::DenseInt32:  return "dense_int32";
        case DataType::SparseFloat: return "sparse_float";
        case DataType::SparseInt32: return "sparse_int32";
        case DataType::Binary:      return "binary";
    }
    return "unknown";
}

// Returns true if the (dtype, metric) pair is a supported combination.
//
//   L2       — DenseFloat, DenseInt32, SparseFloat
//   COSINE   — DenseFloat, DenseInt32, SparseFloat, SparseInt32
//   TANIMOTO — Binary
inline bool is_valid_combination(DataType dtype, Metric metric) {
    switch (metric) {
        case Metric::L2:
            return dtype == DataType::DenseFloat ||
                   dtype == DataType::DenseInt32  ||
                   dtype == DataType::SparseFloat;
        case Metric::COSINE:
            return dtype == DataType::DenseFloat  ||
                   dtype == DataType::DenseInt32  ||
                   dtype == DataType::SparseFloat ||
                   dtype == DataType::SparseInt32;
        case Metric::TANIMOTO:
            return dtype == DataType::Binary;
    }
    return false;
}

} // namespace sparsity

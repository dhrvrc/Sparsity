#include "sparsity/cuda_dense.h"
#include "sparsity/cuda_utils.h"

#include <cuda_runtime.h>
#include <cfloat>
#include <cstdint>

namespace sparsity {

// ---------------------------------------------------------------------------
// topk_kernel
//
// Reduces a row-major distance matrix [n_queries × n_db] to the k_actual
// (= min(k, n_db)) nearest neighbours per query.  Results are written in
// ascending distance order to out_dist / out_idx.
//
// Grid  : (n_queries,)  — one block per query
// Block : (CUDA_TOPK_BLOCK,)  — one warp
//
// Algorithm (two phases):
//
//   Phase 1 — parallel scan (all CUDA_TOPK_BLOCK threads):
//     Each thread strides over [threadIdx.x, threadIdx.x + CUDA_TOPK_BLOCK, ...]
//     maintaining a per-thread max-heap of k_actual candidates in shared memory.
//     The max-heap property ensures the largest distance seen so far is at
//     index 0, allowing O(log k) insertion of a new winner.
//
//   Phase 2 — serial merge (thread 0 only):
//     Thread 0 performs k_actual rounds of selection sort over the pool of
//     CUDA_TOPK_BLOCK * k_actual candidates in shared memory, picking the
//     global minimum each round and marking it used.  Cost is
//     O(CUDA_TOPK_BLOCK * k_actual^2) — acceptable for typical k values and
//     the small CUDA_TOPK_BLOCK=32.
//
// Shared memory layout (dynamic, allocated by launcher):
//   float   s_dist[CUDA_TOPK_BLOCK * k_actual]  — per-thread max-heaps
//   int64_t s_idx [CUDA_TOPK_BLOCK * k_actual]  — corresponding db indices
// ---------------------------------------------------------------------------
__global__ void topk_kernel(const float* __restrict__ distances,
                             uint32_t n_db,
                             uint32_t k,
                             float*   __restrict__ out_dist,
                             int64_t* __restrict__ out_idx,
                             uint32_t n_queries)
{
    extern __shared__ char raw_shmem[];
    const uint32_t k_actual = (k <= n_db) ? k : n_db;

    // Shared memory: float portion then int64_t portion.
    float*   s_dist = reinterpret_cast<float*>(raw_shmem);
    int64_t* s_idx  = reinterpret_cast<int64_t*>(s_dist + CUDA_TOPK_BLOCK * k_actual);

    const uint32_t q = blockIdx.x;
    if (q >= n_queries) return;

    const float* row = distances + (size_t)q * n_db;

    // Pointers to this thread's slice of the shared heaps.
    float*   my_d = s_dist + threadIdx.x * k_actual;
    int64_t* my_i = s_idx  + threadIdx.x * k_actual;

    // -----------------------------------------------------------------------
    // Phase 1: each thread builds its local max-heap of k_actual candidates.
    // -----------------------------------------------------------------------
    for (uint32_t i = 0; i < k_actual; ++i) {
        my_d[i] = FLT_MAX;
        my_i[i] = -1;
    }

    for (uint32_t i = threadIdx.x; i < n_db; i += CUDA_TOPK_BLOCK) {
        const float d = row[i];

        // my_d[0] is the largest distance in the max-heap (the worst candidate).
        // If d is better, replace the root and sift down to restore the heap.
        if (d < my_d[0]) {
            my_d[0] = d;
            my_i[0] = static_cast<int64_t>(i);

            // Sift down.
            uint32_t pos = 0;
            for (;;) {
                const uint32_t left    = 2 * pos + 1;
                const uint32_t right   = 2 * pos + 2;
                uint32_t       largest = pos;
                if (left  < k_actual && my_d[left]  > my_d[largest]) largest = left;
                if (right < k_actual && my_d[right] > my_d[largest]) largest = right;
                if (largest == pos) break;

                const float   tf = my_d[pos]; my_d[pos] = my_d[largest]; my_d[largest] = tf;
                const int64_t ti = my_i[pos]; my_i[pos] = my_i[largest]; my_i[largest] = ti;
                pos = largest;
            }
        }
    }

    // All threads must finish phase 1 before thread 0 reads shared memory.
    __syncwarp();

    // -----------------------------------------------------------------------
    // Phase 2: thread 0 merges all CUDA_TOPK_BLOCK per-thread heaps via
    // selection sort, producing the final sorted top-k.
    // -----------------------------------------------------------------------
    if (threadIdx.x == 0) {
        float*   od = out_dist + (size_t)q * k;
        int64_t* oi = out_idx  + (size_t)q * k;

        const uint32_t total = CUDA_TOPK_BLOCK * k_actual;

        for (uint32_t j = 0; j < k_actual; ++j) {
            float   best_d = FLT_MAX;
            int64_t best_i = -1;
            uint32_t best_c = 0;

            // Find the minimum among all remaining (non-consumed) candidates.
            for (uint32_t c = 0; c < total; ++c) {
                if (s_idx[c] >= 0 && s_dist[c] < best_d) {
                    best_d = s_dist[c];
                    best_i = s_idx[c];
                    best_c = c;
                }
            }

            od[j] = (best_i >= 0) ? best_d : FLT_MAX;
            oi[j] = best_i;
            // Mark as consumed so it is not selected again.
            if (best_i >= 0) s_idx[best_c] = -1;
        }

        // Pad remaining slots when k > n_db (k_actual < k).
        for (uint32_t j = k_actual; j < k; ++j) {
            od[j] = FLT_MAX;
            oi[j] = -1;
        }
    }
}

// ---------------------------------------------------------------------------
// Host launcher
// ---------------------------------------------------------------------------
void cuda_topk_batch(const float* d_distances,
                     uint32_t     n_db,
                     uint32_t     k,
                     float*       d_out_dist,
                     int64_t*     d_out_idx,
                     uint32_t     n_queries)
{
    if (n_queries == 0 || n_db == 0) return;

    const uint32_t k_actual = (k <= n_db) ? k : n_db;
    // Shared memory: CUDA_TOPK_BLOCK slots of (float + int64_t) per k_actual entry.
    const size_t shmem = static_cast<size_t>(CUDA_TOPK_BLOCK) * k_actual
                         * (sizeof(float) + sizeof(int64_t));

    topk_kernel<<<n_queries, CUDA_TOPK_BLOCK, shmem>>>(
        d_distances, n_db, k, d_out_dist, d_out_idx, n_queries);
    SPARSITY_CUDA_CHECK(cudaGetLastError());
}

} // namespace sparsity

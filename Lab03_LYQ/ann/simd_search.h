#pragma once
#include <queue>
#include <cstdint>
#include <cstddef>
#include "simd_utils.h"

// Prefetch distance in vectors. Tuning note:
//   Too small → prefetch arrives too late (memory stall).
//   Too large → wastes cache capacity.
// 4 vectors ahead is a reasonable starting point for large vecdim.
static constexpr int PREFETCH_DIST = 4;

// ---------------------------------------------------------------------------
// simd_flat_search — drop-in replacement for flat_search (flat_scan.h).
//
// Same interface and return type; inner product is accelerated by
// simd_inner_product (4-way unrolled FMA via simd8float32).
//
// Additional optimization: software prefetch of the next vector into L1
// cache while the current inner product is being computed, hiding memory
// latency for large base sets.
// ---------------------------------------------------------------------------
inline std::priority_queue<std::pair<float, uint32_t>>
simd_flat_search(float* base, float* query,
                 size_t base_number, size_t vecdim, size_t k)
{
    std::priority_queue<std::pair<float, uint32_t>> q;

    for (size_t i = 0; i < base_number; ++i) {
        // Prefetch the start of a future vector into L1 (read, temporal locality 1).
        // __builtin_prefetch(addr, rw=0, locality=1..3)
        if (i + PREFETCH_DIST < base_number)
            __builtin_prefetch(base + (i + PREFETCH_DIST) * vecdim, 0, 1);

        float ip  = simd_inner_product(base + i * vecdim, query, vecdim);
        float dis = 1.0f - ip;

        if (q.size() < k) {
            q.push({dis, static_cast<uint32_t>(i)});
        } else if (dis < q.top().first) {
            q.push({dis, static_cast<uint32_t>(i)});
            q.pop();
        }
    }
    return q;
}

#pragma once
#include <queue>
#include <cstdint>
#include <cstddef>
#include "simd_utils.h"

// =============================================================================
// Four incremental versions of SIMD flat search.
// Switch which one main.cc calls to measure per-technique contribution.
//
//  v1 — basic SIMD (1 accumulator, no FMA, no prefetch)
//  v2 — v1 + FMA   (fused multiply-add, single accumulator)
//  v3 — v2 + 4-way loop unrolling (4 independent accumulators)
//  v4 — v3 + software prefetch   (current best, alias: simd_flat_search)
// =============================================================================

static constexpr int PREFETCH_DIST = 4;

// ---------------------------------------------------------------------------
// Inner product helpers — one per optimization level
// ---------------------------------------------------------------------------

// v1: basic SIMD, single accumulator, no FMA
inline float ip_v1(const float* a, const float* b, size_t dim) {
    auto acc = simd8float32::zeros();
    size_t i = 0;
    for (; i + 8 <= dim; i += 8)
        acc += simd8float32(a + i) * simd8float32(b + i);
    float result = acc.reduce_add();
    for (; i < dim; ++i) result += a[i] * b[i];
    return result;
}

// v2: SIMD + FMA, single accumulator
inline float ip_v2(const float* a, const float* b, size_t dim) {
    auto acc = simd8float32::zeros();
    size_t i = 0;
    for (; i + 8 <= dim; i += 8)
        acc = simd8float32::fma(acc, simd8float32(a + i), simd8float32(b + i));
    float result = acc.reduce_add();
    for (; i < dim; ++i) result += a[i] * b[i];
    return result;
}

// v3: SIMD + FMA + 3-way loop unrolling (3 independent accumulators)
//
// Why 3-way for dim=96:
//   Each simd8float32 FMA = 2 NEON vfmaq (two Q-register halves).
//   ARM NEON: 2 FMA units, 4-cycle FMA latency.
//   To hide latency, need 2 units × 4 cycles = 8 independent NEON FMAs in-flight.
//   N accumulators → N×2 independent NEON streams:
//     N=2 → 4 streams: 2 active + 2 stall per 4 cycles = 50% (same as N=1, no gain)
//     N=3 → 6 streams: 3 active + 1 stall per 4 cycles = 75%  ← sweet spot
//     N=4 → 8 streams: 4 active + 0 stall per 4 cycles = 100%, but for dim=96
//            only 3 outer iterations → fixed overhead (14 NEON instr) / FMA (24) = 58%
//   N=3 for dim=96: 4 outer iterations (96÷24=4, exact, no cleanup),
//            fixed overhead = 10 NEON instr / 24 FMA = 42% — better tradeoff.
inline float ip_v3(const float* a, const float* b, size_t dim) {
    auto s0 = simd8float32::zeros();
    auto s1 = simd8float32::zeros();
    auto s2 = simd8float32::zeros();

    size_t i = 0;
    for (; i + 24 <= dim; i += 24) {
        s0 = simd8float32::fma(s0, simd8float32(a + i),      simd8float32(b + i));
        s1 = simd8float32::fma(s1, simd8float32(a + i +  8), simd8float32(b + i +  8));
        s2 = simd8float32::fma(s2, simd8float32(a + i + 16), simd8float32(b + i + 16));
    }
    // Handles dims not divisible by 24 (for dim=96: 96÷24=4, no remainder)
    for (; i + 8 <= dim; i += 8)
        s0 = simd8float32::fma(s0, simd8float32(a + i), simd8float32(b + i));

    s0 += s1; s0 += s2;
    float result = s0.reduce_add();
    for (; i < dim; ++i) result += a[i] * b[i];
    return result;
}

// v4: v3 + software prefetch  (= simd_inner_product in simd_utils.h)
//     reuse the already-defined simd_inner_product
// (no redefinition here; see simd_utils.h)

// ---------------------------------------------------------------------------
// Search functions — same heap logic, different inner product kernels
// ---------------------------------------------------------------------------

inline std::priority_queue<std::pair<float, uint32_t>>
simd_flat_search_v1(float* base, float* query,
                    size_t base_number, size_t vecdim, size_t k)
{
    std::priority_queue<std::pair<float, uint32_t>> q;
    for (size_t i = 0; i < base_number; ++i) {
        float dis = 1.0f - ip_v1(base + i * vecdim, query, vecdim);
        if (q.size() < k) {
            q.push({dis, static_cast<uint32_t>(i)});
        } else if (dis < q.top().first) {
            q.push({dis, static_cast<uint32_t>(i)});
            q.pop();
        }
    }
    return q;
}

inline std::priority_queue<std::pair<float, uint32_t>>
simd_flat_search_v2(float* base, float* query,
                    size_t base_number, size_t vecdim, size_t k)
{
    std::priority_queue<std::pair<float, uint32_t>> q;
    for (size_t i = 0; i < base_number; ++i) {
        float dis = 1.0f - ip_v2(base + i * vecdim, query, vecdim);
        if (q.size() < k) {
            q.push({dis, static_cast<uint32_t>(i)});
        } else if (dis < q.top().first) {
            q.push({dis, static_cast<uint32_t>(i)});
            q.pop();
        }
    }
    return q;
}

inline std::priority_queue<std::pair<float, uint32_t>>
simd_flat_search_v3(float* base, float* query,
                    size_t base_number, size_t vecdim, size_t k)
{
    std::priority_queue<std::pair<float, uint32_t>> q;
    for (size_t i = 0; i < base_number; ++i) {
        float dis = 1.0f - ip_v3(base + i * vecdim, query, vecdim);
        if (q.size() < k) {
            q.push({dis, static_cast<uint32_t>(i)});
        } else if (dis < q.top().first) {
            q.push({dis, static_cast<uint32_t>(i)});
            q.pop();
        }
    }
    return q;
}

inline std::priority_queue<std::pair<float, uint32_t>>
simd_flat_search_v4(float* base, float* query,
                    size_t base_number, size_t vecdim, size_t k)
{
    std::priority_queue<std::pair<float, uint32_t>> q;
    for (size_t i = 0; i < base_number; ++i) {
        if (i + PREFETCH_DIST < base_number)
            __builtin_prefetch(base + (i + PREFETCH_DIST) * vecdim, 0, 1);
        float dis = 1.0f - simd_inner_product(base + i * vecdim, query, vecdim);
        if (q.size() < k) {
            q.push({dis, static_cast<uint32_t>(i)});
        } else if (dis < q.top().first) {
            q.push({dis, static_cast<uint32_t>(i)});
            q.pop();
        }
    }
    return q;
}

// Alias: simd_flat_search → best version (v4)
inline std::priority_queue<std::pair<float, uint32_t>>
simd_flat_search(float* base, float* query,
                 size_t base_number, size_t vecdim, size_t k)
{
    return simd_flat_search_v4(base, query, base_number, vecdim, k);
}

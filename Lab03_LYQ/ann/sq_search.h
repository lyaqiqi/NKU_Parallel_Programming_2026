#pragma once
#include <queue>
#include <vector>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include "simd_utils.h"

// ---------------------------------------------------------------------------
// int8 x int8 inner product → int32
//
// Both base codes AND query are int8. Eliminates the int8→float32 conversion
// chain that made the previous int8×float32 version slower than float32 SIMD.
//
// Per 8 elements:
//   int8×float32 (old): vld1_s8 + vmovl_s8 + 2×vmovl_s16 + 2×vcvtq + 2×vld1q + 2×vfmaq = 10 instr
//   int8×int8    (new): vld1_s8 + vld1_s8 + vmull_s8 + vpaddlq_s16 + vaddq_s32          =  5 instr
//
// Overflow check (int8 in [-127,127]):
//   vmull_s8:  product ∈ [-16129, 16129] → fits in int16 (max ±32767) ✓
//   vpaddlq:   pairwise sum ≤ 2×16129 = 32258 → fits in int32 ✓
//   acc total: dim×32258/2 = 96×16129 = 1.5M → fits in int32 (max ~2.1B) ✓
// ---------------------------------------------------------------------------
inline int32_t int8_int8_ip(const int8_t* __restrict__ a,
                             const int8_t* __restrict__ b,
                             size_t dim)
{
#if defined(USE_NEON)
    int32x4_t acc = vdupq_n_s32(0);
    size_t i = 0;
    for (; i + 8 <= dim; i += 8) {
        int8x8_t  va   = vld1_s8(a + i);
        int8x8_t  vb   = vld1_s8(b + i);
        int16x8_t prod = vmull_s8(va, vb);           // int8×int8 → int16
        acc = vaddq_s32(acc, vpaddlq_s16(prod));     // pairwise widen to int32, accumulate
    }
    // horizontal sum: fold int32x4 → scalar
    int32x2_t s2 = vadd_s32(vget_low_s32(acc), vget_high_s32(acc));
    int32_t result = vget_lane_s32(vpadd_s32(s2, s2), 0);
    for (; i < dim; ++i) result += (int32_t)a[i] * b[i];
    return result;

#elif defined(USE_AVX) && defined(__AVX2__)
    // Process 16 int8 at a time via sign-extension → int16 multiply → madd to int32
    __m256i acc = _mm256_setzero_si256();
    size_t i = 0;
    for (; i + 16 <= dim; i += 16) {
        __m128i va8  = _mm_loadu_si128((const __m128i*)(a + i));
        __m128i vb8  = _mm_loadu_si128((const __m128i*)(b + i));
        __m256i va16 = _mm256_cvtepi8_epi16(va8);           // 16 int8 → 16 int16
        __m256i vb16 = _mm256_cvtepi8_epi16(vb8);
        __m256i prod = _mm256_mullo_epi16(va16, vb16);      // int16×int16 (max ±16129, no overflow)
        // madd with 1s: adjacent int16 pairs summed → int32 (max 32258, fits)
        acc = _mm256_add_epi32(acc,
              _mm256_madd_epi16(prod, _mm256_set1_epi16(1)));
    }
    // horizontal sum of int32x8
    __m128i lo   = _mm256_castsi256_si128(acc);
    __m128i hi   = _mm256_extracti128_si256(acc, 1);
    __m128i s128 = _mm_add_epi32(lo, hi);
    s128 = _mm_hadd_epi32(s128, s128);
    s128 = _mm_hadd_epi32(s128, s128);
    int32_t result = _mm_cvtsi128_si32(s128);
    for (; i < dim; ++i) result += (int32_t)a[i] * b[i];
    return result;

#else
    int32_t result = 0;
    for (size_t i = 0; i < dim; ++i) result += (int32_t)a[i] * b[i];
    return result;
#endif
}

// ---------------------------------------------------------------------------
// SQIndex — Scalar Quantization index (per-vector symmetric int8)
//
// Build:
//   scale[j]    = max(|base[j][d]|) / 127
//   codes[j][d] = round(base[j][d] / scale[j])   ∈ [-127, 127]
//
// Search (two-phase):
//   Phase 1 — coarse (int8×int8):
//     Quantize query → q_int8  (once per query, dim multiplications)
//     For each base vector j: ip_j = int8_int8_ip(codes[j], q_int8)
//     approx_IP_j ≈ scale[j] × scale_q × ip_j
//     Keep top-p by approx distance.   ← 5 NEON instr/8elem vs 6 for float32
//
//   Phase 2 — rerank (float32):
//     Exact inner product for top-p candidates via simd_inner_product.
//     Returns top-k.
//
// Tuning:
//   p=19  → recall matches flat SIMD (0.99995) on DEEP100K
//   p<19  → faster but lower recall
// ---------------------------------------------------------------------------

static constexpr int SQ_PREFETCH_DIST = 16;

struct SQIndex {
    std::vector<int8_t> codes;
    std::vector<float>  scales;
    float*  base        = nullptr;
    size_t  base_number = 0;
    size_t  vecdim      = 0;

    void build(float* base_, size_t n, size_t d) {
        base        = base_;
        base_number = n;
        vecdim      = d;
        codes.resize(n * d);
        scales.resize(n);

        for (size_t j = 0; j < n; ++j) {
            const float* vec = base_ + j * d;
            float max_abs = 0.0f;
            for (size_t di = 0; di < d; ++di) {
                float v = vec[di] < 0.0f ? -vec[di] : vec[di];
                if (v > max_abs) max_abs = v;
            }
            float scale = (max_abs > 0.0f) ? max_abs / 127.0f : 1.0f;
            scales[j] = scale;
            int8_t* dst = codes.data() + j * d;
            for (size_t di = 0; di < d; ++di)
                dst[di] = static_cast<int8_t>(std::round(vec[di] / scale));
        }
    }

    std::priority_queue<std::pair<float, uint32_t>>
    search(float* query, size_t k, size_t p) const {
        if (p < k) p = k;

        // ------------------------------------------------------------------
        // Quantize query to int8 — done ONCE per query call, O(dim) cost.
        // scale_q is constant across all base vectors → doesn't affect ranking.
        // ------------------------------------------------------------------
        float max_abs_q = 0.0f;
        for (size_t d = 0; d < vecdim; ++d) {
            float v = query[d] < 0.0f ? -query[d] : query[d];
            if (v > max_abs_q) max_abs_q = v;
        }
        float scale_q = (max_abs_q > 0.0f) ? max_abs_q / 127.0f : 1.0f;

        std::vector<int8_t> q_int8(vecdim);
        for (size_t d = 0; d < vecdim; ++d)
            q_int8[d] = static_cast<int8_t>(std::round(query[d] / scale_q));

        // ------------------------------------------------------------------
        // Phase 1: coarse ranking — int8×int8 inner product for all 100K vectors
        // approx_IP_j = scale[j] × scale_q × int8_int8_ip(codes[j], q_int8)
        // scale_q included for better distance approximation (doesn't change order)
        // ------------------------------------------------------------------
        std::priority_queue<std::pair<float, uint32_t>> coarse;
        const int8_t* c  = codes.data();
        const int8_t* qb = q_int8.data();

        for (size_t j = 0; j < base_number; ++j) {
            if (j + SQ_PREFETCH_DIST < base_number)
                __builtin_prefetch(c + (j + SQ_PREFETCH_DIST) * vecdim, 0, 1);

            int32_t ip_int  = int8_int8_ip(c + j * vecdim, qb, vecdim);
            float approx_dis = 1.0f - scales[j] * scale_q * (float)ip_int;

            if (coarse.size() < p) {
                coarse.push({approx_dis, static_cast<uint32_t>(j)});
            } else if (approx_dis < coarse.top().first) {
                coarse.push({approx_dis, static_cast<uint32_t>(j)});
                coarse.pop();
            }
        }

        // ------------------------------------------------------------------
        // Phase 2: exact rerank — float32 simd_inner_product for top-p only
        // ------------------------------------------------------------------
        std::priority_queue<std::pair<float, uint32_t>> result;
        while (!coarse.empty()) {
            uint32_t idx = coarse.top().second;
            coarse.pop();
            float ip  = simd_inner_product(base + static_cast<size_t>(idx) * vecdim,
                                           query, vecdim);
            float dis = 1.0f - ip;
            if (result.size() < k) {
                result.push({dis, idx});
            } else if (dis < result.top().first) {
                result.push({dis, idx});
                result.pop();
            }
        }
        return result;
    }
};
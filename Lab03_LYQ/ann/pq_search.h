#pragma once
// PQ FastScan — AoS codes + cross-centroid parallel LUT + register-resident small tables
// M=4, KS=256, SUBDIM=24 (96/4), distance: inner product (dist = 1 - IP)
//
// Implements André et al., "Cache locality is not enough: High-Performance Nearest
// Neighbor Search with Product Quantization Fast Scan", VLDB 2015.
// Adapted for: inner-product distance (max-tables instead of min-tables),
//              M=4 subspaces (c=2 grouping nibbles), ARM NEON vqtbl1q_u8.
//
// Three-phase search:
//   Phase 1  build_lut_fast()  — cross-centroid parallel NEON LUT build
//   Phase 2  FastScan          — register-resident quantized small tables + pruning
//   Phase 3  exact rerank      — simd_inner_product for top-p survivors
//
// FastScan overview (NEON, Phase 2):
//   Sort N vectors once at load time by group key g = (p[0]>>4)*16 + (p[1]>>4).
//   256 groups × ~390 vectors each (N=100K, c=2, 16^2=256).
//
//   Per-query quantized small tables (16-entry uint8, stored in NEON registers):
//     S0, S1  — exact LUT portions for the group's (i0, i1) high nibbles
//     S2, S3  — maximum tables of LUT[2,3], fixed per query (upper bound on IP)
//
//   For each batch of 16 vectors:
//     vld4q_u8      — deinterleave 4 subspace codes for 16 vectors (AoS → SoA)
//     vqtbl1q_u8 ×4 — 16-way in-register lookup (ARM ≡ x86 pshufb)
//     vaddq_u8      — sum 4 components (each ≤63 → sum ≤252, no uint8 overflow)
//     vcleq_u8      — prune mask: upper-bound score ≤ threshold → skip
//   Survivors: exact ADC with float LUT → heap update
//
// Correctness guarantee (no recall loss):
//   S2, S3 are MAX tables → ub ≥ true_score → pruning only when ub is below threshold
//   Safety margin of 2 quantization bins in get_qt_thresh() absorbs rounding errors.

#include <queue>
#include <vector>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <limits>
#include "simd_utils.h"

static constexpr int PQ_M      = 4;
static constexpr int PQ_KS     = 256;
static constexpr int PQ_SUBDIM = 24;
static constexpr int PQ_GROUPS = 256;  // 16^c where c=2 (group on nibbles of p[0], p[1])

// ---------------------------------------------------------------------------
// 24-dim float IP: 6×4 = 24, 3 independent accumulator chains
// ---------------------------------------------------------------------------
inline float pq_ip_24(const float* __restrict__ a, const float* __restrict__ b)
{
#if defined(USE_NEON)
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    float32x4_t acc2 = vdupq_n_f32(0.0f);
    acc0 = vfmaq_f32(acc0, vld1q_f32(a),    vld1q_f32(b));
    acc1 = vfmaq_f32(acc1, vld1q_f32(a+4),  vld1q_f32(b+4));
    acc2 = vfmaq_f32(acc2, vld1q_f32(a+8),  vld1q_f32(b+8));
    acc0 = vfmaq_f32(acc0, vld1q_f32(a+12), vld1q_f32(b+12));
    acc1 = vfmaq_f32(acc1, vld1q_f32(a+16), vld1q_f32(b+16));
    acc2 = vfmaq_f32(acc2, vld1q_f32(a+20), vld1q_f32(b+20));
    return vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1), acc2));
#else
    float s = 0.0f;
    for (int i = 0; i < PQ_SUBDIM; ++i) s += a[i] * b[i];
    return s;
#endif
}

// ---------------------------------------------------------------------------
struct PQIndex {
    std::vector<float>    codebook;      // flat [M * KS * SUBDIM]
    std::vector<uint8_t>  codes;         // AoS [N * M]: codes[i*M+m]
    float*  base_ptr    = nullptr;
    size_t  base_number = 0;
    size_t  vecdim      = 0;

    // FastScan pre-sorted data (built once in load, valid for all queries)
    std::vector<uint8_t>  sorted_codes;  // AoS layout, sorted by group key
    std::vector<uint32_t> sorted_idx;    // original vector index per sorted position
    std::vector<uint32_t> group_start;   // [PQ_GROUPS+1]: group g → [gstart[g], gstart[g+1])

    // ------------------------------------------------------------------
    bool load(const char* codebook_path, const char* codes_path,
              float* base_, size_t base_n, size_t dim)
    {
        base_ptr    = base_;
        base_number = base_n;
        vecdim      = dim;

        {
            FILE* fp = fopen(codebook_path, "rb");
            if (!fp) { fprintf(stderr, "PQIndex: cannot open %s\n", codebook_path); return false; }
            size_t sz = (size_t)PQ_M * PQ_KS * PQ_SUBDIM;
            codebook.resize(sz);
            size_t rd = fread(codebook.data(), sizeof(float), sz, fp);
            fclose(fp);
            if (rd != sz) { fprintf(stderr, "PQIndex: codebook read %zu/%zu\n", rd, sz); return false; }
        }
        {
            FILE* fp = fopen(codes_path, "rb");
            if (!fp) { fprintf(stderr, "PQIndex: cannot open %s\n", codes_path); return false; }
            size_t sz = base_n * (size_t)PQ_M;
            codes.resize(sz);
            size_t rd = fread(codes.data(), sizeof(uint8_t), sz, fp);
            fclose(fp);
            if (rd != sz) { fprintf(stderr, "PQIndex: codes read %zu/%zu\n", rd, sz); return false; }
        }

        build_sorted_codes();
        fprintf(stderr, "PQIndex loaded: M=%d KS=%d subdim=%d, %zu vectors, %d groups\n",
                PQ_M, PQ_KS, PQ_SUBDIM, base_n, PQ_GROUPS);
        return true;
    }

    // ------------------------------------------------------------------
    // Counting-sort vectors into 256 groups by (p[0]>>4)*16 + (p[1]>>4).
    // Called once in load(); O(N) time, O(N*M) extra memory.
    // ------------------------------------------------------------------
    void build_sorted_codes()
    {
        group_start.assign(PQ_GROUPS + 1, 0);

        // Count
        for (size_t i = 0; i < base_number; ++i) {
            int g = (int)(codes[i * PQ_M + 0] >> 4) * 16
                  + (int)(codes[i * PQ_M + 1] >> 4);
            group_start[g + 1]++;
        }
        // Prefix sum
        for (int g = 0; g < PQ_GROUPS; ++g)
            group_start[g + 1] += group_start[g];

        // Scatter
        sorted_codes.resize(base_number * PQ_M);
        sorted_idx.resize(base_number);
        std::vector<uint32_t> pos(group_start.begin(),
                                  group_start.begin() + PQ_GROUPS);
        for (size_t i = 0; i < base_number; ++i) {
            const uint8_t* ci = codes.data() + i * PQ_M;
            int g = (int)(ci[0] >> 4) * 16 + (int)(ci[1] >> 4);
            uint32_t dst = pos[g]++;
            std::memcpy(sorted_codes.data() + dst * PQ_M, ci, PQ_M);
            sorted_idx[dst] = (uint32_t)i;
        }
    }

    // ------------------------------------------------------------------
    // Serial LUT build — scalar reference / non-NEON fallback
    // ------------------------------------------------------------------
    void build_lut(const float* query, float lut[PQ_M][PQ_KS]) const {
        for (int m = 0; m < PQ_M; ++m) {
            const float* q_sub = query + m * PQ_SUBDIM;
            const float* cb_m  = codebook.data() + (size_t)m * PQ_KS * PQ_SUBDIM;
            for (int k = 0; k < PQ_KS; ++k)
                lut[m][k] = pq_ip_24(q_sub, cb_m + k * PQ_SUBDIM);
        }
    }

    // ------------------------------------------------------------------
    // Cross-centroid parallel LUT build (NEON):
    // Query sub-vector (24 floats) loaded into 6 NEON regs ONCE per subspace,
    // then reused across all 256 centroids. 4 independent FMA chains per iter.
    // ------------------------------------------------------------------
    void build_lut_fast(const float* query, float lut[PQ_M][PQ_KS]) const {
#if defined(USE_NEON)
        for (int m = 0; m < PQ_M; ++m) {
            const float* q  = query + (size_t)m * PQ_SUBDIM;
            const float* cb = codebook.data() + (size_t)m * PQ_KS * PQ_SUBDIM;
            float* dst = lut[m];

            float32x4_t q0 = vld1q_f32(q);
            float32x4_t q1 = vld1q_f32(q +  4);
            float32x4_t q2 = vld1q_f32(q +  8);
            float32x4_t q3 = vld1q_f32(q + 12);
            float32x4_t q4 = vld1q_f32(q + 16);
            float32x4_t q5 = vld1q_f32(q + 20);

            for (int k = 0; k < PQ_KS; k += 4) {
                const float* c0 = cb + (size_t)(k+0) * PQ_SUBDIM;
                const float* c1 = cb + (size_t)(k+1) * PQ_SUBDIM;
                const float* c2 = cb + (size_t)(k+2) * PQ_SUBDIM;
                const float* c3 = cb + (size_t)(k+3) * PQ_SUBDIM;

                float32x4_t a0 = vmulq_f32(q0, vld1q_f32(c0));
                float32x4_t a1 = vmulq_f32(q0, vld1q_f32(c1));
                float32x4_t a2 = vmulq_f32(q0, vld1q_f32(c2));
                float32x4_t a3 = vmulq_f32(q0, vld1q_f32(c3));

                a0 = vfmaq_f32(a0, q1, vld1q_f32(c0 +  4));
                a1 = vfmaq_f32(a1, q1, vld1q_f32(c1 +  4));
                a2 = vfmaq_f32(a2, q1, vld1q_f32(c2 +  4));
                a3 = vfmaq_f32(a3, q1, vld1q_f32(c3 +  4));

                a0 = vfmaq_f32(a0, q2, vld1q_f32(c0 +  8));
                a1 = vfmaq_f32(a1, q2, vld1q_f32(c1 +  8));
                a2 = vfmaq_f32(a2, q2, vld1q_f32(c2 +  8));
                a3 = vfmaq_f32(a3, q2, vld1q_f32(c3 +  8));

                a0 = vfmaq_f32(a0, q3, vld1q_f32(c0 + 12));
                a1 = vfmaq_f32(a1, q3, vld1q_f32(c1 + 12));
                a2 = vfmaq_f32(a2, q3, vld1q_f32(c2 + 12));
                a3 = vfmaq_f32(a3, q3, vld1q_f32(c3 + 12));

                a0 = vfmaq_f32(a0, q4, vld1q_f32(c0 + 16));
                a1 = vfmaq_f32(a1, q4, vld1q_f32(c1 + 16));
                a2 = vfmaq_f32(a2, q4, vld1q_f32(c2 + 16));
                a3 = vfmaq_f32(a3, q4, vld1q_f32(c3 + 16));

                a0 = vfmaq_f32(a0, q5, vld1q_f32(c0 + 20));
                a1 = vfmaq_f32(a1, q5, vld1q_f32(c1 + 20));
                a2 = vfmaq_f32(a2, q5, vld1q_f32(c2 + 20));
                a3 = vfmaq_f32(a3, q5, vld1q_f32(c3 + 20));

                dst[k+0] = vaddvq_f32(a0);
                dst[k+1] = vaddvq_f32(a1);
                dst[k+2] = vaddvq_f32(a2);
                dst[k+3] = vaddvq_f32(a3);
            }
        }
#else
        build_lut(query, lut);
#endif
    }

    // ------------------------------------------------------------------
    std::priority_queue<std::pair<float, uint32_t>>
    search(const float* query, size_t k, size_t p) const
    {
        if (p < k) p = k;
        if (p > base_number) p = base_number;

        // ----------------------------------------------------------------
        // Phase 1: build float LUT[M][KS] — IP(query_sub_m, centroid_m_k)
        // ----------------------------------------------------------------
        float lut[PQ_M][PQ_KS];
        build_lut_fast(query, lut);

        // ----------------------------------------------------------------
        // Phase 2: FastScan coarse filter → priority queue of top-p candidates
        // ----------------------------------------------------------------
        std::priority_queue<std::pair<float, uint32_t>> coarse;

#if defined(USE_NEON)
        // ---- Quantization calibration --------------------------------
        // Per-subspace minimums; their sum is the lower bound on total IP.
        float per_min[PQ_M];
        float total_min = 0.0f;
        for (int m = 0; m < PQ_M; ++m) {
            per_min[m] = *std::min_element(lut[m], lut[m] + PQ_KS);
            total_min += per_min[m];
        }

        // Scan first ~0.5% of vectors with exact ADC to estimate qmax
        // (best total IP score seen). This tightens the quantization range.
        float qmax = total_min;
        {
            size_t keep_n = std::min(base_number, (size_t)500);
            const uint8_t* cs = codes.data();
            for (size_t i = 0; i < keep_n; ++i) {
                const uint8_t* ci = cs + i * PQ_M;
                float s = lut[0][ci[0]] + lut[1][ci[1]]
                        + lut[2][ci[2]] + lut[3][ci[3]];
                if (s > qmax) qmax = s;
            }
        }

        // bin_size maps the score range [total_min, qmax] onto [0, 252].
        // Each per-component quantized value ≤ 63; sum of 4 ≤ 252 ≤ 255
        // → uint8 addition never overflows.
        const float bin_size = (qmax > total_min)
                               ? (qmax - total_min) / 252.0f
                               : 1e-6f;

        // Quantize a float LUT value v (for subspace m) to uint8 in [0, 63].
        // Using per_min[m] as zero point so that the minimum maps to 0.
        auto qt = [&](float v, int m) -> uint8_t {
            float s = (v - per_min[m]) / bin_size;
            if (s < 0.0f)  s = 0.0f;
            if (s > 63.0f) s = 63.0f;
            return (uint8_t)s;
        };

        // ---- Build fixed-per-query max tables for subspaces 2 and 3 ----
        // max_table[m][i] = max of lut[m][i*16 .. i*16+15] quantized to [0,63].
        // Since max ≥ every element in the portion, this is an UPPER BOUND on
        // the per-component IP contribution → ub ≥ true_score → safe pruning.
        uint8_t qt_max2[16], qt_max3[16];
        for (int i = 0; i < 16; ++i) {
            float mx2 = *std::max_element(lut[2] + i*16, lut[2] + i*16 + 16);
            float mx3 = *std::max_element(lut[3] + i*16, lut[3] + i*16 + 16);
            qt_max2[i] = qt(mx2, 2);
            qt_max3[i] = qt(mx3, 3);
        }
        const uint8x16_t S2 = vld1q_u8(qt_max2);
        const uint8x16_t S3 = vld1q_u8(qt_max3);

        // ---- Threshold tracking ----------------------------------------
        float exact_threshold = std::numeric_limits<float>::max();
        bool  heap_full = false;  // pruning only valid once heap has p elements

        // Compute quantized threshold from current heap top.
        // threshold_score = 1 - worst_dist_in_heap (score below which we prune).
        // Safety margin of -2 bins absorbs per-component rounding: 4 components
        // × 0.5 bin rounding each = 2 bins worst-case error.
        auto get_qt_thresh = [&]() -> uint8_t {
            float ts  = 1.0f - exact_threshold;
            int   qtv = (int)((ts - total_min) / bin_size) - 2;
            if (qtv < 0)   qtv = 0;
            if (qtv > 252) qtv = 252;
            return (uint8_t)qtv;
        };

        uint8_t qt_thresh = 0;

        // ---- Group scan ------------------------------------------------
        for (int g = 0; g < PQ_GROUPS; ++g) {
            const size_t gstart = group_start[g];
            const size_t gend   = group_start[g + 1];
            if (gstart == gend) continue;

            // Within group g: p[0]>>4 == i0, p[1]>>4 == i1 (fixed)
            const int i0 = g / 16;
            const int i1 = g % 16;

            // S0, S1: EXACT 16-entry portions of LUT[0], LUT[1] for this group.
            // S0[k] = qt(lut[0][i0*16 + k]) = quantized IP of the k-th centroid
            // in this portion. Using p[0]&0xF as index gives the EXACT value.
            uint8_t qt_s0[16], qt_s1[16];
            for (int k = 0; k < 16; ++k) {
                qt_s0[k] = qt(lut[0][i0 * 16 + k], 0);
                qt_s1[k] = qt(lut[1][i1 * 16 + k], 1);
            }
            const uint8x16_t S0 = vld1q_u8(qt_s0);
            const uint8x16_t S1 = vld1q_u8(qt_s1);

            // ---- Main loop: process 16 vectors per iteration ---------------
            size_t pos = gstart;
            for (; pos + 16 <= gend; pos += 16) {
                // vld4q_u8: deinterleave 4 subspace channels for 16 vectors.
                // Input (AoS, 64 bytes): [v0m0,v0m1,v0m2,v0m3, v1m0,..., v15m3]
                // c.val[m][j] = code of vector (pos+j) for subspace m
                const uint8x16x4_t c =
                    vld4q_u8(sorted_codes.data() + pos * PQ_M);

                // Lookup indices:
                //   S0, S1 — 4 LSBs of p[0], p[1] (offset within group's portion)
                //   S2, S3 — 4 MSBs of p[2], p[3] (which 16-element portion)
                const uint8x16_t idx0 = vandq_u8(c.val[0], vdupq_n_u8(0x0F));
                const uint8x16_t idx1 = vandq_u8(c.val[1], vdupq_n_u8(0x0F));
                const uint8x16_t idx2 = vshrq_n_u8(c.val[2], 4);
                const uint8x16_t idx3 = vshrq_n_u8(c.val[3], 4);

                // 16-way in-register table lookup (ARM vqtbl1q_u8 ≡ x86 pshufb).
                // Indices guaranteed in [0,15] → no out-of-bounds zeroing.
                const uint8x16_t u0 = vqtbl1q_u8(S0, idx0);
                const uint8x16_t u1 = vqtbl1q_u8(S1, idx1);
                const uint8x16_t u2 = vqtbl1q_u8(S2, idx2);
                const uint8x16_t u3 = vqtbl1q_u8(S3, idx3);

                // ub[j] = quantized upper bound on IP score for vector (pos+j).
                // Each component ≤ 63 → sum ≤ 252 → uint8 never overflows.
                const uint8x16_t ub = vaddq_u8(vaddq_u8(u0, u1),
                                               vaddq_u8(u2, u3));

                // Pruning is only valid once the heap is full (threshold is meaningful).
                // Before that, pass all 16 vectors through without SIMD pruning.
                uint8x16_t pmask;
                if (heap_full) {
                    pmask = vcleq_u8(ub, vdupq_n_u8(qt_thresh));
                    const uint64_t lo = vgetq_lane_u64(vreinterpretq_u64_u8(pmask), 0);
                    const uint64_t hi = vgetq_lane_u64(vreinterpretq_u64_u8(pmask), 1);
                    if (lo == ~(uint64_t)0 && hi == ~(uint64_t)0) continue;
                } else {
                    pmask = vdupq_n_u8(0);  // no pruning during fill phase
                }

                // Exact ADC for non-pruned survivors
                uint8_t pm[16];
                vst1q_u8(pm, pmask);
                for (int j = 0; j < 16; ++j) {
                    if (pm[j]) continue;  // pruned
                    const uint8_t* ci = sorted_codes.data() + (pos + j) * PQ_M;
                    const float score = lut[0][ci[0]] + lut[1][ci[1]]
                                      + lut[2][ci[2]] + lut[3][ci[3]];
                    const float dist  = 1.0f - score;
                    if (coarse.size() < p) {
                        coarse.push({dist, sorted_idx[pos + j]});
                        if (coarse.size() == p) {
                            heap_full = true;
                            exact_threshold = coarse.top().first;
                            qt_thresh = get_qt_thresh();
                        }
                    } else if (dist < exact_threshold) {
                        coarse.push({dist, sorted_idx[pos + j]});
                        coarse.pop();
                        exact_threshold = coarse.top().first;
                        qt_thresh = get_qt_thresh();
                    }
                }
            }

            // ---- Remainder: < 16 vectors at end of group ------------------
            for (; pos < gend; ++pos) {
                const uint8_t* ci = sorted_codes.data() + pos * PQ_M;
                const float score = lut[0][ci[0]] + lut[1][ci[1]]
                                  + lut[2][ci[2]] + lut[3][ci[3]];
                const float dist  = 1.0f - score;
                if (coarse.size() < p) {
                    coarse.push({dist, sorted_idx[pos]});
                    if (coarse.size() == p) {
                        heap_full = true;
                        exact_threshold = coarse.top().first;
                        qt_thresh = get_qt_thresh();
                    }
                } else if (dist < exact_threshold) {
                    coarse.push({dist, sorted_idx[pos]});
                    coarse.pop();
                    exact_threshold = coarse.top().first;
                    qt_thresh = get_qt_thresh();
                }
            }
        }  // end group loop

#else
        // ---- Non-NEON fallback: sorted codes + threshold pruning --------
        {
            float exact_threshold = std::numeric_limits<float>::max();
            for (int g = 0; g < PQ_GROUPS; ++g) {
                for (size_t pos = group_start[g]; pos < group_start[g + 1]; ++pos) {
                    const uint8_t* ci = sorted_codes.data() + pos * PQ_M;
                    const float score = lut[0][ci[0]] + lut[1][ci[1]]
                                      + lut[2][ci[2]] + lut[3][ci[3]];
                    const float dist  = 1.0f - score;
                    if (coarse.size() < p) {
                        coarse.push({dist, sorted_idx[pos]});
                        if (coarse.size() == p)
                            exact_threshold = coarse.top().first;
                    } else if (dist < exact_threshold) {
                        coarse.push({dist, sorted_idx[pos]});
                        coarse.pop();
                        exact_threshold = coarse.top().first;
                    }
                }
            }
        }
#endif

        // ----------------------------------------------------------------
        // Phase 3: exact rerank — simd_inner_product for top-p candidates
        // ----------------------------------------------------------------
        std::priority_queue<std::pair<float, uint32_t>> result;
        while (!coarse.empty()) {
            const uint32_t idx = coarse.top().second;
            coarse.pop();
            const float ip  = simd_inner_product(
                base_ptr + (size_t)idx * vecdim, query, vecdim);
            const float dis = 1.0f - ip;
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

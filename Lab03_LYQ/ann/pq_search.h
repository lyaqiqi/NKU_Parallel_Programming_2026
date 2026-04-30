#pragma once
// Product Quantization search — M=4, Ks=256, subdim=24 (96/4)
// Distance metric: inner product (IP), consistent with DEEP100K ground truth
//
// LUT[m][k] = IP(query_sub_m, centroid_m_k)
// Approx distance: δ(x,q) = 1 - Σ_m LUT[m][code_m]  (matches official formula)
// Reranking: exact IP via simd_inner_product

#include <queue>
#include <vector>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include "simd_utils.h"

static constexpr int PQ_M      = 4;
static constexpr int PQ_KS     = 256;
static constexpr int PQ_SUBDIM = 24;   // vecdim / PQ_M

// ------------------------------------------------------------------
// SIMD inner product for two 24-dim float vectors
// 24 = 6 × 4  → 6 vld1q_f32 loads, 3 independent accumulators (breaks dep chain)
// ------------------------------------------------------------------
inline float pq_ip_24(const float* __restrict__ a,
                      const float* __restrict__ b)
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

// ------------------------------------------------------------------
// PQIndex — load codebook + codes from files, two-stage search
// ------------------------------------------------------------------
struct PQIndex {
    // codebook[m][k][d] stored flat: index = m*KS*SUBDIM + k*SUBDIM + d
    std::vector<float>   codebook;
    std::vector<uint8_t> codes;
    float*  base_ptr    = nullptr;
    size_t  base_number = 0;
    size_t  vecdim      = 0;

    // Load codebook and codes from binary files.
    // Call before search(); base_ and base_n are needed for reranking.
    bool load(const char* codebook_path, const char* codes_path,
              float* base_, size_t base_n, size_t dim)
    {
        base_ptr    = base_;
        base_number = base_n;
        vecdim      = dim;

        // codebook: M * KS * SUBDIM float32
        {
            FILE* fp = fopen(codebook_path, "rb");
            if (!fp) { fprintf(stderr, "PQIndex: cannot open %s\n", codebook_path); return false; }
            size_t sz = (size_t)PQ_M * PQ_KS * PQ_SUBDIM;
            codebook.resize(sz);
            size_t rd = fread(codebook.data(), sizeof(float), sz, fp);
            fclose(fp);
            if (rd != sz) { fprintf(stderr, "PQIndex: codebook read %zu/%zu\n", rd, sz); return false; }
        }

        // codes: base_n * M uint8
        {
            FILE* fp = fopen(codes_path, "rb");
            if (!fp) { fprintf(stderr, "PQIndex: cannot open %s\n", codes_path); return false; }
            size_t sz = base_n * PQ_M;
            codes.resize(sz);
            size_t rd = fread(codes.data(), sizeof(uint8_t), sz, fp);
            fclose(fp);
            if (rd != sz) { fprintf(stderr, "PQIndex: codes read %zu/%zu\n", rd, sz); return false; }
        }

        fprintf(stderr, "PQIndex loaded: %d subspaces, %d centroids, %zu vectors\n",
                PQ_M, PQ_KS, base_n);
        return true;
    }

    // Build LUT[m][k] = IP( query_sub_m, centroid_m_k )
    // 4 × 256 = 1024 inner products of 24-dim vectors; result is ~4 KB → fits in L1.
    // Approximation: IP(q, x) ≈ Σ_m LUT[m][code_m(x)]
    void build_lut(const float* query, float lut[PQ_M][PQ_KS]) const {
        for (int m = 0; m < PQ_M; ++m) {
            const float* q_sub = query + m * PQ_SUBDIM;
            const float* cb_m  = codebook.data() + m * PQ_KS * PQ_SUBDIM;
            for (int k = 0; k < PQ_KS; ++k) {
                lut[m][k] = pq_ip_24(q_sub, cb_m + k * PQ_SUBDIM);
            }
        }
    }

    std::priority_queue<std::pair<float, uint32_t>>
    search(const float* query, size_t k, size_t p) const {
        if (p < k) p = k;

        // Phase 1: build LUT (4 × 256 floats, ~4 KB)
        float lut[PQ_M][PQ_KS];
        build_lut(query, lut);

        // Phase 2: coarse scan — ADC lookup over all N PQ codes
        // δ_approx(x,q) = 1 - (LUT[0][c0] + LUT[1][c1] + LUT[2][c2] + LUT[3][c3])
        // Maintain max-heap of size p → keeps p smallest δ values
        std::priority_queue<std::pair<float, uint32_t>> coarse;
        const uint8_t* c = codes.data();

        for (size_t i = 0; i < base_number; ++i) {
            const uint8_t* ci = c + i * PQ_M;
            float dist = 1.0f - (lut[0][ci[0]] + lut[1][ci[1]]
                               + lut[2][ci[2]] + lut[3][ci[3]]);

            if (coarse.size() < p) {
                coarse.push({dist, static_cast<uint32_t>(i)});
            } else if (dist < coarse.top().first) {
                coarse.push({dist, static_cast<uint32_t>(i)});
                coarse.pop();
            }
        }

        // Phase 3: exact reranking — IP distance for top-p candidates
        std::priority_queue<std::pair<float, uint32_t>> result;
        while (!coarse.empty()) {
            uint32_t idx = coarse.top().second;
            coarse.pop();
            float ip  = simd_inner_product(
                base_ptr + static_cast<size_t>(idx) * vecdim, query, vecdim);
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
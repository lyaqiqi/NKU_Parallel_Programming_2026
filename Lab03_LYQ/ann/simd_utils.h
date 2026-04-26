#pragma once
#include <cstddef>

// ---------------------------------------------------------------------------
// Platform detection
// ---------------------------------------------------------------------------
#if defined(__ARM_NEON) || defined(__aarch64__)
#   define USE_NEON
#   include <arm_neon.h>
#elif defined(__AVX__)
#   define USE_AVX
#   include <immintrin.h>
#   ifdef __FMA__
#       define USE_FMA
#   endif
#elif defined(__SSE__)
#   define USE_SSE
#   include <xmmintrin.h>
#endif

// ---------------------------------------------------------------------------
// simd8float32 — 256-bit wide (8 x float32) SIMD wrapper
//
// Three backends, selected at compile time:
//   1. ARM NEON  (__ARM_NEON / __aarch64__)  — primary (cluster)
//   2. x86 AVX   (__AVX__)                   — local x86 testing
//   3. Scalar fallback                        — any other platform
//
// Public interface (identical across all backends):
//   simd8float32()                              default-construct
//   explicit simd8float32(const float* x)       load 8 floats (unaligned ok)
//   explicit simd8float32(float val)            broadcast scalar to all 8 lanes
//   static simd8float32 zeros()                 all-zero
//   simd8float32 operator+(rhs)                 element-wise add
//   simd8float32 operator*(rhs)                 element-wise multiply
//   simd8float32& operator+=(rhs)               in-place add
//   simd8float32& operator*=(rhs)               in-place multiply
//   void store(float* dst)                      write 8 floats (unaligned ok)
//   void storeu(float* dst)                     alias for store
//   float reduce_add()                          horizontal sum of 8 lanes
//   static simd8float32 fma(acc, a, b)          acc + a*b  (fused multiply-add)
// ---------------------------------------------------------------------------

// ===========================================================================
// Backend 1: ARM NEON
// ===========================================================================
#if defined(USE_NEON)

struct simd8float32 {
    float32x4x2_t data;

    simd8float32() = default;

    // load 8 floats from memory
    explicit simd8float32(const float* x)
        : data{vld1q_f32(x), vld1q_f32(x + 4)} {}

    // broadcast scalar to all 8 lanes
    explicit simd8float32(float val) {
        data.val[0] = vdupq_n_f32(val);
        data.val[1] = vdupq_n_f32(val);
    }

    static simd8float32 zeros() {
        simd8float32 v;
        v.data.val[0] = vdupq_n_f32(0.0f);
        v.data.val[1] = vdupq_n_f32(0.0f);
        return v;
    }

    simd8float32 operator+(const simd8float32& o) const {
        simd8float32 r;
        r.data.val[0] = vaddq_f32(data.val[0], o.data.val[0]);
        r.data.val[1] = vaddq_f32(data.val[1], o.data.val[1]);
        return r;
    }

    simd8float32 operator*(const simd8float32& o) const {
        simd8float32 r;
        r.data.val[0] = vmulq_f32(data.val[0], o.data.val[0]);
        r.data.val[1] = vmulq_f32(data.val[1], o.data.val[1]);
        return r;
    }

    simd8float32& operator+=(const simd8float32& o) {
        data.val[0] = vaddq_f32(data.val[0], o.data.val[0]);
        data.val[1] = vaddq_f32(data.val[1], o.data.val[1]);
        return *this;
    }

    simd8float32& operator*=(const simd8float32& o) {
        data.val[0] = vmulq_f32(data.val[0], o.data.val[0]);
        data.val[1] = vmulq_f32(data.val[1], o.data.val[1]);
        return *this;
    }

    void store(float* dst) const {
        vst1q_f32(dst,     data.val[0]);
        vst1q_f32(dst + 4, data.val[1]);
    }
    void storeu(float* dst) const { store(dst); }

    // vaddvq_f32: AArch64 horizontal sum — sums 4 lanes to scalar in one instr.
    float reduce_add() const {
        float32x4_t s = vaddq_f32(data.val[0], data.val[1]);
        return vaddvq_f32(s);
    }

    // vfmaq_f32(acc, a, b) = acc + a*b  — single-cycle FMA on Cortex-A series.
    // Avoids the separate VMUL+VADD pair and breaks the mul→add dependency.
    static simd8float32 fma(simd8float32 acc, const simd8float32& a, const simd8float32& b) {
        acc.data.val[0] = vfmaq_f32(acc.data.val[0], a.data.val[0], b.data.val[0]);
        acc.data.val[1] = vfmaq_f32(acc.data.val[1], a.data.val[1], b.data.val[1]);
        return acc;
    }
};

// ===========================================================================
// Backend 2: x86 AVX  (+FMA if available)
// ===========================================================================
#elif defined(USE_AVX)

struct simd8float32 {
    __m256 data;

    simd8float32() = default;

    explicit simd8float32(const float* x)
        : data(_mm256_loadu_ps(x)) {}

    explicit simd8float32(float val)
        : data(_mm256_set1_ps(val)) {}

    static simd8float32 zeros() {
        simd8float32 v;
        v.data = _mm256_setzero_ps();
        return v;
    }

    simd8float32 operator+(const simd8float32& o) const {
        simd8float32 r;
        r.data = _mm256_add_ps(data, o.data);
        return r;
    }

    simd8float32 operator*(const simd8float32& o) const {
        simd8float32 r;
        r.data = _mm256_mul_ps(data, o.data);
        return r;
    }

    simd8float32& operator+=(const simd8float32& o) {
        data = _mm256_add_ps(data, o.data);
        return *this;
    }

    simd8float32& operator*=(const simd8float32& o) {
        data = _mm256_mul_ps(data, o.data);
        return *this;
    }

    void store(float* dst) const { _mm256_storeu_ps(dst, data); }
    void storeu(float* dst) const { store(dst); }

    // Fold 256→128 bit via extract, then two hadd to reach scalar.
    float reduce_add() const {
        __m128 lo = _mm256_castps256_ps128(data);
        __m128 hi = _mm256_extractf128_ps(data, 1);
        __m128 s  = _mm_add_ps(lo, hi);
        s = _mm_hadd_ps(s, s);
        s = _mm_hadd_ps(s, s);
        return _mm_cvtss_f32(s);
    }

    // Use _mm256_fmadd_ps when the compiler has FMA support (-mfma),
    // otherwise fall back to separate mul+add.
    static simd8float32 fma(simd8float32 acc, const simd8float32& a, const simd8float32& b) {
        simd8float32 r;
#if defined(USE_FMA)
        r.data = _mm256_fmadd_ps(a.data, b.data, acc.data);
#else
        r.data = _mm256_add_ps(acc.data, _mm256_mul_ps(a.data, b.data));
#endif
        return r;
    }
};

// ===========================================================================
// Backend 3: Scalar fallback
// ===========================================================================
#else

struct simd8float32 {
    float data[8];

    simd8float32() = default;

    explicit simd8float32(const float* x) {
        for (int i = 0; i < 8; ++i) data[i] = x[i];
    }

    explicit simd8float32(float val) {
        for (int i = 0; i < 8; ++i) data[i] = val;
    }

    static simd8float32 zeros() {
        simd8float32 v;
        for (int i = 0; i < 8; ++i) v.data[i] = 0.0f;
        return v;
    }

    simd8float32 operator+(const simd8float32& o) const {
        simd8float32 r;
        for (int i = 0; i < 8; ++i) r.data[i] = data[i] + o.data[i];
        return r;
    }

    simd8float32 operator*(const simd8float32& o) const {
        simd8float32 r;
        for (int i = 0; i < 8; ++i) r.data[i] = data[i] * o.data[i];
        return r;
    }

    simd8float32& operator+=(const simd8float32& o) {
        for (int i = 0; i < 8; ++i) data[i] += o.data[i];
        return *this;
    }

    simd8float32& operator*=(const simd8float32& o) {
        for (int i = 0; i < 8; ++i) data[i] *= o.data[i];
        return *this;
    }

    void store(float* dst) const { for (int i = 0; i < 8; ++i) dst[i] = data[i]; }
    void storeu(float* dst) const { store(dst); }

    float reduce_add() const {
        float s = 0.0f;
        for (int i = 0; i < 8; ++i) s += data[i];
        return s;
    }

    static simd8float32 fma(simd8float32 acc, const simd8float32& a, const simd8float32& b) {
        for (int i = 0; i < 8; ++i) acc.data[i] += a.data[i] * b.data[i];
        return acc;
    }
};

#endif // platform

// ---------------------------------------------------------------------------
// simd_inner_product — optimized inner product using simd8float32
//
// Key optimizations over the naive single-accumulator loop:
//   1. 4 independent accumulators  → breaks the dependency chain, lets the
//      CPU issue 4 FMAs per cycle instead of waiting for each sum to commit.
//   2. FMA  (simd8float32::fma)    → fuses multiply+add into one instruction,
//      cutting latency and halving register pressure.
//   3. Scalar tail                 → handles dims not divisible by 8.
// ---------------------------------------------------------------------------
inline float simd_inner_product(const float* a, const float* b, size_t dim) {
    auto s0 = simd8float32::zeros();
    auto s1 = simd8float32::zeros();
    auto s2 = simd8float32::zeros();
    auto s3 = simd8float32::zeros();

    size_t i = 0;
    // Main loop: 4-way unrolled, each lane uses FMA
    for (; i + 32 <= dim; i += 32) {
        s0 = simd8float32::fma(s0, simd8float32(a + i),      simd8float32(b + i));
        s1 = simd8float32::fma(s1, simd8float32(a + i +  8), simd8float32(b + i +  8));
        s2 = simd8float32::fma(s2, simd8float32(a + i + 16), simd8float32(b + i + 16));
        s3 = simd8float32::fma(s3, simd8float32(a + i + 24), simd8float32(b + i + 24));
    }
    // Remaining 8-element chunks
    for (; i + 8 <= dim; i += 8) {
        s0 = simd8float32::fma(s0, simd8float32(a + i), simd8float32(b + i));
    }

    // Merge accumulators, then horizontal reduce
    s0 += s1;
    s2 += s3;
    s0 += s2;
    float result = s0.reduce_add();

    // Scalar tail for dims not divisible by 8
    for (; i < dim; ++i) result += a[i] * b[i];
    return result;
}

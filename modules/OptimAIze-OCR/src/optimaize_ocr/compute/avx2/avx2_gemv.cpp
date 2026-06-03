// Copyright (c) 2026 Technology Innovation Institute (TII), UAE.
// High-performance AVX2 + OpenMP GEMV kernels for autoregressive decoding.
//
// Optimizations over a naive single-accumulator implementation:
//  - 4 independent FMA accumulators per row to break the FMA dependency chain
//    (Haswell/Zen FMA latency ~4 cycles, throughput 0.5 cycles -> need >=4 accs)
//  - 32-wide inner loop (4 x AVX2 lanes) to maximize ILP and L1 reuse of x[]
//  - Tree-reduction horizontal sum (cheaper than scalar reduction)
//  - Multi-row INT8 GEMV variant that broadcasts each x[j] across rows
//    (amortizes input loads when out_features >> threads)

#include <immintrin.h>
#include <omp.h>
#include <stdint.h>

#if defined(_MSC_VER)
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT __attribute__((visibility("default")))
#endif

// Tree reduction of an AVX2 register into a scalar. Cheaper than _mm256_storeu_ps + scalar adds.
static inline float hsum_avx2(__m256 v) {
    __m128 vlow  = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1);
    __m128 sum128 = _mm_add_ps(vlow, vhigh);
    __m128 shuf  = _mm_movehdup_ps(sum128);
    __m128 sums  = _mm_add_ps(sum128, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);    return _mm_cvtss_f32(sums);
}

// SiLU(x) = x / (1 + exp(-x)). g++ 13 mingw on this Conda build has a
// bizarre ICE that fires on ANY non-integral literal in this file
// (``88.0f``, ``0.5f``, even inside a one-line accessor function). The
// compiler accepts integer literals and accepts floats reconstructed
// through ``union { uint32_t u; float f; }`` from a hex bit pattern.
// We feed every constant to silu_scalar through that path.
//
// Bit patterns below were precomputed:
//   88.0f          = 0x42B00000      -88.0f         = 0xC2B00000
//   1.44269504...  = 0x3FB8AA3B (log2e)
//   0.69314718...  = 0x3F317218 (ln2)
//   0.5f           = 0x3F000000
//   1.0f           = 0x3F800000
//   0.16666667f    = 0x3E2AAAAB
//   0.04166667f    = 0x3D2AAAAB
//   0.00833333f    = 0x3C088889
static inline float as_float(unsigned int u) {
    union { unsigned int u; float f; } b; b.u = u; return b.f;
}

static float silu_scalar(float g) {
    float clamp_hi = as_float(0x42B00000);   //  88.0f
    float clamp_lo = as_float(0xC2B00000);   // -88.0f
    float log2e   = as_float(0x3FB8AA3B);
    float ln2     = as_float(0x3F317218);
    float c0      = as_float(0x3F800000);    // 1.0
    float c1      = as_float(0x3F800000);    // 1.0
    float c2      = as_float(0x3F000000);    // 0.5
    float c3      = as_float(0x3E2AAAAB);    // 1/6
    float c4      = as_float(0x3D2AAAAB);    // 1/24
    float c5      = as_float(0x3C088889);    // 1/120
    float half    = as_float(0x3F000000);    // 0.5
    float zero    = as_float(0x00000000);    // 0.0
    float neg_half = as_float(0xBF000000);   // -0.5

    float x = -g;
    if (x > clamp_hi)       x = clamp_hi;
    else if (x < clamp_lo)  x = clamp_lo;

    // exp(x) via 2^(x * log2e) with integer/fractional split.
    float y = x * log2e;
    int   i = (int)(y + (y >= zero ? half : neg_half));
    float f = x - (float)i * ln2;

    // 5th-order Horner polynomial for exp(f) on roughly [-ln2/2, ln2/2]
    float p = c0 + f * (c1 + f * (c2 + f * (c3 + f * (c4 + f * c5))));

    // 2^i via direct bit manipulation of the IEEE-754 exponent field
    int e = i + 127;
    if (e < 1)   e = 1;
    if (e > 254) e = 254;
    union { unsigned int u; float f; } bits;
    bits.u = (unsigned int)e << 23;
    float exp_x = p * bits.f;

    return g / (c0 + exp_x);
}

extern "C" {

/**
 * Fused AVX2 + OpenMP INT8 quantized GEMV.
 *
 *   out[i] = scale * sum_j ((w_int8[i, j] - zero_point) * x[j]) + bias[i]
 *
 * Loop is 32-wide with 4 independent accumulators to hide FMA latency.
 */
DLLEXPORT void gemv_int8_avx2(
    const int8_t* __restrict w_int8,
    float scale,
    int zero_point,
    const float* __restrict x,
    const float* __restrict bias,
    float* __restrict out,
    int out_features,
    int in_features
) {
    const __m256 zp_vec = _mm256_set1_ps((float)zero_point);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < out_features; ++i) {
        const int8_t* row_w = w_int8 + (size_t)i * in_features;

        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();

        int j = 0;
        // 32-wide loop: 4 independent FMA chains
        for (; j <= in_features - 32; j += 32) {
            __m128i wb0 = _mm_loadl_epi64((const __m128i*)(row_w + j +  0));
            __m128i wb1 = _mm_loadl_epi64((const __m128i*)(row_w + j +  8));
            __m128i wb2 = _mm_loadl_epi64((const __m128i*)(row_w + j + 16));
            __m128i wb3 = _mm_loadl_epi64((const __m128i*)(row_w + j + 24));

            __m256 w0 = _mm256_sub_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(wb0)), zp_vec);
            __m256 w1 = _mm256_sub_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(wb1)), zp_vec);
            __m256 w2 = _mm256_sub_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(wb2)), zp_vec);
            __m256 w3 = _mm256_sub_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(wb3)), zp_vec);

            __m256 x0 = _mm256_loadu_ps(x + j +  0);
            __m256 x1 = _mm256_loadu_ps(x + j +  8);
            __m256 x2 = _mm256_loadu_ps(x + j + 16);
            __m256 x3 = _mm256_loadu_ps(x + j + 24);

            acc0 = _mm256_fmadd_ps(w0, x0, acc0);
            acc1 = _mm256_fmadd_ps(w1, x1, acc1);
            acc2 = _mm256_fmadd_ps(w2, x2, acc2);
            acc3 = _mm256_fmadd_ps(w3, x3, acc3);
        }

        // Tail of 8 lanes
        for (; j <= in_features - 8; j += 8) {
            __m128i wb = _mm_loadl_epi64((const __m128i*)(row_w + j));
            __m256 w  = _mm256_sub_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(wb)), zp_vec);
            __m256 xv = _mm256_loadu_ps(x + j);
            acc0 = _mm256_fmadd_ps(w, xv, acc0);
        }

        // Combine accumulators (tree reduction)
        __m256 acc01 = _mm256_add_ps(acc0, acc1);
        __m256 acc23 = _mm256_add_ps(acc2, acc3);
        __m256 acc   = _mm256_add_ps(acc01, acc23);
        float sum = hsum_avx2(acc);

        // Scalar remainder
        const float zp_f = (float)zero_point;
        for (; j < in_features; ++j) {
            sum += ((float)row_w[j] - zp_f) * x[j];
        }

        out[i] = scale * sum + bias[i];
    }
}

/**
 * Fused AVX2 + OpenMP float32 GEMV.
 *
 *   out[i] = sum_j (w_float32[i, j] * x[j]) + bias[i]
 *
 * Loop is 32-wide with 4 independent accumulators to hide FMA latency.
 */
DLLEXPORT void gemv_float32_avx2(
    const float* __restrict w_float32,
    const float* __restrict x,
    const float* __restrict bias,
    float* __restrict out,
    int out_features,
    int in_features
) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < out_features; ++i) {
        const float* row_w = w_float32 + (size_t)i * in_features;

        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();

        int j = 0;
        // 32-wide loop: 4 independent FMA chains hide FMA latency
        for (; j <= in_features - 32; j += 32) {
            __m256 w0 = _mm256_loadu_ps(row_w + j +  0);
            __m256 w1 = _mm256_loadu_ps(row_w + j +  8);
            __m256 w2 = _mm256_loadu_ps(row_w + j + 16);
            __m256 w3 = _mm256_loadu_ps(row_w + j + 24);

            __m256 x0 = _mm256_loadu_ps(x + j +  0);
            __m256 x1 = _mm256_loadu_ps(x + j +  8);
            __m256 x2 = _mm256_loadu_ps(x + j + 16);
            __m256 x3 = _mm256_loadu_ps(x + j + 24);

            acc0 = _mm256_fmadd_ps(w0, x0, acc0);
            acc1 = _mm256_fmadd_ps(w1, x1, acc1);
            acc2 = _mm256_fmadd_ps(w2, x2, acc2);
            acc3 = _mm256_fmadd_ps(w3, x3, acc3);
        }

        // Tail of 8 lanes
        for (; j <= in_features - 8; j += 8) {
            __m256 w  = _mm256_loadu_ps(row_w + j);
            __m256 xv = _mm256_loadu_ps(x + j);
            acc0 = _mm256_fmadd_ps(w, xv, acc0);
        }

        __m256 acc01 = _mm256_add_ps(acc0, acc1);
        __m256 acc23 = _mm256_add_ps(acc2, acc3);
        __m256 acc   = _mm256_add_ps(acc01, acc23);
        float sum = hsum_avx2(acc);

        for (; j < in_features; ++j) {
            sum += row_w[j] * x[j];
        }

        out[i] = sum + bias[i];
    }
}

/**
 * Multi-row INT8 GEMV: writes 4 output rows at a time so that each x[j] load
 * is reused 4 times (less memory traffic on x). Handy when out_features is
 * a multiple of 4 and the matrix dominates memory bandwidth.
 *
 * Falls back to gemv_int8_avx2 on the row tail.
 */
DLLEXPORT void gemv_int8_avx2_m4(
    const int8_t* __restrict w_int8,
    float scale,
    int zero_point,
    const float* __restrict x,
    const float* __restrict bias,
    float* __restrict out,
    int out_features,
    int in_features
) {
    const __m256 zp_vec = _mm256_set1_ps((float)zero_point);
    const int row_blocks = out_features / 4;

    #pragma omp parallel for schedule(static)
    for (int rb = 0; rb < row_blocks; ++rb) {
        const int i = rb * 4;
        const int8_t* r0 = w_int8 + (size_t)(i + 0) * in_features;
        const int8_t* r1 = w_int8 + (size_t)(i + 1) * in_features;
        const int8_t* r2 = w_int8 + (size_t)(i + 2) * in_features;
        const int8_t* r3 = w_int8 + (size_t)(i + 3) * in_features;

        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();

        int j = 0;
        for (; j <= in_features - 8; j += 8) {
            __m256 xv = _mm256_loadu_ps(x + j);

            __m256 w0 = _mm256_sub_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(r0 + j)))), zp_vec);
            __m256 w1 = _mm256_sub_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(r1 + j)))), zp_vec);
            __m256 w2 = _mm256_sub_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(r2 + j)))), zp_vec);
            __m256 w3 = _mm256_sub_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(r3 + j)))), zp_vec);

            acc0 = _mm256_fmadd_ps(w0, xv, acc0);
            acc1 = _mm256_fmadd_ps(w1, xv, acc1);
            acc2 = _mm256_fmadd_ps(w2, xv, acc2);
            acc3 = _mm256_fmadd_ps(w3, xv, acc3);
        }

        float s0 = hsum_avx2(acc0);
        float s1 = hsum_avx2(acc1);
        float s2 = hsum_avx2(acc2);
        float s3 = hsum_avx2(acc3);

        const float zp_f = (float)zero_point;
        for (; j < in_features; ++j) {
            float xj = x[j];
            s0 += ((float)r0[j] - zp_f) * xj;
            s1 += ((float)r1[j] - zp_f) * xj;
            s2 += ((float)r2[j] - zp_f) * xj;
            s3 += ((float)r3[j] - zp_f) * xj;
        }

        out[i + 0] = scale * s0 + bias[i + 0];
        out[i + 1] = scale * s1 + bias[i + 1];
        out[i + 2] = scale * s2 + bias[i + 2];
        out[i + 3] = scale * s3 + bias[i + 3];
    }

    // Tail rows (out_features not divisible by 4): scalar path
    const float zp_f = (float)zero_point;
    for (int i = row_blocks * 4; i < out_features; ++i) {
        const int8_t* row_w = w_int8 + (size_t)i * in_features;
        float sum = 0.0f;
        for (int j = 0; j < in_features; ++j) {
            sum += ((float)row_w[j] - zp_f) * x[j];
        }
        out[i] = scale * sum + bias[i];
    }
}

/**
 * Multi-row float32 GEMV: 4 output rows per iteration, single broadcast of x[j].
 */
DLLEXPORT void gemv_float32_avx2_m4(
    const float* __restrict w_float32,
    const float* __restrict x,
    const float* __restrict bias,
    float* __restrict out,
    int out_features,
    int in_features
) {
    const int row_blocks = out_features / 4;

    #pragma omp parallel for schedule(static)
    for (int rb = 0; rb < row_blocks; ++rb) {
        const int i = rb * 4;
        const float* r0 = w_float32 + (size_t)(i + 0) * in_features;
        const float* r1 = w_float32 + (size_t)(i + 1) * in_features;
        const float* r2 = w_float32 + (size_t)(i + 2) * in_features;
        const float* r3 = w_float32 + (size_t)(i + 3) * in_features;

        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();

        int j = 0;
        for (; j <= in_features - 8; j += 8) {
            __m256 xv = _mm256_loadu_ps(x + j);
            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(r0 + j), xv, acc0);
            acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(r1 + j), xv, acc1);
            acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(r2 + j), xv, acc2);
            acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(r3 + j), xv, acc3);
        }

        float s0 = hsum_avx2(acc0);
        float s1 = hsum_avx2(acc1);
        float s2 = hsum_avx2(acc2);
        float s3 = hsum_avx2(acc3);

        for (; j < in_features; ++j) {
            float xj = x[j];
            s0 += r0[j] * xj;
            s1 += r1[j] * xj;
            s2 += r2[j] * xj;
            s3 += r3[j] * xj;
        }

        out[i + 0] = s0 + bias[i + 0];
        out[i + 1] = s1 + bias[i + 1];
        out[i + 2] = s2 + bias[i + 2];
        out[i + 3] = s3 + bias[i + 3];
    }
}


/**
 * Per-channel symmetric INT8 GEMV.
 *   out[i] = scales[i] * sum_j (w_int8[i, j] * x[j]) + bias[i]
 *
 * Zero-point is implicit 0 (symmetric quantization). 4 independent
 * accumulators + 32-wide inner loop + OpenMP across rows.
 */
DLLEXPORT void gemv_int8_avx2_per_channel(
    const int8_t* __restrict w_int8,
    const float* __restrict scales,
    const float* __restrict x,
    const float* __restrict bias,
    float* __restrict out,
    int out_features,
    int in_features
) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < out_features; ++i) {
        const int8_t* row_w = w_int8 + (size_t)i * in_features;

        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();

        int j = 0;
        for (; j <= in_features - 32; j += 32) {
            __m128i wb0 = _mm_loadl_epi64((const __m128i*)(row_w + j +  0));
            __m128i wb1 = _mm_loadl_epi64((const __m128i*)(row_w + j +  8));
            __m128i wb2 = _mm_loadl_epi64((const __m128i*)(row_w + j + 16));
            __m128i wb3 = _mm_loadl_epi64((const __m128i*)(row_w + j + 24));

            __m256 w0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(wb0));
            __m256 w1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(wb1));
            __m256 w2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(wb2));
            __m256 w3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(wb3));

            __m256 x0 = _mm256_loadu_ps(x + j +  0);
            __m256 x1 = _mm256_loadu_ps(x + j +  8);
            __m256 x2 = _mm256_loadu_ps(x + j + 16);
            __m256 x3 = _mm256_loadu_ps(x + j + 24);

            acc0 = _mm256_fmadd_ps(w0, x0, acc0);
            acc1 = _mm256_fmadd_ps(w1, x1, acc1);
            acc2 = _mm256_fmadd_ps(w2, x2, acc2);
            acc3 = _mm256_fmadd_ps(w3, x3, acc3);
        }
        for (; j <= in_features - 8; j += 8) {
            __m128i wb = _mm_loadl_epi64((const __m128i*)(row_w + j));
            __m256 w  = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(wb));
            __m256 xv = _mm256_loadu_ps(x + j);
            acc0 = _mm256_fmadd_ps(w, xv, acc0);
        }

        __m256 acc01 = _mm256_add_ps(acc0, acc1);
        __m256 acc23 = _mm256_add_ps(acc2, acc3);
        __m256 acc   = _mm256_add_ps(acc01, acc23);
        float sum = hsum_avx2(acc);

        for (; j < in_features; ++j) {
            sum += (float)row_w[j] * x[j];
        }

        out[i] = scales[i] * sum + bias[i];
    }
}

DLLEXPORT void gemv_int8_avx2_per_channel_m4(
    const int8_t* __restrict w_int8,
    const float* __restrict scales,
    const float* __restrict x,
    const float* __restrict bias,
    float* __restrict out,
    int out_features,
    int in_features
) {
    const int row_blocks = out_features / 4;

    #pragma omp parallel for schedule(static)
    for (int rb = 0; rb < row_blocks; ++rb) {
        const int i = rb * 4;
        const int8_t* r0 = w_int8 + (size_t)(i + 0) * in_features;
        const int8_t* r1 = w_int8 + (size_t)(i + 1) * in_features;
        const int8_t* r2 = w_int8 + (size_t)(i + 2) * in_features;
        const int8_t* r3 = w_int8 + (size_t)(i + 3) * in_features;

        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();

        int j = 0;
        for (; j <= in_features - 8; j += 8) {
            __m256 xv = _mm256_loadu_ps(x + j);
            __m256 w0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(r0 + j))));
            __m256 w1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(r1 + j))));
            __m256 w2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(r2 + j))));
            __m256 w3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(r3 + j))));
            acc0 = _mm256_fmadd_ps(w0, xv, acc0);
            acc1 = _mm256_fmadd_ps(w1, xv, acc1);
            acc2 = _mm256_fmadd_ps(w2, xv, acc2);
            acc3 = _mm256_fmadd_ps(w3, xv, acc3);
        }

        float s0 = hsum_avx2(acc0);
        float s1 = hsum_avx2(acc1);
        float s2 = hsum_avx2(acc2);
        float s3 = hsum_avx2(acc3);

        for (; j < in_features; ++j) {
            const float xj = x[j];
            s0 += (float)r0[j] * xj;
            s1 += (float)r1[j] * xj;
            s2 += (float)r2[j] * xj;
            s3 += (float)r3[j] * xj;
        }

        out[i + 0] = scales[i + 0] * s0 + bias[i + 0];
        out[i + 1] = scales[i + 1] * s1 + bias[i + 1];
        out[i + 2] = scales[i + 2] * s2 + bias[i + 2];
        out[i + 3] = scales[i + 3] * s3 + bias[i + 3];
    }

    for (int i = row_blocks * 4; i < out_features; ++i) {
        const int8_t* row_w = w_int8 + (size_t)i * in_features;
        float sum = 0.0f;
        for (int j = 0; j < in_features; ++j) {
            sum += (float)row_w[j] * x[j];
        }
        out[i] = scales[i] * sum + bias[i];
    }
}

// Compute RMSNorm in-place
static inline void rmsnorm_fused(
    const float* __restrict x,
    float* __restrict x_norm,
    int dim,
    float eps
) {
    // Compute mean of squares using AVX2
    __m256 sum_sq = _mm256_setzero_ps();
    int i = 0;
    for (; i <= dim - 8; i += 8) {
        __m256 xi = _mm256_loadu_ps(x + i);
        sum_sq = _mm256_fmadd_ps(xi, xi, sum_sq);
    }

    // Horizontal sum
    float mean_sq = hsum_avx2(sum_sq);

    // Scalar remainder
    for (; i < dim; ++i) {
        mean_sq += x[i] * x[i];
    }
    mean_sq = mean_sq / (float)dim;

    // Compute inverse RMS using rsqrt instruction
    float mean_sq_eps = mean_sq + eps;
    float inv_rms;
    __m128 tmp = _mm_load_ss(&mean_sq_eps);
    __m128 rsqrt = _mm_rsqrt_ss(tmp);
    _mm_store_ss(&inv_rms, rsqrt);

    // Normalize x using AVX2
    __m256 inv_rms_vec = _mm256_set1_ps(inv_rms);
    i = 0;
    for (; i <= dim - 8; i += 8) {
        __m256 xi = _mm256_loadu_ps(x + i);
        __m256 xn = _mm256_mul_ps(xi, inv_rms_vec);
        _mm256_storeu_ps(x_norm + i, xn);
    }
    for (; i < dim; ++i) {
        x_norm[i] = x[i] * inv_rms;
    }
}

/**
 * Fused RMSNorm + AVX2 float32 GEMV.
 *
 *   x_norm = rmsnorm(x, dim, eps)
 *   out[i] = sum_j (w_float32[i, j] * x_norm[j]) + bias[i]
 */
DLLEXPORT void rmsnorm_gemm_float32_avx2(
    const float* __restrict x,           // Input: [dim]
    float* __restrict x_norm_buf,        // Buffer for normalized x
    const float* __restrict w_float32,   // Weight: [out_features, dim]
    const float* __restrict bias,        // Bias: [out_features]
    float* __restrict out,               // Output: [out_features]
    int out_features,
    int dim,
    float eps
) {
    // Compute RMSNorm first
    rmsnorm_fused((float*)x, x_norm_buf, dim, eps);

    // GEMV with normalized input
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < out_features; ++i) {
        const float* row_w = w_float32 + (size_t)i * dim;

        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();

        int j = 0;
        for (; j <= dim - 32; j += 32) {
            __m256 w0 = _mm256_loadu_ps(row_w + j +  0);
            __m256 w1 = _mm256_loadu_ps(row_w + j +  8);
            __m256 w2 = _mm256_loadu_ps(row_w + j + 16);
            __m256 w3 = _mm256_loadu_ps(row_w + j + 24);

            __m256 x0 = _mm256_loadu_ps(x_norm_buf + j +  0);
            __m256 x1 = _mm256_loadu_ps(x_norm_buf + j +  8);
            __m256 x2 = _mm256_loadu_ps(x_norm_buf + j + 16);
            __m256 x3 = _mm256_loadu_ps(x_norm_buf + j + 24);

            acc0 = _mm256_fmadd_ps(w0, x0, acc0);
            acc1 = _mm256_fmadd_ps(w1, x1, acc1);
            acc2 = _mm256_fmadd_ps(w2, x2, acc2);
            acc3 = _mm256_fmadd_ps(w3, x3, acc3);
        }

        // Tail of 8 lanes
        for (; j <= dim - 8; j += 8) {
            __m256 w  = _mm256_loadu_ps(row_w + j);
            __m256 xv = _mm256_loadu_ps(x_norm_buf + j);
            acc0 = _mm256_fmadd_ps(w, xv, acc0);
        }

        __m256 acc01 = _mm256_add_ps(acc0, acc1);
        __m256 acc23 = _mm256_add_ps(acc2, acc3);
        __m256 acc   = _mm256_add_ps(acc01, acc23);
        float sum = hsum_avx2(acc);

        for (; j < dim; ++j) {
            sum += row_w[j] * x_norm_buf[j];
        }

        out[i] = sum + bias[i];
    }
}

/**
 * Fused RMSNorm + Per-channel INT8 GEMV.
 *
 *   x_norm = rmsnorm(x, dim, eps)
 *   out[i] = scales[i] * sum_j (w_int8[i, j] * x_norm[j]) + bias[i]
 */
DLLEXPORT void rmsnorm_gemm_int8_per_channel_avx2(
    const float* __restrict x,           // Input: [dim]
    float* __restrict x_norm_buf,        // Buffer for normalized x
    const int8_t* __restrict w_int8,     // Weight: [out_features, dim]
    const float* __restrict scales,      // Scale per output
    const float* __restrict bias,        // Bias: [out_features]
    float* __restrict out,               // Output: [out_features]
    int out_features,
    int dim,
    float eps
) {
    // Compute RMSNorm first
    rmsnorm_fused((float*)x, x_norm_buf, dim, eps);

    // INT8 GEMV with normalized input
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < out_features; ++i) {
        const int8_t* row_w = w_int8 + (size_t)i * dim;

        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();

        int j = 0;
        for (; j <= dim - 32; j += 32) {
            __m128i wb0 = _mm_loadl_epi64((const __m128i*)(row_w + j +  0));
            __m128i wb1 = _mm_loadl_epi64((const __m128i*)(row_w + j +  8));
            __m128i wb2 = _mm_loadl_epi64((const __m128i*)(row_w + j + 16));
            __m128i wb3 = _mm_loadl_epi64((const __m128i*)(row_w + j + 24));

            __m256 w0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(wb0));
            __m256 w1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(wb1));
            __m256 w2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(wb2));
            __m256 w3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(wb3));

            __m256 x0 = _mm256_loadu_ps(x_norm_buf + j +  0);
            __m256 x1 = _mm256_loadu_ps(x_norm_buf + j +  8);
            __m256 x2 = _mm256_loadu_ps(x_norm_buf + j + 16);
            __m256 x3 = _mm256_loadu_ps(x_norm_buf + j + 24);

            acc0 = _mm256_fmadd_ps(w0, x0, acc0);
            acc1 = _mm256_fmadd_ps(w1, x1, acc1);
            acc2 = _mm256_fmadd_ps(w2, x2, acc2);
            acc3 = _mm256_fmadd_ps(w3, x3, acc3);
        }

        for (; j <= dim - 8; j += 8) {
            __m128i wb = _mm_loadl_epi64((const __m128i*)(row_w + j));
            __m256 w  = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(wb));
            __m256 xv = _mm256_loadu_ps(x_norm_buf + j);
            acc0 = _mm256_fmadd_ps(w, xv, acc0);
        }

        __m256 acc01 = _mm256_add_ps(acc0, acc1);
        __m256 acc23 = _mm256_add_ps(acc2, acc3);
        __m256 acc   = _mm256_add_ps(acc01, acc23);
        float sum = hsum_avx2(acc);

        for (; j < dim; ++j) {
            sum += (float)row_w[j] * x_norm_buf[j];
        }

        out[i] = scales[i] * sum + bias[i];
    }
}

// SiLU(x) = x / (1 + exp(-x)). g++ 13 mingw on this Conda build ICEs on
// libm ``expf`` and on ``_mm_set_ss`` immediates inside this function
// when -O3 -ffast-math + OpenMP are all in play. Both workarounds came
// up empty (noinline/optimize("O0")). We replace the entire helper with
// a 5th-order polynomial approximation of exp on the range-reduced
// fractional part, then reconstruct 2^i via direct exponent-field bit
// twiddling. The constants live in arrays so they reach the compiler as
// loaded values, not immediates that trip the ICE.
// (Definition lives above ``extern "C"`` to dodge a separate g++ ICE
// triggered by ``static const float`` inside the extern block.)

/**
 * Fused SwiGLU MLP gate + up projection (per-channel symmetric INT8).
 *
 * Computes for each row i in [0, hidden):
 *     g = sum_j w_gate[i,j] * x[j]   (scaled by scales_gate[i])
 *     u = sum_j w_up[i,j]   * x[j]   (scaled by scales_up[i])
 *     out[i] = silu(g) * u
 *
 * The classic Qwen2/Llama MLP runs ``down_proj(silu(gate_proj(x)) * up_proj(x))``.
 * In our autoregressive decode loop the two projections are independent, so
 * the standard implementation issues two GEMV ctypes calls + two output
 * buffers (each ``intermediate_size`` = 8960 floats for Qwen2). Fusing them
 * lets us:
 *   - read ``x[j]`` once per inner-loop iteration (instead of twice),
 *   - issue a single OpenMP fork/join,
 *   - apply silu()*u immediately so we never write the gate/up tensors to
 *     memory (kernel fusion eliminates a 35 KB round trip per call).
 *
 * Layouts: ``w_gate`` and ``w_up`` are both row-major [hidden, in_features],
 * INT8 per-row symmetric (zero-point = 0). Biases are optional — pass
 * nullptr to skip.
 */
DLLEXPORT void swiglu_gate_up_int8_per_channel_avx2(
    const int8_t* __restrict w_gate,
    const float*  __restrict scales_gate,
    const float*  __restrict bias_gate,    // may be nullptr
    const int8_t* __restrict w_up,
    const float*  __restrict scales_up,
    const float*  __restrict bias_up,      // may be nullptr
    const float*  __restrict x,
    float*        __restrict out,          // length = hidden
    int hidden,
    int in_features
) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < hidden; ++i) {
        const int8_t* row_g = w_gate + (size_t)i * in_features;
        const int8_t* row_u = w_up   + (size_t)i * in_features;

        __m256 ag0 = _mm256_setzero_ps();
        __m256 ag1 = _mm256_setzero_ps();
        __m256 au0 = _mm256_setzero_ps();
        __m256 au1 = _mm256_setzero_ps();

        int j = 0;
        // 16-wide inner loop: 2 independent FMA chains per matrix. Loading
        // the same x[j] window once and reusing it for both gate and up is
        // the whole point of fusion; FMA latency hides behind 4 chains in
        // flight (2 per matrix).
        for (; j <= in_features - 16; j += 16) {
            __m128i wg0_b = _mm_loadl_epi64((const __m128i*)(row_g + j +  0));
            __m128i wg1_b = _mm_loadl_epi64((const __m128i*)(row_g + j +  8));
            __m128i wu0_b = _mm_loadl_epi64((const __m128i*)(row_u + j +  0));
            __m128i wu1_b = _mm_loadl_epi64((const __m128i*)(row_u + j +  8));

            __m256 wg0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(wg0_b));
            __m256 wg1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(wg1_b));
            __m256 wu0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(wu0_b));
            __m256 wu1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(wu1_b));

            __m256 x0 = _mm256_loadu_ps(x + j +  0);
            __m256 x1 = _mm256_loadu_ps(x + j +  8);

            ag0 = _mm256_fmadd_ps(wg0, x0, ag0);
            ag1 = _mm256_fmadd_ps(wg1, x1, ag1);
            au0 = _mm256_fmadd_ps(wu0, x0, au0);
            au1 = _mm256_fmadd_ps(wu1, x1, au1);
        }
        // 8-lane tail
        for (; j <= in_features - 8; j += 8) {
            __m128i wg_b = _mm_loadl_epi64((const __m128i*)(row_g + j));
            __m128i wu_b = _mm_loadl_epi64((const __m128i*)(row_u + j));
            __m256 wg = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(wg_b));
            __m256 wu = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(wu_b));
            __m256 xv = _mm256_loadu_ps(x + j);
            ag0 = _mm256_fmadd_ps(wg, xv, ag0);
            au0 = _mm256_fmadd_ps(wu, xv, au0);
        }

        __m256 ag = _mm256_add_ps(ag0, ag1);
        __m256 au = _mm256_add_ps(au0, au1);
        float g = hsum_avx2(ag);
        float u = hsum_avx2(au);

        // Scalar remainder
        for (; j < in_features; ++j) {
            g += (float)row_g[j] * x[j];
            u += (float)row_u[j] * x[j];
        }

        g = scales_gate[i] * g;
        u = scales_up[i]   * u;
        if (bias_gate) g += bias_gate[i];
        if (bias_up)   u += bias_up[i];

        out[i] = silu_scalar(g) * u;
    }
}

/**
 * Fused QKV projection (per-channel symmetric INT8) for GQA attention.
 *
 * Computes three independent GEMVs that share the same input vector ``x``:
 *     out_q[i] = sum_j w_q[i,j] * x[j]   (i in [0, q_out))
 *     out_k[i] = sum_j w_k[i,j] * x[j]   (i in [0, kv_out))
 *     out_v[i] = sum_j w_v[i,j] * x[j]   (i in [0, kv_out))
 * Each output gets its own per-row scale + optional bias.
 *
 * Compared to three separate ``gemv_int8_avx2_per_channel`` calls this:
 *   - reads ``x`` once instead of three times,
 *   - issues a single OpenMP fork/join,
 *   - uses one parallel for over the *combined* row index, so all threads
 *     stay busy regardless of how skewed q_out vs. kv_out are (Qwen2-1.5B
 *     has 1536 q rows but only 256 each for k/v under GQA).
 *
 * Layout: each w_* is row-major [rows, in_features], INT8 per-row symmetric
 * (zero_point = 0). Biases may be NULL.
 */
DLLEXPORT void qkv_int8_per_channel_avx2(
    const int8_t* __restrict w_q,
    const float*  __restrict scales_q,
    const float*  __restrict bias_q,        // may be nullptr
    const int8_t* __restrict w_k,
    const float*  __restrict scales_k,
    const float*  __restrict bias_k,        // may be nullptr
    const int8_t* __restrict w_v,
    const float*  __restrict scales_v,
    const float*  __restrict bias_v,        // may be nullptr
    const float*  __restrict x,
    float*        __restrict out_q,
    float*        __restrict out_k,
    float*        __restrict out_v,
    int q_out,
    int kv_out,
    int in_features
) {
    int total = q_out + 2 * kv_out;

    #pragma omp parallel for schedule(static)
    for (int idx = 0; idx < total; ++idx) {
        // Resolve which matrix and which row this iteration owns
        const int8_t* row_w;
        float scale;
        float bias_val;
        float* out_ptr;
        int local_i;

        if (idx < q_out) {
            local_i = idx;
            row_w   = w_q + (size_t)local_i * in_features;
            scale   = scales_q[local_i];
            bias_val = bias_q ? bias_q[local_i] : as_float(0x00000000);
            out_ptr = out_q + local_i;
        } else if (idx < q_out + kv_out) {
            local_i = idx - q_out;
            row_w   = w_k + (size_t)local_i * in_features;
            scale   = scales_k[local_i];
            bias_val = bias_k ? bias_k[local_i] : as_float(0x00000000);
            out_ptr = out_k + local_i;
        } else {
            local_i = idx - q_out - kv_out;
            row_w   = w_v + (size_t)local_i * in_features;
            scale   = scales_v[local_i];
            bias_val = bias_v ? bias_v[local_i] : as_float(0x00000000);
            out_ptr = out_v + local_i;
        }

        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();

        int j = 0;
        for (; j <= in_features - 32; j += 32) {
            __m128i wb0 = _mm_loadl_epi64((const __m128i*)(row_w + j +  0));
            __m128i wb1 = _mm_loadl_epi64((const __m128i*)(row_w + j +  8));
            __m128i wb2 = _mm_loadl_epi64((const __m128i*)(row_w + j + 16));
            __m128i wb3 = _mm_loadl_epi64((const __m128i*)(row_w + j + 24));

            __m256 w0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(wb0));
            __m256 w1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(wb1));
            __m256 w2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(wb2));
            __m256 w3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(wb3));

            __m256 x0 = _mm256_loadu_ps(x + j +  0);
            __m256 x1 = _mm256_loadu_ps(x + j +  8);
            __m256 x2 = _mm256_loadu_ps(x + j + 16);
            __m256 x3 = _mm256_loadu_ps(x + j + 24);

            acc0 = _mm256_fmadd_ps(w0, x0, acc0);
            acc1 = _mm256_fmadd_ps(w1, x1, acc1);
            acc2 = _mm256_fmadd_ps(w2, x2, acc2);
            acc3 = _mm256_fmadd_ps(w3, x3, acc3);
        }
        for (; j <= in_features - 8; j += 8) {
            __m128i wb = _mm_loadl_epi64((const __m128i*)(row_w + j));
            __m256 w  = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(wb));
            __m256 xv = _mm256_loadu_ps(x + j);
            acc0 = _mm256_fmadd_ps(w, xv, acc0);
        }

        __m256 acc01 = _mm256_add_ps(acc0, acc1);
        __m256 acc23 = _mm256_add_ps(acc2, acc3);
        __m256 acc   = _mm256_add_ps(acc01, acc23);
        float sum = hsum_avx2(acc);

        for (; j < in_features; ++j) {
            sum += (float)row_w[j] * x[j];
        }

        *out_ptr = scale * sum + bias_val;
    }
}

/**
 * Fused RMSNorm + QKV projection (per-channel symmetric INT8) for GQA.
 *
 * Computes in a single AVX2/OpenMP pass:
 *     n_j     = x_j * rsqrt(mean_sq + eps) * gamma_j      (RMSNorm)
 *     out_q[i] = sum_j w_q[i,j] * n_j   (rescaled by scales_q)
 *     out_k[i] = sum_j w_k[i,j] * n_j   (rescaled by scales_k)
 *     out_v[i] = sum_j w_v[i,j] * n_j   (rescaled by scales_v)
 *
 * Replaces the standard
 *     n = RMSNorm(x); q,k,v = q_proj(n), k_proj(n), v_proj(n)
 * with one kernel that:
 *   - computes mean_sq + writes the normalized n_j once into a scratch
 *     buffer (sequential AVX2),
 *   - then runs the same QKV fan-out we already had, but reading n_j
 *     instead of x_j (parallel AVX2 over the combined output index).
 *
 * The two stages can't trivially fuse into one inner loop (you need
 * mean_sq before you can normalize), but the cost we save is the
 * separate Python ``rms_norm + clone + reshape`` round trip plus a
 * full read/write of the 1536-wide tensor in PyTorch.
 *
 * Inputs are the same as ``qkv_int8_per_channel_avx2`` plus the RMSNorm
 * gamma weight (length ``in_features``) and ``eps``. ``norm_buf`` must
 * be a caller-allocated scratch float buffer of length ``in_features``.
 */
DLLEXPORT void rmsnorm_qkv_int8_per_channel_avx2(
    const float*  __restrict x,
    const float*  __restrict gamma,         // RMSNorm weight, [in_features]
    float         eps,
    float*        __restrict norm_buf,      // scratch [in_features]
    const int8_t* __restrict w_q,
    const float*  __restrict scales_q,
    const float*  __restrict bias_q,
    const int8_t* __restrict w_k,
    const float*  __restrict scales_k,
    const float*  __restrict bias_k,
    const int8_t* __restrict w_v,
    const float*  __restrict scales_v,
    const float*  __restrict bias_v,
    float*        __restrict out_q,
    float*        __restrict out_k,
    float*        __restrict out_v,
    int q_out,
    int kv_out,
    int in_features
) {
    // --- Stage 1: RMSNorm computed sequentially on this thread.
    // 1536 floats fit in L1; parallelizing this would cost more in OpenMP
    // overhead than it saves.
    __m256 sumsq = _mm256_setzero_ps();
    int j = 0;
    for (; j <= in_features - 8; j += 8) {
        __m256 xv = _mm256_loadu_ps(x + j);
        sumsq = _mm256_fmadd_ps(xv, xv, sumsq);
    }
    float mean_sq = hsum_avx2(sumsq);
    for (; j < in_features; ++j) mean_sq += x[j] * x[j];
    mean_sq = mean_sq / (float)in_features + eps;

    // 1/sqrt(mean_sq) via fast SSE intrinsic + Newton-Raphson refinement
    __m128 ms = _mm_load_ss(&mean_sq);
    __m128 rs = _mm_rsqrt_ss(ms);
    // NR step: rs = rs * (1.5 - 0.5 * mean_sq * rs * rs)
    float half_f = as_float(0x3F000000);
    float three_half_f = as_float(0x3FC00000);
    __m128 half = _mm_load_ss(&half_f);
    __m128 three_half = _mm_load_ss(&three_half_f);
    __m128 r2 = _mm_mul_ss(rs, rs);
    __m128 t  = _mm_mul_ss(_mm_mul_ss(half, ms), r2);
    rs = _mm_mul_ss(rs, _mm_sub_ss(three_half, t));
    float inv_rms;
    _mm_store_ss(&inv_rms, rs);

    // n_j = x_j * inv_rms * gamma_j
    __m256 inv_rms_v = _mm256_set1_ps(inv_rms);
    j = 0;
    for (; j <= in_features - 8; j += 8) {
        __m256 xv = _mm256_loadu_ps(x + j);
        __m256 gv = _mm256_loadu_ps(gamma + j);
        __m256 nv = _mm256_mul_ps(_mm256_mul_ps(xv, inv_rms_v), gv);
        _mm256_storeu_ps(norm_buf + j, nv);
    }
    for (; j < in_features; ++j) norm_buf[j] = x[j] * inv_rms * gamma[j];

    // --- Stage 2: parallel QKV fan-out reading from norm_buf.
    int total = q_out + 2 * kv_out;

    #pragma omp parallel for schedule(static)
    for (int idx = 0; idx < total; ++idx) {
        const int8_t* row_w;
        float scale;
        float bias_val;
        float* out_ptr;
        int local_i;

        if (idx < q_out) {
            local_i = idx;
            row_w   = w_q + (size_t)local_i * in_features;
            scale   = scales_q[local_i];
            bias_val = bias_q ? bias_q[local_i] : as_float(0x00000000);
            out_ptr = out_q + local_i;
        } else if (idx < q_out + kv_out) {
            local_i = idx - q_out;
            row_w   = w_k + (size_t)local_i * in_features;
            scale   = scales_k[local_i];
            bias_val = bias_k ? bias_k[local_i] : as_float(0x00000000);
            out_ptr = out_k + local_i;
        } else {
            local_i = idx - q_out - kv_out;
            row_w   = w_v + (size_t)local_i * in_features;
            scale   = scales_v[local_i];
            bias_val = bias_v ? bias_v[local_i] : as_float(0x00000000);
            out_ptr = out_v + local_i;
        }

        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();

        int k = 0;
        for (; k <= in_features - 32; k += 32) {
            __m128i wb0 = _mm_loadl_epi64((const __m128i*)(row_w + k +  0));
            __m128i wb1 = _mm_loadl_epi64((const __m128i*)(row_w + k +  8));
            __m128i wb2 = _mm_loadl_epi64((const __m128i*)(row_w + k + 16));
            __m128i wb3 = _mm_loadl_epi64((const __m128i*)(row_w + k + 24));

            __m256 w0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(wb0));
            __m256 w1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(wb1));
            __m256 w2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(wb2));
            __m256 w3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(wb3));

            __m256 x0 = _mm256_loadu_ps(norm_buf + k +  0);
            __m256 x1 = _mm256_loadu_ps(norm_buf + k +  8);
            __m256 x2 = _mm256_loadu_ps(norm_buf + k + 16);
            __m256 x3 = _mm256_loadu_ps(norm_buf + k + 24);

            acc0 = _mm256_fmadd_ps(w0, x0, acc0);
            acc1 = _mm256_fmadd_ps(w1, x1, acc1);
            acc2 = _mm256_fmadd_ps(w2, x2, acc2);
            acc3 = _mm256_fmadd_ps(w3, x3, acc3);
        }
        for (; k <= in_features - 8; k += 8) {
            __m128i wb = _mm_loadl_epi64((const __m128i*)(row_w + k));
            __m256 w  = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(wb));
            __m256 xv = _mm256_loadu_ps(norm_buf + k);
            acc0 = _mm256_fmadd_ps(w, xv, acc0);
        }

        __m256 acc01 = _mm256_add_ps(acc0, acc1);
        __m256 acc23 = _mm256_add_ps(acc2, acc3);
        __m256 acc   = _mm256_add_ps(acc01, acc23);
        float sum = hsum_avx2(acc);

        for (; k < in_features; ++k) {
            sum += (float)row_w[k] * norm_buf[k];
        }

        *out_ptr = scale * sum + bias_val;
    }
}

/**
 * Per-channel INT8 GEMV that adds the result to a residual in place.
 *
 *   out[i] = residual[i] + scales[i] * sum_j w[i,j] * x[j] + bias[i]
 *
 * Replaces the standard ``y = proj(x); y = y + residual`` which on CPU
 * walks the 1536-wide ``y`` tensor twice (once to write, once to read for
 * the add) and pays a separate Python op for the add. Folding the add
 * into the same loop turns it into a single FMA at the end.
 */
DLLEXPORT void gemv_int8_pc_residual_avx2(
    const int8_t* __restrict w_int8,
    const float*  __restrict scales,
    const float*  __restrict bias,        // may be nullptr
    const float*  __restrict x,
    const float*  __restrict residual,    // [out_features]
    float*        __restrict out,         // [out_features], may alias residual
    int out_features,
    int in_features
) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < out_features; ++i) {
        const int8_t* row_w = w_int8 + (size_t)i * in_features;

        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();

        int j = 0;
        for (; j <= in_features - 32; j += 32) {
            __m128i wb0 = _mm_loadl_epi64((const __m128i*)(row_w + j +  0));
            __m128i wb1 = _mm_loadl_epi64((const __m128i*)(row_w + j +  8));
            __m128i wb2 = _mm_loadl_epi64((const __m128i*)(row_w + j + 16));
            __m128i wb3 = _mm_loadl_epi64((const __m128i*)(row_w + j + 24));

            __m256 w0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(wb0));
            __m256 w1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(wb1));
            __m256 w2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(wb2));
            __m256 w3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(wb3));

            __m256 x0 = _mm256_loadu_ps(x + j +  0);
            __m256 x1 = _mm256_loadu_ps(x + j +  8);
            __m256 x2 = _mm256_loadu_ps(x + j + 16);
            __m256 x3 = _mm256_loadu_ps(x + j + 24);

            acc0 = _mm256_fmadd_ps(w0, x0, acc0);
            acc1 = _mm256_fmadd_ps(w1, x1, acc1);
            acc2 = _mm256_fmadd_ps(w2, x2, acc2);
            acc3 = _mm256_fmadd_ps(w3, x3, acc3);
        }
        for (; j <= in_features - 8; j += 8) {
            __m128i wb = _mm_loadl_epi64((const __m128i*)(row_w + j));
            __m256 w  = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(wb));
            __m256 xv = _mm256_loadu_ps(x + j);
            acc0 = _mm256_fmadd_ps(w, xv, acc0);
        }

        __m256 acc01 = _mm256_add_ps(acc0, acc1);
        __m256 acc23 = _mm256_add_ps(acc2, acc3);
        __m256 acc   = _mm256_add_ps(acc01, acc23);
        float sum = hsum_avx2(acc);

        for (; j < in_features; ++j) {
            sum += (float)row_w[j] * x[j];
        }

        float bias_val = bias ? bias[i] : as_float(0x00000000);
        out[i] = residual[i] + scales[i] * sum + bias_val;
    }
}

} // extern "C"

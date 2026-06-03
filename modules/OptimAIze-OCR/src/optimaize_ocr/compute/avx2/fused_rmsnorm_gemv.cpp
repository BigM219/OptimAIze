// Copyright (c) 2026 Technology Innovation Institute (TII), UAE.
// Fused RMSNorm + AVX2 INT8/FP32 GEMV kernel.
//
// Combines RMSNorm and GEMV into a single kernel to avoid:
//   - Intermediate tensor allocation
//   - Extra memory read/write for the normalized input
//   - Python dispatch overhead
//
// Formula:
//   x_norm = x / sqrt(mean(x^2) + eps)
//   out[i] = w[i] * x_norm + bias[i]
//
// Optimizations:
//   - Compute RMSNorm first (single pass over input)
//   - Fused GEMV with normalized values
//   - Reuse x_norm across all output rows (better cache locality)

#include <immintrin.h>
#include <omp.h>
#include <stdint.h>
#include <math.h>

#if defined(_MSC_VER)
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT __attribute__((visibility("default")))
#endif

static inline float hsum_avx2(__m256 v) {
    __m128 vlow  = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1);
    __m128 sum128 = _mm_add_ps(vlow, vhigh);
    __m128 shuf  = _mm_movehdup_ps(sum128);
    __m128 sums  = _mm_add_ps(sum128, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
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

    // Compute inverse RMS: 1 / sqrt(mean_sq + eps)
    float mean_sq_eps = mean_sq + eps;
    float inv_rms;
    __m128 tmp = _mm_load_ss(&mean_sq_eps);
    __m128 rsqrt = _mm_rsqrt_ss(tmp);
    _mm_store_ss(&inv_rms, rsqrt);

    // Normalize x in-place using AVX2
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

extern "C" {

/**
 * Fused RMSNorm + AVX2 float32 GEMV.
 *
 *   x_norm = rmsnorm(x, dim, eps)
 *   out[i] = sum_j (w_float32[i, j] * x_norm[j]) + bias[i]
 *
 * This kernel fuses RMSNorm and GEMV into a single kernel, avoiding
 * intermediate tensor allocation and extra memory reads/writes.
 */
DLLEXPORT void rmsnorm_gemm_float32_avx2(
    const float* __restrict x,           // Input: [dim]
    float* __restrict x_norm_buf,        // Buffer for normalized x (caller-allocated)
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
    float* __restrict x_norm_buf,        // Buffer for normalized x (caller-allocated)
    const int8_t* __restrict w_int8,     // Weight: [out_features, dim]
    const float* __restrict scales,      // Scale per output: [out_features]
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

} // extern "C"
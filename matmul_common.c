#include "matmul_common.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

void g128_matrix_init(G128Matrix *m, uint32_t num_rows, uint32_t num_cols) {
    m->num_rows = num_rows;
    m->num_cols = num_cols;
    m->num_blocks_row = (num_rows + G128_BLOCK_SIZE - 1) / G128_BLOCK_SIZE;
    m->num_blocks_col = (num_cols + G128_BLOCK_SIZE - 1) / G128_BLOCK_SIZE;

    // Total blocks = N * K / 128 (one block per 128 consecutive K-elements per output row)
    uint64_t num_blocks = (uint64_t)num_rows * num_cols / G128_BLOCK_SIZE;
    m->magnitude  = calloc(num_blocks * 2, sizeof(uint64_t));
    m->sign       = calloc(num_blocks * 2, sizeof(uint64_t));
    m->scales     = malloc(num_blocks * sizeof(uint16_t));
    m->scales_f32 = NULL;
}

void g128_matrix_free(G128Matrix *m) {
    free(m->magnitude);
    free(m->sign);
    free(m->scales);
    free(m->scales_f32);
}

// Correct IEEE 754 FP16 -> FP32 conversion (includes implicit leading 1 for normals)
static inline float half_to_float(uint16_t h) {
    union { uint32_t u; float f; } v;
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t mant = (uint32_t)(h & 0x3FFu);
    if (exp == 0) {
        v.u = sign; // zero (denormals treated as zero)
    } else if (exp == 31) {
        v.u = sign | 0x7F800000u | (mant << 13); // inf/NaN
    } else {
        // exp_f32 = exp_f16 + (127 - 15) = exp_f16 + 112
        v.u = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    return v.f;
}

void matmul_naive_f32(float *A, float *B, float *C, int M, int K, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

void matmul_naive_transpose(float *A, float *B_T, float *C, int M, int K, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B_T[j * K + k];
            }
            C[i * N + j] = sum;
        }
    }
}

// B_T: shape [N, K], num_rows=N, num_cols=K
// Blocks: row j has num_blocks_col = K/128 blocks
// block_idx for (j, bk) = j * num_blocks_col + bk
// Each block: magnitude[block_idx*2+0] = bits 0-63, magnitude[block_idx*2+1] = bits 64-127

void matmul_bitnet_g128(float *A, G128Matrix *B_T, float *C, int M, int K, int N) {
    int nkb = (int)B_T->num_blocks_col;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            int row_base = j * nkb;
            for (int bk = 0; bk < nkb; bk++) {
                int bidx = row_base + bk;
                uint64_t mag0 = B_T->magnitude[bidx * 2 + 0];
                uint64_t mag1 = B_T->magnitude[bidx * 2 + 1];
                uint64_t sgn0 = B_T->sign[bidx * 2 + 0];
                uint64_t sgn1 = B_T->sign[bidx * 2 + 1];
                float scale = half_to_float(B_T->scales[bidx]);
                int k_base = bk * G128_BLOCK_SIZE;

                uint64_t m0 = mag0;
                while (m0) {
                    int bit = __builtin_ctzll(m0);
                    float sv = ((sgn0 >> bit) & 1) ? -scale : scale;
                    sum += A[i * K + k_base + bit] * sv;
                    m0 &= m0 - 1;
                }
                uint64_t m1 = mag1;
                while (m1) {
                    int bit = __builtin_ctzll(m1);
                    float sv = ((sgn1 >> bit) & 1) ? -scale : scale;
                    sum += A[i * K + k_base + 64 + bit] * sv;
                    m1 &= m1 - 1;
                }
            }
            C[i * N + j] = sum;
        }
    }
}

#ifdef __ARM_NEON

// 4-bit nibble → which of 4 elements are non-zero (0xFFFFFFFF) or zero (0x0)
static const uint32_t mag_lut[16][4] = {
    {0,0,0,0},         {~0u,0,0,0},       {0,~0u,0,0},       {~0u,~0u,0,0},
    {0,0,~0u,0},       {~0u,0,~0u,0},     {0,~0u,~0u,0},     {~0u,~0u,~0u,0},
    {0,0,0,~0u},       {~0u,0,0,~0u},     {0,~0u,0,~0u},     {~0u,~0u,0,~0u},
    {0,0,~0u,~0u},     {~0u,0,~0u,~0u},   {0,~0u,~0u,~0u},   {~0u,~0u,~0u,~0u}
};

// Decode 4-bit magnitude/sign nibbles into ±scale NEON vector (zeroed where mag=0) and accumulate
#define LUT_ACCUM(acc, mag_w, sgn_w, b, neg_sc, pos_sc, av) do { \
    uint8_t _mn = ((uint64_t)(mag_w) >> (b)) & 0xF; \
    uint8_t _sn = ((uint64_t)(sgn_w) >> (b)) & 0xF; \
    float32x4_t _sg = vbslq_f32(vld1q_u32(mag_lut[_sn]), (neg_sc), (pos_sc)); \
    float32x4_t _w  = vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(_sg), vld1q_u32(mag_lut[_mn]))); \
    (acc) = vmlaq_f32((acc), _w, (av)); \
} while(0)

void matmul_simd_g128(float *A, G128Matrix *B_T, float *C, int M, int K, int N) {
    int nkb = (int)B_T->num_blocks_col;
    const float *sf = B_T->scales_f32;
    for (int i = 0; i < M; i++) {
        int n4 = (N / 4) * 4;
        // 4-output parallel: A block loaded once, reused across 4 weight rows
        #pragma omp parallel for schedule(static) if(n4 >= 512)
        for (int j = 0; j < n4; j += 4) {
            float32x4_t acc0 = vdupq_n_f32(0.0f);
            float32x4_t acc1 = vdupq_n_f32(0.0f);
            float32x4_t acc2 = vdupq_n_f32(0.0f);
            float32x4_t acc3 = vdupq_n_f32(0.0f);
            for (int bk = 0; bk < nkb; bk++) {
                int bidx0 = (j+0)*nkb+bk, bidx1 = (j+1)*nkb+bk;
                int bidx2 = (j+2)*nkb+bk, bidx3 = (j+3)*nkb+bk;
                uint64_t m00=B_T->magnitude[bidx0*2+0], m01=B_T->magnitude[bidx0*2+1];
                uint64_t s00=B_T->sign[bidx0*2+0],      s01=B_T->sign[bidx0*2+1];
                uint64_t m10=B_T->magnitude[bidx1*2+0], m11=B_T->magnitude[bidx1*2+1];
                uint64_t s10=B_T->sign[bidx1*2+0],      s11=B_T->sign[bidx1*2+1];
                uint64_t m20=B_T->magnitude[bidx2*2+0], m21=B_T->magnitude[bidx2*2+1];
                uint64_t s20=B_T->sign[bidx2*2+0],      s21=B_T->sign[bidx2*2+1];
                uint64_t m30=B_T->magnitude[bidx3*2+0], m31=B_T->magnitude[bidx3*2+1];
                uint64_t s30=B_T->sign[bidx3*2+0],      s31=B_T->sign[bidx3*2+1];
                float sc0 = sf[bidx0], sc1 = sf[bidx1];
                float sc2 = sf[bidx2], sc3 = sf[bidx3];
                float32x4_t ns0=vdupq_n_f32(-sc0), ps0=vdupq_n_f32(sc0);
                float32x4_t ns1=vdupq_n_f32(-sc1), ps1=vdupq_n_f32(sc1);
                float32x4_t ns2=vdupq_n_f32(-sc2), ps2=vdupq_n_f32(sc2);
                float32x4_t ns3=vdupq_n_f32(-sc3), ps3=vdupq_n_f32(sc3);
                const float *ap = &A[i * K + bk * G128_BLOCK_SIZE];
                for (int b = 0; b < 64; b += 4) {
                    float32x4_t av = vld1q_f32(ap + b);
                    LUT_ACCUM(acc0, m00, s00, b, ns0, ps0, av);
                    LUT_ACCUM(acc1, m10, s10, b, ns1, ps1, av);
                    LUT_ACCUM(acc2, m20, s20, b, ns2, ps2, av);
                    LUT_ACCUM(acc3, m30, s30, b, ns3, ps3, av);
                }
                for (int b = 0; b < 64; b += 4) {
                    float32x4_t av = vld1q_f32(ap + 64 + b);
                    LUT_ACCUM(acc0, m01, s01, b, ns0, ps0, av);
                    LUT_ACCUM(acc1, m11, s11, b, ns1, ps1, av);
                    LUT_ACCUM(acc2, m21, s21, b, ns2, ps2, av);
                    LUT_ACCUM(acc3, m31, s31, b, ns3, ps3, av);
                }
            }
            C[i*N+j+0] = vaddvq_f32(acc0);
            C[i*N+j+1] = vaddvq_f32(acc1);
            C[i*N+j+2] = vaddvq_f32(acc2);
            C[i*N+j+3] = vaddvq_f32(acc3);
        }
        // Tail for N not divisible by 4
        for (int j = n4; j < N; j++) {
            float32x4_t acc = vdupq_n_f32(0.0f);
            int rb = j * nkb;
            for (int bk = 0; bk < nkb; bk++) {
                int bidx = rb + bk;
                uint64_t mag0=B_T->magnitude[bidx*2+0], mag1=B_T->magnitude[bidx*2+1];
                uint64_t sgn0=B_T->sign[bidx*2+0],      sgn1=B_T->sign[bidx*2+1];
                float sc = sf[bidx];
                float32x4_t ns=vdupq_n_f32(-sc), ps=vdupq_n_f32(sc);
                const float *ap = &A[i * K + bk * G128_BLOCK_SIZE];
                for (int b = 0; b < 64; b += 4)
                    LUT_ACCUM(acc, mag0, sgn0, b, ns, ps, vld1q_f32(ap + b));
                for (int b = 0; b < 64; b += 4)
                    LUT_ACCUM(acc, mag1, sgn1, b, ns, ps, vld1q_f32(ap + 64 + b));
            }
            C[i*N+j] = vaddvq_f32(acc);
        }
    }
}

#elif defined(__AVX2__)
#include <immintrin.h>

static inline float hsum_ps(__m128 v) {
    __m128 shuf = _mm_movehdup_ps(v);
    __m128 sums = _mm_add_ps(v, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    return _mm_cvtss_f32(_mm_add_ss(sums, shuf));
}

static const uint32_t mag_lut_x86[16][4] __attribute__((aligned(16))) = {
    {0,0,0,0},       {~0u,0,0,0},     {0,~0u,0,0},     {~0u,~0u,0,0},
    {0,0,~0u,0},     {~0u,0,~0u,0},   {0,~0u,~0u,0},   {~0u,~0u,~0u,0},
    {0,0,0,~0u},     {~0u,0,0,~0u},   {0,~0u,0,~0u},   {~0u,~0u,0,~0u},
    {0,0,~0u,~0u},   {~0u,0,~0u,~0u}, {0,~0u,~0u,~0u}, {~0u,~0u,~0u,~0u}
};

/* Decode 4-bit magnitude/sign nibbles into ±scale SSE vector and accumulate.
   blendv_ps selects neg_sc where sign mask MSB=1, pos_sc where MSB=0.
   and_ps zeros out elements where magnitude=0. */
#define LUT_ACCUM_X86(acc, mag_w, sgn_w, b, neg_sc, pos_sc, av) do { \
    uint8_t _mn = ((uint64_t)(mag_w) >> (b)) & 0xF; \
    uint8_t _sn = ((uint64_t)(sgn_w) >> (b)) & 0xF; \
    __m128 _sg = _mm_blendv_ps((pos_sc), (neg_sc), \
        _mm_castsi128_ps(_mm_load_si128((const __m128i*)mag_lut_x86[_sn]))); \
    __m128 _w  = _mm_and_ps(_sg, \
        _mm_castsi128_ps(_mm_load_si128((const __m128i*)mag_lut_x86[_mn]))); \
    (acc) = _mm_add_ps(_mm_mul_ps(_w, (av)), (acc)); \
} while(0)

void matmul_simd_g128(float *A, G128Matrix *B_T, float *C, int M, int K, int N) {
    int nkb = (int)B_T->num_blocks_col;
    const float *sf = B_T->scales_f32;
    for (int i = 0; i < M; i++) {
        int n4 = (N / 4) * 4;
        #pragma omp parallel for schedule(static) if(n4 >= 512)
        for (int j = 0; j < n4; j += 4) {
            __m128 acc0 = _mm_setzero_ps(), acc1 = _mm_setzero_ps();
            __m128 acc2 = _mm_setzero_ps(), acc3 = _mm_setzero_ps();
            for (int bk = 0; bk < nkb; bk++) {
                int bidx0=(j+0)*nkb+bk, bidx1=(j+1)*nkb+bk;
                int bidx2=(j+2)*nkb+bk, bidx3=(j+3)*nkb+bk;
                uint64_t m00=B_T->magnitude[bidx0*2+0], m01=B_T->magnitude[bidx0*2+1];
                uint64_t s00=B_T->sign[bidx0*2+0],      s01=B_T->sign[bidx0*2+1];
                uint64_t m10=B_T->magnitude[bidx1*2+0], m11=B_T->magnitude[bidx1*2+1];
                uint64_t s10=B_T->sign[bidx1*2+0],      s11=B_T->sign[bidx1*2+1];
                uint64_t m20=B_T->magnitude[bidx2*2+0], m21=B_T->magnitude[bidx2*2+1];
                uint64_t s20=B_T->sign[bidx2*2+0],      s21=B_T->sign[bidx2*2+1];
                uint64_t m30=B_T->magnitude[bidx3*2+0], m31=B_T->magnitude[bidx3*2+1];
                uint64_t s30=B_T->sign[bidx3*2+0],      s31=B_T->sign[bidx3*2+1];
                float sc0=sf[bidx0], sc1=sf[bidx1], sc2=sf[bidx2], sc3=sf[bidx3];
                __m128 ns0=_mm_set1_ps(-sc0), ps0=_mm_set1_ps(sc0);
                __m128 ns1=_mm_set1_ps(-sc1), ps1=_mm_set1_ps(sc1);
                __m128 ns2=_mm_set1_ps(-sc2), ps2=_mm_set1_ps(sc2);
                __m128 ns3=_mm_set1_ps(-sc3), ps3=_mm_set1_ps(sc3);
                const float *ap = &A[i * K + bk * G128_BLOCK_SIZE];
                for (int b = 0; b < 64; b += 4) {
                    __m128 av = _mm_loadu_ps(ap + b);
                    LUT_ACCUM_X86(acc0, m00, s00, b, ns0, ps0, av);
                    LUT_ACCUM_X86(acc1, m10, s10, b, ns1, ps1, av);
                    LUT_ACCUM_X86(acc2, m20, s20, b, ns2, ps2, av);
                    LUT_ACCUM_X86(acc3, m30, s30, b, ns3, ps3, av);
                }
                for (int b = 0; b < 64; b += 4) {
                    __m128 av = _mm_loadu_ps(ap + 64 + b);
                    LUT_ACCUM_X86(acc0, m01, s01, b, ns0, ps0, av);
                    LUT_ACCUM_X86(acc1, m11, s11, b, ns1, ps1, av);
                    LUT_ACCUM_X86(acc2, m21, s21, b, ns2, ps2, av);
                    LUT_ACCUM_X86(acc3, m31, s31, b, ns3, ps3, av);
                }
            }
            C[i*N+j+0] = hsum_ps(acc0); C[i*N+j+1] = hsum_ps(acc1);
            C[i*N+j+2] = hsum_ps(acc2); C[i*N+j+3] = hsum_ps(acc3);
        }
        for (int j = n4; j < N; j++) {
            __m128 acc = _mm_setzero_ps();
            int rb = j * nkb;
            for (int bk = 0; bk < nkb; bk++) {
                int bidx = rb + bk;
                uint64_t mag0=B_T->magnitude[bidx*2+0], mag1=B_T->magnitude[bidx*2+1];
                uint64_t sgn0=B_T->sign[bidx*2+0],      sgn1=B_T->sign[bidx*2+1];
                float sc = sf[bidx];
                __m128 ns=_mm_set1_ps(-sc), ps=_mm_set1_ps(sc);
                const float *ap = &A[i * K + bk * G128_BLOCK_SIZE];
                for (int b = 0; b < 64; b += 4)
                    LUT_ACCUM_X86(acc, mag0, sgn0, b, ns, ps, _mm_loadu_ps(ap + b));
                for (int b = 0; b < 64; b += 4)
                    LUT_ACCUM_X86(acc, mag1, sgn1, b, ns, ps, _mm_loadu_ps(ap + 64 + b));
            }
            C[i*N+j] = hsum_ps(acc);
        }
    }
}

#else  // portable scalar fallback

void matmul_simd_g128(float *A, G128Matrix *B_T, float *C, int M, int K, int N) {
    matmul_swar_g128(A, B_T, C, M, K, N);
}

#endif

void matmul_swar_g128(float *A, G128Matrix *B_T, float *C, int M, int K, int N) {
    int nkb = (int)B_T->num_blocks_col;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            int row_base = j * nkb;
            for (int bk = 0; bk < nkb; bk++) {
                int bidx = row_base + bk;
                uint64_t mag0 = B_T->magnitude[bidx * 2 + 0];
                uint64_t mag1 = B_T->magnitude[bidx * 2 + 1];
                uint64_t sgn0 = B_T->sign[bidx * 2 + 0];
                uint64_t sgn1 = B_T->sign[bidx * 2 + 1];
                float scale = half_to_float(B_T->scales[bidx]);
                int k_base = bk * G128_BLOCK_SIZE;

                // Separate positive (mag=1, sign=0) and negative (mag=1, sign=1) bits
                // to allow independent aggregation with a single scale multiply.
                float sum_pos = 0.0f, sum_neg = 0.0f;

                uint64_t pos0 = mag0 & ~sgn0;
                while (pos0) { int bit = __builtin_ctzll(pos0); sum_pos += A[i * K + k_base + bit]; pos0 &= pos0 - 1; }
                uint64_t pos1 = mag1 & ~sgn1;
                while (pos1) { int bit = __builtin_ctzll(pos1); sum_pos += A[i * K + k_base + 64 + bit]; pos1 &= pos1 - 1; }
                uint64_t neg0 = mag0 & sgn0;
                while (neg0) { int bit = __builtin_ctzll(neg0); sum_neg += A[i * K + k_base + bit]; neg0 &= neg0 - 1; }
                uint64_t neg1 = mag1 & sgn1;
                while (neg1) { int bit = __builtin_ctzll(neg1); sum_neg += A[i * K + k_base + 64 + bit]; neg1 &= neg1 - 1; }

                sum += scale * (sum_pos - sum_neg);
            }
            C[i * N + j] = sum;
        }
    }
}

static uint8_t LUT_POS[256][256]; // LUT_POS[mag_byte][A_byte] not useful for float; LUT used for popcount
static int lut_initialized = 0;

void matmul_lut_g128(float *A, G128Matrix *B_T, float *C, int M, int K, int N) {
    (void)LUT_POS; // suppress unused warning
    int nkb = (int)B_T->num_blocks_col;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            int row_base = j * nkb;
            for (int bk = 0; bk < nkb; bk++) {
                int bidx = row_base + bk;
                uint64_t mag0 = B_T->magnitude[bidx * 2 + 0];
                uint64_t mag1 = B_T->magnitude[bidx * 2 + 1];
                uint64_t sgn0 = B_T->sign[bidx * 2 + 0];
                uint64_t sgn1 = B_T->sign[bidx * 2 + 1];
                float scale = half_to_float(B_T->scales[bidx]);
                int k_base = bk * G128_BLOCK_SIZE;

                // Process 8 bits at a time using byte-level LUT approach
                for (int byte = 0; byte < 8; byte++) {
                    uint8_t mb = (mag0 >> (byte * 8)) & 0xFF;
                    uint8_t sb = (sgn0 >> (byte * 8)) & 0xFF;
                    int base = k_base + byte * 8;
                    uint8_t m = mb;
                    while (m) {
                        int bit = __builtin_ctz(m);
                        float sv = ((sb >> bit) & 1) ? -scale : scale;
                        sum += A[i * K + base + bit] * sv;
                        m &= m - 1;
                    }
                }
                for (int byte = 0; byte < 8; byte++) {
                    uint8_t mb = (mag1 >> (byte * 8)) & 0xFF;
                    uint8_t sb = (sgn1 >> (byte * 8)) & 0xFF;
                    int base = k_base + 64 + byte * 8;
                    uint8_t m = mb;
                    while (m) {
                        int bit = __builtin_ctz(m);
                        float sv = ((sb >> bit) & 1) ? -scale : scale;
                        sum += A[i * K + base + bit] * sv;
                        m &= m - 1;
                    }
                }
            }
            C[i * N + j] = sum;
        }
    }
}

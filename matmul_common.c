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

#elif defined(__AVX512F__)

#include <immintrin.h>

static uint32_t avx512_mag_lut[256][16] __attribute__((aligned(64)));
static int avx512_lut_init = 0;

static void init_avx512_lut_once(void) {
    if (avx512_lut_init) return;
    AVX512_SIGN_MASK = _mm256_castsi256_ps(_mm256_set1_epi32((int)0x80000000));
    for (int i = 0; i < 256; i++)
        for (int j = 0; j < 16; j++)
            avx512_mag_lut[i][j] = ((i >> j) & 1) ? 0xFFFFFFFF : 0;
    avx512_lut_init = 1;
}

static __m256 AVX512_SIGN_MASK;

// acc = FMA(av, w, acc) where w = decode(mag[16bits], sgn[16bits], ps, ns)
// ps = +scale broadcast, ns = -scale broadcast (precomputed)
#define LUT_ACCUM_ZMM(acc, mag_w, sgn_w, b, ps256, ns256, av) do { \
    uint8_t _mn_lo = ((uint64_t)(mag_w) >> (b)) & 0xFF; \
    uint8_t _sn_lo = ((uint64_t)(sgn_w) >> (b)) & 0xFF; \
    uint8_t _mn_hi = ((uint64_t)(mag_w) >> ((b)+8)) & 0xFF; \
    uint8_t _sn_hi = ((uint64_t)(sgn_w) >> ((b)+8)) & 0xFF; \
    __m256 _sg_lo = _mm256_blendv_ps((ps256), (ns256), _mm256_castsi256_ps(_mm256_load_si256((const __m256i*)avx512_mag_lut[_sn_lo]))); \
    __m256 _w_lo  = _mm256_and_ps(_sg_lo, _mm256_castsi256_ps(_mm256_load_si256((const __m256i*)avx512_mag_lut[_mn_lo]))); \
    __m256 _sg_hi = _mm256_blendv_ps((ps256), (ns256), _mm256_castsi256_ps(_mm256_load_si256((const __m256i*)avx512_mag_lut[_sn_hi]))); \
    __m256 _w_hi  = _mm256_and_ps(_sg_hi, _mm256_castsi256_ps(_mm256_load_si256((const __m256i*)avx512_mag_lut[_mn_hi]))); \
    __m512 _w = _mm512_insertf32x8(_mm512_castps256_ps512(_w_lo), _w_hi, 1); \
    (acc) = _mm512_fmadd_ps(_w, (av), (acc)); \
} while(0)

// Horizontal sum across ZMM register
static inline float hsum_zmm(__m512 v) {
    __m256 lo = _mm512_castps512_ps256(v);
    __m256 hi = _mm512_extractf32x8_ps(v, 1);
    __m256 s = _mm256_add_ps(lo, hi);
    __m128 s4 = _mm_add_ps(_mm256_castps256_ps128(s), _mm256_extractf128_ps(s, 1));
    s4 = _mm_hadd_ps(s4, s4);
    s4 = _mm_hadd_ps(s4, s4);
    return _mm_cvtss_f32(s4);
}

void matmul_simd_g128(float *A, G128Matrix *B_T, float *C, int M, int K, int N) {
    init_avx512_lut_once();
    int nkb = (int)B_T->num_blocks_col;
    const float *sf = B_T->scales_f32;
    for (int i = 0; i < M; i++) {
        int n8 = (N / 8) * 8;
        #pragma omp parallel for schedule(static) if(n8 >= 512)
        for (int j = 0; j < n8; j += 8) {
            __m512 acc0 = _mm512_setzero_ps();
            __m512 acc1 = _mm512_setzero_ps();
            __m512 acc2 = _mm512_setzero_ps();
            __m512 acc3 = _mm512_setzero_ps();
            __m512 acc4 = _mm512_setzero_ps();
            __m512 acc5 = _mm512_setzero_ps();
            __m512 acc6 = _mm512_setzero_ps();
            __m512 acc7 = _mm512_setzero_ps();
            for (int bk = 0; bk < nkb; bk++) {
                int b0=(j+0)*nkb+bk, b1=(j+1)*nkb+bk;
                int b2=(j+2)*nkb+bk, b3=(j+3)*nkb+bk;
                int b4=(j+4)*nkb+bk, b5=(j+5)*nkb+bk;
                int b6=(j+6)*nkb+bk, b7=(j+7)*nkb+bk;
                uint64_t m00=B_T->magnitude[b0*2+0], m01=B_T->magnitude[b0*2+1];
                uint64_t s00=B_T->sign[b0*2+0],      s01=B_T->sign[b0*2+1];
                uint64_t m10=B_T->magnitude[b1*2+0], m11=B_T->magnitude[b1*2+1];
                uint64_t s10=B_T->sign[b1*2+0],      s11=B_T->sign[b1*2+1];
                uint64_t m20=B_T->magnitude[b2*2+0], m21=B_T->magnitude[b2*2+1];
                uint64_t s20=B_T->sign[b2*2+0],      s21=B_T->sign[b2*2+1];
                uint64_t m30=B_T->magnitude[b3*2+0], m31=B_T->magnitude[b3*2+1];
                uint64_t s30=B_T->sign[b3*2+0],      s31=B_T->sign[b3*2+1];
                uint64_t m40=B_T->magnitude[b4*2+0], m41=B_T->magnitude[b4*2+1];
                uint64_t s40=B_T->sign[b4*2+0],      s41=B_T->sign[b4*2+1];
                uint64_t m50=B_T->magnitude[b5*2+0], m51=B_T->magnitude[b5*2+1];
                uint64_t s50=B_T->sign[b5*2+0],      s51=B_T->sign[b5*2+1];
                uint64_t m60=B_T->magnitude[b6*2+0], m61=B_T->magnitude[b6*2+1];
                uint64_t s60=B_T->sign[b6*2+0],      s61=B_T->sign[b6*2+1];
                uint64_t m70=B_T->magnitude[b7*2+0], m71=B_T->magnitude[b7*2+1];
                uint64_t s70=B_T->sign[b7*2+0],      s71=B_T->sign[b7*2+1];
                float sc0=sf[b0], sc1=sf[b1], sc2=sf[b2], sc3=sf[b3];
                float sc4=sf[b4], sc5=sf[b5], sc6=sf[b6], sc7=sf[b7];
                __m256 ps0=_mm256_set1_ps(sc0), ns0=_mm256_xor_ps(ps0, AVX512_SIGN_MASK);
                __m256 ps1=_mm256_set1_ps(sc1), ns1=_mm256_xor_ps(ps1, AVX512_SIGN_MASK);
                __m256 ps2=_mm256_set1_ps(sc2), ns2=_mm256_xor_ps(ps2, AVX512_SIGN_MASK);
                __m256 ps3=_mm256_set1_ps(sc3), ns3=_mm256_xor_ps(ps3, AVX512_SIGN_MASK);
                __m256 ps4=_mm256_set1_ps(sc4), ns4=_mm256_xor_ps(ps4, AVX512_SIGN_MASK);
                __m256 ps5=_mm256_set1_ps(sc5), ns5=_mm256_xor_ps(ps5, AVX512_SIGN_MASK);
                __m256 ps6=_mm256_set1_ps(sc6), ns6=_mm256_xor_ps(ps6, AVX512_SIGN_MASK);
                __m256 ps7=_mm256_set1_ps(sc7), ns7=_mm256_xor_ps(ps7, AVX512_SIGN_MASK);
                const float *ap = &A[i * K + bk * G128_BLOCK_SIZE];
                // Prefetch next block's weight data
                if (bk + 1 < nkb) {
                    int pf_bidx = j * nkb + bk + 1;
                    _mm_prefetch((const char*)&B_T->magnitude[pf_bidx * 2], _MM_HINT_T1);
                    _mm_prefetch((const char*)&B_T->sign[pf_bidx * 2], _MM_HINT_T1);
                    _mm_prefetch((const char*)&sf[pf_bidx], _MM_HINT_T1);
                }
                for (int b = 0; b < 64; b += 16) {
                    __m512 av = _mm512_load_ps(ap + b);
                    LUT_ACCUM_ZMM(acc0, m00, s00, b, ps0, ns0, av);
                    LUT_ACCUM_ZMM(acc1, m10, s10, b, ps1, ns1, av);
                    LUT_ACCUM_ZMM(acc2, m20, s20, b, ps2, ns2, av);
                    LUT_ACCUM_ZMM(acc3, m30, s30, b, ps3, ns3, av);
                    LUT_ACCUM_ZMM(acc4, m40, s40, b, ps4, ns4, av);
                    LUT_ACCUM_ZMM(acc5, m50, s50, b, ps5, ns5, av);
                    LUT_ACCUM_ZMM(acc6, m60, s60, b, ps6, ns6, av);
                    LUT_ACCUM_ZMM(acc7, m70, s70, b, ps7, ns7, av);
                }
                for (int b = 0; b < 64; b += 16) {
                    __m512 av = _mm512_load_ps(ap + 64 + b);
                    LUT_ACCUM_ZMM(acc0, m01, s01, b, ps0, ns0, av);
                    LUT_ACCUM_ZMM(acc1, m11, s11, b, ps1, ns1, av);
                    LUT_ACCUM_ZMM(acc2, m21, s21, b, ps2, ns2, av);
                    LUT_ACCUM_ZMM(acc3, m31, s31, b, ps3, ns3, av);
                    LUT_ACCUM_ZMM(acc4, m41, s41, b, ps4, ns4, av);
                    LUT_ACCUM_ZMM(acc5, m51, s51, b, ps5, ns5, av);
                    LUT_ACCUM_ZMM(acc6, m61, s61, b, ps6, ns6, av);
                    LUT_ACCUM_ZMM(acc7, m71, s71, b, ps7, ns7, av);
                }
            }
            C[i*N+j+0]=hsum_zmm(acc0); C[i*N+j+1]=hsum_zmm(acc1);
            C[i*N+j+2]=hsum_zmm(acc2); C[i*N+j+3]=hsum_zmm(acc3);
            C[i*N+j+4]=hsum_zmm(acc4); C[i*N+j+5]=hsum_zmm(acc5);
            C[i*N+j+6]=hsum_zmm(acc6); C[i*N+j+7]=hsum_zmm(acc7);
        }
        for (int j = n8; j < N; j++) {
            __m512 acc = _mm512_setzero_ps();
            int rb = j * nkb;
            for (int bk = 0; bk < nkb; bk++) {
                int bidx = rb + bk;
                uint64_t mag0=B_T->magnitude[bidx*2+0], mag1=B_T->magnitude[bidx*2+1];
                uint64_t sgn0=B_T->sign[bidx*2+0],      sgn1=B_T->sign[bidx*2+1];
                __m256 ps = _mm256_set1_ps(sf[bidx]);
                __m256 ns = _mm256_xor_ps(ps, AVX512_SIGN_MASK);
                const float *ap = &A[i * K + bk * G128_BLOCK_SIZE];
                for (int b = 0; b < 64; b += 16)
                    LUT_ACCUM_ZMM(acc, mag0, sgn0, b, ps, ns, _mm512_load_ps(ap + b));
                for (int b = 0; b < 64; b += 16)
                    LUT_ACCUM_ZMM(acc, mag1, sgn1, b, ps, ns, _mm512_load_ps(ap + 64 + b));
            }
            C[i*N+j] = hsum_zmm(acc);
        }
    }
}

// lm_head prefilter: approximate scores using only the first `max_blocks` K-blocks
// (1 block = 128 dimensions, full K = 2048 = 16 blocks)
// Writes N approximate scores to C.
void lm_head_prefilter(float *A, G128Matrix *B_T, float *C, int N, int max_blocks) {
    init_avx512_lut_once();
    int nkb = (int)B_T->num_blocks_col;  // full K / 128
    if (max_blocks <= 0 || max_blocks > nkb) max_blocks = nkb;
    const float *sf = B_T->scales_f32;
    for (int i = 0; i < 1; i++) {  // M=1 for lm_head
        const float *ap = A;
        int n8 = (N / 8) * 8;
        #pragma omp parallel for schedule(static) if(n8 >= 512)
        for (int j = 0; j < n8; j += 8) {
            __m512 acc0 = _mm512_setzero_ps();
            __m512 acc1 = _mm512_setzero_ps();
            __m512 acc2 = _mm512_setzero_ps();
            __m512 acc3 = _mm512_setzero_ps();
            __m512 acc4 = _mm512_setzero_ps();
            __m512 acc5 = _mm512_setzero_ps();
            __m512 acc6 = _mm512_setzero_ps();
            __m512 acc7 = _mm512_setzero_ps();
            for (int bk = 0; bk < max_blocks; bk++) {
                int b0=(j+0)*nkb+bk, b1=(j+1)*nkb+bk;
                int b2=(j+2)*nkb+bk, b3=(j+3)*nkb+bk;
                int b4=(j+4)*nkb+bk, b5=(j+5)*nkb+bk;
                int b6=(j+6)*nkb+bk, b7=(j+7)*nkb+bk;
                uint64_t m00=B_T->magnitude[b0*2+0], m01=B_T->magnitude[b0*2+1];
                uint64_t s00=B_T->sign[b0*2+0],      s01=B_T->sign[b0*2+1];
                uint64_t m10=B_T->magnitude[b1*2+0], m11=B_T->magnitude[b1*2+1];
                uint64_t s10=B_T->sign[b1*2+0],      s11=B_T->sign[b1*2+1];
                uint64_t m20=B_T->magnitude[b2*2+0], m21=B_T->magnitude[b2*2+1];
                uint64_t s20=B_T->sign[b2*2+0],      s21=B_T->sign[b2*2+1];
                uint64_t m30=B_T->magnitude[b3*2+0], m31=B_T->magnitude[b3*2+1];
                uint64_t s30=B_T->sign[b3*2+0],      s31=B_T->sign[b3*2+1];
                uint64_t m40=B_T->magnitude[b4*2+0], m41=B_T->magnitude[b4*2+1];
                uint64_t s40=B_T->sign[b4*2+0],      s41=B_T->sign[b4*2+1];
                uint64_t m50=B_T->magnitude[b5*2+0], m51=B_T->magnitude[b5*2+1];
                uint64_t s50=B_T->sign[b5*2+0],      s51=B_T->sign[b5*2+1];
                uint64_t m60=B_T->magnitude[b6*2+0], m61=B_T->magnitude[b6*2+1];
                uint64_t s60=B_T->sign[b6*2+0],      s61=B_T->sign[b6*2+1];
                uint64_t m70=B_T->magnitude[b7*2+0], m71=B_T->magnitude[b7*2+1];
                uint64_t s70=B_T->sign[b7*2+0],      s71=B_T->sign[b7*2+1];
                float sc0=sf[b0], sc1=sf[b1], sc2=sf[b2], sc3=sf[b3];
                float sc4=sf[b4], sc5=sf[b5], sc6=sf[b6], sc7=sf[b7];
                __m256 ps0=_mm256_set1_ps(sc0), ns0=_mm256_xor_ps(ps0, AVX512_SIGN_MASK);
                __m256 ps1=_mm256_set1_ps(sc1), ns1=_mm256_xor_ps(ps1, AVX512_SIGN_MASK);
                __m256 ps2=_mm256_set1_ps(sc2), ns2=_mm256_xor_ps(ps2, AVX512_SIGN_MASK);
                __m256 ps3=_mm256_set1_ps(sc3), ns3=_mm256_xor_ps(ps3, AVX512_SIGN_MASK);
                __m256 ps4=_mm256_set1_ps(sc4), ns4=_mm256_xor_ps(ps4, AVX512_SIGN_MASK);
                __m256 ps5=_mm256_set1_ps(sc5), ns5=_mm256_xor_ps(ps5, AVX512_SIGN_MASK);
                __m256 ps6=_mm256_set1_ps(sc6), ns6=_mm256_xor_ps(ps6, AVX512_SIGN_MASK);
                __m256 ps7=_mm256_set1_ps(sc7), ns7=_mm256_xor_ps(ps7, AVX512_SIGN_MASK);
                const float *bk_ap = &ap[bk * G128_BLOCK_SIZE];
                for (int b = 0; b < 64; b += 16) {
                    __m512 av = _mm512_load_ps(&bk_ap[b]);
                    LUT_ACCUM_ZMM(acc0, m00, s00, b, ps0, ns0, av);
                    LUT_ACCUM_ZMM(acc1, m10, s10, b, ps1, ns1, av);
                    LUT_ACCUM_ZMM(acc2, m20, s20, b, ps2, ns2, av);
                    LUT_ACCUM_ZMM(acc3, m30, s30, b, ps3, ns3, av);
                    LUT_ACCUM_ZMM(acc4, m40, s40, b, ps4, ns4, av);
                    LUT_ACCUM_ZMM(acc5, m50, s50, b, ps5, ns5, av);
                    LUT_ACCUM_ZMM(acc6, m60, s60, b, ps6, ns6, av);
                    LUT_ACCUM_ZMM(acc7, m70, s70, b, ps7, ns7, av);
                }
                for (int b = 0; b < 64; b += 16) {
                    __m512 av = _mm512_load_ps(&bk_ap[64 + b]);
                    LUT_ACCUM_ZMM(acc0, m01, s01, b, ps0, ns0, av);
                    LUT_ACCUM_ZMM(acc1, m11, s11, b, ps1, ns1, av);
                    LUT_ACCUM_ZMM(acc2, m21, s21, b, ps2, ns2, av);
                    LUT_ACCUM_ZMM(acc3, m31, s31, b, ps3, ns3, av);
                    LUT_ACCUM_ZMM(acc4, m41, s41, b, ps4, ns4, av);
                    LUT_ACCUM_ZMM(acc5, m51, s51, b, ps5, ns5, av);
                    LUT_ACCUM_ZMM(acc6, m61, s61, b, ps6, ns6, av);
                    LUT_ACCUM_ZMM(acc7, m71, s71, b, ps7, ns7, av);
                }
            }
            C[j+0]=hsum_zmm(acc0); C[j+1]=hsum_zmm(acc1);
            C[j+2]=hsum_zmm(acc2); C[j+3]=hsum_zmm(acc3);
            C[j+4]=hsum_zmm(acc4); C[j+5]=hsum_zmm(acc5);
            C[j+6]=hsum_zmm(acc6); C[j+7]=hsum_zmm(acc7);
        }
        for (int j = n8; j < N; j++) {
            __m512 acc = _mm512_setzero_ps();
            int rb = j * nkb;
            for (int bk = 0; bk < max_blocks; bk++) {
                int bidx = rb + bk;
                uint64_t mag0=B_T->magnitude[bidx*2+0], mag1=B_T->magnitude[bidx*2+1];
                uint64_t sgn0=B_T->sign[bidx*2+0],      sgn1=B_T->sign[bidx*2+1];
                __m256 ps = _mm256_set1_ps(sf[bidx]);
                __m256 ns = _mm256_xor_ps(ps, AVX512_SIGN_MASK);
                const float *bk_ap = &ap[bk * G128_BLOCK_SIZE];
                for (int b = 0; b < 64; b += 16)
                    LUT_ACCUM_ZMM(acc, mag0, sgn0, b, ps, ns, _mm512_load_ps(&bk_ap[b]));
                for (int b = 0; b < 64; b += 16)
                    LUT_ACCUM_ZMM(acc, mag1, sgn1, b, ps, ns, _mm512_load_ps(&bk_ap[64 + b]));
            }
            C[j] = hsum_zmm(acc);
        }
    }
}

// Compute exact dot products for selected vocabulary rows only.
// sel_rows[j] maps to actual row r = sel_rows[j] in the weight matrix.
// Writes scores to C[i*N_full + r] for each selected row r.
void matmul_g128_selected(float *A, G128Matrix *B_T, float *C, int M, int K, int N_full, int N_sel, const int *sel_rows) {
    init_avx512_lut_once();
    int nkb = (int)B_T->num_blocks_col;
    const float *sf = B_T->scales_f32;
    for (int i = 0; i < M; i++) {
        int n8 = (N_sel / 8) * 8;
        #pragma omp parallel for schedule(static) if(n8 >= 128)
        for (int si = 0; si < n8; si += 8) {
            __m512 acc0 = _mm512_setzero_ps();
            __m512 acc1 = _mm512_setzero_ps();
            __m512 acc2 = _mm512_setzero_ps();
            __m512 acc3 = _mm512_setzero_ps();
            __m512 acc4 = _mm512_setzero_ps();
            __m512 acc5 = _mm512_setzero_ps();
            __m512 acc6 = _mm512_setzero_ps();
            __m512 acc7 = _mm512_setzero_ps();
            int r0 = sel_rows[si+0], r1 = sel_rows[si+1];
            int r2 = sel_rows[si+2], r3 = sel_rows[si+3];
            int r4 = sel_rows[si+4], r5 = sel_rows[si+5];
            int r6 = sel_rows[si+6], r7 = sel_rows[si+7];
            for (int bk = 0; bk < nkb; bk++) {
                int b0=r0*nkb+bk, b1=r1*nkb+bk;
                int b2=r2*nkb+bk, b3=r3*nkb+bk;
                int b4=r4*nkb+bk, b5=r5*nkb+bk;
                int b6=r6*nkb+bk, b7=r7*nkb+bk;
                uint64_t m00=B_T->magnitude[b0*2+0], m01=B_T->magnitude[b0*2+1];
                uint64_t s00=B_T->sign[b0*2+0],      s01=B_T->sign[b0*2+1];
                uint64_t m10=B_T->magnitude[b1*2+0], m11=B_T->magnitude[b1*2+1];
                uint64_t s10=B_T->sign[b1*2+0],      s11=B_T->sign[b1*2+1];
                uint64_t m20=B_T->magnitude[b2*2+0], m21=B_T->magnitude[b2*2+1];
                uint64_t s20=B_T->sign[b2*2+0],      s21=B_T->sign[b2*2+1];
                uint64_t m30=B_T->magnitude[b3*2+0], m31=B_T->magnitude[b3*2+1];
                uint64_t s30=B_T->sign[b3*2+0],      s31=B_T->sign[b3*2+1];
                uint64_t m40=B_T->magnitude[b4*2+0], m41=B_T->magnitude[b4*2+1];
                uint64_t s40=B_T->sign[b4*2+0],      s41=B_T->sign[b4*2+1];
                uint64_t m50=B_T->magnitude[b5*2+0], m51=B_T->magnitude[b5*2+1];
                uint64_t s50=B_T->sign[b5*2+0],      s51=B_T->sign[b5*2+1];
                uint64_t m60=B_T->magnitude[b6*2+0], m61=B_T->magnitude[b6*2+1];
                uint64_t s60=B_T->sign[b6*2+0],      s61=B_T->sign[b6*2+1];
                uint64_t m70=B_T->magnitude[b7*2+0], m71=B_T->magnitude[b7*2+1];
                uint64_t s70=B_T->sign[b7*2+0],      s71=B_T->sign[b7*2+1];
                float sc0=sf[b0], sc1=sf[b1], sc2=sf[b2], sc3=sf[b3];
                float sc4=sf[b4], sc5=sf[b5], sc6=sf[b6], sc7=sf[b7];
                __m256 ps0=_mm256_set1_ps(sc0), ns0=_mm256_xor_ps(ps0, AVX512_SIGN_MASK);
                __m256 ps1=_mm256_set1_ps(sc1), ns1=_mm256_xor_ps(ps1, AVX512_SIGN_MASK);
                __m256 ps2=_mm256_set1_ps(sc2), ns2=_mm256_xor_ps(ps2, AVX512_SIGN_MASK);
                __m256 ps3=_mm256_set1_ps(sc3), ns3=_mm256_xor_ps(ps3, AVX512_SIGN_MASK);
                __m256 ps4=_mm256_set1_ps(sc4), ns4=_mm256_xor_ps(ps4, AVX512_SIGN_MASK);
                __m256 ps5=_mm256_set1_ps(sc5), ns5=_mm256_xor_ps(ps5, AVX512_SIGN_MASK);
                __m256 ps6=_mm256_set1_ps(sc6), ns6=_mm256_xor_ps(ps6, AVX512_SIGN_MASK);
                __m256 ps7=_mm256_set1_ps(sc7), ns7=_mm256_xor_ps(ps7, AVX512_SIGN_MASK);
                const float *ap = &A[i * K + bk * G128_BLOCK_SIZE];
                for (int b = 0; b < 64; b += 16) {
                    __m512 av = _mm512_load_ps(ap + b);
                    LUT_ACCUM_ZMM(acc0, m00, s00, b, ps0, ns0, av);
                    LUT_ACCUM_ZMM(acc1, m10, s10, b, ps1, ns1, av);
                    LUT_ACCUM_ZMM(acc2, m20, s20, b, ps2, ns2, av);
                    LUT_ACCUM_ZMM(acc3, m30, s30, b, ps3, ns3, av);
                    LUT_ACCUM_ZMM(acc4, m40, s40, b, ps4, ns4, av);
                    LUT_ACCUM_ZMM(acc5, m50, s50, b, ps5, ns5, av);
                    LUT_ACCUM_ZMM(acc6, m60, s60, b, ps6, ns6, av);
                    LUT_ACCUM_ZMM(acc7, m70, s70, b, ps7, ns7, av);
                }
                for (int b = 0; b < 64; b += 16) {
                    __m512 av = _mm512_load_ps(ap + 64 + b);
                    LUT_ACCUM_ZMM(acc0, m01, s01, b, ps0, ns0, av);
                    LUT_ACCUM_ZMM(acc1, m11, s11, b, ps1, ns1, av);
                    LUT_ACCUM_ZMM(acc2, m21, s21, b, ps2, ns2, av);
                    LUT_ACCUM_ZMM(acc3, m31, s31, b, ps3, ns3, av);
                    LUT_ACCUM_ZMM(acc4, m41, s41, b, ps4, ns4, av);
                    LUT_ACCUM_ZMM(acc5, m51, s51, b, ps5, ns5, av);
                    LUT_ACCUM_ZMM(acc6, m61, s61, b, ps6, ns6, av);
                    LUT_ACCUM_ZMM(acc7, m71, s71, b, ps7, ns7, av);
                }
            }
            C[i*N_full + r0]=hsum_zmm(acc0); C[i*N_full + r1]=hsum_zmm(acc1);
            C[i*N_full + r2]=hsum_zmm(acc2); C[i*N_full + r3]=hsum_zmm(acc3);
            C[i*N_full + r4]=hsum_zmm(acc4); C[i*N_full + r5]=hsum_zmm(acc5);
            C[i*N_full + r6]=hsum_zmm(acc6); C[i*N_full + r7]=hsum_zmm(acc7);
        }
        for (int si = n8; si < N_sel; si++) {
            int r = sel_rows[si];
            __m512 acc = _mm512_setzero_ps();
            int rb = r * nkb;
            for (int bk = 0; bk < nkb; bk++) {
                int bidx = rb + bk;
                uint64_t mag0=B_T->magnitude[bidx*2+0], mag1=B_T->magnitude[bidx*2+1];
                uint64_t sgn0=B_T->sign[bidx*2+0],      sgn1=B_T->sign[bidx*2+1];
                __m256 ps = _mm256_set1_ps(sf[bidx]);
                __m256 ns = _mm256_xor_ps(ps, AVX512_SIGN_MASK);
                const float *ap = &A[i * K + bk * G128_BLOCK_SIZE];
                for (int b = 0; b < 64; b += 16)
                    LUT_ACCUM_ZMM(acc, mag0, sgn0, b, ps, ns, _mm512_load_ps(ap + b));
                for (int b = 0; b < 64; b += 16)
                    LUT_ACCUM_ZMM(acc, mag1, sgn1, b, ps, ns, _mm512_load_ps(ap + 64 + b));
            }
            C[i*N_full + r] = hsum_zmm(acc);
        }
    }
}

// Min-heap based top-K selection. O(N log K)
void find_top_k(float *scores, int N, int K, int *out_indices) {
    if (K > N) K = N;
    if (K <= 0) return;
    typedef struct { float score; int idx; } pair_t;
    pair_t *heap = (pair_t*)malloc((size_t)K * sizeof(pair_t));
    for (int i = 0; i < K; i++) {
        heap[i].score = scores[i];
        heap[i].idx = i;
    }
    for (int i = K/2 - 1; i >= 0; i--) {
        int p = i;
        while (1) {
            int smallest = p;
            int left = 2*p + 1, right = 2*p + 2;
            if (left < K && heap[left].score < heap[smallest].score) smallest = left;
            if (right < K && heap[right].score < heap[smallest].score) smallest = right;
            if (smallest == p) break;
            pair_t tmp = heap[p]; heap[p] = heap[smallest]; heap[smallest] = tmp;
            p = smallest;
        }
    }
    float min_top = heap[0].score;
    for (int i = K; i < N; i++) {
        if (scores[i] > min_top) {
            heap[0].score = scores[i];
            heap[0].idx = i;
            int p = 0;
            while (1) {
                int smallest = p;
                int left = 2*p + 1, right = 2*p + 2;
                if (left < K && heap[left].score < heap[smallest].score) smallest = left;
                if (right < K && heap[right].score < heap[smallest].score) smallest = right;
                if (smallest == p) break;
                pair_t tmp = heap[p]; heap[p] = heap[smallest]; heap[smallest] = tmp;
                p = smallest;
            }
            min_top = heap[0].score;
        }
    }
    for (int i = 0; i < K; i++) out_indices[i] = heap[i].idx;
    free(heap);
}

const int lm_head_prefilter_available = 1;

#elif defined(__AVX2__)

#include <immintrin.h>

// 8-bit LUT for AVX2: maps 8-bit pattern to 8 × uint32 mask
// Lane j = all-1s if bit j of index is set, else 0
static uint32_t avx2_mag_lut[256][8] __attribute__((aligned(64)));
static int avx2_lut_init = 0;

static void init_avx2_lut_once(void) {
    if (avx2_lut_init) return;
    for (int i = 0; i < 256; i++)
        for (int j = 0; j < 8; j++)
            avx2_mag_lut[i][j] = ((i >> j) & 1) ? 0xFFFFFFFF : 0;
    avx2_lut_init = 1;
}

// Horitzontal sum across 8-wide YMM → single float
static inline float hsum_avx2(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

#define LUT_ACCUM_AVX2(acc, mag_w, sgn_w, b, sc, av) do { \
    uint8_t _mn = ((uint64_t)(mag_w) >> (b)) & 0xFF; \
    uint8_t _sn = ((uint64_t)(sgn_w) >> (b)) & 0xFF; \
    __m256 _ps = (sc); \
    __m256 _ns = _mm256_xor_ps(_ps, _mm256_set1_ps(-0.0f)); \
    __m256 _sg = _mm256_blendv_ps(_ps, _ns, \
        _mm256_load_si256((const __m256i*)avx2_mag_lut[_sn])); \
    __m256 _w  = _mm256_and_ps(_sg, \
        _mm256_load_si256((const __m256i*)avx2_mag_lut[_mn])); \
    (acc) = _mm256_fmadd_ps(_w, (av), (acc)); \
} while(0)

void matmul_simd_g128(float *A, G128Matrix *B_T, float *C, int M, int K, int N) {
    init_avx2_lut_once();
    int nkb = (int)B_T->num_blocks_col;
    const float *sf = B_T->scales_f32;
    for (int i = 0; i < M; i++) {
        int n8 = (N / 8) * 8;
        #pragma omp parallel for schedule(static) if(n8 >= 512)
        for (int j = 0; j < n8; j += 8) {
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps();
            __m256 acc3 = _mm256_setzero_ps();
            __m256 acc4 = _mm256_setzero_ps();
            __m256 acc5 = _mm256_setzero_ps();
            __m256 acc6 = _mm256_setzero_ps();
            __m256 acc7 = _mm256_setzero_ps();
            for (int bk = 0; bk < nkb; bk++) {
                int b0=(j+0)*nkb+bk, b1=(j+1)*nkb+bk;
                int b2=(j+2)*nkb+bk, b3=(j+3)*nkb+bk;
                int b4=(j+4)*nkb+bk, b5=(j+5)*nkb+bk;
                int b6=(j+6)*nkb+bk, b7=(j+7)*nkb+bk;
                uint64_t m00=B_T->magnitude[b0*2+0], m01=B_T->magnitude[b0*2+1];
                uint64_t s00=B_T->sign[b0*2+0],      s01=B_T->sign[b0*2+1];
                uint64_t m10=B_T->magnitude[b1*2+0], m11=B_T->magnitude[b1*2+1];
                uint64_t s10=B_T->sign[b1*2+0],      s11=B_T->sign[b1*2+1];
                uint64_t m20=B_T->magnitude[b2*2+0], m21=B_T->magnitude[b2*2+1];
                uint64_t s20=B_T->sign[b2*2+0],      s21=B_T->sign[b2*2+1];
                uint64_t m30=B_T->magnitude[b3*2+0], m31=B_T->magnitude[b3*2+1];
                uint64_t s30=B_T->sign[b3*2+0],      s31=B_T->sign[b3*2+1];
                uint64_t m40=B_T->magnitude[b4*2+0], m41=B_T->magnitude[b4*2+1];
                uint64_t s40=B_T->sign[b4*2+0],      s41=B_T->sign[b4*2+1];
                uint64_t m50=B_T->magnitude[b5*2+0], m51=B_T->magnitude[b5*2+1];
                uint64_t s50=B_T->sign[b5*2+0],      s51=B_T->sign[b5*2+1];
                uint64_t m60=B_T->magnitude[b6*2+0], m61=B_T->magnitude[b6*2+1];
                uint64_t s60=B_T->sign[b6*2+0],      s61=B_T->sign[b6*2+1];
                uint64_t m70=B_T->magnitude[b7*2+0], m71=B_T->magnitude[b7*2+1];
                uint64_t s70=B_T->sign[b7*2+0],      s71=B_T->sign[b7*2+1];
                float sc0=sf[b0], sc1=sf[b1], sc2=sf[b2], sc3=sf[b3];
                float sc4=sf[b4], sc5=sf[b5], sc6=sf[b6], sc7=sf[b7];
                __m256 ps0=_mm256_set1_ps(sc0), ps1=_mm256_set1_ps(sc1);
                __m256 ps2=_mm256_set1_ps(sc2), ps3=_mm256_set1_ps(sc3);
                __m256 ps4=_mm256_set1_ps(sc4), ps5=_mm256_set1_ps(sc5);
                __m256 ps6=_mm256_set1_ps(sc6), ps7=_mm256_set1_ps(sc7);
                const float *ap = &A[i * K + bk * G128_BLOCK_SIZE];
                for (int b = 0; b < 64; b += 8) {
                    __m256 av = _mm256_loadu_ps(ap + b);
                    LUT_ACCUM_AVX2(acc0, m00, s00, b, ps0, av);
                    LUT_ACCUM_AVX2(acc1, m10, s10, b, ps1, av);
                    LUT_ACCUM_AVX2(acc2, m20, s20, b, ps2, av);
                    LUT_ACCUM_AVX2(acc3, m30, s30, b, ps3, av);
                    LUT_ACCUM_AVX2(acc4, m40, s40, b, ps4, av);
                    LUT_ACCUM_AVX2(acc5, m50, s50, b, ps5, av);
                    LUT_ACCUM_AVX2(acc6, m60, s60, b, ps6, av);
                    LUT_ACCUM_AVX2(acc7, m70, s70, b, ps7, av);
                }
                for (int b = 0; b < 64; b += 8) {
                    __m256 av = _mm256_loadu_ps(ap + 64 + b);
                    LUT_ACCUM_AVX2(acc0, m01, s01, b, ps0, av);
                    LUT_ACCUM_AVX2(acc1, m11, s11, b, ps1, av);
                    LUT_ACCUM_AVX2(acc2, m21, s21, b, ps2, av);
                    LUT_ACCUM_AVX2(acc3, m31, s31, b, ps3, av);
                    LUT_ACCUM_AVX2(acc4, m41, s41, b, ps4, av);
                    LUT_ACCUM_AVX2(acc5, m51, s51, b, ps5, av);
                    LUT_ACCUM_AVX2(acc6, m61, s61, b, ps6, av);
                    LUT_ACCUM_AVX2(acc7, m71, s71, b, ps7, av);
                }
            }
            C[i*N+j+0]=hsum_avx2(acc0); C[i*N+j+1]=hsum_avx2(acc1);
            C[i*N+j+2]=hsum_avx2(acc2); C[i*N+j+3]=hsum_avx2(acc3);
            C[i*N+j+4]=hsum_avx2(acc4); C[i*N+j+5]=hsum_avx2(acc5);
            C[i*N+j+6]=hsum_avx2(acc6); C[i*N+j+7]=hsum_avx2(acc7);
        }
        for (int j = n8; j < N; j++) {
            __m256 acc = _mm256_setzero_ps();
            int rb = j * nkb;
            for (int bk = 0; bk < nkb; bk++) {
                int bidx = rb + bk;
                uint64_t mag0=B_T->magnitude[bidx*2+0], mag1=B_T->magnitude[bidx*2+1];
                uint64_t sgn0=B_T->sign[bidx*2+0],      sgn1=B_T->sign[bidx*2+1];
                __m256 ps = _mm256_set1_ps(sf[bidx]);
                const float *ap = &A[i * K + bk * G128_BLOCK_SIZE];
                for (int b = 0; b < 64; b += 8)
                    LUT_ACCUM_AVX2(acc, mag0, sgn0, b, ps, _mm256_loadu_ps(ap + b));
                for (int b = 0; b < 64; b += 8)
                    LUT_ACCUM_AVX2(acc, mag1, sgn1, b, ps, _mm256_loadu_ps(ap + 64 + b));
            }
            C[i*N+j] = hsum_avx2(acc);
        }
    }
}

#elif defined(__SSE4_1__)

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

#define LUT_ACCUM_SSE(acc, mag_w, sgn_w, b, neg_sc, pos_sc, av) do { \
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
                int b0=(j+0)*nkb+bk, b1=(j+1)*nkb+bk;
                int b2=(j+2)*nkb+bk, b3=(j+3)*nkb+bk;
                uint64_t m00=B_T->magnitude[b0*2+0], m01=B_T->magnitude[b0*2+1];
                uint64_t s00=B_T->sign[b0*2+0],      s01=B_T->sign[b0*2+1];
                uint64_t m10=B_T->magnitude[b1*2+0], m11=B_T->magnitude[b1*2+1];
                uint64_t s10=B_T->sign[b1*2+0],      s11=B_T->sign[b1*2+1];
                uint64_t m20=B_T->magnitude[b2*2+0], m21=B_T->magnitude[b2*2+1];
                uint64_t s20=B_T->sign[b2*2+0],      s21=B_T->sign[b2*2+1];
                uint64_t m30=B_T->magnitude[b3*2+0], m31=B_T->magnitude[b3*2+1];
                uint64_t s30=B_T->sign[b3*2+0],      s31=B_T->sign[b3*2+1];
                float sc0=sf[b0], sc1=sf[b1], sc2=sf[b2], sc3=sf[b3];
                __m128 ns0=_mm_set1_ps(-sc0), ps0=_mm_set1_ps(sc0);
                __m128 ns1=_mm_set1_ps(-sc1), ps1=_mm_set1_ps(sc1);
                __m128 ns2=_mm_set1_ps(-sc2), ps2=_mm_set1_ps(sc2);
                __m128 ns3=_mm_set1_ps(-sc3), ps3=_mm_set1_ps(sc3);
                const float *ap = &A[i * K + bk * G128_BLOCK_SIZE];
                for (int b = 0; b < 64; b += 4) {
                    __m128 av = _mm_loadu_ps(ap + b);
                    LUT_ACCUM_SSE(acc0, m00, s00, b, ns0, ps0, av);
                    LUT_ACCUM_SSE(acc1, m10, s10, b, ns1, ps1, av);
                    LUT_ACCUM_SSE(acc2, m20, s20, b, ns2, ps2, av);
                    LUT_ACCUM_SSE(acc3, m30, s30, b, ns3, ps3, av);
                }
                for (int b = 0; b < 64; b += 4) {
                    __m128 av = _mm_loadu_ps(ap + 64 + b);
                    LUT_ACCUM_SSE(acc0, m01, s01, b, ns0, ps0, av);
                    LUT_ACCUM_SSE(acc1, m11, s11, b, ns1, ps1, av);
                    LUT_ACCUM_SSE(acc2, m21, s21, b, ns2, ps2, av);
                    LUT_ACCUM_SSE(acc3, m31, s31, b, ns3, ps3, av);
                }
            }
            C[i*N+j+0]=hsum_ps(acc0); C[i*N+j+1]=hsum_ps(acc1);
            C[i*N+j+2]=hsum_ps(acc2); C[i*N+j+3]=hsum_ps(acc3);
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
                    LUT_ACCUM_SSE(acc, mag0, sgn0, b, ns, ps, _mm_loadu_ps(ap + b));
                for (int b = 0; b < 64; b += 4)
                    LUT_ACCUM_SSE(acc, mag1, sgn1, b, ns, ps, _mm_loadu_ps(ap + 64 + b));
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

// Stubs for platforms without AVX-512 (lm_head_prefilter_available = 0)
// model_infer.c checks the flag before calling these.
#ifndef __AVX512F__
void lm_head_prefilter(float *A, G128Matrix *B_T, float *C, int N, int max_blocks) {
    (void)A; (void)B_T; (void)C; (void)N; (void)max_blocks;
}
void matmul_g128_selected(float *A, G128Matrix *B_T, float *C, int M, int K, int N_full, int N_sel, const int *sel_rows) {
    (void)A; (void)B_T; (void)C; (void)M; (void)K; (void)N_full; (void)N_sel; (void)sel_rows;
}
void find_top_k(float *scores, int N, int K, int *out_indices) {
    (void)scores;
    for (int i = 0; i < K && i < N; i++) out_indices[i] = i;
}
const int lm_head_prefilter_available = 0;
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

static uint8_t LUT_POS[256][256];
static int lut_initialized = 0;

void matmul_lut_g128(float *A, G128Matrix *B_T, float *C, int M, int K, int N) {
    (void)LUT_POS; (void)lut_initialized;
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

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
    m->packed_pos = NULL;
    m->packed_neg = NULL;
    m->scales     = malloc(num_blocks * sizeof(uint16_t));
    m->scales_f32 = NULL;
}

void g128_matrix_free(G128Matrix *m) {
    free(m->magnitude);
    free(m->sign);
    free(m->packed_pos);
    free(m->packed_neg);
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

void matmul_simd_g128_tiled8_neon(float *A, G128Matrix *B_T, float *C, int M, int K, int N);

void matmul_simd_g128(float *A, G128Matrix *B_T, float *C, int M, int K, int N) {
    if (B_T->tiles8 != NULL && B_T->num_tile_groups8 > 0 && N % 8 == 0) {
        matmul_simd_g128_tiled8_neon(A, B_T, C, M, K, N);
        return;
    }
    
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

void matmul_simd_g128_tiled8_neon(float *A, G128Matrix *B_T, float *C, int M, int K, int N) {
    int nkb = (int)B_T->num_blocks_col;
    const float *sf = B_T->scales_f32;
    const TileBlock8 *tiles = B_T->tiles8;
    int n8 = (N / 8) * 8;
    int n_j = n8 / 8;
    int total_jobs = M * n_j;
    
    #ifdef DEBUG
    assert(N % 8 == 0 && "All production matmul shapes are N%8==0");
    #endif
    
    #pragma omp parallel for schedule(static)
    for (int idx = 0; idx < total_jobs; idx++) {
        int i = idx / n_j;
        int j = (idx % n_j) * 8;
        float32x4_t acc0 = vdupq_n_f32(0.0f);
        float32x4_t acc1 = vdupq_n_f32(0.0f);
        float32x4_t acc2 = vdupq_n_f32(0.0f);
        float32x4_t acc3 = vdupq_n_f32(0.0f);
        float32x4_t acc4 = vdupq_n_f32(0.0f);
        float32x4_t acc5 = vdupq_n_f32(0.0f);
        float32x4_t acc6 = vdupq_n_f32(0.0f);
        float32x4_t acc7 = vdupq_n_f32(0.0f);
        int tg = j / 8;
        
        for (int bk = 0; bk < nkb; bk++) {
            int b0=(j+0)*nkb+bk, b1=(j+1)*nkb+bk, b2=(j+2)*nkb+bk, b3=(j+3)*nkb+bk;
            int b4=(j+4)*nkb+bk, b5=(j+5)*nkb+bk, b6=(j+6)*nkb+bk, b7=(j+7)*nkb+bk;
            float32x4_t sc0=vdupq_n_f32(sf[b0]), sc1=vdupq_n_f32(sf[b1]);
            float32x4_t sc2=vdupq_n_f32(sf[b2]), sc3=vdupq_n_f32(sf[b3]);
            float32x4_t sc4=vdupq_n_f32(sf[b4]), sc5=vdupq_n_f32(sf[b5]);
            float32x4_t sc6=vdupq_n_f32(sf[b6]), sc7=vdupq_n_f32(sf[b7]);
            float32x4_t ns0=vnegq_f32(sc0), ns1=vnegq_f32(sc1);
            float32x4_t ns2=vnegq_f32(sc2), ns3=vnegq_f32(sc3);
            float32x4_t ns4=vnegq_f32(sc4), ns5=vnegq_f32(sc5);
            float32x4_t ns6=vnegq_f32(sc6), ns7=vnegq_f32(sc7);
            const float *ap = &A[i * K + bk * G128_BLOCK_SIZE];
            const TileBlock8 *tb = &tiles[tg * nkb + bk];
            
            for (int b = 0; b < 64; b += 4) {
                float32x4_t av = vld1q_f32(ap + b);
                uint64_t p0=tb->pos[0][b/4], n0=tb->neg[0][b/4];
                uint64_t p1=tb->pos[1][b/4], n1=tb->neg[1][b/4];
                uint64_t p2=tb->pos[2][b/4], n2=tb->neg[2][b/4];
                uint64_t p3=tb->pos[3][b/4], n3=tb->neg[3][b/4];
                uint64_t p4=tb->pos[4][b/4], n4=tb->neg[4][b/4];
                uint64_t p5=tb->pos[5][b/4], n5=tb->neg[5][b/4];
                uint64_t p6=tb->pos[6][b/4], n6=tb->neg[6][b/4];
                uint64_t p7=tb->pos[7][b/4], n7=tb->neg[7][b/4];
                
                uint32x4_t pm0 = vreinterpretq_u32_u64(vdupq_n_u64(p0));
                uint32x4_t nm0 = vreinterpretq_u32_u64(vdupq_n_u64(n0));
                uint32x4_t pm1 = vreinterpretq_u32_u64(vdupq_n_u64(p1));
                uint32x4_t nm1 = vreinterpretq_u32_u64(vdupq_n_u64(n1));
                uint32x4_t pm2 = vreinterpretq_u32_u64(vdupq_n_u64(p2));
                uint32x4_t nm2 = vreinterpretq_u32_u64(vdupq_n_u64(n2));
                uint32x4_t pm3 = vreinterpretq_u32_u64(vdupq_n_u64(p3));
                uint32x4_t nm3 = vreinterpretq_u32_u64(vdupq_n_u64(n3));
                uint32x4_t pm4 = vreinterpretq_u32_u64(vdupq_n_u64(p4));
                uint32x4_t nm4 = vreinterpretq_u32_u64(vdupq_n_u64(n4));
                uint32x4_t pm5 = vreinterpretq_u32_u64(vdupq_n_u64(p5));
                uint32x4_t nm5 = vreinterpretq_u32_u64(vdupq_n_u64(n5));
                uint32x4_t pm6 = vreinterpretq_u32_u64(vdupq_n_u64(p6));
                uint32x4_t nm6 = vreinterpretq_u32_u64(vdupq_n_u64(n6));
                uint32x4_t pm7 = vreinterpretq_u32_u64(vdupq_n_u64(p7));
                uint32x4_t nm7 = vreinterpretq_u32_u64(vdupq_n_u64(n7));
                
                acc0 = vmlaq_f32(acc0, vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(sc0), pm0)), av);
                acc0 = vmlaq_f32(acc0, vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(ns0), nm0)), av);
                acc1 = vmlaq_f32(acc1, vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(sc1), pm1)), av);
                acc1 = vmlaq_f32(acc1, vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(ns1), nm1)), av);
                acc2 = vmlaq_f32(acc2, vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(sc2), pm2)), av);
                acc2 = vmlaq_f32(acc2, vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(ns2), nm2)), av);
                acc3 = vmlaq_f32(acc3, vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(sc3), pm3)), av);
                acc3 = vmlaq_f32(acc3, vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(ns3), nm3)), av);
                acc4 = vmlaq_f32(acc4, vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(sc4), pm4)), av);
                acc4 = vmlaq_f32(acc4, vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(ns4), nm4)), av);
                acc5 = vmlaq_f32(acc5, vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(sc5), pm5)), av);
                acc5 = vmlaq_f32(acc5, vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(ns5), nm5)), av);
                acc6 = vmlaq_f32(acc6, vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(sc6), pm6)), av);
                acc6 = vmlaq_f32(acc6, vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(ns6), nm6)), av);
                acc7 = vmlaq_f32(acc7, vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(sc7), pm7)), av);
                acc7 = vmlaq_f32(acc7, vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(ns7), nm7)), av);
            }
            for (int b = 0; b < 64; b += 4) {
                float32x4_t av = vld1q_f32(ap + 64 + b);
                uint64_t p0=tb->pos[0][4+b/4], n0=tb->neg[0][4+b/4];
                uint64_t p1=tb->pos[1][4+b/4], n1=tb->neg[1][4+b/4];
                uint64_t p2=tb->pos[2][4+b/4], n2=tb->neg[2][4+b/4];
                uint64_t p3=tb->pos[3][4+b/4], n3=tb->neg[3][4+b/4];
                uint64_t p4=tb->pos[4][4+b/4], n4=tb->neg[4][4+b/4];
                uint64_t p5=tb->pos[5][4+b/4], n5=tb->neg[5][4+b/4];
                uint64_t p6=tb->pos[6][4+b/4], n6=tb->neg[6][4+b/4];
                uint64_t p7=tb->pos[7][4+b/4], n7=tb->neg[7][4+b/4];
                
                uint32x4_t pm0 = vreinterpretq_u32_u64(vdupq_n_u64(p0));
                uint32x4_t nm0 = vreinterpretq_u32_u64(vdupq_n_u64(n0));
                uint32x4_t pm1 = vreinterpretq_u32_u64(vdupq_n_u64(p1));
                uint32x4_t nm1 = vreinterpretq_u32_u64(vdupq_n_u64(n1));
                uint32x4_t pm2 = vreinterpretq_u32_u64(vdupq_n_u64(p2));
                uint32x4_t nm2 = vreinterpretq_u32_u64(vdupq_n_u64(n2));
                uint32x4_t pm3 = vreinterpretq_u32_u64(vdupq_n_u64(p3));
                uint32x4_t nm3 = vreinterpretq_u32_u64(vdupq_n_u64(n3));
                uint32x4_t pm4 = vreinterpretq_u32_u64(vdupq_n_u64(p4));
                uint32x4_t nm4 = vreinterpretq_u32_u64(vdupq_n_u64(n4));
                uint32x4_t pm5 = vreinterpretq_u32_u64(vdupq_n_u64(p5));
                uint32x4_t nm5 = vreinterpretq_u32_u64(vdupq_n_u64(n5));
                uint32x4_t pm6 = vreinterpretq_u32_u64(vdupq_n_u64(p6));
                uint32x4_t nm6 = vreinterpretq_u32_u64(vdupq_n_u64(n6));
                uint32x4_t pm7 = vreinterpretq_u32_u64(vdupq_n_u64(p7));
                uint32x4_t nm7 = vreinterpretq_u32_u64(vdupq_n_u64(n7));
                
                acc0 = vmlaq_f32(acc0, vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(sc0), pm0)), av);
                acc0 = vmlaq_f32(acc0, vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(ns0), nm0)), av);
                acc1 = vmlaq_f32(acc1, vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(sc1), pm1)), av);
                acc1 = vmlaq_f32(acc1, vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(ns1), nm1)), av);
                acc2 = vmlaq_f32(acc2, vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(sc2), pm2)), av);
                acc2 = vmlaq_f32(acc2, vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(ns2), nm2)), av);
                acc3 = vmlaq_f32(acc3, vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(sc3), pm3)), av);
                acc3 = vmlaq_f32(acc3, vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(ns3), nm3)), av);
                acc4 = vmlaq_f32(acc4, vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(sc4), pm4)), av);
                acc4 = vmlaq_f32(acc4, vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(ns4), nm4)), av);
                acc5 = vmlaq_f32(acc5, vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(sc5), pm5)), av);
                acc5 = vmlaq_f32(acc5, vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(ns5), nm5)), av);
                acc6 = vmlaq_f32(acc6, vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(sc6), pm6)), av);
                acc6 = vmlaq_f32(acc6, vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(ns6), nm6)), av);
                acc7 = vmlaq_f32(acc7, vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(sc7), pm7)), av);
                acc7 = vmlaq_f32(acc7, vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(ns7), nm7)), av);
            }
        }
        C[i*N+j+0] = vaddvq_f32(acc0); C[i*N+j+1] = vaddvq_f32(acc1);
        C[i*N+j+2] = vaddvq_f32(acc2); C[i*N+j+3] = vaddvq_f32(acc3);
        C[i*N+j+4] = vaddvq_f32(acc4); C[i*N+j+5] = vaddvq_f32(acc5);
        C[i*N+j+6] = vaddvq_f32(acc6); C[i*N+j+7] = vaddvq_f32(acc7);
    }
    
    for (int i = 0; i < M; i++) {
        for (int j = n8; j < N; j++) {
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

// Pre-separated pos/neg bitmap matmul: no runtime mask computation.
// packed_pos holds mask_bits(weight==+scale), packed_neg holds mask_bits(weight==-scale).
// pos and neg values are pre-packed as 32-bit chunks (lower 16 bits = first 16-elem group,
// upper 16 bits = second 16-elem group), same layout as old 'packed'.
#define LUT_ACCUM_POS(acc, mask_val, sc, av) \
    (acc) = _mm512_mask3_fmadd_ps((av), (sc), (acc), (__mmask16)(uint32_t)(mask_val))
#define LUT_ACCUM_NEG(acc, mask_val, sc, av) \
    (acc) = _mm512_mask3_fnmadd_ps((av), (sc), (acc), (__mmask16)(uint32_t)(mask_val))

// Process positive-weight bitmaps for 8 output rows, one packed word at a time.
// Each word covers 32 elements (two 16-elem groups). Loads 8 uint64 values
// then processes lower/upper halves with shared activation loads.
#define PROCESS_WORD_POS(pword, ap_off) do { \
    uint64_t _p0=pos[b0*4+(pword)],_p1=pos[b1*4+(pword)]; \
    uint64_t _p2=pos[b2*4+(pword)],_p3=pos[b3*4+(pword)]; \
    uint64_t _p4=pos[b4*4+(pword)],_p5=pos[b5*4+(pword)]; \
    uint64_t _p6=pos[b6*4+(pword)],_p7=pos[b7*4+(pword)]; \
    __m512 _av = _mm512_load_ps(ap + (ap_off)); \
    LUT_ACCUM_POS(acc0,_p0,scv0,_av); LUT_ACCUM_POS(acc1,_p1,scv1,_av); \
    LUT_ACCUM_POS(acc2,_p2,scv2,_av); LUT_ACCUM_POS(acc3,_p3,scv3,_av); \
    LUT_ACCUM_POS(acc4,_p4,scv4,_av); LUT_ACCUM_POS(acc5,_p5,scv5,_av); \
    LUT_ACCUM_POS(acc6,_p6,scv6,_av); LUT_ACCUM_POS(acc7,_p7,scv7,_av); \
    _av = _mm512_load_ps(ap + (ap_off) + 16); \
    LUT_ACCUM_POS(acc0,(uint32_t)(_p0>>32),scv0,_av); \
    LUT_ACCUM_POS(acc1,(uint32_t)(_p1>>32),scv1,_av); \
    LUT_ACCUM_POS(acc2,(uint32_t)(_p2>>32),scv2,_av); \
    LUT_ACCUM_POS(acc3,(uint32_t)(_p3>>32),scv3,_av); \
    LUT_ACCUM_POS(acc4,(uint32_t)(_p4>>32),scv4,_av); \
    LUT_ACCUM_POS(acc5,(uint32_t)(_p5>>32),scv5,_av); \
    LUT_ACCUM_POS(acc6,(uint32_t)(_p6>>32),scv6,_av); \
    LUT_ACCUM_POS(acc7,(uint32_t)(_p7>>32),scv7,_av); \
} while(0)

#define PROCESS_WORD_NEG(pword, ap_off) do { \
    uint64_t _n0=neg[b0*4+(pword)],_n1=neg[b1*4+(pword)]; \
    uint64_t _n2=neg[b2*4+(pword)],_n3=neg[b3*4+(pword)]; \
    uint64_t _n4=neg[b4*4+(pword)],_n5=neg[b5*4+(pword)]; \
    uint64_t _n6=neg[b6*4+(pword)],_n7=neg[b7*4+(pword)]; \
    __m512 _av = _mm512_load_ps(ap + (ap_off)); \
    LUT_ACCUM_NEG(acc0,_n0,scv0,_av); LUT_ACCUM_NEG(acc1,_n1,scv1,_av); \
    LUT_ACCUM_NEG(acc2,_n2,scv2,_av); LUT_ACCUM_NEG(acc3,_n3,scv3,_av); \
    LUT_ACCUM_NEG(acc4,_n4,scv4,_av); LUT_ACCUM_NEG(acc5,_n5,scv5,_av); \
    LUT_ACCUM_NEG(acc6,_n6,scv6,_av); LUT_ACCUM_NEG(acc7,_n7,scv7,_av); \
    _av = _mm512_load_ps(ap + (ap_off) + 16); \
    LUT_ACCUM_NEG(acc0,(uint32_t)(_n0>>32),scv0,_av); \
    LUT_ACCUM_NEG(acc1,(uint32_t)(_n1>>32),scv1,_av); \
    LUT_ACCUM_NEG(acc2,(uint32_t)(_n2>>32),scv2,_av); \
    LUT_ACCUM_NEG(acc3,(uint32_t)(_n3>>32),scv3,_av); \
    LUT_ACCUM_NEG(acc4,(uint32_t)(_n4>>32),scv4,_av); \
    LUT_ACCUM_NEG(acc5,(uint32_t)(_n5>>32),scv5,_av); \
    LUT_ACCUM_NEG(acc6,(uint32_t)(_n6>>32),scv6,_av); \
    LUT_ACCUM_NEG(acc7,(uint32_t)(_n7>>32),scv7,_av); \
} while(0)

#define PROCESS_WORD_FUSED8(pword, ap_off) do { \
    const TileBlock8 *tb = &tiles[tg * nkb + bk]; \
    uint64_t _p0=tb->pos[0][pword], _p1=tb->pos[1][pword]; \
    uint64_t _p2=tb->pos[2][pword], _p3=tb->pos[3][pword]; \
    uint64_t _p4=tb->pos[4][pword], _p5=tb->pos[5][pword]; \
    uint64_t _p6=tb->pos[6][pword], _p7=tb->pos[7][pword]; \
    uint64_t _n0=tb->neg[0][pword], _n1=tb->neg[1][pword]; \
    uint64_t _n2=tb->neg[2][pword], _n3=tb->neg[3][pword]; \
    uint64_t _n4=tb->neg[4][pword], _n5=tb->neg[5][pword]; \
    uint64_t _n6=tb->neg[6][pword], _n7=tb->neg[7][pword]; \
    __m512 _av = _mm512_load_ps(ap + (ap_off)); \
    acc0 = _mm512_mask3_fmadd_ps(_av,scv0,acc0,(__mmask16)(uint32_t)_p0); \
    acc0 = _mm512_mask3_fnmadd_ps(_av,scv0,acc0,(__mmask16)(uint32_t)_n0); \
    acc1 = _mm512_mask3_fmadd_ps(_av,scv1,acc1,(__mmask16)(uint32_t)_p1); \
    acc1 = _mm512_mask3_fnmadd_ps(_av,scv1,acc1,(__mmask16)(uint32_t)_n1); \
    acc2 = _mm512_mask3_fmadd_ps(_av,scv2,acc2,(__mmask16)(uint32_t)_p2); \
    acc2 = _mm512_mask3_fnmadd_ps(_av,scv2,acc2,(__mmask16)(uint32_t)_n2); \
    acc3 = _mm512_mask3_fmadd_ps(_av,scv3,acc3,(__mmask16)(uint32_t)_p3); \
    acc3 = _mm512_mask3_fnmadd_ps(_av,scv3,acc3,(__mmask16)(uint32_t)_n3); \
    acc4 = _mm512_mask3_fmadd_ps(_av,scv4,acc4,(__mmask16)(uint32_t)_p4); \
    acc4 = _mm512_mask3_fnmadd_ps(_av,scv4,acc4,(__mmask16)(uint32_t)_n4); \
    acc5 = _mm512_mask3_fmadd_ps(_av,scv5,acc5,(__mmask16)(uint32_t)_p5); \
    acc5 = _mm512_mask3_fnmadd_ps(_av,scv5,acc5,(__mmask16)(uint32_t)_n5); \
    acc6 = _mm512_mask3_fmadd_ps(_av,scv6,acc6,(__mmask16)(uint32_t)_p6); \
    acc6 = _mm512_mask3_fnmadd_ps(_av,scv6,acc6,(__mmask16)(uint32_t)_n6); \
    acc7 = _mm512_mask3_fmadd_ps(_av,scv7,acc7,(__mmask16)(uint32_t)_p7); \
    acc7 = _mm512_mask3_fnmadd_ps(_av,scv7,acc7,(__mmask16)(uint32_t)_n7); \
    _av = _mm512_load_ps(ap + (ap_off) + 16); \
    acc0 = _mm512_mask3_fmadd_ps(_av,scv0,acc0,(__mmask16)(uint32_t)(_p0>>32)); \
    acc0 = _mm512_mask3_fnmadd_ps(_av,scv0,acc0,(__mmask16)(uint32_t)(_n0>>32)); \
    acc1 = _mm512_mask3_fmadd_ps(_av,scv1,acc1,(__mmask16)(uint32_t)(_p1>>32)); \
    acc1 = _mm512_mask3_fnmadd_ps(_av,scv1,acc1,(__mmask16)(uint32_t)(_n1>>32)); \
    acc2 = _mm512_mask3_fmadd_ps(_av,scv2,acc2,(__mmask16)(uint32_t)(_p2>>32)); \
    acc2 = _mm512_mask3_fnmadd_ps(_av,scv2,acc2,(__mmask16)(uint32_t)(_n2>>32)); \
    acc3 = _mm512_mask3_fmadd_ps(_av,scv3,acc3,(__mmask16)(uint32_t)(_p3>>32)); \
    acc3 = _mm512_mask3_fnmadd_ps(_av,scv3,acc3,(__mmask16)(uint32_t)(_n3>>32)); \
    acc4 = _mm512_mask3_fmadd_ps(_av,scv4,acc4,(__mmask16)(uint32_t)(_p4>>32)); \
    acc4 = _mm512_mask3_fnmadd_ps(_av,scv4,acc4,(__mmask16)(uint32_t)(_n4>>32)); \
    acc5 = _mm512_mask3_fmadd_ps(_av,scv5,acc5,(__mmask16)(uint32_t)(_p5>>32)); \
    acc5 = _mm512_mask3_fnmadd_ps(_av,scv5,acc5,(__mmask16)(uint32_t)(_n5>>32)); \
    acc6 = _mm512_mask3_fmadd_ps(_av,scv6,acc6,(__mmask16)(uint32_t)(_p6>>32)); \
    acc6 = _mm512_mask3_fnmadd_ps(_av,scv6,acc6,(__mmask16)(uint32_t)(_n6>>32)); \
    acc7 = _mm512_mask3_fmadd_ps(_av,scv7,acc7,(__mmask16)(uint32_t)(_p7>>32)); \
    acc7 = _mm512_mask3_fnmadd_ps(_av,scv7,acc7,(__mmask16)(uint32_t)(_n7>>32)); \
} while(0)

static inline float hsum_zmm(__m512 v) {
    __m256 lo = _mm512_castps512_ps256(v);
    __m256 hi = _mm512_extractf32x8_ps(v, 1);
    __m256 s = _mm256_add_ps(lo, hi);
    __m128 s4 = _mm_add_ps(_mm256_castps256_ps128(s), _mm256_extractf128_ps(s, 1));
    s4 = _mm_hadd_ps(s4, s4);
    s4 = _mm_hadd_ps(s4, s4);
    return _mm_cvtss_f32(s4);
}

void matmul_simd_g128_tiled8(float *A, G128Matrix *B_T, float *C, int M, int K, int N) {
    int nkb = (int)B_T->num_blocks_col;
    const float *sf = B_T->scales_f32;
    const TileBlock8 *tiles = B_T->tiles8;
    int n8 = (N / 8) * 8;
    int n_j = n8 / 8;
    int total_jobs = M * n_j;
    
    #ifdef DEBUG
    assert(N % 8 == 0 && "All production matmul shapes are N%8==0");
    #endif
    
    #pragma omp parallel for schedule(static)
    for (int idx = 0; idx < total_jobs; idx++) {
        int i = idx / n_j;
        int j = (idx % n_j) * 8;
        __m512 acc0 = _mm512_setzero_ps();
        __m512 acc1 = _mm512_setzero_ps();
        __m512 acc2 = _mm512_setzero_ps();
        __m512 acc3 = _mm512_setzero_ps();
        __m512 acc4 = _mm512_setzero_ps();
        __m512 acc5 = _mm512_setzero_ps();
        __m512 acc6 = _mm512_setzero_ps();
        __m512 acc7 = _mm512_setzero_ps();
        int tg = j / 8;
        
        for (int bk = 0; bk < nkb; bk++) {
            int b0=(j+0)*nkb+bk, b1=(j+1)*nkb+bk, b2=(j+2)*nkb+bk, b3=(j+3)*nkb+bk;
            int b4=(j+4)*nkb+bk, b5=(j+5)*nkb+bk, b6=(j+6)*nkb+bk, b7=(j+7)*nkb+bk;
            __m512 scv0=_mm512_set1_ps(sf[b0]), scv1=_mm512_set1_ps(sf[b1]);
            __m512 scv2=_mm512_set1_ps(sf[b2]), scv3=_mm512_set1_ps(sf[b3]);
            __m512 scv4=_mm512_set1_ps(sf[b4]), scv5=_mm512_set1_ps(sf[b5]);
            __m512 scv6=_mm512_set1_ps(sf[b6]), scv7=_mm512_set1_ps(sf[b7]);
            const float *ap = &A[i * K + bk * G128_BLOCK_SIZE];
            
            if (bk + 2 < nkb) {
                const TileBlock8 *tb_pf = &tiles[tg * nkb + bk + 2];
                _mm_prefetch((const char*)tb_pf,               _MM_HINT_T1);
                _mm_prefetch((const char*)tb_pf + 64,          _MM_HINT_T1);
                _mm_prefetch((const char*)tb_pf + 128,         _MM_HINT_T1);
                _mm_prefetch((const char*)tb_pf + 192,         _MM_HINT_T1);
                _mm_prefetch((const char*)tb_pf + 256,         _MM_HINT_T1);
                _mm_prefetch((const char*)tb_pf + 320,         _MM_HINT_T1);
                _mm_prefetch((const char*)tb_pf + 384,         _MM_HINT_T1);
                _mm_prefetch((const char*)tb_pf + 448,         _MM_HINT_T1);
                _mm_prefetch((const char*)tb_pf + 512,         _MM_HINT_T1);
                _mm_prefetch(ap + G128_BLOCK_SIZE, _MM_HINT_T0);
            } else if (bk + 1 < nkb) {
                _mm_prefetch(ap + G128_BLOCK_SIZE, _MM_HINT_T0);
            }
            
            PROCESS_WORD_FUSED8(0, 0);
            PROCESS_WORD_FUSED8(1, 32);
            PROCESS_WORD_FUSED8(2, 64);
            PROCESS_WORD_FUSED8(3, 96);
        }
        C[i*N+j+0]=hsum_zmm(acc0); C[i*N+j+1]=hsum_zmm(acc1);
        C[i*N+j+2]=hsum_zmm(acc2); C[i*N+j+3]=hsum_zmm(acc3);
        C[i*N+j+4]=hsum_zmm(acc4); C[i*N+j+5]=hsum_zmm(acc5);
        C[i*N+j+6]=hsum_zmm(acc6); C[i*N+j+7]=hsum_zmm(acc7);
    }
    
    for (int i = 0; i < M; i++) {
        for (int j = n8; j < N; j++) {
            __m512 acc = _mm512_setzero_ps();
            int rb = j * nkb;
            for (int bk = 0; bk < nkb; bk++) {
                int bidx = rb + bk;
                uint64_t pkp0= B_T->packed_pos[bidx*4+0], pkp1= B_T->packed_pos[bidx*4+1];
                uint64_t pkp2= B_T->packed_pos[bidx*4+2], pkp3= B_T->packed_pos[bidx*4+3];
                uint64_t pkn0= B_T->packed_neg[bidx*4+0], pkn1= B_T->packed_neg[bidx*4+1];
                uint64_t pkn2= B_T->packed_neg[bidx*4+2], pkn3= B_T->packed_neg[bidx*4+3];
                uint32_t cp_lo[4]={(uint32_t)pkp0,(uint32_t)(pkp0>>32),(uint32_t)pkp1,(uint32_t)(pkp1>>32)};
                uint32_t cp_hi[4]={(uint32_t)pkp2,(uint32_t)(pkp2>>32),(uint32_t)pkp3,(uint32_t)(pkp3>>32)};
                uint32_t cn_lo[4]={(uint32_t)pkn0,(uint32_t)(pkn0>>32),(uint32_t)pkn1,(uint32_t)(pkn1>>32)};
                uint32_t cn_hi[4]={(uint32_t)pkn2,(uint32_t)(pkn2>>32),(uint32_t)pkn3,(uint32_t)(pkn3>>32)};
                __m512 scv = _mm512_set1_ps(sf[bidx]);
                const float *ap = &A[i * K + bk * G128_BLOCK_SIZE];
                for (int bi = 0; bi < 4; bi++) {
                    __m512 av = _mm512_load_ps(ap + bi * 16);
                    acc = _mm512_mask3_fmadd_ps(av, scv, acc, (__mmask16)cp_lo[bi]);
                    acc = _mm512_mask3_fnmadd_ps(av, scv, acc, (__mmask16)cn_lo[bi]);
                }
                for (int bi = 0; bi < 4; bi++) {
                    __m512 av = _mm512_load_ps(ap + 64 + bi * 16);
                    acc = _mm512_mask3_fmadd_ps(av, scv, acc, (__mmask16)cp_hi[bi]);
                    acc = _mm512_mask3_fnmadd_ps(av, scv, acc, (__mmask16)cn_hi[bi]);
                }
            }
            C[i*N+j] = hsum_zmm(acc);
        }
    }
}

void matmul_simd_g128(float *A, G128Matrix *B_T, float *C, int M, int K, int N) {
    if (B_T->tiles8 != NULL && B_T->num_tile_groups8 > 0 && N % 8 == 0) {
        matmul_simd_g128_tiled8(A, B_T, C, M, K, N);
        return;
    }
    
    int nkb = (int)B_T->num_blocks_col;
    const float *sf = B_T->scales_f32;
    const uint64_t *pos = B_T->packed_pos;
    const uint64_t *neg = B_T->packed_neg;
    int n8 = (N / 8) * 8;
    int n_j = n8 / 8;
    int total_jobs = M * n_j;
    if (total_jobs >= 32) {
        #pragma omp parallel for schedule(static)
        for (int idx = 0; idx < total_jobs; idx++) {
            int i = idx / n_j;
            int j = (idx % n_j) * 8;
            __m512 acc0 = _mm512_setzero_ps();
            __m512 acc1 = _mm512_setzero_ps();
            __m512 acc2 = _mm512_setzero_ps();
            __m512 acc3 = _mm512_setzero_ps();
            __m512 acc4 = _mm512_setzero_ps();
            __m512 acc5 = _mm512_setzero_ps();
            __m512 acc6 = _mm512_setzero_ps();
            __m512 acc7 = _mm512_setzero_ps();
            int r0=(j+0)*nkb, r1=(j+1)*nkb, r2=(j+2)*nkb, r3=(j+3)*nkb;
            int r4=(j+4)*nkb, r5=(j+5)*nkb, r6=(j+6)*nkb, r7=(j+7)*nkb;
            for (int bk = 0; bk < nkb; bk++) {
                int b0=r0+bk, b1=r1+bk, b2=r2+bk, b3=r3+bk;
                int b4=r4+bk, b5=r5+bk, b6=r6+bk, b7=r7+bk;
                __m512 scv0=_mm512_set1_ps(sf[b0]), scv1=_mm512_set1_ps(sf[b1]);
                __m512 scv2=_mm512_set1_ps(sf[b2]), scv3=_mm512_set1_ps(sf[b3]);
                __m512 scv4=_mm512_set1_ps(sf[b4]), scv5=_mm512_set1_ps(sf[b5]);
                __m512 scv6=_mm512_set1_ps(sf[b6]), scv7=_mm512_set1_ps(sf[b7]);
                const float *ap = &A[i * K + bk * G128_BLOCK_SIZE];
                if (bk + 1 < nkb) {
                    _mm_prefetch((const char*)&pos[(b0+1)*4], _MM_HINT_T0);
                    _mm_prefetch((const char*)&pos[(b1+1)*4], _MM_HINT_T0);
                    _mm_prefetch((const char*)&pos[(b2+1)*4], _MM_HINT_T0);
                    _mm_prefetch((const char*)&pos[(b3+1)*4], _MM_HINT_T0);
                    _mm_prefetch((const char*)&pos[(b4+1)*4], _MM_HINT_T0);
                    _mm_prefetch((const char*)&pos[(b5+1)*4], _MM_HINT_T0);
                    _mm_prefetch((const char*)&pos[(b6+1)*4], _MM_HINT_T0);
                    _mm_prefetch((const char*)&pos[(b7+1)*4], _MM_HINT_T0);
                    _mm_prefetch((const char*)&neg[(b0+1)*4], _MM_HINT_T1);
                    _mm_prefetch((const char*)&neg[(b4+1)*4], _MM_HINT_T1);
                    _mm_prefetch(ap + G128_BLOCK_SIZE, _MM_HINT_T0);
                    _mm_prefetch((const char*)&sf[b0+1], _MM_HINT_T0);
                }
                PROCESS_WORD_POS(0, 0);   PROCESS_WORD_NEG(0, 0);
                PROCESS_WORD_POS(1, 32);  PROCESS_WORD_NEG(1, 32);
                PROCESS_WORD_POS(2, 64);  PROCESS_WORD_NEG(2, 64);
                PROCESS_WORD_POS(3, 96);  PROCESS_WORD_NEG(3, 96);
            }
            C[i*N+j+0]=hsum_zmm(acc0); C[i*N+j+1]=hsum_zmm(acc1);
            C[i*N+j+2]=hsum_zmm(acc2); C[i*N+j+3]=hsum_zmm(acc3);
            C[i*N+j+4]=hsum_zmm(acc4); C[i*N+j+5]=hsum_zmm(acc5);
            C[i*N+j+6]=hsum_zmm(acc6); C[i*N+j+7]=hsum_zmm(acc7);
        }
    } else {
        for (int i = 0; i < M; i++) {
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
                int r0=(j+0)*nkb, r1=(j+1)*nkb, r2=(j+2)*nkb, r3=(j+3)*nkb;
                int r4=(j+4)*nkb, r5=(j+5)*nkb, r6=(j+6)*nkb, r7=(j+7)*nkb;
                for (int bk = 0; bk < nkb; bk++) {
                    int b0=r0+bk, b1=r1+bk, b2=r2+bk, b3=r3+bk;
                    int b4=r4+bk, b5=r5+bk, b6=r6+bk, b7=r7+bk;
                    __m512 scv0=_mm512_set1_ps(sf[b0]), scv1=_mm512_set1_ps(sf[b1]);
                    __m512 scv2=_mm512_set1_ps(sf[b2]), scv3=_mm512_set1_ps(sf[b3]);
                    __m512 scv4=_mm512_set1_ps(sf[b4]), scv5=_mm512_set1_ps(sf[b5]);
                    __m512 scv6=_mm512_set1_ps(sf[b6]), scv7=_mm512_set1_ps(sf[b7]);
                    const float *ap = &A[i * K + bk * G128_BLOCK_SIZE];
                    if (bk + 1 < nkb) {
                        _mm_prefetch((const char*)&pos[(b0+1)*4], _MM_HINT_T0);
                        _mm_prefetch((const char*)&pos[(b1+1)*4], _MM_HINT_T0);
                        _mm_prefetch((const char*)&pos[(b2+1)*4], _MM_HINT_T0);
                        _mm_prefetch((const char*)&pos[(b3+1)*4], _MM_HINT_T0);
                        _mm_prefetch((const char*)&pos[(b4+1)*4], _MM_HINT_T0);
                        _mm_prefetch((const char*)&pos[(b5+1)*4], _MM_HINT_T0);
                        _mm_prefetch((const char*)&pos[(b6+1)*4], _MM_HINT_T0);
                        _mm_prefetch((const char*)&pos[(b7+1)*4], _MM_HINT_T0);
                        _mm_prefetch((const char*)&neg[(b0+1)*4], _MM_HINT_T1);
                        _mm_prefetch((const char*)&neg[(b4+1)*4], _MM_HINT_T1);
                        _mm_prefetch(ap + G128_BLOCK_SIZE, _MM_HINT_T0);
                        _mm_prefetch((const char*)&sf[b0+1], _MM_HINT_T0);
                    }
                    PROCESS_WORD_POS(0, 0);   PROCESS_WORD_NEG(0, 0);
                    PROCESS_WORD_POS(1, 32);  PROCESS_WORD_NEG(1, 32);
                    PROCESS_WORD_POS(2, 64);  PROCESS_WORD_NEG(2, 64);
                    PROCESS_WORD_POS(3, 96);  PROCESS_WORD_NEG(3, 96);
                }
                C[i*N+j+0]=hsum_zmm(acc0); C[i*N+j+1]=hsum_zmm(acc1);
                C[i*N+j+2]=hsum_zmm(acc2); C[i*N+j+3]=hsum_zmm(acc3);
                C[i*N+j+4]=hsum_zmm(acc4); C[i*N+j+5]=hsum_zmm(acc5);
                C[i*N+j+6]=hsum_zmm(acc6); C[i*N+j+7]=hsum_zmm(acc7);
            }
        }
    }
    for (int i = 0; i < M; i++) {
        for (int j = n8; j < N; j++) {
            __m512 acc = _mm512_setzero_ps();
            int rb = j * nkb;
            for (int bk = 0; bk < nkb; bk++) {
                int bidx = rb + bk;
                uint64_t pkp0= B_T->packed_pos[bidx*4+0], pkp1= B_T->packed_pos[bidx*4+1];
                uint64_t pkp2= B_T->packed_pos[bidx*4+2], pkp3= B_T->packed_pos[bidx*4+3];
                uint64_t pkn0= B_T->packed_neg[bidx*4+0], pkn1= B_T->packed_neg[bidx*4+1];
                uint64_t pkn2= B_T->packed_neg[bidx*4+2], pkn3= B_T->packed_neg[bidx*4+3];
                uint32_t cp_lo[4]={(uint32_t)pkp0,(uint32_t)(pkp0>>32),(uint32_t)pkp1,(uint32_t)(pkp1>>32)};
                uint32_t cp_hi[4]={(uint32_t)pkp2,(uint32_t)(pkp2>>32),(uint32_t)pkp3,(uint32_t)(pkp3>>32)};
                uint32_t cn_lo[4]={(uint32_t)pkn0,(uint32_t)(pkn0>>32),(uint32_t)pkn1,(uint32_t)(pkn1>>32)};
                uint32_t cn_hi[4]={(uint32_t)pkn2,(uint32_t)(pkn2>>32),(uint32_t)pkn3,(uint32_t)(pkn3>>32)};
                __m512 scv = _mm512_set1_ps(sf[bidx]);
                const float *ap = &A[i * K + bk * G128_BLOCK_SIZE];
                for (int bi = 0; bi < 4; bi++) {
                    __m512 av = _mm512_load_ps(ap + bi * 16);
                    LUT_ACCUM_POS(acc, cp_lo[bi], scv, av);
                    LUT_ACCUM_NEG(acc, cn_lo[bi], scv, av);
                }
                for (int bi = 0; bi < 4; bi++) {
                    __m512 av = _mm512_load_ps(ap + 64 + bi * 16);
                    LUT_ACCUM_POS(acc, cp_hi[bi], scv, av);
                    LUT_ACCUM_NEG(acc, cn_hi[bi], scv, av);
                }
            }
            C[i*N+j] = hsum_zmm(acc);
        }
    }
}

// lm_head_prefilter: stub kept for interface compatibility.
// Vocabulary-level prefiltering is held out — use full lm_head matmul instead.
void lm_head_prefilter(float *A, G128Matrix *B_T, float *C, int N, int max_blocks) {
    (void)A; (void)B_T; (void)C; (void)N; (void)max_blocks;
}

// Compute exact dot products for a selected subset of vocabulary rows.
// Uses pre-separated pos/neg bitmaps.
void matmul_g128_selected(float *A, G128Matrix *B_T, float *C, int M, int K, int N_full, int N_sel, const int *sel_rows) {
    int nkb = (int)B_T->num_blocks_col;
    const float *sf = B_T->scales_f32;
    const uint64_t *pos = B_T->packed_pos;
    const uint64_t *neg = B_T->packed_neg;
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
            int row0=sel_rows[si+0], row1=sel_rows[si+1];
            int row2=sel_rows[si+2], row3=sel_rows[si+3];
            int row4=sel_rows[si+4], row5=sel_rows[si+5];
            int row6=sel_rows[si+6], row7=sel_rows[si+7];
            int r0=row0*nkb, r1=row1*nkb, r2=row2*nkb, r3=row3*nkb;
            int r4=row4*nkb, r5=row5*nkb, r6=row6*nkb, r7=row7*nkb;
            for (int bk = 0; bk < nkb; bk++) {
                int b0=r0+bk, b1=r1+bk, b2=r2+bk, b3=r3+bk;
                int b4=r4+bk, b5=r5+bk, b6=r6+bk, b7=r7+bk;
                __m512 scv0=_mm512_set1_ps(sf[b0]), scv1=_mm512_set1_ps(sf[b1]);
                __m512 scv2=_mm512_set1_ps(sf[b2]), scv3=_mm512_set1_ps(sf[b3]);
                __m512 scv4=_mm512_set1_ps(sf[b4]), scv5=_mm512_set1_ps(sf[b5]);
                __m512 scv6=_mm512_set1_ps(sf[b6]), scv7=_mm512_set1_ps(sf[b7]);
                const float *ap = &A[i * K + bk * G128_BLOCK_SIZE];
                PROCESS_WORD_POS(0, 0);   PROCESS_WORD_NEG(0, 0);
                PROCESS_WORD_POS(1, 32);  PROCESS_WORD_NEG(1, 32);
                PROCESS_WORD_POS(2, 64);  PROCESS_WORD_NEG(2, 64);
                PROCESS_WORD_POS(3, 96);  PROCESS_WORD_NEG(3, 96);
            }
            C[i*N_full+row0]=hsum_zmm(acc0); C[i*N_full+row1]=hsum_zmm(acc1);
            C[i*N_full+row2]=hsum_zmm(acc2); C[i*N_full+row3]=hsum_zmm(acc3);
            C[i*N_full+row4]=hsum_zmm(acc4); C[i*N_full+row5]=hsum_zmm(acc5);
            C[i*N_full+row6]=hsum_zmm(acc6); C[i*N_full+row7]=hsum_zmm(acc7);
        }
        for (int si = n8; si < N_sel; si++) {
            int r = sel_rows[si];
            __m512 acc = _mm512_setzero_ps();
            int rb = r * nkb;
            for (int bk = 0; bk < nkb; bk++) {
                int bidx = rb + bk;
                uint64_t pkp0= B_T->packed_pos[bidx*4+0], pkp1= B_T->packed_pos[bidx*4+1];
                uint64_t pkp2= B_T->packed_pos[bidx*4+2], pkp3= B_T->packed_pos[bidx*4+3];
                uint64_t pkn0= B_T->packed_neg[bidx*4+0], pkn1= B_T->packed_neg[bidx*4+1];
                uint64_t pkn2= B_T->packed_neg[bidx*4+2], pkn3= B_T->packed_neg[bidx*4+3];
                uint32_t cp_lo[4]={(uint32_t)pkp0,(uint32_t)(pkp0>>32),(uint32_t)pkp1,(uint32_t)(pkp1>>32)};
                uint32_t cp_hi[4]={(uint32_t)pkp2,(uint32_t)(pkp2>>32),(uint32_t)pkp3,(uint32_t)(pkp3>>32)};
                uint32_t cn_lo[4]={(uint32_t)pkn0,(uint32_t)(pkn0>>32),(uint32_t)pkn1,(uint32_t)(pkn1>>32)};
                uint32_t cn_hi[4]={(uint32_t)pkn2,(uint32_t)(pkn2>>32),(uint32_t)pkn3,(uint32_t)(pkn3>>32)};
                __m512 scv = _mm512_set1_ps(sf[bidx]);
                const float *ap = &A[i * K + bk * G128_BLOCK_SIZE];
                for (int bi = 0; bi < 4; bi++) {
                    __m512 av = _mm512_load_ps(ap + bi * 16);
                    LUT_ACCUM_POS(acc, cp_lo[bi], scv, av);
                    LUT_ACCUM_NEG(acc, cn_lo[bi], scv, av);
                }
                for (int bi = 0; bi < 4; bi++) {
                    __m512 av = _mm512_load_ps(ap + 64 + bi * 16);
                    LUT_ACCUM_POS(acc, cp_hi[bi], scv, av);
                    LUT_ACCUM_NEG(acc, cn_hi[bi], scv, av);
                }
            }
            C[i*N_full + r] = hsum_zmm(acc);
        }
    }
}

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

const int lm_head_prefilter_available = 0;

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
    if (B_T->tiles8 != NULL && B_T->num_tile_groups8 > 0 && N % 8 == 0) {
        matmul_simd_g128_tiled8_avx2(A, B_T, C, M, K, N);
        return;
    }
    
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

#define PROCESS_WORD_AVX2_FUSED(acc, pos_byte, neg_byte, sc, av) do { \
    __m256 _pm = _mm256_castsi256_ps(                                   \
        _mm256_load_si256((const __m256i*)avx2_mag_lut[pos_byte]));    \
    __m256 _nm = _mm256_castsi256_ps(                                   \
        _mm256_load_si256((const __m256i*)avx2_mag_lut[neg_byte]));    \
    __m256 _ns = _mm256_xor_ps(sc, _mm256_set1_ps(-0.0f));             \
    (acc) = _mm256_fmadd_ps(_mm256_and_ps(sc, _pm), (av), (acc));      \
    (acc) = _mm256_fmadd_ps(_mm256_and_ps(_ns, _nm), (av), (acc));     \
} while(0)

void matmul_simd_g128_tiled8_avx2(float *A, G128Matrix *B_T, float *C, int M, int K, int N) {
    init_avx2_lut_once();
    int nkb = (int)B_T->num_blocks_col;
    const float *sf = B_T->scales_f32;
    const TileBlock8 *tiles = B_T->tiles8;
    int n8 = (N / 8) * 8;
    int n_j = n8 / 8;
    int total_jobs = M * n_j;
    
    #ifdef DEBUG
    assert(N % 8 == 0 && "All production matmul shapes are N%8==0");
    #endif
    
    #pragma omp parallel for schedule(static)
    for (int idx = 0; idx < total_jobs; idx++) {
        int i = idx / n_j;
        int j = (idx % n_j) * 8;
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();
        __m256 acc4 = _mm256_setzero_ps();
        __m256 acc5 = _mm256_setzero_ps();
        __m256 acc6 = _mm256_setzero_ps();
        __m256 acc7 = _mm256_setzero_ps();
        int tg = j / 8;
        
        for (int bk = 0; bk < nkb; bk++) {
            int b0=(j+0)*nkb+bk, b1=(j+1)*nkb+bk, b2=(j+2)*nkb+bk, b3=(j+3)*nkb+bk;
            int b4=(j+4)*nkb+bk, b5=(j+5)*nkb+bk, b6=(j+6)*nkb+bk, b7=(j+7)*nkb+bk;
            __m256 sc0=_mm256_set1_ps(sf[b0]), sc1=_mm256_set1_ps(sf[b1]);
            __m256 sc2=_mm256_set1_ps(sf[b2]), sc3=_mm256_set1_ps(sf[b3]);
            __m256 sc4=_mm256_set1_ps(sf[b4]), sc5=_mm256_set1_ps(sf[b5]);
            __m256 sc6=_mm256_set1_ps(sf[b6]), sc7=_mm256_set1_ps(sf[b7]);
            __m256 ns0=_mm256_xor_ps(sc0, _mm256_set1_ps(-0.0f));
            __m256 ns1=_mm256_xor_ps(sc1, _mm256_set1_ps(-0.0f));
            __m256 ns2=_mm256_xor_ps(sc2, _mm256_set1_ps(-0.0f));
            __m256 ns3=_mm256_xor_ps(sc3, _mm256_set1_ps(-0.0f));
            __m256 ns4=_mm256_xor_ps(sc4, _mm256_set1_ps(-0.0f));
            __m256 ns5=_mm256_xor_ps(sc5, _mm256_set1_ps(-0.0f));
            __m256 ns6=_mm256_xor_ps(sc6, _mm256_set1_ps(-0.0f));
            __m256 ns7=_mm256_xor_ps(sc7, _mm256_set1_ps(-0.0f));
            const float *ap = &A[i * K + bk * G128_BLOCK_SIZE];
            
            if (bk + 2 < nkb) {
                const TileBlock8 *tb_pf = &tiles[tg * nkb + bk + 2];
                _mm_prefetch((const char*)tb_pf, _MM_HINT_T1);
                _mm_prefetch((const char*)tb_pf + 64, _MM_HINT_T1);
                _mm_prefetch((const char*)tb_pf + 128, _MM_HINT_T1);
                _mm_prefetch((const char*)tb_pf + 192, _MM_HINT_T1);
                _mm_prefetch((const char*)tb_pf + 256, _MM_HINT_T1);
                _mm_prefetch((const char*)tb_pf + 320, _MM_HINT_T1);
                _mm_prefetch((const char*)tb_pf + 384, _MM_HINT_T1);
                _mm_prefetch((const char*)tb_pf + 448, _MM_HINT_T1);
                _mm_prefetch((const char*)tb_pf + 512, _MM_HINT_T1);
                _mm_prefetch(ap + G128_BLOCK_SIZE, _MM_HINT_T0);
            } else if (bk + 1 < nkb) {
                _mm_prefetch(ap + G128_BLOCK_SIZE, _MM_HINT_T0);
            }
            
            const TileBlock8 *tb = &tiles[tg * nkb + bk];
            for (int b = 0; b < 64; b += 8) {
                __m256 av = _mm256_loadu_ps(ap + b);
                uint8_t p0=((uint8_t*)tb->pos[0])[b/8], n0=((uint8_t*)tb->neg[0])[b/8];
                uint8_t p1=((uint8_t*)tb->pos[1])[b/8], n1=((uint8_t*)tb->neg[1])[b/8];
                uint8_t p2=((uint8_t*)tb->pos[2])[b/8], n2=((uint8_t*)tb->neg[2])[b/8];
                uint8_t p3=((uint8_t*)tb->pos[3])[b/8], n3=((uint8_t*)tb->neg[3])[b/8];
                uint8_t p4=((uint8_t*)tb->pos[4])[b/8], n4=((uint8_t*)tb->neg[4])[b/8];
                uint8_t p5=((uint8_t*)tb->pos[5])[b/8], n5=((uint8_t*)tb->neg[5])[b/8];
                uint8_t p6=((uint8_t*)tb->pos[6])[b/8], n6=((uint8_t*)tb->neg[6])[b/8];
                uint8_t p7=((uint8_t*)tb->pos[7])[b/8], n7=((uint8_t*)tb->neg[7])[b/8];
                PROCESS_WORD_AVX2_FUSED(acc0, p0, n0, sc0, av);
                PROCESS_WORD_AVX2_FUSED(acc1, p1, n1, sc1, av);
                PROCESS_WORD_AVX2_FUSED(acc2, p2, n2, sc2, av);
                PROCESS_WORD_AVX2_FUSED(acc3, p3, n3, sc3, av);
                PROCESS_WORD_AVX2_FUSED(acc4, p4, n4, sc4, av);
                PROCESS_WORD_AVX2_FUSED(acc5, p5, n5, sc5, av);
                PROCESS_WORD_AVX2_FUSED(acc6, p6, n6, sc6, av);
                PROCESS_WORD_AVX2_FUSED(acc7, p7, n7, sc7, av);
            }
            for (int b = 0; b < 64; b += 8) {
                __m256 av = _mm256_loadu_ps(ap + 64 + b);
                uint8_t p0=((uint8_t*)tb->pos[0])[8+b/8], n0=((uint8_t*)tb->neg[0])[8+b/8];
                uint8_t p1=((uint8_t*)tb->pos[1])[8+b/8], n1=((uint8_t*)tb->neg[1])[8+b/8];
                uint8_t p2=((uint8_t*)tb->pos[2])[8+b/8], n2=((uint8_t*)tb->neg[2])[8+b/8];
                uint8_t p3=((uint8_t*)tb->pos[3])[8+b/8], n3=((uint8_t*)tb->neg[3])[8+b/8];
                uint8_t p4=((uint8_t*)tb->pos[4])[8+b/8], n4=((uint8_t*)tb->neg[4])[8+b/8];
                uint8_t p5=((uint8_t*)tb->pos[5])[8+b/8], n5=((uint8_t*)tb->neg[5])[8+b/8];
                uint8_t p6=((uint8_t*)tb->pos[6])[8+b/8], n6=((uint8_t*)tb->neg[6])[8+b/8];
                uint8_t p7=((uint8_t*)tb->pos[7])[8+b/8], n7=((uint8_t*)tb->neg[7])[8+b/8];
                PROCESS_WORD_AVX2_FUSED(acc0, p0, n0, sc0, av);
                PROCESS_WORD_AVX2_FUSED(acc1, p1, n1, sc1, av);
                PROCESS_WORD_AVX2_FUSED(acc2, p2, n2, sc2, av);
                PROCESS_WORD_AVX2_FUSED(acc3, p3, n3, sc3, av);
                PROCESS_WORD_AVX2_FUSED(acc4, p4, n4, sc4, av);
                PROCESS_WORD_AVX2_FUSED(acc5, p5, n5, sc5, av);
                PROCESS_WORD_AVX2_FUSED(acc6, p6, n6, sc6, av);
                PROCESS_WORD_AVX2_FUSED(acc7, p7, n7, sc7, av);
            }
        }
        C[i*N+j+0]=hsum_avx2(acc0); C[i*N+j+1]=hsum_avx2(acc1);
        C[i*N+j+2]=hsum_avx2(acc2); C[i*N+j+3]=hsum_avx2(acc3);
        C[i*N+j+4]=hsum_avx2(acc4); C[i*N+j+5]=hsum_avx2(acc5);
        C[i*N+j+6]=hsum_avx2(acc6); C[i*N+j+7]=hsum_avx2(acc7);
    }
    
    for (int i = 0; i < M; i++) {
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

static int _diag_test_f32(const char *name, const float *got, const float *exp, int n) {
    int ok = 1;
    for (int i = 0; i < n; i++) {
        float d = fabsf(got[i] - exp[i]);
        if (d > 1e-5f && d > 1e-5f * fabsf(exp[i])) {
            fprintf(stdout, "  %-40s FAIL lane %d: got %8.4f exp %8.4f\n", name, i, got[i], exp[i]);
            fflush(stdout);
            ok = 0;
        }
    }
    if (ok) { fprintf(stdout, "  %-40s PASS\n", name); fflush(stdout); }
    return ok;
}

void avx512_diagnostic(void) {
    static int done = 0;
    if (done) return; done = 1;
    fprintf(stdout, "[DIAG] AVX-512 instruction diagnostic\n");
    fflush(stdout);
    float av_in[16], exp[16], got[16];
    for (int i = 0; i < 16; i++) av_in[i] = (float)(i + 1);
    float sc_val = 3.0f;
    __m512 av = _mm512_loadu_ps(av_in);
    __m512 sc = _mm512_set1_ps(sc_val);
    __m512 zero = _mm512_setzero_ps();
    __m512i one = _mm512_set1_epi32(1);

    // 1. Unmasked FMADD (control)
    __m512 acc = _mm512_fmadd_ps(av, sc, zero);
    _mm512_storeu_ps(got, acc);
    for (int i = 0; i < 16; i++) exp[i] = av_in[i] * sc_val;
    _diag_test_f32("VFMADD (unmasked)", got, exp, 16);

    // 2. Merge-masked VFMADD (odd lanes)
    __mmask16 k_odd = 0xAAAA;
    acc = zero;
    acc = _mm512_mask_fmadd_ps(acc, k_odd, sc, av);
    _mm512_storeu_ps(got, acc);
    for (int i = 0; i < 16; i++) exp[i] = (i & 1) ? av_in[i] * sc_val : 0.0f;
    _diag_test_f32("VFMADD (merge-mask, odd lanes)", got, exp, 16);

    // 3. Merge-masked VFNMADD (odd lanes)
    acc = zero;
    acc = _mm512_mask_fnmadd_ps(acc, k_odd, sc, av);
    _mm512_storeu_ps(got, acc);
    for (int i = 0; i < 16; i++) exp[i] = (i & 1) ? -(av_in[i] * sc_val) : 0.0f;
    _diag_test_f32("VFNMADD (merge-mask, odd lanes)", got, exp, 16);

    // 4. Sequential merge-masked VFMADD + VFMADD(nsc) (mask kernel pattern)
    __mmask16 k_pos = 0x0005;  // lanes 0,2
    __mmask16 k_neg = 0x0050;  // lanes 4,6
    acc = zero;
    acc = _mm512_mask_fmadd_ps(acc, k_pos, sc, av);
    __m512 nsc = _mm512_xor_ps(sc, _mm512_set1_ps(-0.0f));
    acc = _mm512_mask_fmadd_ps(acc, k_neg, nsc, av);
    _mm512_storeu_ps(got, acc);
    for (int i = 0; i < 16; i++) {
        if (i == 0 || i == 2) exp[i] = sc_val * av_in[i];
        else if (i == 4 || i == 6) exp[i] = -sc_val * av_in[i];
        else exp[i] = 0.0f;
    }
    _diag_test_f32("mask kernel (VFMADD+VFMADD-nsc)", got, exp, 16);

    // 5. Zero-masked VMOVAPS
    acc = _mm512_maskz_mov_ps(k_odd, av);
    _mm512_storeu_ps(got, acc);
    for (int i = 0; i < 16; i++) exp[i] = (i & 1) ? av_in[i] : 0.0f;
    _diag_test_f32("VMOVAPS (zero-mask)", got, exp, 16);

    // 6. Zero-masked VPBROADCASTD (0x80000000)
    __m512i sgn_bits = _mm512_maskz_set1_epi32(k_odd, 0x80000000);
    _mm512_storeu_ps(got, _mm512_castsi512_ps(sgn_bits));
    for (int i = 0; i < 16; i++) exp[i] = (i & 1) ? -0.0f : 0.0f;
    _diag_test_f32("VPBROADCASTD zero 0x80000000", got, exp, 16);

    // 7. Zero-masked VPSLLD (1 << 31)
    sgn_bits = _mm512_maskz_slli_epi32(k_odd, one, 31);
    _mm512_storeu_ps(got, _mm512_castsi512_ps(sgn_bits));
    _diag_test_f32("VPSLLD zero 1<<31", got, exp, 16);

    // 8. Merge-masked VPBLENDMD (±av)
    __m512 neg_av = _mm512_xor_ps(av, _mm512_set1_ps(-0.0f));
    acc = _mm512_mask_blend_ps(k_odd, av, neg_av);
    _mm512_storeu_ps(got, acc);
    for (int i = 0; i < 16; i++) exp[i] = (i & 1) ? -av_in[i] : av_in[i];
    _diag_test_f32("VPBLENDMD merge ±av", got, exp, 16);

    // 9. Zero-masked blend via AND+XOR (mask-to-vector)
    __m512i mag_bits = _mm512_maskz_set1_epi32(k_odd, -1);
    sgn_bits = _mm512_maskz_slli_epi32(k_neg, one, 31);
    __m512 sf = _mm512_castsi512_ps(sgn_bits);
    __m512 w = _mm512_and_ps(_mm512_xor_ps(sc, sf), _mm512_castsi512_ps(mag_bits));
    acc = _mm512_fmadd_ps(w, av, zero);
    _mm512_storeu_ps(got, acc);
    for (int i = 0; i < 16; i++) {
        if (i == 0 || i == 2) exp[i] = sc_val * av_in[i];
        else if (i == 4 || i == 6) exp[i] = -sc_val * av_in[i];
        else exp[i] = 0.0f;
    }
    _diag_test_f32("mask-to-vector (zero-mask + AND+XOR+FMADD)", got, exp, 16);

    fprintf(stdout, "[DIAG] AVX-512 diagnostic complete\n");
    fflush(stdout);
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

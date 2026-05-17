#pragma once

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define BITS_PER_LONG 64
#define G128_BLOCK_SIZE 128

typedef struct __attribute__((aligned(64))) {
    uint64_t pos[8][4];   // 256 bytes: pos bitmaps for 8 rows
    uint64_t neg[8][4];   // 256 bytes: neg bitmaps
    float    scales[8];   // 32 bytes: one scale per row
    float    _pad[7];     // 28 bytes pad → 576 bytes total
} TileBlock8;

typedef struct {
    uint32_t num_rows;      
    uint32_t num_cols;      
    uint32_t num_blocks_row;
    uint32_t num_blocks_col;
    uint64_t *magnitude;
    uint64_t *sign;
    uint64_t *packed_pos;   // weight==+scale bitmap: 4 × uint64 per block (AVX-512)
    uint64_t *packed_neg;   // weight==-scale bitmap: 4 × uint64 per block (AVX-512)
    uint16_t *scales;
    float    *scales_f32;
    TileBlock8 *tiles8;     // 8-row tiled layout for optimized matmul
    uint64_t num_tile_groups8;  // N/8
    uint32_t total_tiles8;      // num_tile_groups8 * num_blocks_col
} G128Matrix;

void g128_matrix_init(G128Matrix *m, uint32_t num_rows, uint32_t num_cols);
void g128_matrix_free(G128Matrix *m);

void matmul_naive_f32(float *A, float *B, float *C, int M, int K, int N);
void matmul_naive_transpose(float *A, float *B_T, float *C, int M, int K, int N);

void matmul_bitnet_g128(float *A, G128Matrix *B_T, float *C, int M, int K, int N);
void matmul_simd_g128(float *A, G128Matrix *B_T, float *C, int M, int K, int N);
void matmul_swar_g128(float *A, G128Matrix *B_T, float *C, int M, int K, int N);
void matmul_lut_g128(float *A, G128Matrix *B_T, float *C, int M, int K, int N);

// lm_head prefilter helpers
extern const int lm_head_prefilter_available;
void lm_head_prefilter(float *A, G128Matrix *B_T, float *C, int N, int max_blocks);
void matmul_g128_selected(float *A, G128Matrix *B_T, float *C, int M, int K, int N_full, int N_sel, const int *sel_rows);
void find_top_k(float *scores, int N, int K, int *out_indices);
void avx512_diagnostic(void);

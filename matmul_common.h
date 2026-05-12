#pragma once

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define BITS_PER_LONG 64
#define G128_BLOCK_SIZE 128

typedef struct {
    uint32_t num_rows;      
    uint32_t num_cols;      
    uint32_t num_blocks_row;
    uint32_t num_blocks_col;
    uint64_t *magnitude;
    uint64_t *sign;
    uint16_t *scales;
    float    *scales_f32;
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

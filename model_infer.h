#pragma once

#include "matmul_common.h"
#include <stdint.h>
#include <stdbool.h>

#define HIDDEN_SIZE 2048
#define INTERMEDIATE_SIZE 6144
#define NUM_HEADS 16
#define HEAD_DIM 128
#define NUM_KV_HEADS 8
#define NUM_LAYERS 28
#define VOCAB_SIZE 151669
#define MAX_SEQ_LEN 512

typedef struct {
    float *data;
    int num_rows, num_cols;
} FP32Matrix;

typedef struct {
    FP32Matrix ln1, ln2, q_norm, k_norm;
    G128Matrix q_proj, k_proj, v_proj, o_proj;
    G128Matrix gate_proj, up_proj, down_proj;
} LayerWeights;

typedef struct {
    uint64_t decode_count;      // number of decode steps profiled
    double   matmul_ns;         // cumulative ns in G128 matmuls (decode only)
    double   attn_ns;           // cumulative ns in attention core (decode only)
    double   logits_ns;         // cumulative ns in final embed projection (decode only)
    double   total_ns;          // cumulative ns across entire model_decode
} ProfileStats;

typedef struct {
    LayerWeights layers[NUM_LAYERS];
    G128Matrix embed;
    FP32Matrix final_norm;
    float *hidden, *normalized, *residual, *q, *k, *v, *attn_out, *attn_weights;
    float *gate_out, *up_out, *mlp_act;
    float kv_k[NUM_LAYERS][NUM_KV_HEADS][MAX_SEQ_LEN][HEAD_DIM];
    float kv_v[NUM_LAYERS][NUM_KV_HEADS][MAX_SEQ_LEN][HEAD_DIM];
    float inv_freq[HEAD_DIM/2];
    float attn_scale;
    int kv_len;
    bool loaded;
    ProfileStats profile;
} ModelState;

int model_load(ModelState *s, const char *dir);
void model_free(ModelState *s);
int model_prefill(ModelState *s, int32_t *tokens, int n, float *logits);
int model_decode(ModelState *s, int32_t token, float *logits);
void model_get_profile(ModelState *s, ProfileStats *out);
void model_reset_profile(ModelState *s);

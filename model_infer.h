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
#define LM_HEAD_CANDIDATES 4096

typedef enum {
    MATMUL_Q_PROJ    = 0,
    MATMUL_K_PROJ    = 1,
    MATMUL_V_PROJ    = 2,
    MATMUL_O_PROJ    = 3,
    MATMUL_GATE_PROJ = 4,
    MATMUL_UP_PROJ   = 5,
    MATMUL_DOWN_PROJ = 6,
    MATMUL_COUNT     = 7
} MatmulType;

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
    uint64_t decode_count;           // number of decode steps profiled
    double   matmul_ns;              // cumulative ns in G128 matmuls (decode only)
    double   attn_ns;                // cumulative ns in attention core (decode only)
    double   logits_ns;              // cumulative ns in final embed projection (decode only)
    double   total_ns;               // cumulative ns across entire model_decode
    double   per_matmul_ns[MATMUL_COUNT];   // per-matmul-type ns
    uint64_t per_matmul_calls[MATMUL_COUNT];   // per-type call count
    uint64_t per_matmul_elements[MATMUL_COUNT]; // per-type ternary elements processed
} ProfileStats;

typedef struct {
    LayerWeights layers[NUM_LAYERS];
    G128Matrix embed;
    FP32Matrix final_norm;
    float *hidden, *normalized, *residual, *q, *k, *v, *attn_out, *attn_weights;
    float *gate_out, *up_out, *mlp_act;
    float *approx_logits;
    int lm_head_candidates[LM_HEAD_CANDIDATES];
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
const char* model_matmul_path(void);
const char* model_compile_info(void);
int model_omp_max_threads(void);
long model_struct_size(void);
long model_debug_offset_loaded(void);
long model_debug_offset_kv_len(void);
void model_set_omp_threads(int n);
int model_affinity_cpu_count(void);

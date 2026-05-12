#include "model_infer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif

static inline float half_to_float(uint16_t h) {
    union { uint32_t u; float f; } v;
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t mant = (uint32_t)(h & 0x3FFu);
    if (exp == 0) v.u = sign;
    else if (exp == 31) v.u = sign | 0x7F800000u | (mant << 13);
    else v.u = sign | ((exp + 112u) << 23) | (mant << 13);
    return v.f;
}

static int load_g128(G128Matrix *m, const char *base) {
    char path[512];
    snprintf(path, sizeof(path), "%s_magnitude.bin", base);
    FILE *f = fopen(path, "rb");
    if (!f) return -1;

    uint32_t header_size, nlen;
    fread(&header_size, 4, 1, f);
    fread(&nlen, 4, 1, f);
    fseek(f, nlen, SEEK_CUR);  // skip name

    uint32_t nd;
    fread(&nd, 4, 1, f);  // num_dims
    uint64_t rows = 0, cols = 0;
    for (uint32_t i = 0; i < nd; i++) {
        uint64_t d; fread(&d, 8, 1, f);
        if (i == 0) rows = d; else if (i == 1) cols = d;
    }

    m->num_rows = (uint32_t)rows;
    m->num_cols = (uint32_t)(nd > 1 ? cols : rows);
    m->num_blocks_row = (m->num_rows + G128_BLOCK_SIZE - 1) / G128_BLOCK_SIZE;
    m->num_blocks_col = (m->num_cols + G128_BLOCK_SIZE - 1) / G128_BLOCK_SIZE;
    uint64_t nb = (uint64_t)m->num_rows * m->num_cols / G128_BLOCK_SIZE;

    m->magnitude = malloc(nb * 2 * sizeof(uint64_t));
    m->sign      = malloc(nb * 2 * sizeof(uint64_t));
    m->scales    = malloc(nb * sizeof(uint16_t));

    fseek(f, header_size, SEEK_SET);
    fread(m->magnitude, 2 * sizeof(uint64_t), nb, f);
    fclose(f);

    snprintf(path, sizeof(path), "%s_sign.bin", base);
    f = fopen(path, "rb"); if (!f) return -1;
    fseek(f, header_size, SEEK_SET);
    fread(m->sign, 2 * sizeof(uint64_t), nb, f);
    fclose(f);

    snprintf(path, sizeof(path), "%s_scales.bin", base);
    f = fopen(path, "rb"); if (!f) return -1;
    fseek(f, header_size, SEEK_SET);
    fread(m->scales, sizeof(uint16_t), nb, f);
    fclose(f);

    // precompute FP32 scales once to avoid per-call half_to_float during inference
    m->scales_f32 = malloc(nb * sizeof(float));
    for (uint64_t bi = 0; bi < nb; bi++)
        m->scales_f32[bi] = half_to_float(m->scales[bi]);

    return 0;
}

static int load_fp32(FP32Matrix *m, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    uint32_t hdr, nlen;
    fread(&hdr, 4, 1, f); fread(&nlen, 4, 1, f);
    fseek(f, nlen, SEEK_CUR);
    uint32_t nd; fread(&nd, 4, 1, f);
    uint64_t d0 = 0, d1 = 0;
    for (uint32_t i = 0; i < nd; i++) { uint64_t d; fread(&d, 8, 1, f); if (i == 0) d0 = d; else d1 = d; }
    uint32_t dc; fread(&dc, 4, 1, f);
    m->num_rows = d0; m->num_cols = d1 ? d1 : 1;
    m->data = malloc((uint64_t)m->num_rows * m->num_cols * 4);
    fread(m->data, 4, (uint64_t)m->num_rows * m->num_cols, f);
    fclose(f);
    return 0;
}

static void embed_lookup(G128Matrix *e, int tid, float *out) {
    int nkb = (int)e->num_blocks_col;
    int block_start = tid * nkb;
    for (int b = 0; b < nkb; b++) {
        int bidx = block_start + b;
        uint64_t mlo = e->magnitude[bidx*2+0], mhi = e->magnitude[bidx*2+1];
        uint64_t slo = e->sign[bidx*2+0],      shi = e->sign[bidx*2+1];
        float sc = e->scales_f32[bidx];
        float *o = out + b * G128_BLOCK_SIZE;
        for (int i = 0; i < 64; i++)
            o[i]    = ((mlo >> i) & 1) ? (((slo >> i) & 1) ? -sc : sc) : 0.0f;
        for (int i = 0; i < 64; i++)
            o[64+i] = ((mhi >> i) & 1) ? (((shi >> i) & 1) ? -sc : sc) : 0.0f;
    }
}

static void rms_norm(float *in, float *w, float *out, int n) {
    float rms = 0; for (int i = 0; i < n; i++) rms += in[i] * in[i];
    rms = sqrtf(rms / n + 1e-6f); float r = 1.0f / rms;
    for (int i = 0; i < n; i++) out[i] = in[i] * r * w[i];
}

static void rms_norm_head(float *x, float *w, int nh, int hd) {
    for (int h = 0; h < nh; h++) {
        float rms = 0; for (int d = 0; d < hd; d++) rms += x[h*hd+d] * x[h*hd+d];
        rms = sqrtf(rms / hd + 1e-6f); float r = 1.0f / rms;
        for (int d = 0; d < hd; d++) x[h*hd+d] *= r * w[d];
    }
}

static void silu(float *in, float *out, int n) {
    for (int i = 0; i < n; i++) { float x = in[i]; out[i] = x / (1.0f + expf(-x)); }
}

static void softmax(float *s, int n) {
    float m = s[0]; for (int i = 1; i < n; i++) if (s[i] > m) m = s[i];
    float sum = 0; for (int i = 0; i < n; i++) { s[i] = expf(s[i] - m); sum += s[i]; }
    for (int i = 0; i < n; i++) s[i] /= sum;
}

static void apply_rope(float *q, int seqlen, int nh, int hd, float *invf, float as, int pos_offset) {
    int half = hd / 2;
    for (int pos = 0; pos < seqlen; pos++) {
        for (int h = 0; h < nh; h++) {
            int off = pos * nh * hd + h * hd;
            for (int i = 0; i < half; i++) {
                float a = (float)(pos + pos_offset) * invf[i];
                float c = cosf(a) * as, s = sinf(a) * as;
                float q0 = q[off + i], q1 = q[off + half + i];
                q[off + i] = q0 * c - q1 * s;
                q[off + half + i] = q0 * s + q1 * c;
            }
        }
    }
}

// decode_pos: ignored during prefill; for decode = position of the new token in the sequence
static void forward_layer(ModelState *s, int lid, int seqlen, int prefill, int decode_pos) {
    LayerWeights *lw = &s->layers[lid];
    float *h = s->hidden, *n = s->normalized, *res = s->residual;
    float *q = s->q, *k = s->k, *v = s->v, *ao = s->attn_out, *aw = s->attn_weights;
    float *go = s->gate_out, *uo = s->up_out, *ma = s->mlp_act;

    memcpy(res, h, seqlen * HIDDEN_SIZE * 4);
    for (int i = 0; i < seqlen; i++) rms_norm(&h[i*HIDDEN_SIZE], lw->ln1.data, &n[i*HIDDEN_SIZE], HIDDEN_SIZE);

    matmul_simd_g128(n, &lw->q_proj, q, seqlen, HIDDEN_SIZE, HIDDEN_SIZE);
    matmul_simd_g128(n, &lw->k_proj, k, seqlen, HIDDEN_SIZE, NUM_KV_HEADS*HEAD_DIM);
    matmul_simd_g128(n, &lw->v_proj, v, seqlen, HIDDEN_SIZE, NUM_KV_HEADS*HEAD_DIM);

    for (int i = 0; i < seqlen; i++) {
        rms_norm_head(&q[i*HIDDEN_SIZE], lw->q_norm.data, NUM_HEADS, HEAD_DIM);
        rms_norm_head(&k[i*NUM_KV_HEADS*HEAD_DIM], lw->k_norm.data, NUM_KV_HEADS, HEAD_DIM);
    }

    int rope_offset = prefill ? 0 : decode_pos;
    apply_rope(q, seqlen, NUM_HEADS, HEAD_DIM, s->inv_freq, s->attn_scale, rope_offset);
    apply_rope(k, seqlen, NUM_KV_HEADS, HEAD_DIM, s->inv_freq, s->attn_scale, rope_offset);

    // KV cache write (head-major layout: kv_k[lid][head][pos][dim])
    if (prefill) {
        for (int i = 0; i < seqlen; i++) {
            for (int h = 0; h < NUM_KV_HEADS; h++) {
                memcpy(&s->kv_k[lid][h][i][0], &k[i*NUM_KV_HEADS*HEAD_DIM + h*HEAD_DIM], HEAD_DIM*4);
                memcpy(&s->kv_v[lid][h][i][0], &v[i*NUM_KV_HEADS*HEAD_DIM + h*HEAD_DIM], HEAD_DIM*4);
            }
        }
    } else {
        for (int h = 0; h < NUM_KV_HEADS; h++) {
            memcpy(&s->kv_k[lid][h][decode_pos][0], &k[h*HEAD_DIM], HEAD_DIM*4);
            memcpy(&s->kv_v[lid][h][decode_pos][0], &v[h*HEAD_DIM], HEAD_DIM*4);
        }
    }

    int kv_len = prefill ? seqlen : decode_pos + 1;
    for (int h = 0; h < NUM_HEADS; h++) {
        int qkh = h / (NUM_HEADS / NUM_KV_HEADS);
        float *k_cache_head = &s->kv_k[lid][qkh][0][0];
        float *v_cache_head = &s->kv_v[lid][qkh][0][0];
        for (int i = 0; i < seqlen; i++) {
            int query_pos = prefill ? i : decode_pos;
            float *q_head = &q[i*HIDDEN_SIZE + h*HEAD_DIM];
            float *aw_row = &aw[i * kv_len];
            for (int j = 0; j < kv_len; j++) {
                if (j > query_pos) { aw_row[j] = -1e38f; continue; }
                float sc = 0;
                float *kc = &k_cache_head[j * HEAD_DIM];
                for (int d = 0; d < HEAD_DIM; d++) sc += q_head[d] * kc[d];
                aw_row[j] = sc / sqrtf((float)HEAD_DIM);
            }
            softmax(aw_row, kv_len);
        }
        for (int i = 0; i < seqlen; i++) {
            float *aw_row = &aw[i * kv_len];
            float *ao_head = &ao[i*HIDDEN_SIZE + h*HEAD_DIM];
            for (int d = 0; d < HEAD_DIM; d++) {
                float val = 0;
                for (int j = 0; j < kv_len; j++) val += aw_row[j] * v_cache_head[j * HEAD_DIM + d];
                ao_head[d] = val;
            }
        }
    }

    matmul_simd_g128(ao, &lw->o_proj, h, seqlen, HIDDEN_SIZE, HIDDEN_SIZE);
    for (int i = 0; i < seqlen*HIDDEN_SIZE; i++) h[i] += res[i];
    memcpy(res, h, seqlen*HIDDEN_SIZE*4);

    for (int i = 0; i < seqlen; i++) rms_norm(&h[i*HIDDEN_SIZE], lw->ln2.data, &n[i*HIDDEN_SIZE], HIDDEN_SIZE);

    matmul_simd_g128(n, &lw->gate_proj, go, seqlen, HIDDEN_SIZE, INTERMEDIATE_SIZE);
    matmul_simd_g128(n, &lw->up_proj, uo, seqlen, HIDDEN_SIZE, INTERMEDIATE_SIZE);
    silu(go, ma, seqlen*INTERMEDIATE_SIZE);
    for (int i = 0; i < seqlen*INTERMEDIATE_SIZE; i++) go[i] = ma[i] * uo[i];
    matmul_simd_g128(go, &lw->down_proj, h, seqlen, INTERMEDIATE_SIZE, HIDDEN_SIZE);
    for (int i = 0; i < seqlen*HIDDEN_SIZE; i++) h[i] += res[i];
}

int model_load(ModelState *s, const char *dir) {
    memset(s, 0, sizeof(ModelState));
    char p[512];

    snprintf(p, sizeof(p), "%s/weight_model_embed_tokens_weight", dir);
    if (load_g128(&s->embed, p) != 0) return -1;

    for (int i = 0; i < NUM_LAYERS; i++) {
        snprintf(p, sizeof(p), "%s/weight_model_layers_%d_input_layernorm_weight.bin", dir, i);
        if (load_fp32(&s->layers[i].ln1, p) != 0) return -1;
        snprintf(p, sizeof(p), "%s/weight_model_layers_%d_post_attention_layernorm_weight.bin", dir, i);
        if (load_fp32(&s->layers[i].ln2, p) != 0) return -1;
        snprintf(p, sizeof(p), "%s/weight_model_layers_%d_self_attn_q_norm_weight.bin", dir, i);
        if (load_fp32(&s->layers[i].q_norm, p) != 0) return -1;
        snprintf(p, sizeof(p), "%s/weight_model_layers_%d_self_attn_k_norm_weight.bin", dir, i);
        if (load_fp32(&s->layers[i].k_norm, p) != 0) return -1;

        snprintf(p, sizeof(p), "%s/weight_model_layers_%d_self_attn_q_proj_weight", dir, i);
        if (load_g128(&s->layers[i].q_proj, p) != 0) return -1;
        snprintf(p, sizeof(p), "%s/weight_model_layers_%d_self_attn_k_proj_weight", dir, i);
        if (load_g128(&s->layers[i].k_proj, p) != 0) return -1;
        snprintf(p, sizeof(p), "%s/weight_model_layers_%d_self_attn_v_proj_weight", dir, i);
        if (load_g128(&s->layers[i].v_proj, p) != 0) return -1;
        snprintf(p, sizeof(p), "%s/weight_model_layers_%d_self_attn_o_proj_weight", dir, i);
        if (load_g128(&s->layers[i].o_proj, p) != 0) return -1;
        snprintf(p, sizeof(p), "%s/weight_model_layers_%d_mlp_gate_proj_weight", dir, i);
        if (load_g128(&s->layers[i].gate_proj, p) != 0) return -1;
        snprintf(p, sizeof(p), "%s/weight_model_layers_%d_mlp_up_proj_weight", dir, i);
        if (load_g128(&s->layers[i].up_proj, p) != 0) return -1;
        snprintf(p, sizeof(p), "%s/weight_model_layers_%d_mlp_down_proj_weight", dir, i);
        if (load_g128(&s->layers[i].down_proj, p) != 0) return -1;
    }

    snprintf(p, sizeof(p), "%s/weight_model_norm_weight.bin", dir);
    if (load_fp32(&s->final_norm, p) != 0) return -1;

    s->hidden = calloc(MAX_SEQ_LEN * HIDDEN_SIZE, 4);
    s->normalized = calloc(MAX_SEQ_LEN * HIDDEN_SIZE, 4);
    s->residual = calloc(MAX_SEQ_LEN * HIDDEN_SIZE, 4);
    s->q = calloc(MAX_SEQ_LEN * HIDDEN_SIZE, 4);
    s->k = calloc(MAX_SEQ_LEN * NUM_KV_HEADS * HEAD_DIM, 4);
    s->v = calloc(MAX_SEQ_LEN * NUM_KV_HEADS * HEAD_DIM, 4);
    s->attn_out = calloc(MAX_SEQ_LEN * HIDDEN_SIZE, 4);
    s->attn_weights = calloc(MAX_SEQ_LEN * MAX_SEQ_LEN, 4);
    s->gate_out = calloc(MAX_SEQ_LEN * INTERMEDIATE_SIZE, 4);
    s->up_out = calloc(MAX_SEQ_LEN * INTERMEDIATE_SIZE, 4);
    s->mlp_act = calloc(MAX_SEQ_LEN * INTERMEDIATE_SIZE, 4);

    // YaRN RoPE: blend interpolated (low-freq) and extrapolated (high-freq) inv_freq
    {
        float base = 1000000.0f, factor = 4.0f, beta_fast = 32.0f, beta_slow = 1.0f;
        float orig_max_pos = 8192.0f;
        float dim_f = (float)HEAD_DIM;
        int half_dim = HEAD_DIM / 2;
        float pi = 3.14159265358979323846f;
        float low_f  = (dim_f * logf(orig_max_pos / (beta_fast * 2.0f * pi))) / (2.0f * logf(base));
        float high_f = (dim_f * logf(orig_max_pos / (beta_slow * 2.0f * pi))) / (2.0f * logf(base));
        int low  = (int)floorf(low_f); if (low  < 0)            low  = 0;
        int high = (int)ceilf(high_f); if (high > half_dim - 1) high = half_dim - 1;
        for (int i = 0; i < half_dim; i++) {
            float pf        = powf(base, (2.0f * i) / dim_f);
            float inv_extrap = 1.0f / pf;
            float inv_interp = 1.0f / (factor * pf);
            float ramp = (high != low) ? ((float)i - low) / (float)(high - low) : (i >= high ? 1.0f : 0.0f);
            if (ramp < 0.0f) ramp = 0.0f;
            if (ramp > 1.0f) ramp = 1.0f;
            s->inv_freq[i] = inv_interp * ramp + inv_extrap * (1.0f - ramp);
        }
    }
    s->attn_scale = 1.0f + 0.1f * logf(4.0f);

    s->loaded = true;
    return 0;
}

static void free_g128(G128Matrix *m) {
    free(m->magnitude); free(m->sign); free(m->scales); free(m->scales_f32);
}

void model_free(ModelState *s) {
    free_g128(&s->embed);
    free(s->final_norm.data);
    for (int i = 0; i < NUM_LAYERS; i++) {
        LayerWeights *lw = &s->layers[i];
        free(lw->ln1.data); free(lw->ln2.data); free(lw->q_norm.data); free(lw->k_norm.data);
        free_g128(&lw->q_proj); free_g128(&lw->k_proj); free_g128(&lw->v_proj);
        free_g128(&lw->o_proj); free_g128(&lw->gate_proj);
        free_g128(&lw->up_proj); free_g128(&lw->down_proj);
    }
    free(s->hidden); free(s->normalized); free(s->residual);
    free(s->q); free(s->k); free(s->v); free(s->attn_out);
    free(s->attn_weights);
    free(s->gate_out); free(s->up_out); free(s->mlp_act);
}

int model_prefill(ModelState *s, int32_t *tokens, int n, float *logits) {
    if (!s->loaded || n > MAX_SEQ_LEN) return -1;
    s->kv_len = 0;

    for (int i = 0; i < n; i++) {
        int tid = tokens[i];
        if (tid < 0 || tid >= s->embed.num_rows) return -1;
        embed_lookup(&s->embed, tid, &s->hidden[i*HIDDEN_SIZE]);
    }

    for (int i = 0; i < NUM_LAYERS; i++) forward_layer(s, i, n, 1, 0);
    s->kv_len = n;

    rms_norm(&s->hidden[(n-1)*HIDDEN_SIZE], s->final_norm.data, s->normalized, HIDDEN_SIZE);
    matmul_simd_g128(s->normalized, &s->embed, logits, 1, HIDDEN_SIZE, (int)s->embed.num_rows);
    return 0;
}

int model_decode(ModelState *s, int32_t token, float *logits) {
    if (!s->loaded) return -1;
    if (token < 0 || token >= s->embed.num_rows) return -1;
    embed_lookup(&s->embed, token, s->hidden);

    int pos = s->kv_len;
    if (pos >= MAX_SEQ_LEN) return -1;

    for (int i = 0; i < NUM_LAYERS; i++) forward_layer(s, i, 1, 0, pos);
    s->kv_len = pos + 1;

    rms_norm(s->hidden, s->final_norm.data, s->normalized, HIDDEN_SIZE);
    matmul_simd_g128(s->normalized, &s->embed, logits, 1, HIDDEN_SIZE, (int)s->embed.num_rows);
    return 0;
}

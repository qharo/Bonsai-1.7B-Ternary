// Expose POSIX clock_gettime even under -std=c11 — must precede ALL system includes
#ifndef __MACH__
#define _GNU_SOURCE
#endif

#include "model_infer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef __MACH__
#include <mach/mach_time.h>
#else
#include <time.h>
#endif
#include <sched.h>

// 64-byte aligned calloc for AVX-512 (safe on all POSIX platforms)
static inline void *aligned_calloc(size_t alignment, size_t size) {
    void *ptr = NULL;
    if (posix_memalign(&ptr, alignment, size) != 0) return NULL;
    memset(ptr, 0, size);
    return ptr;
}

static inline uint64_t now_ns(void) {
#ifdef __MACH__
    static mach_timebase_info_data_t info = {0};
    if (info.denom == 0) mach_timebase_info(&info);
    return mach_absolute_time() * info.numer / info.denom;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
#endif
}

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

    posix_memalign((void**)&m->magnitude, 64, nb * 2 * sizeof(uint64_t));
    posix_memalign((void**)&m->sign, 64, nb * 2 * sizeof(uint64_t));
    m->scales = (uint16_t*)malloc(nb * sizeof(uint16_t));

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
    posix_memalign((void**)&m->scales_f32, 64, nb * sizeof(float));
    for (uint64_t bi = 0; bi < nb; bi++)
        m->scales_f32[bi] = half_to_float(m->scales[bi]);

    // build pre-separated pos/neg packed arrays for AVX-512:
    // 4 × uint64 per block per array. Each uint64 holds two 32-bit masks:
    // lower 16 bits = first 16-elem group, upper 16 bits = second 16-elem group.
    // pos_mask = mag & ~sign, neg_mask = mag & sign — computed once at load time.
    posix_memalign((void**)&m->packed_pos, 64, nb * 4 * sizeof(uint64_t));
    posix_memalign((void**)&m->packed_neg, 64, nb * 4 * sizeof(uint64_t));
    for (uint64_t bi = 0; bi < nb; bi++) {
        uint64_t m0 = m->magnitude[bi*2+0], m1 = m->magnitude[bi*2+1];
        uint64_t s0 = m->sign[bi*2+0],      s1 = m->sign[bi*2+1];
        uint64_t p0 = m0 & ~s0, p1 = m1 & ~s1;
        uint64_t n0 = m0 & s0,  n1 = m1 & s1;
#define PK16(bits) ((uint32_t)(bits) & 0xFFFF)
        m->packed_pos[bi*4+0] = (uint64_t)PK16(p0>>16) << 32 | PK16(p0);
        m->packed_pos[bi*4+1] = (uint64_t)PK16(p0>>48) << 32 | PK16(p0>>32);
        m->packed_pos[bi*4+2] = (uint64_t)PK16(p1>>16) << 32 | PK16(p1);
        m->packed_pos[bi*4+3] = (uint64_t)PK16(p1>>48) << 32 | PK16(p1>>32);
        m->packed_neg[bi*4+0] = (uint64_t)PK16(n0>>16) << 32 | PK16(n0);
        m->packed_neg[bi*4+1] = (uint64_t)PK16(n0>>48) << 32 | PK16(n0>>32);
        m->packed_neg[bi*4+2] = (uint64_t)PK16(n1>>16) << 32 | PK16(n1);
        m->packed_neg[bi*4+3] = (uint64_t)PK16(n1>>48) << 32 | PK16(n1>>32);
#undef PK16
    }

    // Build 8-row tiled layout for optimized matmul
    m->num_tile_groups8 = m->num_rows / 8;
    m->total_tiles8 = (uint32_t)(m->num_tile_groups8 * m->num_blocks_col);
    if (m->num_tile_groups8 > 0 && m->total_tiles8 > 0) {
        posix_memalign((void**)&m->tiles8, 64, 
            m->total_tiles8 * sizeof(TileBlock8));
        
        for (uint64_t tg = 0; tg < m->num_tile_groups8; tg++) {
            for (uint32_t bk = 0; bk < m->num_blocks_col; bk++) {
                TileBlock8 *tb = &m->tiles8[tg * m->num_blocks_col + bk];
                for (int r = 0; r < 8; r++) {
                    uint64_t row = tg * 8 + r;
                    if (row >= m->num_rows) break;
                    uint64_t bi = row * m->num_blocks_col + bk;
                    for (int w = 0; w < 4; w++) {
                        tb->pos[r][w] = m->packed_pos[bi*4 + w];
                        tb->neg[r][w] = m->packed_neg[bi*4 + w];
                    }
                    tb->scales[r] = m->scales_f32[bi];
                }
            }
        }
    }

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

#ifdef __AVX512F__
#include <immintrin.h>

static inline __m512 exp_approx512(__m512 x) {
    __m512 one = _mm512_set1_ps(1.0f);
    __m512 a = _mm512_fmadd_ps(x, _mm512_set1_ps(1.0f/256.0f), one);
    __m512 b = a; for (int i = 0; i < 8; i++) b = _mm512_mul_ps(b, b);
    return b;
}

static void silu_mul(float *gate, float *up, float *out, int n) {
    for (int i = 0; i <= n - 16; i += 16) {
        __m512 g = _mm512_load_ps(gate + i);
        __m512 u = _mm512_load_ps(up + i);
        __m512 neg_g = _mm512_xor_ps(g, _mm512_set1_ps(-0.0f));
        __m512 exp_neg_g = exp_approx512(neg_g);
        __m512 denom = _mm512_add_ps(_mm512_set1_ps(1.0f), exp_neg_g);
        __m512 sig = _mm512_div_ps(g, denom);
        _mm512_store_ps(out + i, _mm512_mul_ps(sig, u));
    }
    for (int i = (n / 16) * 16; i < n; i++) {
        float x = gate[i]; out[i] = (x / (1.0f + expf(-x))) * up[i];
    }
}
#else
static void silu_mul(float *gate, float *up, float *out, int n) {
    for (int i = 0; i < n; i++) {
        float x = gate[i]; out[i] = (x / (1.0f + expf(-x))) * up[i];
    }
}
#endif

static void softmax(float *s, int n) {
    float m = s[0]; for (int i = 1; i < n; i++) if (s[i] > m) m = s[i];
    float sum = 0; for (int i = 0; i < n; i++) { s[i] = expf(s[i] - m); sum += s[i]; }
    for (int i = 0; i < n; i++) s[i] /= sum;
}

#ifdef __AVX512F__

static inline float hsum_zmm_local(__m512 v) {
    __m256 lo = _mm512_castps512_ps256(v);
    __m256 hi = _mm512_extractf32x8_ps(v, 1);
    __m256 s2 = _mm256_add_ps(lo, hi);
    __m128 s4 = _mm_add_ps(_mm256_castps256_ps128(s2), _mm256_extractf128_ps(s2, 1));
    s4 = _mm_hadd_ps(s4, s4);
    s4 = _mm_hadd_ps(s4, s4);
    return _mm_cvtss_f32(s4);
}

static void attn_qk_dot_avx512(const float *q_head, const float *k_base, int kv_len,
                                float *aw_row, float inv_sqrt_hd) {
    for (int j = 0; j < kv_len; j++) {
        const float *kc = &k_base[j * HEAD_DIM];
        __m512 acc = _mm512_setzero_ps();
        for (int d = 0; d < HEAD_DIM; d += 16)
            acc = _mm512_fmadd_ps(_mm512_load_ps(q_head + d), _mm512_load_ps(kc + d), acc);
        aw_row[j] = hsum_zmm_local(acc) * inv_sqrt_hd;
    }
}

static void attn_vsum_avx512(const float *aw_row, const float *v_base, int kv_len,
                              float *ao_head) {
    for (int d = 0; d < HEAD_DIM; d += 16) {
        __m512 vsum = _mm512_setzero_ps();
        for (int j = 0; j < kv_len; j++) {
            __m512 w = _mm512_set1_ps(aw_row[j]);
            vsum = _mm512_fmadd_ps(w, _mm512_load_ps(&v_base[j * HEAD_DIM + d]), vsum);
        }
        _mm512_store_ps(ao_head + d, vsum);
    }
}

#endif

static void apply_rope(float *q, int seqlen, int nh, int hd, float rope_cos[][HEAD_DIM/2], float rope_sin[][HEAD_DIM/2], int pos_offset) {
    int half = hd / 2;
    for (int pos = 0; pos < seqlen; pos++) {
        int rp = pos + pos_offset;
        float *rc = rope_cos[rp], *rs = rope_sin[rp];
        for (int h = 0; h < nh; h++) {
            int off = pos * nh * hd + h * hd;
            for (int i = 0; i < half; i++) {
                float c = rc[i], s = rs[i];
                float q0 = q[off + i], q1 = q[off + half + i];
                q[off + i] = q0 * c - q1 * s;
                q[off + half + i] = q0 * s + q1 * c;
            }
        }
    }
}

// Profile helper: accumulate timing + element count for a single matmul
static inline void profile_matmul(ModelState *s, int type, uint64_t t0, int seqlen, int K, int N) {
    uint64_t t1 = now_ns();
    double dt = (double)(t1 - t0);
    s->profile.matmul_ns += dt;
    s->profile.per_matmul_ns[type] += dt;
    s->profile.per_matmul_calls[type]++;
    s->profile.per_matmul_elements[type] += (uint64_t)seqlen * (uint64_t)K * (uint64_t)N;
}

// decode_pos: ignored during prefill; for decode = position of the new token in the sequence
static void forward_layer(ModelState *s, int lid, int seqlen, int prefill, int decode_pos) {
    LayerWeights *lw = &s->layers[lid];
    float *h = s->hidden, *n = s->normalized, *res = s->residual;
    float *q = s->q, *k = s->k, *v = s->v, *ao = s->attn_out, *aw = s->attn_weights;
    float *go = s->gate_out, *uo = s->up_out;
    uint64_t _t0 = 0, _t1 = 0;

    memcpy(res, h, seqlen * HIDDEN_SIZE * 4);
    for (int i = 0; i < seqlen; i++) rms_norm(&h[i*HIDDEN_SIZE], lw->ln1.data, &n[i*HIDDEN_SIZE], HIDDEN_SIZE);

    if (!prefill) _t0 = now_ns();
    matmul_simd_g128(n, &lw->q_proj, q, seqlen, HIDDEN_SIZE, HIDDEN_SIZE);
    if (!prefill) profile_matmul(s, MATMUL_Q_PROJ, _t0, seqlen, HIDDEN_SIZE, HIDDEN_SIZE);

    if (!prefill) _t0 = now_ns();
    matmul_simd_g128(n, &lw->k_proj, k, seqlen, HIDDEN_SIZE, NUM_KV_HEADS*HEAD_DIM);
    if (!prefill) profile_matmul(s, MATMUL_K_PROJ, _t0, seqlen, HIDDEN_SIZE, NUM_KV_HEADS*HEAD_DIM);

    if (!prefill) _t0 = now_ns();
    matmul_simd_g128(n, &lw->v_proj, v, seqlen, HIDDEN_SIZE, NUM_KV_HEADS*HEAD_DIM);
    if (!prefill) profile_matmul(s, MATMUL_V_PROJ, _t0, seqlen, HIDDEN_SIZE, NUM_KV_HEADS*HEAD_DIM);

    for (int i = 0; i < seqlen; i++) {
        rms_norm_head(&q[i*HIDDEN_SIZE], lw->q_norm.data, NUM_HEADS, HEAD_DIM);
        rms_norm_head(&k[i*NUM_KV_HEADS*HEAD_DIM], lw->k_norm.data, NUM_KV_HEADS, HEAD_DIM);
    }

    int rope_offset = prefill ? 0 : decode_pos;
    apply_rope(q, seqlen, NUM_HEADS, HEAD_DIM, s->rope_cos, s->rope_sin, rope_offset);
    apply_rope(k, seqlen, NUM_KV_HEADS, HEAD_DIM, s->rope_cos, s->rope_sin, rope_offset);

    if (!prefill) _t0 = now_ns();
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
    float inv_sqrt_hd = 1.0f / sqrtf((float)HEAD_DIM);
    for (int h = 0; h < NUM_HEADS; h++) {
        int qkh = h / (NUM_HEADS / NUM_KV_HEADS);
        float *k_cache_head = &s->kv_k[lid][qkh][0][0];
        float *v_cache_head = &s->kv_v[lid][qkh][0][0];
#ifdef __AVX512F__
        for (int i = 0; i < seqlen; i++) {
            int query_pos = prefill ? i : decode_pos;
            float *q_head = &q[i*HIDDEN_SIZE + h*HEAD_DIM];
            float *aw_row = &aw[i * kv_len];
            attn_qk_dot_avx512(q_head, k_cache_head, kv_len, aw_row, inv_sqrt_hd);
            for (int j = query_pos + 1; j < kv_len; j++) aw_row[j] = -1e38f;
            softmax(aw_row, kv_len);
        }
        for (int i = 0; i < seqlen; i++) {
            float *aw_row = &aw[i * kv_len];
            float *ao_head = &ao[i*HIDDEN_SIZE + h*HEAD_DIM];
            attn_vsum_avx512(aw_row, v_cache_head, kv_len, ao_head);
        }
#else
        for (int i = 0; i < seqlen; i++) {
            int query_pos = prefill ? i : decode_pos;
            float *q_head = &q[i*HIDDEN_SIZE + h*HEAD_DIM];
            float *aw_row = &aw[i * kv_len];
            for (int j = 0; j < kv_len; j++) {
                if (j > query_pos) { aw_row[j] = -1e38f; continue; }
                float sc = 0;
                float *kc = &k_cache_head[j * HEAD_DIM];
                for (int d = 0; d < HEAD_DIM; d++) sc += q_head[d] * kc[d];
                aw_row[j] = sc * inv_sqrt_hd;
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
#endif
    }
    if (!prefill) { _t1 = now_ns(); s->profile.attn_ns += (double)(_t1 - _t0); }

    if (!prefill) _t0 = now_ns();
    matmul_simd_g128(ao, &lw->o_proj, h, seqlen, HIDDEN_SIZE, HIDDEN_SIZE);
    if (!prefill) profile_matmul(s, MATMUL_O_PROJ, _t0, seqlen, HIDDEN_SIZE, HIDDEN_SIZE);
    for (int i = 0; i < seqlen*HIDDEN_SIZE; i++) h[i] += res[i];
    memcpy(res, h, seqlen*HIDDEN_SIZE*4);

    for (int i = 0; i < seqlen; i++) rms_norm(&h[i*HIDDEN_SIZE], lw->ln2.data, &n[i*HIDDEN_SIZE], HIDDEN_SIZE);

    if (!prefill) _t0 = now_ns();
    matmul_simd_g128(n, &lw->gate_proj, go, seqlen, HIDDEN_SIZE, INTERMEDIATE_SIZE);
    if (!prefill) profile_matmul(s, MATMUL_GATE_PROJ, _t0, seqlen, HIDDEN_SIZE, INTERMEDIATE_SIZE);

    if (!prefill) _t0 = now_ns();
    matmul_simd_g128(n, &lw->up_proj, uo, seqlen, HIDDEN_SIZE, INTERMEDIATE_SIZE);
    if (!prefill) profile_matmul(s, MATMUL_UP_PROJ, _t0, seqlen, HIDDEN_SIZE, INTERMEDIATE_SIZE);
    silu_mul(go, uo, go, seqlen*INTERMEDIATE_SIZE);

    if (!prefill) _t0 = now_ns();
    matmul_simd_g128(go, &lw->down_proj, h, seqlen, INTERMEDIATE_SIZE, HIDDEN_SIZE);
    if (!prefill) profile_matmul(s, MATMUL_DOWN_PROJ, _t0, seqlen, INTERMEDIATE_SIZE, HIDDEN_SIZE);
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

    s->hidden       = aligned_calloc(64, (size_t)MAX_SEQ_LEN * HIDDEN_SIZE * 4);
    s->normalized   = aligned_calloc(64, (size_t)MAX_SEQ_LEN * HIDDEN_SIZE * 4);
    s->residual     = aligned_calloc(64, (size_t)MAX_SEQ_LEN * HIDDEN_SIZE * 4);
    s->q            = aligned_calloc(64, (size_t)MAX_SEQ_LEN * HIDDEN_SIZE * 4);
    s->k            = aligned_calloc(64, (size_t)MAX_SEQ_LEN * NUM_KV_HEADS * HEAD_DIM * 4);
    s->v            = aligned_calloc(64, (size_t)MAX_SEQ_LEN * NUM_KV_HEADS * HEAD_DIM * 4);
    s->attn_out     = aligned_calloc(64, (size_t)MAX_SEQ_LEN * HIDDEN_SIZE * 4);
    s->attn_weights = aligned_calloc(64, (size_t)MAX_SEQ_LEN * MAX_SEQ_LEN * 4);
    s->gate_out     = aligned_calloc(64, (size_t)MAX_SEQ_LEN * INTERMEDIATE_SIZE * 4);
    s->up_out       = aligned_calloc(64, (size_t)MAX_SEQ_LEN * INTERMEDIATE_SIZE * 4);
    s->mlp_act      = aligned_calloc(64, (size_t)MAX_SEQ_LEN * INTERMEDIATE_SIZE * 4);
    s->approx_logits = aligned_calloc(64, (size_t)VOCAB_SIZE * 4);

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

    for (int pos = 0; pos < MAX_SEQ_LEN; pos++) {
        for (int i = 0; i < HEAD_DIM / 2; i++) {
            float a = (float)pos * s->inv_freq[i];
            s->rope_cos[pos][i] = cosf(a) * s->attn_scale;
            s->rope_sin[pos][i] = sinf(a) * s->attn_scale;
        }
    }

    // Log tiling status for deployment verification
    uint64_t total_tiles = 0;
    size_t tile_memory = 0;
    for (int i = 0; i < NUM_LAYERS; i++) {
        total_tiles += s->layers[i].q_proj.total_tiles8;
        total_tiles += s->layers[i].k_proj.total_tiles8;
        total_tiles += s->layers[i].v_proj.total_tiles8;
        total_tiles += s->layers[i].o_proj.total_tiles8;
        total_tiles += s->layers[i].gate_proj.total_tiles8;
        total_tiles += s->layers[i].up_proj.total_tiles8;
        total_tiles += s->layers[i].down_proj.total_tiles8;
        tile_memory += s->layers[i].q_proj.total_tiles8 * sizeof(TileBlock8);
        tile_memory += s->layers[i].k_proj.total_tiles8 * sizeof(TileBlock8);
        tile_memory += s->layers[i].v_proj.total_tiles8 * sizeof(TileBlock8);
        tile_memory += s->layers[i].o_proj.total_tiles8 * sizeof(TileBlock8);
        tile_memory += s->layers[i].gate_proj.total_tiles8 * sizeof(TileBlock8);
        tile_memory += s->layers[i].up_proj.total_tiles8 * sizeof(TileBlock8);
        tile_memory += s->layers[i].down_proj.total_tiles8 * sizeof(TileBlock8);
    }
    fprintf(stderr, "[TILING] 8-row tiles: %llu blocks, %.2f MB\n", 
            (unsigned long long)total_tiles, tile_memory / (1024.0 * 1024.0));
    fprintf(stderr, "[TILING] AVX-512 tiled kernel: %s\n", 
            (total_tiles > 0) ? "ACTIVE" : "FALLBACK");

    s->loaded = true;
    return 0;
}

static void free_g128(G128Matrix *m) {
    free(m->magnitude); free(m->sign); 
    free(m->packed_pos); free(m->packed_neg); 
    free(m->scales); free(m->scales_f32);
    free(m->tiles8);
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
    free(s->approx_logits);
}

int model_prefill(ModelState *s, int32_t *tokens, int n, float *logits) {
    if (!s->loaded || n > MAX_SEQ_LEN) return -1;
    s->kv_len = 0;

    for (int i = 0; i < n; i++) {
        int tid = tokens[i];
        if (tid < 0 || (uint32_t)tid >= s->embed.num_rows) return -1;
        embed_lookup(&s->embed, tid, &s->hidden[i*HIDDEN_SIZE]);
    }

    for (int i = 0; i < NUM_LAYERS; i++) forward_layer(s, i, n, 1, 0);
    s->kv_len = n;

    rms_norm(&s->hidden[(n-1)*HIDDEN_SIZE], s->final_norm.data, s->normalized, HIDDEN_SIZE);
    int vocab_n = (int)s->embed.num_rows;
    if (lm_head_prefilter_available) {
        lm_head_prefilter(s->normalized, &s->embed, s->approx_logits, vocab_n, LM_HEAD_PREFILTER_BLOCKS);
        find_top_k(s->approx_logits, vocab_n, LM_HEAD_CANDIDATES, s->lm_head_candidates);
        for (int i = 0; i < vocab_n; i++) logits[i] = -1e38f;
        matmul_g128_selected(s->normalized, &s->embed, logits, 1, HIDDEN_SIZE, vocab_n,
                            LM_HEAD_CANDIDATES, s->lm_head_candidates);
    } else {
        matmul_simd_g128(s->normalized, &s->embed, logits, 1, HIDDEN_SIZE, vocab_n);
    }
    return 0;
}

int model_decode(ModelState *s, int32_t token, float *logits) {
    if (!s->loaded) return -1;
    if (token < 0 || (uint32_t)token >= s->embed.num_rows) return -1;

    uint64_t _tstart = now_ns();
    embed_lookup(&s->embed, token, s->hidden);

    int pos = s->kv_len;
    if (pos >= MAX_SEQ_LEN) return -1;

    for (int i = 0; i < NUM_LAYERS; i++) forward_layer(s, i, 1, 0, pos);
    s->kv_len = pos + 1;

    uint64_t _lt0 = now_ns();
    rms_norm(s->hidden, s->final_norm.data, s->normalized, HIDDEN_SIZE);
    int vocab_n = (int)s->embed.num_rows;
    if (lm_head_prefilter_available) {
        // Phase 1: approximate scores from first N blocks of dimensions
        lm_head_prefilter(s->normalized, &s->embed, s->approx_logits, vocab_n, LM_HEAD_PREFILTER_BLOCKS);
        // Phase 2: find top-K candidate rows
        find_top_k(s->approx_logits, vocab_n, LM_HEAD_CANDIDATES, s->lm_head_candidates);
        // Phase 3: zero out all logits, then compute exact full-dim scores for candidates
        for (int i = 0; i < vocab_n; i++) logits[i] = -1e38f;
        matmul_g128_selected(s->normalized, &s->embed, logits, 1, HIDDEN_SIZE, vocab_n, LM_HEAD_CANDIDATES, s->lm_head_candidates);
    } else {
        matmul_simd_g128(s->normalized, &s->embed, logits, 1, HIDDEN_SIZE, vocab_n);
    }
    s->profile.logits_ns += (double)(now_ns() - _lt0);
    s->profile.total_ns += (double)(now_ns() - _tstart);
    s->profile.decode_count++;
    return 0;
}

void model_get_profile(ModelState *s, ProfileStats *out) {
    out->decode_count = s->profile.decode_count;
    out->matmul_ns    = s->profile.matmul_ns;
    out->attn_ns      = s->profile.attn_ns;
    out->logits_ns    = s->profile.logits_ns;
    out->total_ns     = s->profile.total_ns;
    for (int i = 0; i < MATMUL_COUNT; i++) {
        out->per_matmul_ns[i]       = s->profile.per_matmul_ns[i];
        out->per_matmul_calls[i]    = s->profile.per_matmul_calls[i];
        out->per_matmul_elements[i] = s->profile.per_matmul_elements[i];
    }
}

void model_reset_profile(ModelState *s) {
    memset(&s->profile, 0, sizeof(s->profile));
}

const char* model_matmul_path(void) {
#ifdef __ARM_NEON
    return "NEON";
#elif defined(__AVX512F__)
    return "AVX512";
#elif defined(__AVX2__)
    return "AVX2";
#elif defined(__SSE4_1__)
    return "SSE4.1";
#else
    return "scalar (SWAR)";
#endif
}

const char* model_compile_info(void) {
#ifdef _OPENMP
    return "OMP enabled";
#else
    return "OMP disabled";
#endif
}

int model_omp_max_threads(void) {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

// Debug: return offset of `loaded` field within ModelState
#include <stddef.h>
long model_struct_size(void) {
    return (long)sizeof(ModelState);
}
long model_debug_offset_loaded(void) {
    return (long)offsetof(ModelState, loaded);
}
long model_debug_offset_kv_len(void) {
    return (long)offsetof(ModelState, kv_len);
}

void model_set_omp_threads(int n) {
#ifdef _OPENMP
    char buf[16];
    snprintf(buf, sizeof(buf), "%d", n);
    setenv("OMP_NUM_THREADS", buf, 1);
    for (int attempt = 0; attempt < 3; attempt++) {
        omp_set_num_threads(n);
        if (omp_get_max_threads() == n) break;
        fprintf(stderr, "[OMP] attempt %d: set(%d) but max=%d\n", attempt, n, omp_get_max_threads());
    }
#else
    (void)n;
#endif
}

int model_affinity_cpu_count(void) {
#ifdef __linux__
    cpu_set_t cs;
    CPU_ZERO(&cs);
    if (sched_getaffinity(0, sizeof(cs), &cs) == 0)
        return CPU_COUNT(&cs);
#endif
    return -1;
}



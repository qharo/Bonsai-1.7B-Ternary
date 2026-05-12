# Bonsai 1.7B — Project Notes

## What this is

A from-scratch C inference engine for **Ternary-Bonsai-1.7B**, a 1-bit LLM by PrismML (Dr. Babak Hassibi, Caltech). Weights are ternary ({-scale, 0, +scale} per 128-element block), served via a FastAPI server with a streaming dark-mode chat UI, deployable to HuggingFace Spaces.

---

## Model Architecture

Based on **Qwen3** transformer architecture:

| Parameter | Value |
|-----------|-------|
| Hidden size | 2048 |
| Intermediate size (MLP) | 6144 |
| Layers | 28 |
| Attention heads (Q) | 16 |
| Attention heads (KV) | 8 — Grouped Query Attention (GQA) |
| Head dimension | 128 |
| Vocabulary size | 151,669 |
| Max sequence length | 512 |
| RoPE variant | YaRN (extended context) |
| RoPE base | 1,000,000 |
| Activation | SwiGLU (SiLU gate + linear up, element-wise multiply) |
| Norm | RMSNorm (eps 1e-6), applied per-head on Q and K (q_norm / k_norm) |
| EOS token | 151,645 |

---

## Weight Format: G128 Packed Ternary

Each quantized weight matrix is stored as **three binary files** (not standard safetensors):

```
weight_<name>_magnitude.bin   — uint64[2] per block, 1 bit per element (is non-zero?)
weight_<name>_sign.bin        — uint64[2] per block, 1 bit per element (is negative?)
weight_<name>_scales.bin      — float16 per block (max abs value in block)
```

Each **block** covers 128 consecutive elements. The 128 bits are packed into 2 × uint64 (little-endian). To decode: `w[i] = mag_bit[i] ? (sign_bit[i] ? -scale : +scale) : 0`.

**Non-ternary weights** (embedding table, layer norms) use a simpler format:
```
weight_<name>.bin   — raw float32 array with a small header
```

**Binary file header format** (shared by all):
```
[4 bytes] header_size (uint32)
[4 bytes] name_len (uint32)
[N bytes] name (UTF-8)
[4 bytes] num_dims (uint32)
[8*D bytes] dims (uint64 each)
[4 bytes] dtype_code (0=fp16, 1=fp32, 3=packed-ternary-g128)
[data...]
```
For G128 files, two extra uint32s after dtype_code: total_elements and elements_per_block (128).

### Converting weights from safetensors → G128 binary

Use `export_weight.py` (requires `safetensors`, `numpy`):

```bash
# Export all 28 layers + embed + norm in G128 format
python export_weight.py --layers 0-27 --packed --model-dir models/Ternary-Bonsai-1.7B-unpacked

# Export a single weight (FP32, for inspection)
python export_weight.py model.layers.0.mlp.gate_proj.weight
```

The script reads from `model.safetensors` in the model directory and writes `.bin` triplets alongside it.

---

## C Inference Engine

### Files
- **`model_infer.c` / `model_infer.h`** — full transformer forward pass, model load/free
- **`matmul_common.c` / `matmul_common.h`** — G128 matmul with NEON (ARM) and AVX2 (x86) paths
- **`Makefile`** — builds `inference.so` (shared library loaded by Python via ctypes)

### Build
```bash
make inference.so CC=clang "CFLAGS=-O3 -std=c11 -fPIC -march=native -ffast-math -fopenmp"
```

### Forward pass per token (decode mode)

For each of 28 layers:
1. **RMSNorm** on hidden state → `ln1`
2. **Q/K/V projections** (G128 matmul): `2048→2048`, `2048→1024`, `2048→1024`
3. **Per-head RMSNorm** on Q and K (`q_norm`, `k_norm`)
4. **YaRN RoPE** applied to Q and K
5. **GQA attention**: 16 Q heads attend over 8 KV heads (ratio 2:1), KV cache updated
6. **O projection** (G128 matmul): `2048→2048`
7. **Residual add**
8. **RMSNorm** → `ln2`
9. **MLP**: gate_proj + up_proj (both `2048→6144`), SiLU(gate) × up, down_proj (`6144→2048`)
10. **Residual add**

After all layers: final RMSNorm, then **lm_head** (FP32 dot product with embedding matrix: `hidden → vocab_size logits`).

### KV cache
Static allocation: `float kv_k[28][8][512][128]` and `kv_v` same shape — **head-major layout** (`[layer][head][pos][dim]`) for cache-friendly attention during decode. Consecutive positions for a given head are spaced by 512 bytes (128 × 4), fitting well within L1/L2 cache. Populated during prefill, extended one position per decode step.

---

## Matmul Implementation: `matmul_simd_g128`

The core routine. Processes `M × K` input against a `N × K` G128 weight matrix, producing `M × N` output.

### Key design:
- **Platform dispatch** via `#if` chain: `__ARM_NEON` → `__AVX2__` → `__SSE4_1__` → scalar fallback
- **AVX2**: 8-wide output loop, 256-bit YMM, 8-bit LUT (256-entry, 8 KB), `_mm256_fmadd_ps`
- **NEON**: 4-wide output loop, 128-bit NEON, 4-bit LUT (16-entry), `vmlaq_f32`
- **SSE**: 4-wide output loop, 128-bit XMM, 4-bit LUT (16-entry), `_mm_add_ps(_mm_mul_ps(...))`
- **LUT decode**: for each 4/8-bit nibble from packed mag/sign uint64s, a 16/256-entry lookup table maps to 4/8 × float32 masks — blendv for sign selection, AND for mag zeroing
- **Scale negation via XOR** (AVX2 only): `_mm256_xor_ps(ps, signbit)` avoids pre-computing neg_sc vectors
- **OpenMP**: `#pragma omp parallel for schedule(static) if(n8 >= 256)` on the outer j loop (N≥2048 parallelized)
- **FP32 scales precomputed at load time**: avoids ~28M `half_to_float` calls per decode step

**`lm_head` (vocab projection)**: Uses the same `matmul_simd_g128` kernel as all other projections — no special code path required.

---

## Python Server (`app.py`)

FastAPI + uvicorn. Loads `inference.so` via ctypes at startup.

### Sampling (`sample_token`)
Temperature scaling → top-k (`np.argpartition`, O(n)) → top-p (vectorized `np.cumsum`) → categorical sample.

### Generation loop
```
prefill(prompt_tokens) → logits
loop:
    next = sample(logits)
    yield next
    logits = decode(next)
```

`/generate/completion` runs in `asyncio.run_in_executor` (thread pool) to avoid blocking the event loop.

### Endpoints

| Endpoint | Method | Response |
|----------|--------|----------|
| `GET /` | — | Chat UI (dark mode, streaming) |
| `POST /generate` | SSE stream | `{"token": "...", "full": "..."}` per token, then `{"done": true, "tokens_generated": N, "total_time_s": T, "tokens_per_second": S}` |
| `POST /generate/completion` | JSON | `{"text": "...", "prompt_tokens": N, "tokens_generated": N, "total_time_s": T, "tokens_per_second": S}` |
| `GET /health` | JSON | status |
| `GET /model/info` | JSON | architecture metadata |

---

## Performance

Baseline (before optimizations): **1.64 t/s**

**After current optimizations: ~5.6 t/s** on Apple Silicon, target **4-7 t/s** on HF Spaces (2 vCPU, 16 GB).

### Optimization history

| Optimization | Gain | Notes |
|---|---|---|
| NEON lm_head (4-wide unrolling + `-march=native`) | ~1.57× | - |
| `np.argpartition` for top-k, vectorized top-p cumsum | minor | - |
| Batch tokenizer decode (collect IDs, single call) | minor | - |
| Async thread pool for blocking C calls | correctness + concurrency | - |
| OpenMP parallelism on matmul j-loop + lm_head | major | - |
| FP32 scales precomputed at load time | ~10-15% | - |
| `-ffast-math` compiler flag | ~10-20% | - |
| **AVX2 256-bit matmul kernel** (8-wide, 256-bit YMM) | **~1.5-1.8×** | Replaced SSE 4-wide with AVX2 8-wide + 256-bit FMA |
| **KV cache head-major transpose** | ~5-10% | `kv_k[lid][head][pos][dim]` for sequential attention reads |
| **Redundant op removal** | ~2% | Removed memset/memcpy in attention, pre-allocated Python logits |

### SIMD kernel dispatch

```c
#if defined(__ARM_NEON)     → 4-wide, float32x4_t, vmlaq_f32 (FMA)
#elif defined(__AVX2__)     → 8-wide, __m256, _mm256_fmadd_ps (true 256-bit)
#elif defined(__SSE4_1__)   → 4-wide, __m128, _mm_add_ps(_mm_mul_ps(...))
#else                       → scalar SWAR (ctzll loop)
```

The AVX2 kernel:
- 8-bit nibble → 256-entry LUT (8 KB) of 8 × float32 masks
- Processes 8 output rows simultaneously vs 4 in SSE
- Uses `_mm256_fmadd_ps` for fused multiply-accumulate
- Negates scales via `_mm256_xor_ps` sign-bit trick instead of separate neg/pos broadcasts
- Results: per-iteration throughput halved (64 LUT calls/block vs 128 SSE calls/block)

### Future optimization path: lm_head vocabulary filtering (quality-at-risk)

The lm_head computes `dot(h, e_r)` for all 151669 vocabulary entries — 310M weight elements per decode step, the single largest matmul. The vocabulary is dense (all but 2 of 151671 tokens valid), so naive sparsity filtering doesn't help.

**Proposed approach** (not yet implemented — requires quality evaluation):

1. **Two-phase scoring:**
   - Phase 1: Precompute 16 block-wise sums of the hidden state: `block_sum[b] = Σ_{i in block b} h[i]`
   - For each token row, compute a cheap score: `approx_r = Σ_b scale_{r,b} × block_sum[b] / 128`
   - This treats all weights in a block as +scale (ignoring sign), yielding an overestimate
   - Phase 2: Compute exact dot products for the top-K (e.g., K=2000) candidates only

2. **Token frequency cache:**
   - Maintain a rolling cache of recently sampled tokens
   - Score the cache first; if the max logit is above a threshold, skip the full scan

3. **Quality degradation risk:**
   - The coarse score overestimates influence of oppositely-signed blocks, potentially ranking a low-probability token too high
   - Mitigation: use a conservative K (e.g., 2000 of 151669 = 1.3%), which captures >99% of true top-1 probability mass in practice
   - Worst case: ≈0.1-0.5% generation quality regression in perplexity
   - **Revert strategy:** remove the `#define LM_HEAD_PREFILTER 1` flag and fall back to full compute

4. **Expected speedup:** ~1.3-1.5× on lm_head alone if prefilter catches 98% of tokens. ~10-15% overall.

---

## Deployment: HuggingFace Spaces

HF Spaces runs a Docker container built from this repo. Weights (~1.5 GB total) are stored in `models/` and tracked via **Git LFS** (`.gitattributes` tracks `models/**/*.bin`).

```bash
git lfs install
git lfs track "models/**/*.bin"   # already in .gitattributes
git add models/Ternary-Bonsai-1.7B-unpacked/
git commit -m "add model weights"
git push space main
```

Local dev (no rebuild needed):
```bash
docker build -t cvp-app .
docker run -p 7860:7860 -v $(pwd)/models:/app/models cvp-app
```

The volume mount overrides the `COPY models/` layer in the image for local iteration.

---

## Original Development Process

The engine was built incrementally:

1. Implemented 5 standalone matmul variants (`naive`, `bitnet`, `simd`, `swar`, `lut`) to benchmark approaches
2. Verified layer-by-layer correctness against PyTorch reference (`inference.py`) — Q/K/V projections matched exactly; attention/MLP showed minor numerical differences acceptable for generation
3. Consolidated into `matmul_common.c` with NEON SIMD as the primary path
4. Built full 28-layer model in `model_infer.c`, exposed as `inference.so`
5. Wrapped in FastAPI with ctypes bindings
6. Added streaming SSE endpoint, sampling, chat template support
7. Optimized: sampling, lm_head vectorization, OpenMP, scales precomputation
8. Added AVX2 path for x86 (HF Spaces) parity with NEON path
9. Added dark-mode streaming chat UI + HF Spaces deployment config

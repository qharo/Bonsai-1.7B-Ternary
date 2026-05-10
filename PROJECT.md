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
Static allocation: `float kv_k[28][512][8][128]` and `kv_v` same shape (28 layers × 512 max positions × 8 KV heads × 128 head dim). Populated during prefill, extended one position per decode step.

---

## Matmul Implementation: `matmul_simd_g128`

The core routine. Processes `M × K` input against a `N × K` G128 weight matrix, producing `M × N` output.

**Key design decisions:**
- **4-wide output loop**: processes 4 output rows simultaneously to expose independent accumulators
- **4-bit nibble LUT**: decodes 4 magnitude bits + 4 sign bits → 4 float weights using a 16-entry lookup table
- **Platform dispatch** via `#ifdef`:
  - `__ARM_NEON` → `float32x4_t` + `vmlaq_f32` + `vbslq_f32`
  - `__AVX2__` → `__m128` + `_mm_add_ps(_mm_mul_ps(...))` + `_mm_blendv_ps` (SSE4.1, auto-fused to FMA)
  - fallback → scalar SWAR
- **OpenMP**: `#pragma omp parallel for schedule(static) if(n4 >= 512)` on the outer j loop
- **FP32 scales precomputed at load time** (`scales_f32` field on `G128Matrix`): avoids ~28M `half_to_float` calls per decode step

**`lm_head` (vocab projection)**: plain C with `restrict` + 4-wide unrolling + OpenMP — lets the compiler auto-vectorize with NEON/AVX2 without explicit intrinsics.

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

| Optimization | Gain |
|---|---|
| NEON lm_head (4-wide unrolling + `-march=native`) | ~1.57× |
| `np.argpartition` for top-k, vectorized top-p cumsum | minor |
| Batch tokenizer decode (collect IDs, single call) | minor |
| Async thread pool for blocking C calls | correctness + concurrency |
| OpenMP parallelism on matmul j-loop + lm_head | major |
| FP32 scales precomputed at load time | ~10-15% |
| `-ffast-math` compiler flag | ~10-20% |

**After all optimizations: ~5.6 t/s** on Apple Silicon (M-series, ARM64 Docker).

On HF Spaces (x86_64): AVX2 path in `matmul_simd_g128` provides equivalent SIMD performance; OpenMP scales with vCPU count.

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

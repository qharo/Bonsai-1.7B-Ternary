This file is a merged representation of a subset of the codebase, containing files not matching ignore patterns, combined into a single document by Repomix.

# File Summary

## Purpose
This file contains a packed representation of a subset of the repository's contents that is considered the most important context.
It is designed to be easily consumable by AI systems for analysis, code review,
or other automated processes.

## File Format
The content is organized as follows:
1. This summary section
2. Repository information
3. Directory structure
4. Repository files (if enabled)
5. Multiple file entries, each consisting of:
  a. A header with the file path (## File: path/to/file)
  b. The full contents of the file in a code block

## Usage Guidelines
- This file should be treated as read-only. Any changes should be made to the
  original repository files, not this packed version.
- When processing this file, use the file path to distinguish
  between different files in the repository.
- Be aware that this file may contain sensitive information. Handle it with
  the same level of security as you would the original repository.

## Notes
- Some files may have been excluded based on .gitignore rules and Repomix's configuration
- Binary files are not included in this packed representation. Please refer to the Repository Structure section for a complete list of file paths, including binary files
- Files matching these patterns are excluded: **.json, ./models/**
- Files matching patterns in .gitignore are excluded
- Files matching default ignore patterns are excluded
- Files are sorted by Git change count (files with more changes are at the bottom)

# Directory Structure
```
static/
  index.html
tts/
  onnx/
    english/
      bos_before_voice.npy
      tokenizer.model
  pocket_tts_onnx.py
.dockerignore
.gitattributes
.gitignore
app.py
Dockerfile
export_weight.py
inference.py
Makefile
matmul_common.c
matmul_common.h
model_infer.c
model_infer.h
PROJECT.md
README.md
requirements.txt
tiny_tts_onnx.py
```

# Files

## File: .dockerignore
````
.venv/
__pycache__/
*.pyc
*.so
*.o
````

## File: .gitignore
````
__pycache__/
*.pyc
*.pyo
*.o
*.dSYM/
.venv/
venv/
tokenizer/

# HuggingFace cache
.cache/

# Compiled binaries
inference.so

# Original safetensors — not used by inference, too large for HF Spaces
models/**/*.safetensors

# fp32 versions of weights that have g128 equivalents — inference uses g128 only
models/**/weight_model_embed_tokens_weight.bin
models/**/weight_model_layers_*_mlp_*_proj_weight.bin
models/**/weight_model_layers_*_self_attn_[qkvo]_proj_weight.bin

# Unused g128 exports of fp32-only weights (layernorms, q/k norms, final norm)
models/**/*_layernorm_weight_magnitude.bin
models/**/*_layernorm_weight_sign.bin
models/**/*_layernorm_weight_scales.bin
models/**/*_q_norm_weight_magnitude.bin
models/**/*_q_norm_weight_sign.bin
models/**/*_q_norm_weight_scales.bin
models/**/*_k_norm_weight_magnitude.bin
models/**/*_k_norm_weight_sign.bin
models/**/*_k_norm_weight_scales.bin
models/**/weight_model_norm_weight_magnitude.bin
models/**/weight_model_norm_weight_sign.bin
models/**/weight_model_norm_weight_scales.bin
````

## File: export_weight.py
````python
# initial weight export script
import argparse
import struct
import numpy as np
import safetensors.numpy
from pathlib import Path


def sanitize_name(name: str) -> str:
    return name.replace("/", ".").replace(":", "_")


def extract_ternary_g128(arr: np.ndarray) -> tuple:
    """Extract ternary values and scales from G128 quantized weight.

    Returns:
        magnitude: packed as uint64_t[2] per block (128 bits, little-endian)
        sign: packed as uint64_t[2] per block (128 bits, little-endian)
        scales: FP16 per block
    """
    total_elements = arr.size
    num_blocks = total_elements // 128

    # Reshape into blocks of 128
    arr_flat = arr.flat[:num_blocks * 128].reshape(num_blocks, 128)

    # Extract magnitude: 1 if non-zero, 0 otherwise
    magnitude = (np.abs(arr_flat) > 0).astype(np.uint8)

    # Extract sign: 1 if negative, 0 otherwise
    sign = (arr_flat < 0).astype(np.uint8)

    # Extract scales: max abs value per block as FP16
    scales = np.max(np.abs(arr_flat), axis=1).astype(np.float16)

    # Pack into 2x uint64_t per block (128 bits = 16 bytes)
    # Each block: 128 bits packed into 2 × uint64
    def pack_bits(bits: np.ndarray) -> np.ndarray:
        # bits shape: (num_blocks, 128)
        # Pack 32 bits at a time using packbits, then reshape
        packed = np.packbits(bits, axis=1, bitorder='little')
        # packed shape: (num_blocks, 16) - 16 bytes
        # Reinterpret as 2 × uint64 (8 bytes each)
        packed_u64 = packed.view(np.uint64)
        return packed_u64.reshape(num_blocks, 2)

    magnitude_packed = pack_bits(magnitude)
    sign_packed = pack_bits(sign)

    return magnitude_packed, sign_packed, scales


def pack_weight(name: str, arr, out_path: Path) -> None:
    name_bytes = name.encode("utf-8")
    name_len = len(name_bytes)
    num_dims = len(arr.shape)
    dtype_code = 1  # 0 = float16, 1 = float32

    total_elements = arr.size
    data_bytes = arr.nbytes

    header_size = 4 + 4 + name_len + 4 + num_dims * 8 + 4

    with open(out_path, "wb") as f:
        f.write(struct.pack("<I", header_size))
        f.write(struct.pack("<I", name_len))
        f.write(name_bytes)
        f.write(struct.pack("<I", num_dims))
        for dim in arr.shape:
            f.write(struct.pack("<Q", dim))
        f.write(struct.pack("<I", dtype_code))
        f.write(arr.tobytes())

    print(f"Exported: {name}")
    print(f"  Shape: {list(arr.shape)}  dtype: {arr.dtype}  numel: {total_elements:,}")
    print(f"  Data bytes: {data_bytes:,}  Header: {header_size}")
    print(f"  Total file size: {header_size + data_bytes:,} bytes")
    print(f"  Saved to: {out_path}")


def pack_weight_ternary_g128(name: str, arr_shape: tuple, num_blocks: int,
                            magnitude, sign, scales,
                            out_path_mag: Path, out_path_sign: Path, out_path_scales: Path) -> None:
    name_bytes = name.encode("utf-8")
    name_len = len(name_bytes)
    dtype_code = 3  # 3 = packed ternary G128 (128 bits per block)

    shape = arr_shape
    num_dims = len(shape)
    total_elements = num_blocks * 128
    elements_per_block = 128

    header_size = 4 + 4 + name_len + 4 + num_dims * 8 + 4 + 4 + 4

    with open(out_path_mag, "wb") as f:
        f.write(struct.pack("<I", header_size))
        f.write(struct.pack("<I", name_len))
        f.write(name_bytes)
        f.write(struct.pack("<I", num_dims))
        for dim in shape:
            f.write(struct.pack("<Q", dim))
        f.write(struct.pack("<I", dtype_code))
        f.write(struct.pack("<I", total_elements))
        f.write(struct.pack("<I", elements_per_block))
        f.write(magnitude.tobytes())

    with open(out_path_sign, "wb") as f:
        f.write(struct.pack("<I", header_size))
        f.write(struct.pack("<I", name_len))
        f.write(name_bytes)
        f.write(struct.pack("<I", num_dims))
        for dim in shape:
            f.write(struct.pack("<Q", dim))
        f.write(struct.pack("<I", dtype_code))
        f.write(struct.pack("<I", total_elements))
        f.write(struct.pack("<I", elements_per_block))
        f.write(sign.tobytes())

    with open(out_path_scales, "wb") as f:
        f.write(struct.pack("<I", header_size))
        f.write(struct.pack("<I", name_len))
        f.write(name_bytes)
        f.write(struct.pack("<I", num_dims))
        for dim in shape:
            f.write(struct.pack("<Q", dim))
        f.write(struct.pack("<I", 0))  # dtype_code 0 = FP16 for scales
        f.write(struct.pack("<I", total_elements))
        f.write(struct.pack("<I", elements_per_block))
        f.write(scales.tobytes())

    print(f"Exported: {name}")
    print(f"  Shape: {list(shape)}  numel: {total_elements:,}  blocks: {num_blocks:,}")
    print(f"  Magnitude: {magnitude.nbytes:,} bytes ({magnitude.shape})")
    print(f"  Sign: {sign.nbytes:,} bytes ({sign.shape})")
    print(f"  Scales: {scales.nbytes:,} bytes (FP16)")
    print(f"  Saved magnitude to: {out_path_mag}")
    print(f"  Saved sign to: {out_path_sign}")
    print(f"  Saved scales to: {out_path_scales}")


LAYER_WEIGHTS = [
    "input_layernorm.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
    "self_attn.o_proj.weight",
    "post_attention_layernorm.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
]


def export_batch(model_dir: Path, layers: list, packed: bool = False):
    safetensors_path = model_dir / "model.safetensors"
    print(f"Loading {safetensors_path} ...")
    tensors = safetensors.numpy.load_file(str(safetensors_path))

    total = 0
    for layer_idx in layers:
        for weight_suffix in LAYER_WEIGHTS:
            name = f"model.layers.{layer_idx}.{weight_suffix}"
            if name not in tensors:
                print(f"SKIP (not found): {name}")
                continue
            arr = tensors[name].astype(np.float32)
            safe_name = sanitize_name(name).replace(".", "_")

            if packed:
                if arr.size % 128 != 0:
                    print(f"SKIP {name}: size {arr.size} not divisible by 128")
                    continue
                magnitude, sign, scales = extract_ternary_g128(arr)
                num_blocks = arr.size // 128
                out_path_mag = model_dir / f"weight_{safe_name}_magnitude.bin"
                out_path_sign = model_dir / f"weight_{safe_name}_sign.bin"
                out_path_scales = model_dir / f"weight_{safe_name}_scales.bin"
                pack_weight_ternary_g128(name, arr.shape, num_blocks,
                                         magnitude, sign, scales,
                                         out_path_mag, out_path_sign, out_path_scales)
            else:
                out_path = model_dir / f"weight_{safe_name}.bin"
                pack_weight(name, arr, out_path)
            total += 1
    print(f"\nBatch export complete: {total} weights exported")


def main():
    parser = argparse.ArgumentParser(description="Export a model weight to binary float32.")
    parser.add_argument("weight_name", nargs="?", help="Full weight name (e.g. model.layers.0.mlp.up_proj.weight)")
    parser.add_argument("--model-dir", default="models/Ternary-Bonsai-1.7B-unpacked",
                        help="Path to model directory")
    parser.add_argument("--packed", action="store_true",
                        help="Export as packed ternary G128 (magnitude + sign + scales)")
    parser.add_argument("--layers", type=str, default=None,
                        help="Export all layer weights. Comma-sep list (e.g. '0-7' or '0,3,5'). "
                             "Includes embed_tokens and model.norm automatically.")
    parser.add_argument("--embed", action="store_true",
                        help="Export embed_tokens (use with --layers)")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)

    if args.layers is not None:
        # Parse layer range
        parts = args.layers.replace(" ", "").split(",")
        layers = []
        for p in parts:
            if "-" in p:
                start, end = p.split("-")
                layers.extend(range(int(start), int(end) + 1))
            else:
                layers.append(int(p))
        layers = sorted(set(layers))
        packed = args.packed
        print(f"Batch export for layers: {layers}, packed={packed}")

        # Export embed_tokens first
        if args.embed or True:
            name = "model.embed_tokens.weight"
            tensors = safetensors.numpy.load_file(str(model_dir / "model.safetensors"))
            if name in tensors:
                arr = tensors[name].astype(np.float32)
                safe_name = sanitize_name(name).replace(".", "_")
                if packed:
                    if arr.size % 128 == 0:
                        magnitude, sign, scales = extract_ternary_g128(arr)
                        num_blocks = arr.size // 128
                        out_path_mag = model_dir / f"weight_{safe_name}_magnitude.bin"
                        out_path_sign = model_dir / f"weight_{safe_name}_sign.bin"
                        out_path_scales = model_dir / f"weight_{safe_name}_scales.bin"
                        pack_weight_ternary_g128(name, arr.shape, num_blocks,
                                                 magnitude, sign, scales,
                                                 out_path_mag, out_path_sign, out_path_scales)
                    else:
                        print(f"SKIP (not divisible by 128): {name}")
                else:
                    out_path = model_dir / f"weight_{safe_name}.bin"
                    pack_weight(name, arr, out_path)
            else:
                print(f"SKIP (not found): {name}")

        # Export model.norm
        name = "model.norm.weight"
        tensors = safetensors.numpy.load_file(str(model_dir / "model.safetensors"))
        if name in tensors:
            arr = tensors[name].astype(np.float32)
            safe_name = sanitize_name(name).replace(".", "_")
            if packed:
                if arr.size % 128 == 0:
                    magnitude, sign, scales = extract_ternary_g128(arr)
                    num_blocks = arr.size // 128
                    out_path_mag = model_dir / f"weight_{safe_name}_magnitude.bin"
                    out_path_sign = model_dir / f"weight_{safe_name}_sign.bin"
                    out_path_scales = model_dir / f"weight_{safe_name}_scales.bin"
                    pack_weight_ternary_g128(name, arr.shape, num_blocks,
                                             magnitude, sign, scales,
                                             out_path_mag, out_path_sign, out_path_scales)
                else:
                    print(f"SKIP (not divisible by 128): {name}")
            else:
                out_path = model_dir / f"weight_{safe_name}.bin"
                pack_weight(name, arr, out_path)
        else:
            print(f"SKIP (not found): {name}")

        export_batch(model_dir, layers, packed)
        return

    if not args.weight_name:
        parser.print_help()
        return

    safetensors_path = model_dir / "model.safetensors"

    print(f"Loading {safetensors_path} ...")
    tensors = safetensors.numpy.load_file(str(safetensors_path))

    name = args.weight_name
    if name not in tensors:
        print(f"ERROR: weight '{name}' not found.")
        print("Available weights:")
        for n in sorted(tensors.keys()):
            print(f"  {n}")
        return

    arr = tensors[name].astype(np.float32)
    safe_name = sanitize_name(name).replace(".", "_")

    if arr.size % 128 != 0:
        raise ValueError(f"Array size {arr.size} is not divisible by 128")

    if args.packed:
        magnitude, sign, scales = extract_ternary_g128(arr)
        num_blocks = arr.size // 128
        out_path_mag = model_dir / f"weight_{safe_name}_magnitude.bin"
        out_path_sign = model_dir / f"weight_{safe_name}_sign.bin"
        out_path_scales = model_dir / f"weight_{safe_name}_scales.bin"
        pack_weight_ternary_g128(name, arr.shape, num_blocks,
                          magnitude, sign, scales,
                          out_path_mag, out_path_sign, out_path_scales)
    else:
        out_path = model_dir / f"weight_{safe_name}.bin"
        pack_weight(name, arr, out_path)


if __name__ == "__main__":
    main()
````

## File: inference.py
````python
# final inference file
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    model_path = "models/Ternary-Bonsai-1.7B-unpacked"
    prompt = "What's the weather like, usually, in Nice, France?'"

    print(f"Loading tokenizer from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    print(f"Loading model from {model_path} ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )

    print(f"Model loaded. Device: {model.device}, dtype: {model.dtype}")
    print(f"\nPrompt: {prompt}\n")

    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    inputs = tokenizer(input_text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    print(f"Input tokens: {input_ids.shape[1]}")

    print("\nGenerating...\n")
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

    response_ids = outputs[0][input_ids.shape[1]:]
    response = tokenizer.decode(response_ids, skip_special_tokens=True)
    print(f"\n> {prompt}")
    print(response)


if __name__ == "__main__":
    main()
````

## File: PROJECT.md
````markdown
# Bonsai 1.7B — Project Notes

## What this is

A from-scratch C inference engine for **Ternary-Bonsai-1.7B**, a 1.58-bit LLM by PrismML (Dr. Babak Hassibi, Caltech). Weights are ternary ({-scale, 0, +scale} per 128-element block), served via a FastAPI server with a streaming dark-mode chat UI, deployable to HuggingFace Spaces.

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
````

## File: .gitattributes
````
models/tinytts/** filter=lfs diff=lfs merge=lfs -text
````

## File: Makefile
````
CC = clang
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    CFLAGS_BASE = -O3 -std=c11 -Wall -fPIC -march=native -ffast-math
    FRAMEWORKS = -framework Accelerate
else
    # Linux / HF Spaces: cap at x86-64-v2 to avoid AVX-512 SIGILL on runtime nodes
    CFLAGS_BASE = -O3 -std=c11 -Wall -fPIC -march=x86-64-v2 -ffast-math
    FRAMEWORKS =
endif

# Detect if compiler supports OpenMP
OPENMP_FLAG =
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
    # Try clang with OpenMP first, fall back to gcc if needed
    OPENMP_TEST := $(shell $(CC) -fopenmp -E - < /dev/null 2>/dev/null && echo yes)
    ifeq ($(OPENMP_TEST),yes)
        OPENMP_FLAG = -fopenmp
    else
        # Try gcc
        GCC_EXISTS := $(shell which gcc 2>/dev/null)
        ifneq ($(GCC_EXISTS),)
            CC = gcc
            OPENMP_FLAG = -fopenmp
        endif
    endif
endif

CFLAGS = $(CFLAGS_BASE) $(OPENMP_FLAG)

MATMUL_FILES = matmul_naive.c matmul_bitnet.c matmul_simd.c matmul_swar.c matmul_lut.c

all: model_run matmul_test inference.so

model_run: model.c $(MATMUL_FILES) matmul_common.h
	$(CC) $(CFLAGS) -o model_run model.c $(MATMUL_FILES) $(FRAMEWORKS)

matmul_test: matmul_test.c $(MATMUL_FILES) matmul_common.h
	$(CC) $(CFLAGS) -o matmul_test matmul_test.c $(MATMUL_FILES) $(FRAMEWORKS)

inference.so: model_infer.c model_infer.h matmul_common.h matmul_common.c
	$(CC) $(CFLAGS) -shared -o $@ model_infer.c matmul_common.c -lm

matmul_naive.o: matmul_naive.c matmul_common.h
	$(CC) $(CFLAGS) -c -o matmul_naive.o matmul_naive.c

matmul_bitnet.o: matmul_bitnet.c matmul_common.h
	$(CC) $(CFLAGS) -c -o matmul_bitnet.o matmul_bitnet.c

matmul_simd.o: matmul_simd.c matmul_common.h
	$(CC) $(CFLAGS) -c -o matmul_simd.o matmul_simd.c

matmul_swar.o: matmul_swar.c matmul_common.h
	$(CC) $(CFLAGS) -c -o matmul_swar.o matmul_swar.c

matmul_lut.o: matmul_lut.c matmul_common.h
	$(CC) $(CFLAGS) -c -o matmul_lut.o matmul_lut.c

clean:
	rm -f model_run matmul_test inference.so *.o

.PHONY: all clean
````

## File: README.md
````markdown
---
title: Bonsai 1.7B
emoji: 🌿
colorFrom: green
colorTo: green
sdk: docker
pinned: false
app_port: 7860
models:
  - prism-ml/Ternary-Bonsai-1.7B-mlx-2bit
tags:
  - text-generation
  - 1-58-bit
  - bonsai
license: mit
---

# Bonsai 1.7B — A 1.58-bit LLM

[![Model](https://img.shields.io/badge/Model-PrismML%2FTernary--Bonsai--1.7B-blue)](https://huggingface.co/prism-ml/Ternary-Bonsai-1.7B-mlx-2bit)
[![Space](https://img.shields.io/badge/Space-Demo-green)](https://huggingface.co/spaces/qhar0h/Bonsai-1.7B)

A 1.58-bit language model inference server with a streaming chat UI.

**Model by:** [PrismML / Caltech](https://huggingface.co/PrismML)

## Model weights

The `models/` directory contains the pre-converted G128 ternary format weights tracked via Git LFS:

```
models/Ternary-Bonsai-1.7B-unpacked/
  weight_model_embed_tokens_weight.bin
  weight_model_norm_weight.bin
  weight_model_layers_*_*.bin   (magnitude / sign / scales triplets)
  tokenizer files
```

## Local development

```bash
docker build -t cvp-app .
docker run -p 7860:7860 -v $(pwd)/models:/app/models cvp-app
# open http://localhost:7860
```

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `GET /` | — | Chat UI |
| `POST /generate` | SSE stream | Streaming token generation |
| `POST /generate/completion` | JSON | Full response with timing stats |
| `GET /health` | — | Status check |
| `GET /model/info` | — | Model metadata |

## Links

- **Official Model:** [prism-ml/Ternary-Bonsai-1.7B-mlx-2bit](https://huggingface.co/prism-ml/Ternary-Bonsai-1.7B-mlx-2bit)
- **Organization:** [PrismML](https://huggingface.co/PrismML)
- **Paper:** [1.58-bit LLMs](https://arxiv.org/abs/2402.17764)
````

## File: requirements.txt
````
fastapi>=0.109.0
uvicorn[standard]>=0.27.0
transformers>=4.37.0
numpy>=1.24.0
tokenizers>=0.15.0
jinja2>=3.1.0
onnxruntime>=1.17.0
g2p-en>=2.0.0
huggingface_hub
scipy>=1.10.0
````

## File: tts/pocket_tts_onnx.py
````python
"""
PocketTTS ONNX - bundle-aware ONNX inference for Pocket TTS.

NO LONGER USED  - PHASE 2 MAYBE
"""

import json
import os
import queue
import threading
import time
import wave
from pathlib import Path
from typing import Generator, Optional, Union

import numpy as np
import onnxruntime as ort
import sentencepiece as spm
from huggingface_hub import hf_hub_download
from safetensors import safe_open

try:
    import soundfile as sf

    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False

try:
    import scipy.signal

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class PocketTTSOnnx:
    HF_REPO_ID = "kyutai/pocket-tts"
    DEFAULT_LANGUAGE = "english_2026-04"
    VALID_PRECISIONS = ("int8", "fp32")
    TOKENS_PER_SECOND_ESTIMATE = 3.0
    GEN_SECONDS_PADDING = 2.0
    MIN_FRAMES_BEFORE_EOS = 8

    def __init__(
        self,
        models_dir: str = "onnx",
        language: str = DEFAULT_LANGUAGE,
        tokenizer_path: Optional[str] = None,
        precision: str = "int8",
        device: str = "auto",
        temperature: float = 0.7,
        lsd_steps: int = 1,
    ):
        if precision not in self.VALID_PRECISIONS:
            raise ValueError(f"precision must be one of {self.VALID_PRECISIONS}, got '{precision}'")

        self.models_root = Path(models_dir)
        self.language = self._normalize_language(language)
        self.bundle_dir = self._resolve_bundle_dir(self.models_root, self.language)
        self.metadata = self._load_metadata(self.bundle_dir)

        self.precision = precision
        self.temperature = temperature
        self.lsd_steps = lsd_steps
        self.providers = self._get_providers(device)

        self.sample_rate = int(self.metadata["sample_rate"])
        self.frame_rate = float(self.metadata["frame_rate"])
        self.samples_per_frame = int(self.metadata["samples_per_frame"])
        self.frame_duration = self.samples_per_frame / self.sample_rate
        self.latent_dim = int(self.metadata["latent_dim"])
        self.conditioning_dim = int(self.metadata["conditioning_dim"])
        self.pad_with_spaces_for_short_inputs = bool(
            self.metadata.get("pad_with_spaces_for_short_inputs", False)
        )
        self.remove_semicolons = bool(self.metadata.get("remove_semicolons", False))
        self.model_recommended_frames_after_eos = self.metadata.get(
            "model_recommended_frames_after_eos"
        )
        self.max_token_per_chunk = int(self.metadata.get("max_token_per_chunk", 50))
        self.insert_bos_before_voice = bool(self.metadata.get("insert_bos_before_voice", False))
        self.predefined_voices = tuple(self.metadata.get("predefined_voices", []))

        tokenizer_file = tokenizer_path or str(self.bundle_dir / self.metadata["tokenizer_file"])
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.Load(tokenizer_file)

        self.bos_before_voice = None
        bos_file = self.metadata.get("bos_before_voice_file")
        if bos_file:
            self.bos_before_voice = np.load(self.bundle_dir / bos_file).astype(np.float32)

        self.flow_state_manifest = self.metadata["flow_lm_state_manifest"]
        self.mimi_state_manifest = self.metadata["mimi_state_manifest"]

        self._load_models()
        self._precompute_flow_buffers()
        self._voice_cache: dict[str, np.ndarray] = {}
        self._voice_state_cache: dict[str, dict[str, np.ndarray]] = {}

    @staticmethod
    def _normalize_language(language: str) -> str:
        if language == "english":
            return "english_2026-04"
        return language.replace("_2026_", "_2026-")

    @staticmethod
    def _resolve_bundle_dir(models_root: Path, language: str) -> Path:
        candidate = models_root / language
        if candidate.is_dir():
            return candidate
        if (models_root / "bundle.json").exists():
            return models_root
        raise FileNotFoundError(
            f"Could not find ONNX bundle for '{language}' under {models_root}."
        )

    @staticmethod
    def _load_metadata(bundle_dir: Path) -> dict:
        metadata_path = bundle_dir / "bundle.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing bundle metadata: {metadata_path}")
        return json.loads(metadata_path.read_text())

    def _get_providers(self, device: str) -> list[str]:
        if device == "cpu":
            return ["CPUExecutionProvider"]
        available = ort.get_available_providers()
        if device == "cuda" and "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    def _make_session_options(self) -> ort.SessionOptions:
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = os.cpu_count() or 2
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        return opts

    def _model_file(self, stem: str) -> str:
        if self.precision == "int8":
            quantized = self.bundle_dir / f"{stem}_int8.onnx"
            if quantized.exists():
                return quantized.name
        fp32 = self.bundle_dir / f"{stem}.onnx"
        if fp32.exists():
            return fp32.name
        raise FileNotFoundError(f"Missing ONNX file for {stem} in {self.bundle_dir}")

    def _load_models(self):
        opts = self._make_session_options()

        try:
            encoder_file = self._model_file("mimi_encoder")
            self.mimi_encoder = ort.InferenceSession(
                str(self.bundle_dir / encoder_file), sess_options=opts, providers=self.providers
            )
        except FileNotFoundError:
            self.mimi_encoder = None
        self.text_conditioner = ort.InferenceSession(
            str(self.bundle_dir / self._model_file("text_conditioner")),
            sess_options=opts,
            providers=self.providers,
        )
        self.flow_lm_main = ort.InferenceSession(
            str(self.bundle_dir / self._model_file("flow_lm_main")),
            sess_options=opts,
            providers=self.providers,
        )
        self.flow_lm_flow = ort.InferenceSession(
            str(self.bundle_dir / self._model_file("flow_lm_flow")),
            sess_options=opts,
            providers=self.providers,
        )
        self.mimi_decoder = ort.InferenceSession(
            str(self.bundle_dir / self._model_file("mimi_decoder")),
            sess_options=opts,
            providers=self.providers,
        )

    def _precompute_flow_buffers(self):
        dt = 1.0 / self.lsd_steps
        self._st_buffers = []
        for j in range(self.lsd_steps):
            s = j / self.lsd_steps
            t = s + dt
            self._st_buffers.append(
                (
                    np.array([[s]], dtype=np.float32),
                    np.array([[t]], dtype=np.float32),
                )
            )

    @staticmethod
    def _numpy_dtype(dtype: str):
        return {
            "float32": np.float32,
            "float16": np.float16,
            "int64": np.int64,
            "bool": np.bool_,
        }[dtype]

    def _make_filled_array(self, shape: list[int], dtype, fill: str) -> np.ndarray:
        if fill == "nan":
            return np.full(shape, np.nan, dtype=dtype)
        if fill == "ones":
            return np.ones(shape, dtype=dtype)
        return np.zeros(shape, dtype=dtype)

    def _init_state(self, manifest: list[dict]) -> dict[str, np.ndarray]:
        state = {}
        for entry in manifest:
            dtype = self._numpy_dtype(entry["dtype"])
            state[entry["input_name"]] = self._make_filled_array(
                entry["shape"], dtype=dtype, fill=entry["fill"]
            )
        return state

    @staticmethod
    def _clone_state(state: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {key: value.copy() for key, value in state.items()}

    def _update_state_from_outputs(
        self,
        state: dict[str, np.ndarray],
        result: list[np.ndarray],
        manifest: list[dict],
        output_offset: int,
    ):
        for entry in manifest:
            state[entry["input_name"]] = result[output_offset + entry["index"]]

    def _load_audio(self, path: Union[str, Path]) -> np.ndarray:
        path = Path(path)

        if path.suffix.lower() == ".wav":
            with wave.open(str(path), "rb") as wav_file:
                sr = wav_file.getframerate()
                raw_data = wav_file.readframes(-1)
                audio = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
        else:
            if not HAS_SOUNDFILE:
                raise ImportError("soundfile required for non-wav voice cloning inputs.")
            audio, sr = sf.read(str(path))
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            audio = audio.astype(np.float32)

        if sr != self.sample_rate:
            if not HAS_SCIPY:
                raise ImportError("scipy required for resampling.")
            gcd = np.gcd(int(sr), int(self.sample_rate))
            up = int(self.sample_rate // gcd)
            down = int(sr // gcd)
            audio = scipy.signal.resample_poly(audio, up, down, axis=-1).astype(np.float32)

        return audio.reshape(1, 1, -1)

    def encode_voice(self, audio_path: Union[str, Path]) -> np.ndarray:
        if self.mimi_encoder is None:
            raise RuntimeError("mimi_encoder.onnx was not found; voice cloning is unavailable.")
        audio = self._load_audio(audio_path)
        embeddings = self.mimi_encoder.run(None, {"audio": audio})[0]
        while embeddings.ndim > 3:
            embeddings = embeddings.squeeze(0)
        if embeddings.ndim < 3:
            embeddings = embeddings[None]
        return embeddings.astype(np.float32, copy=False)

    def _prepare_voice_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        embeddings = np.asarray(embeddings, dtype=np.float32)
        while embeddings.ndim > 3:
            embeddings = embeddings.squeeze(0)
        if embeddings.ndim < 3:
            embeddings = embeddings.reshape(1, -1, embeddings.shape[-1])
        if self.insert_bos_before_voice and self.bos_before_voice is not None:
            embeddings = np.concatenate([self.bos_before_voice, embeddings], axis=1)
        return embeddings

    def _hf_model_state(self, filename: str) -> dict[str, dict[str, np.ndarray]]:
        cached = hf_hub_download(repo_id=self.HF_REPO_ID, filename=filename)
        result: dict[str, dict[str, np.ndarray]] = {}
        with safe_open(cached, framework="np") as handle:
            for key in handle.keys():
                module_name, tensor_key = key.split("/", 1)
                result.setdefault(module_name, {})
                result[module_name][tensor_key] = handle.get_tensor(key)
        return result

    @staticmethod
    def _import_model_state_file(source: Union[str, Path]) -> dict[str, dict[str, np.ndarray]]:
        result: dict[str, dict[str, np.ndarray]] = {}
        with safe_open(str(source), framework="np") as handle:
            for key in handle.keys():
                module_name, tensor_key = key.split("/", 1)
                result.setdefault(module_name, {})
                result[module_name][tensor_key] = handle.get_tensor(key)
        return result

    @staticmethod
    def _derive_step(module_state: dict[str, np.ndarray]) -> np.ndarray:
        if "step" in module_state:
            return np.asarray(module_state["step"], dtype=np.int64).reshape(1)
        if "offset" in module_state and "end_offset" not in module_state:
            return np.asarray(module_state["offset"], dtype=np.int64).reshape(1)
        if "current_end" in module_state:
            return np.array([module_state["current_end"].shape[0]], dtype=np.int64)
        return np.array([0], dtype=np.int64)

    def _adapt_state_tensor(self, source: np.ndarray, entry: dict) -> np.ndarray:
        target_shape = tuple(entry["shape"])
        target_dtype = self._numpy_dtype(entry["dtype"])
        source = np.asarray(source, dtype=target_dtype)

        if source.shape == target_shape:
            return source.copy()

        if source.size == np.prod(target_shape, dtype=np.int64):
            return source.reshape(target_shape).copy()

        target = self._make_filled_array(list(target_shape), target_dtype, entry["fill"])
        if source.ndim != len(target_shape):
            return target

        slices = tuple(slice(0, min(src, dst)) for src, dst in zip(source.shape, target_shape))
        if all(s.start == s.stop for s in slices):
            return target
        target[slices] = source[slices]
        return target

    def _state_from_model_state(
        self, model_state: dict[str, dict[str, np.ndarray]], manifest: list[dict]
    ) -> dict[str, np.ndarray]:
        state = self._init_state(manifest)
        for entry in manifest:
            module_state = model_state.get(entry["module"], {})
            tensor = module_state.get(entry["key"])
            if tensor is None and entry["key"] == "step":
                tensor = self._derive_step(module_state)
            if tensor is None:
                continue
            state[entry["input_name"]] = self._adapt_state_tensor(tensor, entry)
        return state

    def _condition_with_voice_embeddings(self, embeddings: np.ndarray) -> dict[str, np.ndarray]:
        voice_embeddings = self._prepare_voice_embeddings(embeddings)
        state = self._init_state(self.flow_state_manifest)
        empty_seq = np.zeros((1, 0, self.latent_dim), dtype=np.float32)
        result = self.flow_lm_main.run(
            None,
            {"sequence": empty_seq, "text_embeddings": voice_embeddings, **state},
        )
        self._update_state_from_outputs(state, result, self.flow_state_manifest, output_offset=2)
        return state

    def prepare_voice_state(self, voice: Union[str, Path, np.ndarray]) -> dict[str, np.ndarray]:
        if isinstance(voice, np.ndarray):
            return self._condition_with_voice_embeddings(voice)

        voice_str = str(voice)
        if voice_str in self._voice_state_cache:
            return self._clone_state(self._voice_state_cache[voice_str])

        if voice_str in self.predefined_voices:
            local_embed = self.bundle_dir / f"{voice_str}.safetensors"
            if local_embed.exists():
                model_state = self._import_model_state_file(local_embed)
            else:
                filename = f"languages/{self.language}/embeddings/{voice_str}.safetensors"
                model_state = self._hf_model_state(filename)
            state = self._state_from_model_state(model_state, self.flow_state_manifest)
            self._voice_state_cache[voice_str] = self._clone_state(state)
            return state

        voice_path = Path(voice_str)
        if voice_path.exists() and voice_path.suffix == ".safetensors":
            model_state = self._import_model_state_file(voice_path)
            return self._state_from_model_state(model_state, self.flow_state_manifest)

        if voice_str in self._voice_cache:
            embeddings = self._voice_cache[voice_str]
        elif voice_path.exists():
            embeddings = self.encode_voice(voice_path)
            self._voice_cache[voice_str] = embeddings
        else:
            raise ValueError(f"Voice '{voice}' not found.")
        return self._condition_with_voice_embeddings(embeddings)

    def _prepare_text_prompt(self, text: str) -> tuple[str, int]:
        text = text.strip()
        if not text:
            raise ValueError("Text cannot be empty")
        text = text.replace("\n", " ").replace("\r", " ").replace("  ", " ")
        if self.remove_semicolons:
            text = text.replace(";", ",")

        number_of_words = len(text.split())
        frames_after_eos_guess = 3 if number_of_words <= 4 else 1

        if not text[0].isupper():
            text = text[0].upper() + text[1:]
        if text[-1].isalnum():
            text = text + "."
        if self.pad_with_spaces_for_short_inputs and len(text.split()) < 5:
            text = " " * 8 + text
        return text, frames_after_eos_guess

    def _tokenize(self, text: str) -> np.ndarray:
        prepared, _ = self._prepare_text_prompt(text)
        token_ids = self.tokenizer.Encode(prepared)
        return np.array(token_ids, dtype=np.int64).reshape(1, -1)

    @staticmethod
    def _find_boundary_indices(tokens: list[int], boundary_tokens: set[int]) -> list[int]:
        indices = [0]
        previous_was_boundary = False
        for index, token in enumerate(tokens):
            if token in boundary_tokens:
                previous_was_boundary = True
            else:
                if previous_was_boundary:
                    indices.append(index)
                previous_was_boundary = False
        indices.append(len(tokens))
        return indices

    def _segments_from_boundaries(
        self, tokens: list[int], boundary_indices: list[int]
    ) -> list[tuple[int, str]]:
        segments = []
        for i in range(len(boundary_indices) - 1):
            start = boundary_indices[i]
            end = boundary_indices[i + 1]
            text = self.tokenizer.Decode(tokens[start:end])
            segments.append((end - start, text))
        return segments

    def _split_into_best_sentences(self, text: str) -> list[str]:
        prepared, _ = self._prepare_text_prompt(text)
        prepared = prepared.strip()
        tokens = self.tokenizer.Encode(prepared)

        eos_tokens = set(self.tokenizer.Encode(".!...?")[1:])
        boundaries = self._find_boundary_indices(tokens, eos_tokens)
        segments = self._segments_from_boundaries(tokens, boundaries)

        fallback_tokens = set(self.tokenizer.Encode(",;:")[1:])
        refined_segments = []
        for count, segment_text in segments:
            if count <= self.max_token_per_chunk:
                refined_segments.append((count, segment_text))
                continue
            sub_tokens = self.tokenizer.Encode(segment_text.strip())
            sub_boundaries = self._find_boundary_indices(sub_tokens, fallback_tokens)
            sub_segments = self._segments_from_boundaries(sub_tokens, sub_boundaries)
            if len(sub_segments) > 1:
                refined_segments.extend(sub_segments)
            else:
                refined_segments.append((count, segment_text))

        chunks = []
        current_chunk = ""
        current_count = 0
        for count, segment_text in refined_segments:
            if not current_chunk:
                current_chunk = segment_text
                current_count = count
                continue
            if current_count + count > self.max_token_per_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = segment_text
                current_count = count
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks

    def _estimate_max_gen_len(self, token_count: int) -> int:
        gen_len_sec = token_count / self.TOKENS_PER_SECOND_ESTIMATE + self.GEN_SECONDS_PADDING
        return int(np.ceil(gen_len_sec * self.frame_rate))

    def _run_flow_lm_chunk(
        self,
        initial_state: dict[str, np.ndarray],
        text_ids: np.ndarray,
        max_frames: Optional[int],
        frames_after_eos: int,
    ) -> Generator[np.ndarray, None, None]:
        state = self._clone_state(initial_state)
        text_embeddings = self.text_conditioner.run(None, {"token_ids": text_ids})[0]
        if text_embeddings.ndim == 2:
            text_embeddings = text_embeddings[None]

        empty_seq = np.zeros((1, 0, self.latent_dim), dtype=np.float32)
        empty_text = np.zeros((1, 0, self.conditioning_dim), dtype=np.float32)

        result = self.flow_lm_main.run(
            None,
            {"sequence": empty_seq, "text_embeddings": text_embeddings, **state},
        )
        self._update_state_from_outputs(state, result, self.flow_state_manifest, output_offset=2)

        curr = np.full((1, 1, self.latent_dim), np.nan, dtype=np.float32)
        eos_step = None
        frame_limit = max_frames or self._estimate_max_gen_len(text_ids.shape[1])
        dt = 1.0 / self.lsd_steps

        for step in range(frame_limit):
            result = self.flow_lm_main.run(
                None,
                {"sequence": curr, "text_embeddings": empty_text, **state},
            )
            conditioning = result[0]
            eos_logit = result[1]
            self._update_state_from_outputs(state, result, self.flow_state_manifest, output_offset=2)

            if step >= self.MIN_FRAMES_BEFORE_EOS and eos_logit[0][0] > -4.0 and eos_step is None:
                eos_step = step
            if eos_step is not None and step >= eos_step + frames_after_eos:
                break

            if self.temperature > 0:
                std = np.sqrt(self.temperature)
                x = np.random.normal(0.0, std, (1, self.latent_dim)).astype(np.float32)
            else:
                x = np.zeros((1, self.latent_dim), dtype=np.float32)

            for s_arr, t_arr in self._st_buffers:
                flow = self.flow_lm_flow.run(
                    None,
                    {"c": conditioning, "s": s_arr, "t": t_arr, "x": x},
                )[0]
                x = x + flow * dt

            latent = x.reshape(1, 1, self.latent_dim)
            yield latent
            curr = latent

    def generate_latents(
        self,
        text: str,
        voice: Union[str, Path, np.ndarray],
        max_frames: Optional[int] = None,
        frames_after_eos: Optional[int] = None,
    ) -> np.ndarray:
        base_state = self.prepare_voice_state(voice)
        latent_chunks = []

        for chunk in self._split_into_best_sentences(text):
            _, guess = self._prepare_text_prompt(chunk)
            effective_frames = (
                frames_after_eos
                if frames_after_eos is not None
                else (self.model_recommended_frames_after_eos or (guess + 2))
            )
            text_ids = self._tokenize(chunk)
            latent_chunks.extend(
                self._run_flow_lm_chunk(base_state, text_ids, max_frames, effective_frames)
            )

        if not latent_chunks:
            return np.zeros((1, 0, self.latent_dim), dtype=np.float32)
        return np.concatenate(latent_chunks, axis=1)

    def decode_latents(self, latents: np.ndarray, chunk_size: int = 15) -> np.ndarray:
        state = self._init_state(self.mimi_state_manifest)
        audio_chunks = []

        for index in range(0, latents.shape[1], chunk_size):
            chunk = latents[:, index : index + chunk_size, :]
            result = self.mimi_decoder.run(None, {"latent": chunk, **state})
            audio_chunks.append(result[0].reshape(-1))
            self._update_state_from_outputs(state, result, self.mimi_state_manifest, output_offset=1)

        if not audio_chunks:
            return np.zeros((0,), dtype=np.float32)
        return np.concatenate(audio_chunks)

    def _decode_worker(self, latent_queue: queue.Queue, audio_chunks: list, decode_chunk_size: int = 12):
        mimi_state = self._init_state(self.mimi_state_manifest)
        buffered = []
        decoded = 0

        while True:
            item = latent_queue.get()
            if item is None:
                break
            buffered.append(item)

            if len(buffered) - decoded >= decode_chunk_size:
                chunk = np.concatenate(buffered[decoded : decoded + decode_chunk_size], axis=1)
                result = self.mimi_decoder.run(None, {"latent": chunk, **mimi_state})
                audio_chunks.append(result[0].reshape(-1))
                self._update_state_from_outputs(
                    mimi_state, result, self.mimi_state_manifest, output_offset=1
                )
                decoded += decode_chunk_size

        if decoded < len(buffered):
            chunk = np.concatenate(buffered[decoded:], axis=1)
            result = self.mimi_decoder.run(None, {"latent": chunk, **mimi_state})
            audio_chunks.append(result[0].reshape(-1))

    def generate(
        self,
        text: str,
        voice: Union[str, Path, np.ndarray],
        max_frames: Optional[int] = None,
        frames_after_eos: Optional[int] = None,
    ) -> np.ndarray:
        base_state = self.prepare_voice_state(voice)
        full_audio = []

        for chunk in self._split_into_best_sentences(text):
            _, guess = self._prepare_text_prompt(chunk)
            effective_frames = (
                frames_after_eos
                if frames_after_eos is not None
                else (self.model_recommended_frames_after_eos or (guess + 2))
            )
            text_ids = self._tokenize(chunk)

            latent_queue: queue.Queue = queue.Queue()
            audio_chunks: list[np.ndarray] = []
            decoder = threading.Thread(
                target=self._decode_worker,
                args=(latent_queue, audio_chunks),
                daemon=True,
            )
            decoder.start()

            for latent in self._run_flow_lm_chunk(base_state, text_ids, max_frames, effective_frames):
                latent_queue.put(latent)
            latent_queue.put(None)
            decoder.join()

            if audio_chunks:
                full_audio.append(np.concatenate(audio_chunks))

        if not full_audio:
            return np.zeros((0,), dtype=np.float32)
        return np.concatenate(full_audio)

    def stream(
        self,
        text: str,
        voice: Union[str, Path, np.ndarray],
        max_frames: Optional[int] = None,
        frames_after_eos: Optional[int] = None,
        first_chunk_frames: int = 2,
        target_buffer_sec: float = 0.2,
        max_chunk_frames: int = 15,
    ) -> Generator[np.ndarray, None, None]:
        base_state = self.prepare_voice_state(voice)

        for chunk_text in self._split_into_best_sentences(text):
            _, guess = self._prepare_text_prompt(chunk_text)
            effective_frames = (
                frames_after_eos
                if frames_after_eos is not None
                else (self.model_recommended_frames_after_eos or (guess + 2))
            )
            text_ids = self._tokenize(chunk_text)

            mimi_state = self._init_state(self.mimi_state_manifest)
            generated_latents = []
            decoded_frames = 0
            playback_start_time = None
            start_time = time.time()

            for latent in self._run_flow_lm_chunk(base_state, text_ids, max_frames, effective_frames):
                generated_latents.append(latent)
                pending = len(generated_latents) - decoded_frames
                chunk_size = 0

                if playback_start_time is None:
                    if pending >= first_chunk_frames:
                        chunk_size = first_chunk_frames
                else:
                    elapsed = time.time() - start_time
                    audio_decoded_sec = decoded_frames * self.frame_duration
                    playback_elapsed = elapsed - playback_start_time
                    buffer_sec = audio_decoded_sec - playback_elapsed

                    if buffer_sec < target_buffer_sec and pending >= 1:
                        chunk_size = min(pending, 3)
                    elif pending >= max_chunk_frames:
                        chunk_size = max_chunk_frames

                if chunk_size > 0:
                    latents_chunk = np.concatenate(
                        generated_latents[decoded_frames : decoded_frames + chunk_size], axis=1
                    )
                    result = self.mimi_decoder.run(None, {"latent": latents_chunk, **mimi_state})
                    self._update_state_from_outputs(
                        mimi_state, result, self.mimi_state_manifest, output_offset=1
                    )
                    decoded_frames += chunk_size
                    if playback_start_time is None:
                        playback_start_time = time.time() - start_time
                    yield result[0].reshape(-1)

            if decoded_frames < len(generated_latents):
                latents_chunk = np.concatenate(generated_latents[decoded_frames:], axis=1)
                result = self.mimi_decoder.run(None, {"latent": latents_chunk, **mimi_state})
                yield result[0].reshape(-1)

    def save_audio(self, audio: np.ndarray, path: Union[str, Path]):
        if not HAS_SOUNDFILE:
            raise ImportError("soundfile required.")
        sf.write(str(path), audio, self.sample_rate)

    @property
    def device(self) -> str:
        if "CUDAExecutionProvider" in self.providers:
            return "cuda"
        return "cpu"

    def __repr__(self) -> str:
        return (
            f"PocketTTSOnnx("
            f"language={self.language!r}, "
            f"device={self.device!r}, "
            f"precision={self.precision!r}, "
            f"temperature={self.temperature}, "
            f"lsd_steps={self.lsd_steps}, "
            f"sample_rate={self.sample_rate})"
        )
````

## File: tiny_tts_onnx.py
````python
"""
TinyTTS ONNX — CPU inference, no PyTorch/transformers/numba.
Model and tokenizer loaded from models/tinytts/ (local, not HF Hub).

NO LONGER USED, PHASE 2

"""
import os, re
import numpy as np
import onnxruntime as ort
from g2p_en import G2p
from tokenizers import Tokenizer

_MODEL_DIR = os.path.join(os.path.dirname(__file__), "models", "tinytts")

# ── Full multilingual phoneme table (from tiny_tts.text.symbols) ──

_ZH_SYMBOLS = ["E","En","a","ai","an","ang","ao","b","c","ch","d","e","ei","en","eng","er","f","g","h","i","i0","ia","ian","iang","iao","ie","in","ing","iong","ir","iu","j","k","l","m","n","o","ong","ou","p","q","r","s","sh","t","u","ua","uai","uan","uang","ui","un","uo","v","van","ve","vn","w","x","y","z","zh","AA","EE","OO"]
_JA_SYMBOLS = ["N","a","a:","b","by","ch","d","dy","e","e:","f","g","gy","h","hy","i","i:","j","k","ky","m","my","n","ny","o","o:","p","py","q","r","ry","s","sh","t","ts","ty","u","u:","w","y","z","zy"]
_EN_SYMBOLS = ["aa","ae","ah","ao","aw","ay","b","ch","d","dh","eh","er","ey","f","g","hh","ih","iy","jh","k","l","m","n","ng","ow","oy","p","r","s","sh","t","th","uh","uw","V","w","y","z","zh"]
_KR_SYMBOLS = ['ᄌ','ᅥ','ᆫ','ᅦ','ᄋ','ᅵ','ᄅ','ᅴ','ᄀ','ᅡ','ᄎ','ᅪ','ᄑ','ᅩ','ᄐ','ᄃ','ᅢ','ᅮ','ᆼ','ᅳ','ᄒ','ᄆ','ᆯ','ᆷ','ᄂ','ᄇ','ᄉ','ᆮ','ᄁ','ᅬ','ᅣ','ᄄ','ᆨ','ᄍ','ᅧ','ᄏ','ᆸ','ᅭ','(','ᄊ',')','ᅲ','ᅨ','ᄈ','ᅱ','ᅯ','ᅫ','ᅰ','ᅤ','~','\\','[',']','/','^',':','ㄸ','*']
_ES_SYMBOLS = ["N","Q","a","b","d","e","f","g","h","i","j","k","l","m","n","o","p","s","t","u","v","w","x","y","z","ɑ","æ","ʃ","ʑ","ç","ɯ","ɪ","ɔ","ɛ","ɹ","ð","ə","ɫ","ɥ","ɸ","ʊ","ɾ","ʒ","θ","β","ŋ","ɦ","ɡ","r","ɲ","ʝ","ɣ","ʎ","ˈ","ˌ","ː"]
_FR_SYMBOLS = ["\u0303","œ","ø","ʁ","ɒ","ʌ","ɜ","ɐ"]
_DE_SYMBOLS = ["ʏ","̩"]
_RU_SYMBOLS = ["ɭ","ʲ","ɕ","\"","ɵ","^","ɬ"]

_PUNCTUATION = ["!", "?", "…", ",", ".", "'", "-", "¿", "¡", "SP", "UNK"]

_NORMAL_SYMBOLS = sorted(set(_ZH_SYMBOLS + _JA_SYMBOLS + _EN_SYMBOLS + _KR_SYMBOLS + _ES_SYMBOLS + _FR_SYMBOLS + _DE_SYMBOLS + _RU_SYMBOLS))
_SYMBOLS = ["_"] + _NORMAL_SYMBOLS + _PUNCTUATION
_SYM_TO_ID = {s: i for i, s in enumerate(_SYMBOLS)}

# ── Constants ──
_UNK_ID = _SYM_TO_ID["UNK"]
_PAD_ID = _SYM_TO_ID["_"]
_EN_TONE_OFFSET = 7  # num_zh_tones(6) + num_ja_tones(1)

# Export constants for app.py
__all__ = ["TinyTTSOnnx", "_PAD_ID", "_UNK_ID"]

# ── ARPAbet → tiny-tts mapping ──
_ARPA_MAP = {
    "AA": "aa", "AE": "ae", "AH": "ah", "AO": "ao", "AW": "aw", "AY": "ay",
    "B": "b", "CH": "ch", "D": "d", "DH": "dh", "EH": "eh", "ER": "er",
    "EY": "ey", "F": "f", "G": "g", "HH": "hh", "IH": "ih", "IY": "iy",
    "JH": "jh", "K": "k", "L": "l", "M": "m", "N": "n", "NG": "ng",
    "OW": "ow", "OY": "oy", "P": "p", "R": "r", "S": "s", "SH": "sh",
    "T": "t", "TH": "th", "UH": "uh", "UW": "uw", "V": "v", "W": "w",
    "Y": "y", "Z": "z", "ZH": "zh",
}
_ARPABET_SET = set(_ARPA_MAP.keys())

_PUNCT_MAP = {
    "：": ",", "；": ",", "，": ",", "。": ".", "！": "!",
    "？": "?", "\n": ".", "·": ",", "、": ",", "…": "…", "v": "V",
}


def _insert_blanks(lst, item):
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result


def _parse_arpabet(ph):
    m = re.match(r'^([A-Z]+)(\d)$', ph)
    if m:
        return m.group(1), int(m.group(2)) + 1
    return ph, 0


def _map_phoneme(symbol):
    if symbol in _PUNCT_MAP:
        symbol = _PUNCT_MAP[symbol]
    if symbol in _SYM_TO_ID:
        return symbol
    return "UNK"


class TinyTTSOnnx:

    def __init__(self):
        self.sample_rate = 44100
        self.frame_rate = 44100.0 / 512.0
        self.predefined_voices = ["MALE", "FEMALE"]

        onnx_path = os.path.join(_MODEL_DIR, "tinytts_fp16.onnx")
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = os.cpu_count() or 2
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            onnx_path, sess_options=opts, providers=["CPUExecutionProvider"]
        )

        tok_path = os.path.join(_MODEL_DIR, "tokenizer.json")
        if os.path.exists(tok_path):
            self.tokenizer = Tokenizer.from_file(tok_path)
        else:
            self.tokenizer = None

        self.g2p = G2p()

    def _tokenize_words(self, text: str) -> list[str]:
        words = []
        if self.tokenizer:
            encoded = self.tokenizer.encode(text)
            tokens = encoded.tokens
            current = ""
            for t in tokens:
                if t in ("[CLS]", "[SEP]"):
                    continue
                if t.startswith("##"):
                    current += t[2:]
                else:
                    if current:
                        words.append(current)
                    current = t
            if current:
                words.append(current)
        else:
            words = text.split()
        return [w for w in words if w]

    def _text_to_ids(self, text: str):
        text = text.strip().lower()
        text = re.sub(r"(\d+)\.(\d+)", r"\1 point \2", text)
        text = re.sub(r"(\d+)\s*-\s*(\d+)", r"\1 to \2", text)

        words = self._tokenize_words(text)
        phones = []
        tones = []

        for word in words:
            if not word:
                continue
            try:
                arpa_phones = self.g2p(word)
            except Exception:
                arpa_phones = []

            for ph in arpa_phones:
                if ph == " ":
                    continue
                if ph in _ARPABET_SET or re.match(r'^[A-Z]+\d$', ph):
                    arpa_base, tone = _parse_arpabet(ph)
                    tts_ph = _ARPA_MAP.get(arpa_base, arpa_base.lower())
                    phones.append(_map_phoneme(tts_ph))
                    tones.append(tone)
                elif ph in _PUNCTUATION or ph in _PUNCT_MAP:
                    tts_ph = _PUNCT_MAP.get(ph, ph)
                    phones.append(_map_phoneme(tts_ph))
                    tones.append(0)
                else:
                    phones.append(_map_phoneme(ph))
                    tones.append(0)

        phones = ["_"] + phones + ["_"]
        tones = [0] + tones + [0]

        phone_ids = [_SYM_TO_ID.get(ph, _UNK_ID) for ph in phones]
        tone_ids = [t + _EN_TONE_OFFSET for t in tones]
        lang_ids = [2] * len(phone_ids)

        phone_ids = _insert_blanks(phone_ids, _PAD_ID)
        tone_ids = _insert_blanks(tone_ids, 0)
        lang_ids = _insert_blanks(lang_ids, 0)

        return phone_ids, tone_ids, lang_ids

    def phonemize(self, text: str, add_padding: bool = False):
        """
        Convert text to phoneme IDs.
        This is FAST (~1ms per word) and can run during LLM generation.
        Args:
            text: Text to phonemize
            add_padding: If True, adds "_" padding at start/end (for single-phrase generation)
                         If False, returns raw phonemes (for accumulation across multiple calls)
        Returns tuple: (phone_ids, tone_ids, lang_ids) - lists of integers
        """
        text = text.strip().lower()
        text = re.sub(r"(\d+)\.(\d+)", r"\1 point \2", text)
        text = re.sub(r"(\d+)\s*-\s*(\d+)", r"\1 to \2", text)

        words = self._tokenize_words(text)
        phones = []
        tones = []

        for word in words:
            if not word:
                continue
            try:
                arpa_phones = self.g2p(word)
            except Exception:
                arpa_phones = []

            for ph in arpa_phones:
                if ph == " ":
                    continue
                if ph in _ARPABET_SET or re.match(r'^[A-Z]+\d$', ph):
                    arpa_base, tone = _parse_arpabet(ph)
                    tts_ph = _ARPA_MAP.get(arpa_base, arpa_base.lower())
                    phones.append(_map_phoneme(tts_ph))
                    tones.append(tone)
                elif ph in _PUNCTUATION or ph in _PUNCT_MAP:
                    tts_ph = _PUNCT_MAP.get(ph, ph)
                    phones.append(_map_phoneme(tts_ph))
                    tones.append(0)
                else:
                    phones.append(_map_phoneme(ph))
                    tones.append(0)

        if add_padding:
            phones = ["_"] + phones + ["_"]
            tones = [0] + tones + [0]

        phone_ids = [_SYM_TO_ID.get(ph, _UNK_ID) for ph in phones]
        tone_ids = [t + _EN_TONE_OFFSET for t in tones]
        lang_ids = [2] * len(phone_ids)

        return phone_ids, tone_ids, lang_ids

    def generate_from_phones(self, phone_ids, tone_ids, lang_ids, voice: str = "MALE"):
        """
        Generate audio from pre-computed phoneme IDs.
        This skips the g2p phonemization step for faster inference.
        Args:
            phone_ids: list of phone IDs (already padded with blanks)
            tone_ids: list of tone IDs (already padded with blanks)
            lang_ids: list of language IDs (already padded with blanks)
        """
        T = len(phone_ids)

        x = np.array(phone_ids, dtype=np.int64)[None, :]
        x_len = np.array([T], dtype=np.int64)
        tone = np.array(tone_ids, dtype=np.int64)[None, :]
        lang = np.array(lang_ids, dtype=np.int64)[None, :]
        bert = np.zeros((1, 1024, T), dtype=np.float16)
        ja_bert = np.zeros((1, 768, T), dtype=np.float16)
        sid_arr = np.array([0], dtype=np.int64)
        noise_scale = np.array([0.667], dtype=np.float16)
        noise_scale_w = np.array([0.8], dtype=np.float16)
        length_scale = np.array([1.0], dtype=np.float16)

        audio = self.session.run(None, {
            "x": x, "x_lengths": x_len, "sid": sid_arr,
            "tone": tone, "language": lang,
            "bert": bert, "ja_bert": ja_bert,
            "noise_scale": noise_scale,
            "noise_scale_w": noise_scale_w,
            "length_scale": length_scale,
        })[0]

        return audio[0, 0].astype(np.float32)

    def generate(self, text: str, voice: str = "MALE"):
        phone_ids, tone_ids, lang_ids = self._text_to_ids(text)
        return self.generate_from_phones(phone_ids, tone_ids, lang_ids, voice)

    def stream(self, text: str, voice: str = "MALE"):
        yield self.generate(text, voice)
````

## File: matmul_common.h
````
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
void find_top_k(float *scores, int N, int K, int *out_indices, void *heap_buffer);
void avx512_diagnostic(void);
````

## File: static/index.html
````html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Bonsai 1.7B — A 1.58-bit LLM</title>
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg:        #0d1117;
  --surface:   #161b22;
  --surface2:  #21262d;
  --border:    #30363d;
  --text:      #e6edf3;
  --muted:     #8b949e;
  --accent:    #3fb950;
  --user-bg:   #1f6feb;
  --error:     #f85149;
  --r:         12px;
}

body {
  background: var(--bg);
  color: var(--text);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
  font-size: 14px;
  height: 100dvh;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

/* ── Header ─────────────────────────────────────────────── */
header {
  background: var(--surface);
  border-bottom: 1px solid var(--border);
  padding: 11px 20px;
  display: flex;
  align-items: center;
  gap: 10px;
  flex-shrink: 0;
}

.h-logo { font-size: 20px; line-height: 1; }

.h-title { font-weight: 600; font-size: 15px; }
.h-sub   { font-size: 11px; color: var(--muted); margin-top: 1px; }

.h-badge {
  margin-left: auto;
  display: flex;
  align-items: center;
  gap: 6px;
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: 20px;
  padding: 4px 11px;
  font-size: 12px;
  color: var(--muted);
  white-space: nowrap;
}

.live-dot {
  width: 7px; height: 7px;
  border-radius: 50%;
  background: var(--accent);
  box-shadow: 0 0 5px var(--accent);
  flex-shrink: 0;
}

/* ── Messages ────────────────────────────────────────────── */
.messages {
  flex: 1;
  overflow-y: auto;
  padding: 28px 20px 16px;
  display: flex;
  flex-direction: column;
  gap: 22px;
}

.messages::-webkit-scrollbar { width: 5px; }
.messages::-webkit-scrollbar-track { background: transparent; }
.messages::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

/* Welcome splash */
.welcome {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 12px;
  color: var(--muted);
  text-align: center;
  padding: 40px 20px;
}
.welcome-icon { font-size: 44px; }
.welcome h2   { font-size: 20px; font-weight: 600; color: var(--text); }
.welcome p    { font-size: 13px; max-width: 340px; line-height: 1.65; }
.welcome a    { color: var(--accent); text-decoration: none; }
.welcome a:hover { text-decoration: underline; }

/* Message rows */
.row {
  display: flex;
  flex-direction: column;
  max-width: 78%;
  position: relative;
}
.row.user { align-self: flex-end; align-items: flex-end; }
.row.bot  { align-self: flex-start; align-items: flex-start; }

.bubble {
  padding: 11px 15px;
  border-radius: var(--r);
  line-height: 1.65;
  white-space: pre-wrap;
  word-break: break-word;
}

.row.user .bubble {
  background: var(--user-bg);
  border-radius: var(--r) var(--r) 3px var(--r);
  color: #fff;
}

.row.bot .bubble {
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: var(--r) var(--r) var(--r) 3px;
}

.row.bot .bubble.error { color: var(--error); }

/* Blinking cursor */
.cursor {
  display: inline-block;
  width: 2px; height: 1em;
  background: var(--accent);
  margin-left: 1px;
  vertical-align: text-bottom;
  animation: blink .75s step-end infinite;
}
@keyframes blink { 50% { opacity: 0; } }

/* Stats line */
.stats {
  font-size: 11px;
  color: var(--muted);
  margin-top: 5px;
  padding: 0 3px;
}

/* ── Input area ──────────────────────────────────────────── */
.input-area {
  background: var(--surface);
  border-top: 1px solid var(--border);
  padding: 13px 18px;
  display: flex;
  gap: 10px;
  align-items: flex-end;
  flex-shrink: 0;
}

textarea {
  flex: 1;
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: var(--r);
  color: var(--text);
  padding: 10px 14px;
  font: inherit;
  font-size: 14px;
  resize: none;
  min-height: 44px;
  max-height: 140px;
  line-height: 1.55;
  outline: none;
  transition: border-color .15s;
  overflow-y: auto;
}
textarea:focus       { border-color: #58a6ff; }
textarea::placeholder { color: var(--muted); }

.send, .stop {
  background: var(--accent);
  color: #0d1117;
  border: none;
  border-radius: var(--r);
  width: 44px; height: 44px;
  display: flex; align-items: center; justify-content: center;
  cursor: pointer;
  flex-shrink: 0;
  font-size: 18px; font-weight: 700; line-height: 1;
  transition: opacity .15s;
}

.stop {
  background: var(--error);
  color: #fff;
}

.send:disabled, .stop:disabled {
  background: var(--surface2);
  color: var(--muted);
  cursor: not-allowed;
}

.send:hover:not(:disabled), .stop:hover:not(:disabled) {
  opacity: .82;
}
</style>
</head>
<body>

<header>
  <span class="h-logo">🌿</span>
  <div>
    <div class="h-title">Bonsai 1.7B</div>
    <div class="h-sub">A 1.58-bit LLM</div>
  </div>
  <div class="h-badge">
    <div class="live-dot"></div>
    <span id="badge-text">live</span>
  </div>
</header>

<div class="messages" id="msgs">
  <div class="welcome" id="welcome">
    <div class="welcome-icon">🌿</div>
    <h2>Bonsai 1.7B</h2>
    <p>A 1.58-bit language model.<br>
       Model: <a href="https://huggingface.co/prism-ml/Ternary-Bonsai-1.7B-mlx-2bit" target="_blank" rel="noopener">prism-ml/Ternary-Bonsai-1.7B-mlx-2bit</a><br>
       by <a href="https://huggingface.co/PrismML" target="_blank" rel="noopener">PrismML / Caltech</a></p>
  </div>
</div>

<div class="input-area">
  <textarea id="inp" placeholder="How can I help you today? (Enter to send, Shift+Enter for newline)" rows="1" autofocus></textarea>
  <button class="send" id="send-btn" title="Send">↑</button>
</div>

<script>
const msgsEl   = document.getElementById('msgs');
const welcomeEl= document.getElementById('welcome');
const inpEl    = document.getElementById('inp');
const sendBtn  = document.getElementById('send-btn');
const badgeEl  = document.getElementById('badge-text');

let busy = false;
let generating = false;

// Auto-resize textarea
inpEl.addEventListener('input', () => {
  inpEl.style.height = 'auto';
  inpEl.style.height = Math.min(inpEl.scrollHeight, 140) + 'px';
});

inpEl.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
});

sendBtn.addEventListener('click', () => {
  if (generating) {
    stopGeneration();
  } else {
    send();
  }
});

function scrollBottom() {
  msgsEl.scrollTop = msgsEl.scrollHeight;
}

function appendUser(text) {
  welcomeEl.style.display = 'none';
  const row = document.createElement('div');
  row.className = 'row user';
  const b = document.createElement('div');
  b.className = 'bubble';
  b.textContent = text;
  row.appendChild(b);
  msgsEl.appendChild(row);
  scrollBottom();
}

function appendBot() {
  const row = document.createElement('div');
  row.className = 'row bot';
  const b = document.createElement('div');
  b.className = 'bubble';
  const cur = document.createElement('span');
  cur.className = 'cursor';
  b.appendChild(cur);
  row.appendChild(b);
  msgsEl.appendChild(row);
  scrollBottom();
  return { row, b, cur };
}

async function stopGeneration() {
  try {
    await fetch('/stop', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({})
    });
  } catch (e) {}
  generating = false;
  busy = false;
  sendBtn.innerHTML = '↑';
  sendBtn.className = 'send';
  sendBtn.disabled = false;
  inpEl.focus();
}

async function send() {
  const prompt = inpEl.value.trim();
  if (!prompt || busy) return;

  busy = true;
  generating = true;
  sendBtn.innerHTML = '⏹';
  sendBtn.className = 'stop';
  inpEl.value = '';
  inpEl.style.height = 'auto';

  appendUser(prompt);
  const { row, b, cur } = appendBot();
  let text = '';
  let lastFullText = '';

  const SYSTEM_PROMPT = "You are a helpful, concise assistant. Answer questions directly and clearly.";

  try {
    const res = await fetch('/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt, max_new_tokens: 512, system_prompt: SYSTEM_PROMPT })
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);

    const reader = res.body.getReader();
    const dec    = new TextDecoder();
    let buf = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const parts = buf.split('\n\n');
      buf = parts.pop();

      for (const part of parts) {
        if (!part.startsWith('data: ')) continue;
        let d;
        try { d = JSON.parse(part.slice(6)); } catch (e) { 
          console.error('[SSE] JSON parse error:', e.message, 'data:', part.substring(0, 100));
          continue; 
        }

        if (d.stopped) {
          cur.remove();
          const s = document.createElement('div');
          s.className = 'stats';
          s.textContent = 'Stopped by user';
          row.appendChild(s);
          break;
        }

        if (d.done) {
          cur.remove();
          const s = document.createElement('div');
          s.className = 'stats';
          s.textContent =
            `${d.tokens_per_second.toFixed(1)} t/s · ${d.total_time_s.toFixed(1)}s · ${d.tokens_generated} tokens`;
          row.appendChild(s);
          badgeEl.textContent = `${d.tokens_per_second.toFixed(1)} t/s`;
        }

        if (d.token !== undefined) {
          text += d.token;
          if (d.full !== lastFullText) {
            b.innerHTML = escapeHtml(d.full) + '<span class="cursor"></span>';
            lastFullText = d.full;
          }
        }
      }
    }
  } catch (err) {
    cur.remove();
    b.classList.add('error');
    b.textContent = `Error: ${err.message}`;
  } finally {
    generating = false;
    busy = false;
    sendBtn.innerHTML = '↑';
    sendBtn.className = 'send';
    sendBtn.disabled = false;
    inpEl.focus();
  }
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}
</script>
</body>
</html>
````

## File: Dockerfile
````dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    clang \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy source files
COPY model_infer.c model_infer.h matmul_common.c matmul_common.h Makefile ./
COPY app.py requirements.txt ./
COPY static/ ./static/

# Build inference library — Makefile auto-detects platform & SIMD level
RUN make inference.so

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Model weights: copied from repo via Git LFS (HF Spaces) or overridden at runtime
# Local dev: docker run -v $(pwd)/models:/app/models cvp-app
COPY models/ ./models/

EXPOSE 7860

ENV MODEL_DIR=/app/models/Ternary-Bonsai-1.7B-unpacked
ENV OMP_SCHEDULE=static
ENV OMP_WAIT_POLICY=passive
ENV OMP_PROC_BIND=close

CMD ["python", "app.py"]
````

## File: model_infer.h
````
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
#define LM_HEAD_CANDIDATES 16384
#define LM_HEAD_PREFILTER_BLOCKS 2

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
    void *topk_heap;
    float kv_k[NUM_LAYERS][NUM_KV_HEADS][MAX_SEQ_LEN][HEAD_DIM];
    float kv_v[NUM_LAYERS][NUM_KV_HEADS][MAX_SEQ_LEN][HEAD_DIM];
    float rope_cos[MAX_SEQ_LEN][HEAD_DIM/2];
    float rope_sin[MAX_SEQ_LEN][HEAD_DIM/2];
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
````

## File: model_infer.c
````cpp
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
    s->topk_heap     = aligned_calloc(64, (size_t)LM_HEAD_CANDIDATES * 8);

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
    free(s->topk_heap);
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
        find_top_k(s->approx_logits, vocab_n, LM_HEAD_CANDIDATES, s->lm_head_candidates, s->topk_heap);
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
        find_top_k(s->approx_logits, vocab_n, LM_HEAD_CANDIDATES, s->lm_head_candidates, s->topk_heap);
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
````

## File: matmul_common.c
````cpp
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
    
    if (!tiles || n_j <= 0) return;
    
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
            const TileBlock8 *tb = &tiles[tg * nkb + bk];
            float32x4_t sc0=vdupq_n_f32(tb->scales[0]), sc1=vdupq_n_f32(tb->scales[1]);
            float32x4_t sc2=vdupq_n_f32(tb->scales[2]), sc3=vdupq_n_f32(tb->scales[3]);
            float32x4_t sc4=vdupq_n_f32(tb->scales[4]), sc5=vdupq_n_f32(tb->scales[5]);
            float32x4_t sc6=vdupq_n_f32(tb->scales[6]), sc7=vdupq_n_f32(tb->scales[7]);
            float32x4_t ns0=vnegq_f32(sc0), ns1=vnegq_f32(sc1);
            float32x4_t ns2=vnegq_f32(sc2), ns3=vnegq_f32(sc3);
            float32x4_t ns4=vnegq_f32(sc4), ns5=vnegq_f32(sc5);
            float32x4_t ns6=vnegq_f32(sc6), ns7=vnegq_f32(sc7);
            const float *ap = &A[i * K + bk * G128_BLOCK_SIZE];
            
            for (int w = 0; w < 4; w++) {
                uint64_t p0=tb->pos[0][w], n0=tb->neg[0][w];
                uint64_t p1=tb->pos[1][w], n1=tb->neg[1][w];
                uint64_t p2=tb->pos[2][w], n2=tb->neg[2][w];
                uint64_t p3=tb->pos[3][w], n3=tb->neg[3][w];
                uint64_t p4=tb->pos[4][w], n4=tb->neg[4][w];
                uint64_t p5=tb->pos[5][w], n5=tb->neg[5][w];
                uint64_t p6=tb->pos[6][w], n6=tb->neg[6][w];
                uint64_t p7=tb->pos[7][w], n7=tb->neg[7][w];
                
                for (int bit = 0; bit < 64; bit += 4) {
                    float32x4_t av = vld1q_f32(ap + w * 32 + bit);
                    uint32_t pm0 = (uint32_t)(p0 >> bit) & 0xF;
                    uint32_t nm0 = (uint32_t)(n0 >> bit) & 0xF;
                    uint32_t pm1 = (uint32_t)(p1 >> bit) & 0xF;
                    uint32_t nm1 = (uint32_t)(n1 >> bit) & 0xF;
                    uint32_t pm2 = (uint32_t)(p2 >> bit) & 0xF;
                    uint32_t nm2 = (uint32_t)(n2 >> bit) & 0xF;
                    uint32_t pm3 = (uint32_t)(p3 >> bit) & 0xF;
                    uint32_t nm3 = (uint32_t)(n3 >> bit) & 0xF;
                    uint32_t pm4 = (uint32_t)(p4 >> bit) & 0xF;
                    uint32_t nm4 = (uint32_t)(n4 >> bit) & 0xF;
                    uint32_t pm5 = (uint32_t)(p5 >> bit) & 0xF;
                    uint32_t nm5 = (uint32_t)(n5 >> bit) & 0xF;
                    uint32_t pm6 = (uint32_t)(p6 >> bit) & 0xF;
                    uint32_t nm6 = (uint32_t)(n6 >> bit) & 0xF;
                    uint32_t pm7 = (uint32_t)(p7 >> bit) & 0xF;
                    uint32_t nm7 = (uint32_t)(n7 >> bit) & 0xF;
                    
                    uint32x4_t mask_p0 = vld1q_u32(mag_lut[pm0]);
                    uint32x4_t mask_n0 = vld1q_u32(mag_lut[nm0]);
                    uint32x4_t mask_p1 = vld1q_u32(mag_lut[pm1]);
                    uint32x4_t mask_n1 = vld1q_u32(mag_lut[nm1]);
                    uint32x4_t mask_p2 = vld1q_u32(mag_lut[pm2]);
                    uint32x4_t mask_n2 = vld1q_u32(mag_lut[nm2]);
                    uint32x4_t mask_p3 = vld1q_u32(mag_lut[pm3]);
                    uint32x4_t mask_n3 = vld1q_u32(mag_lut[nm3]);
                    uint32x4_t mask_p4 = vld1q_u32(mag_lut[pm4]);
                    uint32x4_t mask_n4 = vld1q_u32(mag_lut[nm4]);
                    uint32x4_t mask_p5 = vld1q_u32(mag_lut[pm5]);
                    uint32x4_t mask_n5 = vld1q_u32(mag_lut[nm5]);
                    uint32x4_t mask_p6 = vld1q_u32(mag_lut[pm6]);
                    uint32x4_t mask_n6 = vld1q_u32(mag_lut[nm6]);
                    uint32x4_t mask_p7 = vld1q_u32(mag_lut[pm7]);
                    uint32x4_t mask_n7 = vld1q_u32(mag_lut[nm7]);
                    
                    acc0 = vmlaq_f32(acc0, vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(sc0), mask_p0)), av);
                    acc0 = vmlaq_f32(acc0, vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(ns0), mask_n0)), av);
                    acc1 = vmlaq_f32(acc1, vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(sc1), mask_p1)), av);
                    acc1 = vmlaq_f32(acc1, vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(ns1), mask_n1)), av);
                    acc2 = vmlaq_f32(acc2, vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(sc2), mask_p2)), av);
                    acc2 = vmlaq_f32(acc2, vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(ns2), mask_n2)), av);
                    acc3 = vmlaq_f32(acc3, vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(sc3), mask_p3)), av);
                    acc3 = vmlaq_f32(acc3, vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(ns3), mask_n3)), av);
                    acc4 = vmlaq_f32(acc4, vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(sc4), mask_p4)), av);
                    acc4 = vmlaq_f32(acc4, vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(ns4), mask_n4)), av);
                    acc5 = vmlaq_f32(acc5, vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(sc5), mask_p5)), av);
                    acc5 = vmlaq_f32(acc5, vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(ns5), mask_n5)), av);
                    acc6 = vmlaq_f32(acc6, vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(sc6), mask_p6)), av);
                    acc6 = vmlaq_f32(acc6, vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(ns6), mask_n6)), av);
                    acc7 = vmlaq_f32(acc7, vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(sc7), mask_p7)), av);
                    acc7 = vmlaq_f32(acc7, vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(ns7), mask_n7)), av);
                }
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

#define PROCESS_WORD_FUSED8(tb, pword, ap_off) do { \
    uint64_t _p0=(tb)->pos[0][pword], _p1=(tb)->pos[1][pword]; \
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
    
    if (!tiles || n_j <= 0) return;
    
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
            const TileBlock8 *tb = &tiles[tg * nkb + bk];
            __m512 scv0=_mm512_set1_ps(tb->scales[0]), scv1=_mm512_set1_ps(tb->scales[1]);
            __m512 scv2=_mm512_set1_ps(tb->scales[2]), scv3=_mm512_set1_ps(tb->scales[3]);
            __m512 scv4=_mm512_set1_ps(tb->scales[4]), scv5=_mm512_set1_ps(tb->scales[5]);
            __m512 scv6=_mm512_set1_ps(tb->scales[6]), scv7=_mm512_set1_ps(tb->scales[7]);
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
            
            PROCESS_WORD_FUSED8(tb, 0, 0);
            PROCESS_WORD_FUSED8(tb, 1, 32);
            PROCESS_WORD_FUSED8(tb, 2, 64);
            PROCESS_WORD_FUSED8(tb, 3, 96);
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

void lm_head_prefilter(float *A, G128Matrix *B_T, float *C, int N, int max_blocks) {
    int nkb = (int)B_T->num_blocks_col;
    if (max_blocks > nkb) max_blocks = nkb;
    if (N > (int)B_T->num_rows) N = (int)B_T->num_rows;
    const float *sf = B_T->scales_f32;
    const uint64_t *pos = B_T->packed_pos;
    const uint64_t *neg = B_T->packed_neg;
    int n8 = (N / 8) * 8;

    #pragma omp parallel for schedule(static)
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
        for (int bk = 0; bk < max_blocks; bk++) {
            int b0=r0+bk, b1=r1+bk, b2=r2+bk, b3=r3+bk;
            int b4=r4+bk, b5=r5+bk, b6=r6+bk, b7=r7+bk;
            __m512 scv0=_mm512_set1_ps(sf[b0]), scv1=_mm512_set1_ps(sf[b1]);
            __m512 scv2=_mm512_set1_ps(sf[b2]), scv3=_mm512_set1_ps(sf[b3]);
            __m512 scv4=_mm512_set1_ps(sf[b4]), scv5=_mm512_set1_ps(sf[b5]);
            __m512 scv6=_mm512_set1_ps(sf[b6]), scv7=_mm512_set1_ps(sf[b7]);
            const float *ap = &A[bk * G128_BLOCK_SIZE];
            PROCESS_WORD_POS(0, 0);   PROCESS_WORD_NEG(0, 0);
            PROCESS_WORD_POS(1, 32);  PROCESS_WORD_NEG(1, 32);
            PROCESS_WORD_POS(2, 64);  PROCESS_WORD_NEG(2, 64);
            PROCESS_WORD_POS(3, 96);  PROCESS_WORD_NEG(3, 96);
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
            uint64_t pkp0= B_T->packed_pos[bidx*4+0], pkp1= B_T->packed_pos[bidx*4+1];
            uint64_t pkp2= B_T->packed_pos[bidx*4+2], pkp3= B_T->packed_pos[bidx*4+3];
            uint64_t pkn0= B_T->packed_neg[bidx*4+0], pkn1= B_T->packed_neg[bidx*4+1];
            uint64_t pkn2= B_T->packed_neg[bidx*4+2], pkn3= B_T->packed_neg[bidx*4+3];
            uint32_t cp_lo[4]={(uint32_t)pkp0,(uint32_t)(pkp0>>32),(uint32_t)pkp1,(uint32_t)(pkp1>>32)};
            uint32_t cp_hi[4]={(uint32_t)pkp2,(uint32_t)(pkp2>>32),(uint32_t)pkp3,(uint32_t)(pkp3>>32)};
            uint32_t cn_lo[4]={(uint32_t)pkn0,(uint32_t)(pkn0>>32),(uint32_t)pkn1,(uint32_t)(pkn1>>32)};
            uint32_t cn_hi[4]={(uint32_t)pkn2,(uint32_t)(pkn2>>32),(uint32_t)pkn3,(uint32_t)(pkn3>>32)};
            __m512 scv = _mm512_set1_ps(sf[bidx]);
            const float *ap = &A[bk * G128_BLOCK_SIZE];
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
        C[j] = hsum_zmm(acc);
    }
}

// Compute exact dot products for a selected subset of vocabulary rows.
// Uses pre-separated pos/neg bitmaps.
void matmul_g128_selected(float *A, G128Matrix *B_T, float *C, int M, int K, int N_full, int N_sel, const int *sel_rows) {
    int nkb = (int)B_T->num_blocks_col;
    const float *sf = B_T->scales_f32;
    const uint64_t *pos = B_T->packed_pos;
    const uint64_t *neg = B_T->packed_neg;
    int max_row = (int)B_T->num_rows;
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
            if (r < 0 || r >= max_row) continue;
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

void find_top_k(float *scores, int N, int K, int *out_indices, void *heap_buffer) {
    if (K > N) K = N;
    if (K <= 0) return;
    typedef struct { float score; int idx; } pair_t;
    pair_t *heap = heap_buffer ? (pair_t*)heap_buffer : (pair_t*)malloc((size_t)K * sizeof(pair_t));
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
    if (!heap_buffer) free(heap);
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
    
    if (!tiles || n_j <= 0) return;
    
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
            const TileBlock8 *tb = &tiles[tg * nkb + bk];
            __m256 sc0=_mm256_set1_ps(tb->scales[0]), sc1=_mm256_set1_ps(tb->scales[1]);
            __m256 sc2=_mm256_set1_ps(tb->scales[2]), sc3=_mm256_set1_ps(tb->scales[3]);
            __m256 sc4=_mm256_set1_ps(tb->scales[4]), sc5=_mm256_set1_ps(tb->scales[5]);
            __m256 sc6=_mm256_set1_ps(tb->scales[6]), sc7=_mm256_set1_ps(tb->scales[7]);
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

#ifdef __AVX512F__

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

#endif

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
void find_top_k(float *scores, int N, int K, int *out_indices, void *heap_buffer) {
    (void)scores; (void)heap_buffer;
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
````

## File: app.py
````python
#!/usr/bin/env python3
# frontend
"""FastAPI server for Bonsai 1.7B inference with streaming."""

import os
import json
import ctypes
import time
import signal
import sys
import asyncio
import numpy as np
from contextlib import asynccontextmanager

# we need only tokenizers
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
from transformers import AutoTokenizer
import uvicorn

# Model config
MODEL_DIR = os.environ.get("MODEL_DIR", "./models/Ternary-Bonsai-1.7B-unpacked")
MAX_SEQ_LEN = 512
DEFAULT_TEMP = 0.7
DEFAULT_TOP_P = 0.85
DEFAULT_TOP_K = 20
DEFAULT_MAX_TOKENS = 256
VOCAB_SIZE = 151669
STOP_EOS = 151645

# Global state
_lib = None
ModelState = None
ProfileStats = None
_model = None
_tokenizer = None
G128Matrix = None
FP32Matrix = None
LayerWeights = None

def load_library():
    global _lib, ModelState, ProfileStats, G128Matrix, FP32Matrix, LayerWeights

    lib_path = os.path.join(os.path.dirname(__file__), "inference.so")
    if not os.path.exists(lib_path):
        lib_path = "./inference.so"
    _lib = ctypes.CDLL(lib_path)

    class G128MatrixStruct(ctypes.Structure):
        _fields_ = [
            ("num_rows", ctypes.c_uint32),
            ("num_cols", ctypes.c_uint32),
            ("num_blocks_row", ctypes.c_uint32),
            ("num_blocks_col", ctypes.c_uint32),
            ("magnitude", ctypes.c_void_p),
            ("sign", ctypes.c_void_p),
            ("packed_pos", ctypes.c_void_p),
            ("packed_neg", ctypes.c_void_p),
            ("scales", ctypes.c_void_p),
            ("scales_f32", ctypes.c_void_p),
            ("tiles8", ctypes.c_void_p),
            ("num_tile_groups8", ctypes.c_uint64),
            ("total_tiles8", ctypes.c_uint32),
        ]

    class FP32MatrixStruct(ctypes.Structure):
        _fields_ = [
            ("data", ctypes.c_void_p),
            ("num_rows", ctypes.c_int),
            ("num_cols", ctypes.c_int),
        ]

    class LayerWeightsStruct(ctypes.Structure):
        _fields_ = [
            ("ln1", FP32MatrixStruct),
            ("ln2", FP32MatrixStruct),
            ("q_norm", FP32MatrixStruct),
            ("k_norm", FP32MatrixStruct),
            ("q_proj", G128MatrixStruct),
            ("k_proj", G128MatrixStruct),
            ("v_proj", G128MatrixStruct),
            ("o_proj", G128MatrixStruct),
            ("gate_proj", G128MatrixStruct),
            ("up_proj", G128MatrixStruct),
            ("down_proj", G128MatrixStruct),
        ]

    class ProfileStatsStruct(ctypes.Structure):
        _fields_ = [
            ("decode_count", ctypes.c_uint64),
            ("matmul_ns", ctypes.c_double),
            ("attn_ns", ctypes.c_double),
            ("logits_ns", ctypes.c_double),
            ("total_ns", ctypes.c_double),
            ("per_matmul_ns", ctypes.c_double * 7),
            ("per_matmul_calls", ctypes.c_uint64 * 7),
            ("per_matmul_elements", ctypes.c_uint64 * 7),
        ]

    class ModelStateStruct(ctypes.Structure):
        _fields_ = [
            ("layers", LayerWeightsStruct * 28),
            ("embed", G128MatrixStruct),
            ("final_norm", FP32MatrixStruct),
            ("hidden", ctypes.c_void_p),
            ("normalized", ctypes.c_void_p),
            ("residual", ctypes.c_void_p),
            ("q", ctypes.c_void_p),
            ("k", ctypes.c_void_p),
            ("v", ctypes.c_void_p),
            ("attn_out", ctypes.c_void_p),
            ("attn_weights", ctypes.c_void_p),
            ("gate_out", ctypes.c_void_p),
            ("up_out", ctypes.c_void_p),
            ("mlp_act", ctypes.c_void_p),
            ("approx_logits", ctypes.c_void_p),
            ("lm_head_candidates", ctypes.c_int * 16384),
            ("topk_heap", ctypes.c_void_p),
            ("kv_k", ctypes.c_float * (28 * 512 * 8 * 128)),
            ("kv_v", ctypes.c_float * (28 * 512 * 8 * 128)),
            ("rope_cos", ctypes.c_float * (512 * 64)),
            ("rope_sin", ctypes.c_float * (512 * 64)),
            ("inv_freq", ctypes.c_float * 64),
            ("attn_scale", ctypes.c_float),
            ("kv_len", ctypes.c_int),
            ("loaded", ctypes.c_bool),
            ("profile", ProfileStatsStruct),
        ]

    G128Matrix = G128MatrixStruct
    FP32Matrix = FP32MatrixStruct
    LayerWeights = LayerWeightsStruct
    ModelState = ModelStateStruct
    ProfileStats = ProfileStatsStruct

def _detect_container_cpus():
    # cgroup v2: /sys/fs/cgroup/cpu.max  →  "quota period" or "max period"
    try:
        with open('/sys/fs/cgroup/cpu.max') as f:
            quota_str, period_str = f.read().strip().split()
            if quota_str != 'max':
                return max(1, round(int(quota_str) / int(period_str)))
    except Exception:
        pass
    # cgroup v1
    try:
        with open('/sys/fs/cgroup/cpu/cpu.cfs_quota_us') as fq, \
             open('/sys/fs/cgroup/cpu/cpu.cfs_period_us') as fp:
            quota, period = int(fq.read()), int(fp.read())
            if quota > 0:
                return max(1, round(quota / period))
    except Exception:
        pass
    # Fallback: sched_getaffinity (may be inflated inside containers)
    try:
        return len(os.sched_getaffinity(0))
    except Exception:
        return os.cpu_count() or 2


def init_model():
    global _model, _tokenizer, _lib, ModelState, ProfileStats

    if _model is not None:
        return

    # Detect the real allocated CPU count before the OMP pool is created.
    # os.sched_getaffinity() can return the host CPU count on cgroup-limited pods;
    # cgroup quota is the authoritative number that matches actual scheduling capacity.
    n_cpus = _detect_container_cpus()
    os.environ['OMP_NUM_THREADS'] = str(n_cpus)
    os.environ['OMP_THREAD_LIMIT'] = str(n_cpus)
    os.environ['OMP_DYNAMIC'] = 'FALSE'
    os.environ['OMP_PROC_BIND'] = 'close'
    os.environ['OMP_PLACES'] = 'cores'
    # passive: idle OMP threads sleep instead of spinning, so they don't consume
    # the CPU quota allocated to the working threads.
    os.environ['OMP_WAIT_POLICY'] = 'passive'

    load_library()

    _lib.model_load.argtypes = [ctypes.POINTER(ModelState), ctypes.c_char_p]
    _lib.model_load.restype = ctypes.c_int
    _lib.model_free.argtypes = [ctypes.POINTER(ModelState)]
    _lib.model_prefill.argtypes = [ctypes.POINTER(ModelState), ctypes.POINTER(ctypes.c_int32), ctypes.c_int, ctypes.POINTER(ctypes.c_float)]
    _lib.model_prefill.restype = ctypes.c_int
    _lib.model_decode.argtypes = [ctypes.POINTER(ModelState), ctypes.c_int32, ctypes.POINTER(ctypes.c_float)]
    _lib.model_decode.restype = ctypes.c_int
    _lib.model_get_profile.argtypes = [ctypes.POINTER(ModelState), ctypes.POINTER(ProfileStats)]
    _lib.model_get_profile.restype = None
    _lib.model_reset_profile.argtypes = [ctypes.POINTER(ModelState)]
    _lib.model_reset_profile.restype = None
    _lib.model_matmul_path.restype = ctypes.c_char_p
    _lib.model_compile_info.restype = ctypes.c_char_p
    _lib.model_omp_max_threads.argtypes = []
    _lib.model_omp_max_threads.restype = ctypes.c_int
    _lib.model_set_omp_threads.argtypes = [ctypes.c_int]
    _lib.model_set_omp_threads.restype = None
    _lib.model_affinity_cpu_count.argtypes = []
    _lib.model_affinity_cpu_count.restype = ctypes.c_int

    c_affinity = _lib.model_affinity_cpu_count()
    if c_affinity > 0 and c_affinity < n_cpus:
        n_cpus = c_affinity
        os.environ['OMP_NUM_THREADS'] = str(n_cpus)
        os.environ['OMP_THREAD_LIMIT'] = str(n_cpus)

    _lib.model_set_omp_threads(ctypes.c_int(n_cpus))

    try:
        _lib.model_struct_size.argtypes = []
        _lib.model_struct_size.restype = ctypes.c_long
        c_size = _lib.model_struct_size()
        py_size = ctypes.sizeof(ModelState)
        if c_size != py_size:
            print(f"[WARN] ModelState struct size mismatch: C says {c_size}, Python says {py_size}")
            print("[WARN] inference.so and app.py are out of sync — expect incorrect output")
    except AttributeError:
        pass

    t0 = time.perf_counter()
    _model = ModelState()
    ret = _lib.model_load(ctypes.byref(_model), MODEL_DIR.encode())
    if ret != 0:
        raise RuntimeError(f"Failed to load model from {MODEL_DIR}")
    signal.signal(signal.SIGSEGV, _sigsegv_handler)
    if hasattr(_lib, 'avx512_diagnostic'):
        _lib.avx512_diagnostic.argtypes = []
        _lib.avx512_diagnostic.restype = None
        _lib.avx512_diagnostic()
    load_s = time.perf_counter() - t0

    global _logits_buf
    _logits_buf = (ctypes.c_float * VOCAB_SIZE)()

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)

    log_startup_diagnostics(load_s)
    log_pod_info()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    init_model()
    yield
    # Shutdown
    if _model and _lib:
        _lib.model_free(ctypes.byref(_model))

app = FastAPI(title="Bonsai 1.7B Inference", lifespan=lifespan)

@app.get("/")
async def index():
    return FileResponse("static/index.html")

class GenerateRequest(BaseModel):
    prompt: str
    system_prompt: str = ""
    max_new_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = DEFAULT_TEMP
    top_p: float = DEFAULT_TOP_P
    top_k: int = DEFAULT_TOP_K
    stop_tokens: list = None


class StopRequest(BaseModel):
    session_id: str = "default"


_stop_flags: dict[str, bool] = {}

def _is_tiled_active():
    if not _model or not _model.loaded:
        return False
    try:
        for i in range(28):
            if _model.layers[i].q_proj.tiles8 is not None:
                return True
    except Exception:
        pass
    return False

def np_softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def sample_token(logits, temperature, top_p, top_k):
    logits = np.frombuffer(logits, dtype=np.float32).copy()
    if temperature <= 0:
        return int(np.argmax(logits))
    logits /= temperature
    if top_k > 0:
        # argpartition is O(n) vs argsort O(n log n)
        top_k = min(top_k, len(logits))
        indices = np.argpartition(logits, -top_k)[-top_k:]
        mask = np.ones_like(logits, dtype=bool)
        mask[indices] = False
        logits[mask] = -1e9
    if top_p < 1.0:
        probs = np_softmax(logits)
        sorted_idx = np.argsort(probs)[::-1]
        cumsum = np.cumsum(probs[sorted_idx])
        cutoff = np.searchsorted(cumsum, top_p) + 1
        mask = np.ones_like(logits, dtype=bool)
        mask[sorted_idx[:cutoff]] = False
        logits[mask] = -1e9
    probs = np_softmax(logits)
    return int(np.random.choice(len(probs), p=probs))

_logits_buf = None
_max_token_array = None
_last_prefill_s = 0.0
_last_prompt_tokens = 0

def generate_tokens(prompt_tokens, max_new, temp, top_p, top_k, stop_ids):
    global _logits_buf, _last_prefill_s, _last_prompt_tokens
    t0 = time.perf_counter()
    token_array = (ctypes.c_int32 * len(prompt_tokens))(*prompt_tokens)
    ret = _lib.model_prefill(ctypes.byref(_model), token_array, len(prompt_tokens), _logits_buf)
    _last_prefill_s = time.perf_counter() - t0
    _last_prompt_tokens = len(prompt_tokens)
    if ret != 0:
        raise RuntimeError("Prefill failed")

    for i in range(max_new):
        next_token = sample_token(_logits_buf, temp, top_p, top_k)
        yield next_token
        if next_token in stop_ids:
            break
        ret = _lib.model_decode(ctypes.byref(_model), ctypes.c_int32(next_token), _logits_buf)
        if ret != 0:
            break

@app.post("/generate")
async def generate(req: GenerateRequest):
    messages = []
    if req.system_prompt:
        messages.append({"role": "system", "content": req.system_prompt})
    messages.append({"role": "user", "content": req.prompt})

    text = _tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    tokens = _tokenizer.encode(text, add_special_tokens=False)
    stop_ids = set(req.stop_tokens) if req.stop_tokens else {STOP_EOS}

    def generate_and_decode():
        full_text = ""
        n_tokens = 0
        start = time.perf_counter()
        kernel_type = "tiled8" if _is_tiled_active() else "fallback"
        for token in generate_tokens(tokens, req.max_new_tokens, req.temperature, req.top_p, req.top_k, stop_ids):
            if token == STOP_EOS:
                break
            txt = _tokenizer.decode([token], skip_special_tokens=True)
            full_text += txt
            n_tokens += 1
            yield f"data: {json.dumps({'token': txt, 'full': full_text, 'kernel_type': kernel_type})}\n\n"
        elapsed = time.perf_counter() - start
        tps = n_tokens / elapsed if elapsed > 0 else 0.0
        log_profile(f"generate: {n_tokens} tokens, {tps:.1f} t/s")
        yield f"data: {json.dumps({'done': True, 'full': full_text, 'tokens_generated': n_tokens, 'total_time_s': round(elapsed, 3), 'tokens_per_second': round(tps, 2), 'kernel_type': kernel_type})}\n\n"

    return StreamingResponse(generate_and_decode(), media_type="text/event-stream")

@app.post("/generate/completion")
async def generate_completion(req: GenerateRequest):
    messages = []
    if req.system_prompt:
        messages.append({"role": "system", "content": req.system_prompt})
    messages.append({"role": "user", "content": req.prompt})

    text = _tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    tokens = _tokenizer.encode(text, add_special_tokens=False)
    stop_ids = set(req.stop_tokens) if req.stop_tokens else {STOP_EOS}
    prompt_len = len(tokens)

    def _sync_generate():
        token_ids = []
        start = time.perf_counter()
        for token in generate_tokens(tokens, req.max_new_tokens, req.temperature, req.top_p, req.top_k, stop_ids):
            if token == STOP_EOS:
                break
            token_ids.append(token)
        elapsed = time.perf_counter() - start
        return token_ids, elapsed

    loop = asyncio.get_event_loop()
    token_ids, elapsed = await loop.run_in_executor(None, _sync_generate)

    full_text = _tokenizer.decode(token_ids, skip_special_tokens=True)
    n_tokens = len(token_ids)
    tps = n_tokens / elapsed if elapsed > 0 else 0.0

    log_profile(f"completion: {n_tokens} tokens, {tps:.1f} t/s")

    return {
        "text": full_text,
        "prompt_tokens": prompt_len,
        "tokens_generated": n_tokens,
        "total_time_s": round(elapsed, 3),
        "tokens_per_second": round(tps, 2),
    }


@app.post("/stop")
async def stop_generation(req: StopRequest = None):
    session_id = req.session_id if req else "default"
    _stop_flags[session_id] = True
    return {"stopped": True, "session_id": session_id}

@app.get("/health")
async def health():
    import subprocess
    cpu = ""
    simd = []
    try:
        cpu = open("/proc/cpuinfo").read()
        flags_line = next(l for l in cpu.splitlines() if l.startswith("flags"))
        flags = flags_line.split(":")[1].split()
        for f in ("avx512f", "avx2", "avx", "sse4_2"):
            if f in flags:
                simd.append(f)
        model_name = next(l for l in cpu.splitlines() if "model name" in l).split(":")[1].strip()
    except Exception:
        model_name = "unknown"
    import ctypes.util
    omp_threads = int(os.environ.get("OMP_NUM_THREADS", 0)) or os.cpu_count()
    return {
        "status": "ok",
        "model": "Bonsai 1.7B",
        "vocab_size": VOCAB_SIZE,
        "cpu": model_name,
        "simd": simd,
        "omp_threads": omp_threads,
    }

@app.get("/profile")
async def profile():
    if not _model:
        return {"error": "model not loaded"}
    p = ProfileStats()
    _lib.model_get_profile(ctypes.byref(_model), ctypes.byref(p))
    return _format_profile(p)

def _format_profile(p):
    c = p.decode_count
    if c == 0:
        return {"decode_count": 0}
    matmul_ms = p.matmul_ns / 1e6
    attn_ms = p.attn_ns / 1e6
    logits_ms = p.logits_ns / 1e6
    total_ms = p.total_ns / 1e6
    # Per-matmul breakdown
    per_matmul = {}
    total_elements = 0
    for i, name in enumerate(_MATMUL_TYPE_NAMES):
        calls = p.per_matmul_calls[i]
        ns = p.per_matmul_ns[i]
        elems = p.per_matmul_elements[i]
        total_elements += elems
        if calls > 0:
            per_matmul[name] = {
                "calls": calls,
                "ms": round(ns / 1e6, 2),
                "ms_per_call": round(ns / 1e6 / calls, 2) if calls else 0,
                "elements": elems,
            }
    # Uop efficiency: est. ~13 uops per 16 elements in LUT_ACCUM_ZMM
    est_uops = (total_elements / 16) * 13 if total_elements else 0
    uop_slots = c * 16 * 2.9e9 * 3  # 16 cores × freq × ~3 uops/cycle × decode_count seconds
    # Actually uop_slots should be per step, not total
    decode_time_ns = p.total_ns  # total decode time
    uop_slots_per_step = 16 * 2.9e9 * 3 * (decode_time_ns / 1e9) / c if c > 0 else 1
    uop_efficiency = (est_uops / c) / uop_slots_per_step * 100 if uop_slots_per_step > 0 else 0
    # Effective bandwidth (bytes read = elements × 0.28125 for G128 ternary)
    bytes_read = total_elements * 0.28125 if c > 0 else 0
    bw_gbs = (bytes_read / 1e9) / (matmul_ms / 1000 / c) if matmul_ms > 0 and c > 0 else 0
    return {
        "decode_count": c,
        "kv_len": _model.kv_len if _model else 0,
        "avg_per_step_ms": {
            "matmul": round(matmul_ms / c, 2),
            "attention": round(attn_ms / c, 2),
            "logits": round(logits_ms / c, 2),
            "other": round((total_ms - matmul_ms - attn_ms - logits_ms) / c, 2),
            "total": round(total_ms / c, 2),
        },
        "pct": {
            "matmul": f"{matmul_ms/total_ms*100:.1f}%",
            "attention": f"{attn_ms/total_ms*100:.1f}%",
            "logits": f"{logits_ms/total_ms*100:.1f}%",
            "other": f"{(total_ms-matmul_ms-attn_ms-logits_ms)/total_ms*100:.1f}%",
        },
        "cumulative_s": {
            "matmul": round(matmul_ms / 1000, 2),
            "attention": round(attn_ms / 1000, 2),
            "logits": round(logits_ms / 1000, 2),
            "total": round(total_ms / 1000, 2),
        },
        "per_matmul": per_matmul,
        "diagnostic": {
            "total_ternary_elements": total_elements,
            "est_uops": int(est_uops),
            "uop_efficiency_pct": round(uop_efficiency, 1),
            "eff_bw_gbs": round(bw_gbs, 2),
        },
    }

def _sigsegv_handler(signum, frame):
    print(f"[FATAL] SIGSEGV received at {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print("[FATAL] This is likely a memory corruption bug in inference.so", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(139)

def log_profile(label=""):
    if not _model:
        return
    try:
        p = ProfileStats()
        _lib.model_get_profile(ctypes.byref(_model), ctypes.byref(p))
        d = _format_profile(p)
        d["label"] = label
        d["prefill_s"] = round(_last_prefill_s, 3)
        d["prompt_tokens"] = _last_prompt_tokens
        print(f"[PROFILE] {json.dumps(d)}", flush=True)
    except Exception as e:
        print(f"[PROFILE ERROR] {label}: {e}", flush=True)

def log_startup_diagnostics(load_s):
    diag = {
        "model_load_s": round(load_s, 2),
        "inference_so": _lib.model_matmul_path().decode(),
        "compile": _lib.model_compile_info().decode(),
        "vocab_size": VOCAB_SIZE,
        "hidden_size": 2048,
        "intermediate_size": 6144,
        "num_layers": 28,
        "num_heads": 16,
        "num_kv_heads": 8,
        "head_dim": 128,
        "max_seq_len": MAX_SEQ_LEN,
        "model_dir": MODEL_DIR,
    }
    print(f"[DIAG] {json.dumps(diag)}", flush=True)

_MATMUL_TYPE_NAMES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]

def log_pod_info():
    info = {"env": {}}
    for var in ("OMP_NUM_THREADS", "OMP_SCHEDULE", "OMP_WAIT_POLICY", "OMP_PROC_BIND", "MODEL_DIR"):
        info["env"][var] = os.environ.get(var, "(unset)")
    info["cpu_count"] = os.cpu_count()
    try:
        omp_actual = _lib.model_omp_max_threads()
        info["omp_actual_threads"] = omp_actual
    except Exception:
        pass
    try:
        affinity_cpus = _lib.model_affinity_cpu_count()
        info["affinity_cpus"] = affinity_cpus
    except Exception:
        pass
    try:
        with open("/proc/cpuinfo") as f:
            cpu = f.read()
        for line in cpu.splitlines():
            if line.startswith("model name"):
                info["cpu"] = line.split(":")[1].strip()
                break
        flags_line = next(l for l in cpu.splitlines() if l.startswith("flags"))
        flags = flags_line.split(":")[1].split()
        info["simd"] = [f for f in ("avx512f","avx512_vnni","avx2","avx","sse4_2","sse4_1","ssse3","neon") if f in flags]
    except Exception:
        info["cpu"] = "unknown"
        info["simd"] = []
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if "MemTotal" in line:
                    kb = int(line.split()[1])
                    info["mem_total_gb"] = round(kb / 1024 / 1024, 1)
                elif "MemAvailable" in line:
                    kb = int(line.split()[1])
                    info["mem_avail_gb"] = round(kb / 1024 / 1024, 1)
    except Exception:
        pass
    print(f"[POD] {json.dumps(info)}", flush=True)

@app.post("/profile/reset")
async def profile_reset():
    if not _model:
        return {"error": "model not loaded"}
    log_profile("reset")
    _lib.model_reset_profile(ctypes.byref(_model))
    return {"reset": True}

@app.get("/model/info")
async def model_info():
    tiled_active = False
    if _model and _model.loaded:
        try:
            for i in range(28):
                layer = _model.layers[i]
                if layer.q_proj.tiles8 is not None:
                    tiled_active = True
                    break
        except Exception:
            pass

    info = {
        "llm": {
            "vocab_size": _model.embed.num_rows if _model else None,
            "hidden_size": 2048,
            "num_layers": 28,
            "num_heads": 16,
            "num_kv_heads": 8,
            "head_dim": 128,
            "max_seq_len": MAX_SEQ_LEN
        } if _model else {"error": "LLM not loaded"},
        "tiled_kernel_active": tiled_active,
        "matmul_backend": "avx512_tiled8" if tiled_active else "avx512_fallback" if _model else None,
    }
    return info


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
````

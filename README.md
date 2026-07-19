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

A from-scratch, highly optimized C inference engine for **Ternary-Bonsai-1.7B**, served via a FastAPI server with a streaming dark-mode chat UI. 

**Model by:** [PrismML / Caltech](https://huggingface.co/PrismML)

---

## 🧠 Core Engineering Philosophies

LLM inference is almost entirely memory-bandwidth bound. To get the most out of the CPU, I built this engine from scratch, bypassing standard PyTorch paradigms in favor of bare-metal systems optimization:

*   **1.58-bit G128 Ternary Packing:** Instead of standard quantization, I packed the weights into 1.58 bits (magnitude, sign, and block scale). This reduces the memory footprint by ~16x, turning a memory-bound monster into a bitwise puzzle.
*   **Load-Time Precomputation:** I avoid doing the same math twice. I precompute FP32 scales and pre-separate positive/negative bitmaps at startup, ensuring the hot decode loop executes zero branching and zero type-conversions.
*   **Hand-Tuned SIMD Kernels:** Because compilers can't auto-vectorize custom bit-packed math, I wrote bare-metal AVX-512, AVX2, and NEON kernels. They utilize lookup tables (LUTs) and masked fused-multiply-adds (FMA) to process up to 512 elements per clock cycle.
*   **Zero-Allocation & Head-Major Memory:** Dynamic allocation kills latency, so I allocate all intermediate tensors once at boot. I also lay out the KV cache head-major to guarantee perfectly sequential, cache-line-friendly memory reads during attention.
*   **Two-Phase `lm_head` Prefilter:** Computing 151,000 exact dot products per token is wasteful. I use a cheap, approximate block-sum to instantly eliminate 89% of the vocabulary, reserving heavy math only for the true candidates.
*   **Container-Aware Runtime Tuning:** Cloud schedulers often report the wrong CPU counts. I read cgroup quotas directly to prevent thread over-subscription, pairing it with passive OpenMP policies so idle threads sleep instead of burning CPU quota.

---

## 📦 Model Weights

The `model_weights_repacked/` directory contains the pre-converted G128 ternary format weights tracked via Git LFS. The folder is flattened for direct, zero-nesting access by the C engine:

```text
model_weights_repacked/
├── tokenizer.json
├── weight_model_embed_tokens_weight_magnitude.bin
├── weight_model_embed_tokens_weight_sign.bin
├── weight_model_embed_tokens_weight_scales.bin
└── ... (all other layer triplets & fp32 norms)
```

---

## 🚀 Local Development

```bash
# Build the C inference engine and Python environment
docker build -t cvp-app .

# Run the container (weights are baked in, or mount them locally)
docker run -p 7860:7860 cvp-app
# OR for local iteration without rebuilding the image:
docker run -p 7860:7860 -v $(pwd)/model_weights_repacked:/app/model_weights_repacked cvp-app

# Open the streaming chat UI
# http://localhost:7860
```

---

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `GET /` | — | Chat UI |
| `POST /generate` | SSE stream | Streaming token generation |
| `POST /generate/completion` | JSON | Full response with timing stats |
| `GET /health` | — | Status & SIMD check |
| `GET /model/info` | — | Architecture metadata |
| `GET /profile` | — | Deep-dive C engine profiling stats |

---

## 🔗 Links

- **Official Model:** [prism-ml/Ternary-Bonsai-1.7B-mlx-2bit](https://huggingface.co/prism-ml/Ternary-Bonsai-1.7B-mlx-2bit)
- **Organization:** [PrismML](https://huggingface.co/PrismML)
- **Paper:** [1.58-bit LLMs](https://arxiv.org/abs/2402.17764)

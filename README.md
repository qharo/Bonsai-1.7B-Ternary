# Bonsai 1.7B — A 1.58-bit LLM
[![Model](https://img.shields.io/badge/Model-PrismML%2FTernary--Bonsai--1.7B-blue)](https://huggingface.co/prism-ml/Ternary-Bonsai-1.7B-mlx-2bit)
[![Space](https://img.shields.io/badge/Space-Demo-green)](https://huggingface.co/spaces/qhar0h/Bonsai-1.7B)

An optimized C inference engine for **Ternary-Bonsai-1.7B**, served via a FastAPI server with a streaming dark-mode chat UI. 

**Model by:** [PrismML / Caltech](https://huggingface.co/PrismML)

---

## Optimizations

LLM inference is almost entirely memory-bandwidth bound. To get the most out of the CPU, I focused on moving as little data as possible:

*   **1.58-bit G128 Ternary Packing:** Instead of standard quantization, I packed the weights into 1.58 bits (magnitude, sign, and block scale). This reduces the memory footprint by ~16x
*   **Load-Time Precomputation:** Precompute FP32 scales and pre-separate positive/negative bitmaps at startup
*   **SIMD Kernels:** I used lookup tables (LUTs) and masked fused-multiply-adds (FMA) to process up to 512 elements per clock cycle.
*   **Two-Phase `lm_head` Prefilter** 
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

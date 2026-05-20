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
  - ternary
  - bonsai
license: mit
---

# Bonsai 1.7B — A 1.58-bit Ternary LLM

[![Model](https://img.shields.io/badge/Model-PrismML%2FTernary--Bonsai--1.7B-blue)](https://huggingface.co/prism-ml/Ternary-Bonsai-1.7B-mlx-2bit)
[![Space](https://img.shields.io/badge/Space-Demo-green)](https://huggingface.co/spaces/qhar0h/Bonsai-1.7B)

A 1.58-bit ternary language model inference server with a streaming chat UI.

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

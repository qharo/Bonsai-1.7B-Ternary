---
title: Bonsai 1.7B
emoji: 🌿
colorFrom: green
colorTo: teal
sdk: docker
pinned: false
app_port: 7860
---

# Bonsai 1.7B

A 1-bit language model inference server with a dark-mode streaming chat UI.

Built on the [Ternary-Bonsai-1.7B](https://huggingface.co/PrismML) architecture by PrismML / Caltech. Weights are stored in a custom G128 ternary format and served via a C inference engine with NEON/AVX2 SIMD acceleration.

## Model weights

The `models/` directory contains the pre-converted G128 binary weights tracked via Git LFS:

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

# Bonsai 1.7B — A 1.58-bit LLM with Voice Chat

<div align="center">

[![Hugging Face](https://img.shields.io/static/v1?label=%F0%9F%A4%97%20Model&message=HuggingFace&color=orange)](https://huggingface.co/prism-ml/Ternary-Bonsai-1.7B-mlx-2bit)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Space-Live%20Demo-blue)](https://huggingface.co/spaces/qhar0h/Bonsai)
</div>

An optimized C inference engine for **Ternary-Bonsai-1.7B**, served via a FastAPI server

**Model by:** [PrismML / Caltech](https://huggingface.co/PrismML)

---

## 🚀 Optimizations

LLM inference is almost entirely memory-bandwidth bound. To get the most out of the CPU, I focused on moving as little data as possible:

*   **1.58-bit G128 Ternary Packing:** Instead of standard quantization, I packed the weights into 1.58 bits (magnitude, sign, and block scale). This reduces the memory footprint by ~16x
*   **Load-Time Precomputation:** Precompute FP32 scales and pre-separate positive/negative bitmaps at startup
*   **SIMD Kernels:** I used lookup tables (LUTs) and masked fused-multiply-adds (FMA) to process up to 512 elements per clock cycle.
*   **Two-Phase `lm_head` Prefilter** 
---

## 🎤 Voice Pipeline

The interface is a single voice-first chat UI. You can type or dictate, and the model responds with both text and optional spoken audio.

**Speech-to-Text**
- **Voice Activity Detection (VAD):** WebRTC VAD detects speech boundaries client-side.
- **Transcription:** [faster-whisper tiny](https://huggingface.co/Systran/faster-whisper-tiny) (int8) transcribes the complete audio once the user releases the mic button (or spacebar).

**LLM Inference**
- The full Bonsai 1.7B runs on the optimized C engine using 2 OMP threads. It generates ~7-10 tokens per second on AMD EPYC 7R13.

**Text-to-Speech**
- **Synthesis:** piper-tts (local ONNX) renders each complete sentence.
- **Interleaved Streaming:** The LLM generates sentences in a background thread while TTS synthesizes the previous sentence.

---

## 📦 Model Weights

The `model_weights_repacked/` directory contains the pre-converted G128 ternary format weights tracked via Git LFS. 

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
| `GET /` | — | Voice chat UI (dictation + TTS) |
| `WS /ws/voice` | WebSocket | Dictation, chat, and audio streaming |
| `POST /generate` | SSE stream | Text-only streaming (legacy) |
| `POST /generate/completion` | JSON | Full response with timing stats |
| `GET /health` | — | Status & SIMD check |
| `GET /health/voice` | — | Voice pipeline diagnostics |
| `GET /model/info` | — | Architecture metadata |
| `GET /profile` | — | Deep-dive C engine profiling stats |

---

## 🔗 Links

- **Official Model:** [prism-ml/Ternary-Bonsai-1.7B-mlx-2bit](https://huggingface.co/prism-ml/Ternary-Bonsai-1.7B-mlx-2bit)
- **Organization:** [PrismML](https://huggingface.co/PrismML)
- **Paper:** [PrismML Whitepaper](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/bonsai-27b-whitepaper.pdf)
[1.58-bit LLMs](https://arxiv.org/abs/2402.17764)

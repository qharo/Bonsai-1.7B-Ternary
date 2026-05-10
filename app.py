#!/usr/bin/env python3
"""FastAPI server for BitNet 1.7B inference with streaming."""

import os
import sys
import json
import ctypes
import time
import asyncio
import numpy as np
from contextlib import asynccontextmanager
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
_model = None
_tts = None
_tokenizer = None
G128Matrix = None
FP32Matrix = None
LayerWeights = None

def load_library():
    global _lib, ModelState, G128Matrix, FP32Matrix, LayerWeights
    
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
            ("scales", ctypes.c_void_p),
            ("scales_f32", ctypes.c_void_p),
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
            ("attn_probs", ctypes.c_void_p),
            ("gate_out", ctypes.c_void_p),
            ("up_out", ctypes.c_void_p),
            ("mlp_act", ctypes.c_void_p),
            ("kv_k", ctypes.c_float * (28 * 512 * 8 * 128)),
            ("kv_v", ctypes.c_float * (28 * 512 * 8 * 128)),
            ("inv_freq", ctypes.c_float * 64),
            ("attn_scale", ctypes.c_float),
            ("kv_len", ctypes.c_int),
            ("loaded", ctypes.c_bool),
        ]
    
    G128Matrix = G128MatrixStruct
    FP32Matrix = FP32MatrixStruct
    LayerWeights = LayerWeightsStruct
    ModelState = ModelStateStruct

def init_model():
    global _model, _tokenizer, _lib, _tts, ModelState
    
    if _model is not None:
        return
    
    load_library()
    
    _lib.model_load.argtypes = [ctypes.POINTER(ModelState), ctypes.c_char_p]
    _lib.model_load.restype = ctypes.c_int
    _lib.model_free.argtypes = [ctypes.POINTER(ModelState)]
    _lib.model_prefill.argtypes = [ctypes.POINTER(ModelState), ctypes.POINTER(ctypes.c_int32), ctypes.c_int, ctypes.POINTER(ctypes.c_float)]
    _lib.model_prefill.restype = ctypes.c_int
    _lib.model_decode.argtypes = [ctypes.POINTER(ModelState), ctypes.c_int32, ctypes.POINTER(ctypes.c_float)]
    _lib.model_decode.restype = ctypes.c_int
    
    _model = ModelState()
    ret = _lib.model_load(ctypes.byref(_model), MODEL_DIR.encode())
    if ret != 0:
        raise RuntimeError(f"Failed to load model from {MODEL_DIR}")
    
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    print(f"Model loaded: vocab={_model.embed.num_rows}")
    
    # Initialize TTS
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tts"))
    from pocket_tts_onnx import PocketTTSOnnx
    _tts = PocketTTSOnnx(
        models_dir=os.path.join(os.path.dirname(__file__), "tts", "onnx"),
        language="english_2026-04",
        precision="int8",
    )
    print(f"TTS loaded: sample_rate={_tts.sample_rate}, voices={_tts.predefined_voices}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    init_model()
    yield
    # Shutdown
    if _model and _lib:
        _lib.model_free(ctypes.byref(_model))

app = FastAPI(title="BitNet 1.7B Inference", lifespan=lifespan)

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
    stream_audio: bool = False
    voice: str = "alba"


class StopRequest(BaseModel):
    session_id: str = "default"


_stop_flags: dict[str, bool] = {}

def np_softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def sample_token(logits, temperature, top_p, top_k):
    logits = np.array(logits[:VOCAB_SIZE], dtype=np.float32)
    if temperature > 0:
        logits = logits / temperature
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

def generate_tokens(prompt_tokens, max_new, temp, top_p, top_k, stop_ids):
    logits = (ctypes.c_float * VOCAB_SIZE)()
    token_array = (ctypes.c_int32 * len(prompt_tokens))(*prompt_tokens)
    ret = _lib.model_prefill(ctypes.byref(_model), token_array, len(prompt_tokens), logits)
    if ret != 0:
        raise RuntimeError("Prefill failed")
    
    for i in range(max_new):
        next_token = sample_token(logits, temp, top_p, top_k)
        yield next_token
        if next_token in stop_ids:
            break
        ret = _lib.model_decode(ctypes.byref(_model), ctypes.c_int32(next_token), logits)
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
        for token in generate_tokens(tokens, req.max_new_tokens, req.temperature, req.top_p, req.top_k, stop_ids):
            if token == STOP_EOS:
                break
            txt = _tokenizer.decode([token], skip_special_tokens=True)
            full_text += txt
            n_tokens += 1
            yield f"data: {json.dumps({'token': txt, 'full': full_text})}\n\n"
        elapsed = time.perf_counter() - start
        tps = n_tokens / elapsed if elapsed > 0 else 0.0
        yield f"data: {json.dumps({'done': True, 'full': full_text, 'tokens_generated': n_tokens, 'total_time_s': round(elapsed, 3), 'tokens_per_second': round(tps, 2)})}\n\n"

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

    return {
        "text": full_text,
        "prompt_tokens": prompt_len,
        "tokens_generated": n_tokens,
        "total_time_s": round(elapsed, 3),
        "tokens_per_second": round(tps, 2),
    }


import re

SENTENCE_END_RE = re.compile(r'([.!?]+)(\s|$)')
ABBREVIATIONS = {'Mr', 'Mrs', 'Ms', 'Dr', 'Prof', 'Sr', 'Jr', 'vs', 'etc', 'Inc', 'Ltd', 'Co', 'St', 'Ave', 'Blvd', 'Rd'}

def is_sentence_boundary(text: str, pos: int) -> bool:
    if pos <= 0 or pos >= len(text):
        return False
    if text[pos] not in '.!?':
        return False
    next_char = text[pos + 1] if pos + 1 < len(text) else ' '
    valid_next = {' ', '\t', '\n', '\r', '"', "'", ')', ']', '}'}
    if next_char not in valid_next:
        return False
    before = text[:pos]
    words = before.split()
    if not words:
        return True
    last_word = words[-1].rstrip('.')
    if last_word in ABBREVIATIONS:
        return False
    if last_word.endswith('.') and len(last_word) <= 3:
        return False
    if text[pos:pos+3] == '...':
        return True
    return True


def find_sentence_boundaries(text: str) -> list[int]:
    boundaries = []
    for i, char in enumerate(text):
        if is_sentence_boundary(text, i):
            boundaries.append(i)
    return boundaries


@app.post("/generate/voice")
async def generate_voice(req: GenerateRequest):
    if not _tts:
        raise HTTPException(status_code=503, detail="TTS not loaded")
    
    session_id = "default"
    _stop_flags[session_id] = False
    
    messages = []
    if req.system_prompt:
        messages.append({"role": "system", "content": req.system_prompt})
    messages.append({"role": "user", "content": req.prompt})
    
    text = _tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    tokens = _tokenizer.encode(text, add_special_tokens=False)
    stop_ids = set(req.stop_tokens) if req.stop_tokens else {STOP_EOS}
    
    def generate_voice_stream():
        full_text = ""
        n_tokens = 0
        last_boundary = 0
        start = time.perf_counter()
        
        for token in generate_tokens(tokens, req.max_new_tokens, req.temperature, req.top_p, req.top_k, stop_ids):
            if _stop_flags.get(session_id, False):
                yield f"data: {json.dumps({'stopped': True, 'full': full_text})}\n\n"
                return
            
            if token == STOP_EOS:
                break
            txt = _tokenizer.decode([token], skip_special_tokens=True)
            full_text += txt
            n_tokens += 1
            
            boundaries = find_sentence_boundaries(full_text)
            if boundaries and boundaries[-1] > last_boundary:
                sentence_end = boundaries[-1]
                sentence_text = full_text[last_boundary:sentence_end + 1].strip()
                if sentence_text and len(sentence_text) > 3:
                    try:
                        audio = _tts.generate(sentence_text, voice=req.voice, max_frames=188, frames_after_eos=0)
                        audio_bytes = audio.tobytes()
                        response_data = {
                            'token': txt,
                            'full': full_text,
                            'audio': audio_bytes.hex(),
                            'sample_rate': _tts.sample_rate,
                            'sentence_start': last_boundary,
                            'sentence_end': sentence_end + 1
                        }
                        yield f"data: {json.dumps(response_data)}\n\n"
                        last_boundary = sentence_end + 1
                    except Exception as e:
                        print(f"TTS error: {e}")
                        yield f"data: {json.dumps({'token': txt, 'full': full_text, 'tts_error': str(e)})}\n\n"
                else:
                    yield f"data: {json.dumps({'token': txt, 'full': full_text})}\n\n"
            else:
                yield f"data: {json.dumps({'token': txt, 'full': full_text})}\n\n"
        
        if last_boundary < len(full_text):
            remaining = full_text[last_boundary:].strip()
            if remaining:
                try:
                    audio = _tts.generate(remaining, voice=req.voice, max_frames=188, frames_after_eos=0)
                    audio_bytes = audio.tobytes()
                    response_data = {
                        'audio': audio_bytes.hex(),
                        'sample_rate': _tts.sample_rate,
                        'sentence_start': last_boundary,
                        'sentence_end': len(full_text),
                        'final': True
                    }
                    yield f"data: {json.dumps(response_data)}\n\n"
                except Exception as e:
                    print(f"TTS error (final): {e}")
        
        elapsed = time.perf_counter() - start
        tps = n_tokens / elapsed if elapsed > 0 else 0.0
        yield f"data: {json.dumps({
            'done': True,
            'full': full_text,
            'tokens_generated': n_tokens,
            'total_time_s': round(elapsed, 3),
            'tokens_per_second': round(tps, 2)
        })}\n\n"
        _stop_flags[session_id] = False
    
    return StreamingResponse(generate_voice_stream(), media_type="text/event-stream")


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
        "model": "BitNet-1.7B",
        "vocab_size": VOCAB_SIZE,
        "cpu": model_name,
        "simd": simd,
        "omp_threads": omp_threads,
    }

@app.get("/model/info")
async def model_info():
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
        "tts": {
            "sample_rate": _tts.sample_rate,
            "frame_rate": _tts.frame_rate,
            "voices": list(_tts.predefined_voices),
        } if _tts else {"error": "TTS not loaded"},
    }
    return info


class TTSRequest(BaseModel):
    text: str
    voice: str = "alba"


@app.post("/tts/generate")
async def tts_generate(req: TTSRequest):
    if not _tts:
        raise HTTPException(status_code=503, detail="TTS not loaded")
    
    start = time.time()
    audio = _tts.generate(req.text, voice=req.voice)
    elapsed = time.time() - start
    
    duration = len(audio) / _tts.sample_rate
    
    import io
    import scipy.io.wavfile as wavfile
    buffer = io.BytesIO()
    wavfile.write(buffer, _tts.sample_rate, audio)
    buffer.seek(0)
    
    return {
        "audio_base64": buffer.read().hex(),
        "duration_s": round(duration, 3),
        "generation_time_s": round(elapsed, 3),
        "rtf": round(elapsed / duration, 3) if duration > 0 else 0,
    }


@app.post("/tts/stream")
async def tts_stream(req: TTSRequest):
    if not _tts:
        raise HTTPException(status_code=503, detail="TTS not loaded")
    
    def generate_audio_chunks():
        for chunk in _tts.stream(req.text, voice=req.voice):
            yield chunk.tobytes()
    
    return StreamingResponse(
        generate_audio_chunks(),
        media_type="audio/pcm",
        headers={
            "X-Sample-Rate": str(_tts.sample_rate),
            "X-Channels": "1",
        }
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)

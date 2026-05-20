#!/usr/bin/env python3
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

# Suppress transformers PyTorch advisory warnings (tokenizer works without PyTorch)
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

    # Double-check: C sched_getaffinity may see a different (tighter) affinity than
    # Python/cgroup. Use whichever gives the smaller count so we never over-subscribe.
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

#!/usr/bin/env python3
"""Pre-download voice models for Docker build to avoid startup timeout."""
import os
import sys

# Ensure models are cached in the image
os.environ.setdefault("HF_HOME", "/app/.cache/huggingface")
os.environ.setdefault("TRANSFORMERS_CACHE", "/app/.cache/huggingface")

def download_stt():
    print("[DOWNLOAD] Pre-loading faster-whisper tiny model...")
    try:
        from faster_whisper import WhisperModel
        model = WhisperModel("tiny", device="cpu", compute_type="int8")
        print("[DOWNLOAD] STT model ready")
        return True
    except Exception as e:
        print(f"[DOWNLOAD ERROR] STT: {e}")
        return False

def download_tts():
    print("[DOWNLOAD] Downloading piper voice model...")
    try:
        from huggingface_hub import hf_hub_download
        
        repo_id = "rhasspy/piper-voices"
        
        # Use a good English voice (lessac medium)
        model_path = hf_hub_download(
            repo_id, 
            "en/en_US/lessac/medium/en_US-lessac-medium.onnx",
            local_dir="/app/piper_voice",
            local_dir_use_symlinks=False
        )
        config_path = hf_hub_download(
            repo_id,
            "en/en_US/lessac/medium/en_US-lessac-medium.onnx.json",
            local_dir="/app/piper_voice",
            local_dir_use_symlinks=False
        )
        
        print(f"[DOWNLOAD] TTS model: {model_path}")
        print(f"[DOWNLOAD] TTS config: {config_path}")
        return True
    except Exception as e:
        print(f"[DOWNLOAD ERROR] TTS: {e}")
        return False

if __name__ == "__main__":
    ok_stt = download_stt()
    ok_tts = download_tts()
    
    if ok_stt and ok_tts:
        print("[DOWNLOAD] All models ready")
        sys.exit(0)
    else:
        print("[DOWNLOAD] Some models failed to download")
        sys.exit(1)

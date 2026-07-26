#!/usr/bin/env python3
"""Voice pipeline: VAD → STT → LLM → TTS with extensive logging."""

import asyncio
import io
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import wave
from collections import deque
from typing import AsyncIterator, Callable, List, Optional, Tuple

import numpy as np

# Ensure logging shows INFO messages (HF Spaces suppresses them by default)
logging.basicConfig(level=logging.INFO, format='%(name)s: %(message)s')
logger = logging.getLogger("voice")

# ── Audio utilities ─────────────────────────────────────────

def pcm_to_wav(pcm_bytes: bytes, sample_rate: int = 16000, channels: int = 1) -> bytes:
    """Wrap raw PCM s16le bytes in a WAV container."""
    wav_io = io.BytesIO()
    with wave.open(wav_io, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return wav_io.getvalue()


def convert_audio_to_pcm(input_bytes: bytes, input_format: str = "auto", sample_rate: int = 16000) -> bytes:
    """Convert audio (WebM/Opus/OGG/etc) to 16kHz mono PCM s16le using ffmpeg.
    
    input_format: 'auto' lets ffmpeg detect from magic bytes, or specify 'webm', 'ogg', etc.
    """
    t0 = time.perf_counter()
    
    # Log first 8 bytes as hex for debugging
    header_hex = input_bytes[:8].hex()
    logger.info(f"ffmpeg: input blob {len(input_bytes)} bytes, header={header_hex}")
    
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
    ]
    
    # Only force format if explicitly specified (not 'auto')
    if input_format and input_format != "auto":
        cmd.extend(["-f", input_format])
    
    cmd.extend([
        "-i", "pipe:0",
        "-ar", str(sample_rate), "-ac", "1",
        "-f", "s16le", "-acodec", "pcm_s16le",
        "pipe:1"
    ])
    
    try:
        result = subprocess.run(cmd, input=input_bytes, capture_output=True, timeout=30)
        if result.returncode != 0:
            err = result.stderr.decode('utf-8', errors='replace')[:300]
            logger.error(f"ffmpeg conversion failed: {err}")
            return b''
        proc_s = time.perf_counter() - t0
        logger.info(f"ffmpeg: converted {len(input_bytes)} bytes → {len(result.stdout)} bytes PCM in {proc_s:.2f}s")
        return result.stdout
    except subprocess.TimeoutExpired:
        logger.error("ffmpeg conversion timed out after 30s")
        return b''
    except Exception as e:
        logger.error(f"ffmpeg conversion exception: {type(e).__name__}: {e}")
        return b''


# ── VAD ─────────────────────────────────────────────────────

class VADBuffer:
    """WebRTC VAD-based speech detection on 16kHz 16-bit PCM."""
    
    def __init__(self, aggressiveness: int = 3, frame_ms: int = 30):
        try:
            import webrtcvad
            self.vad = webrtcvad.Vad(aggressiveness)
            self.available = True
            logger.info(f"VAD: Initialized (aggressiveness={aggressiveness}, frame_ms={frame_ms})")
        except Exception as e:
            logger.error(f"VAD: Failed to import webrtcvad: {e}")
            self.vad = None
            self.available = False
        
        self.frame_ms = frame_ms
        self.sample_rate = 16000
        self.frame_bytes = int(self.sample_rate * frame_ms / 1000) * 2  # 960 bytes for 30ms at 16kHz
        
        # State
        self.ring_buffer = bytearray()
        self.speech_active = False
        self.speech_buffer = bytearray()
        self.silence_count = 0
        self.speech_count = 0
        self.total_frames = 0
        self.total_audio_bytes = 0
        
        # Thresholds
        self.min_speech_frames = 10       # 300ms minimum speech
        self.max_silence_frames = 33      # ~1s silence triggers end
        self.max_duration_frames = 1000   # 30s max utterance
        
    def process(self, pcm_chunk: bytes) -> Optional[bytes]:
        """
        Process PCM audio chunk. Returns complete utterance bytes when silence detected.
        """
        if not self.available:
            return None
        
        self.total_audio_bytes += len(pcm_chunk)
        self.ring_buffer.extend(pcm_chunk)
        utterance = None
        frames_processed = 0
        
        while len(self.ring_buffer) >= self.frame_bytes:
            frame = bytes(self.ring_buffer[:self.frame_bytes])
            self.ring_buffer = self.ring_buffer[self.frame_bytes:]
            self.total_frames += 1
            frames_processed += 1
            
            try:
                is_speech = self.vad.is_speech(frame, self.sample_rate)
            except Exception as e:
                logger.warning(f"VAD frame processing error: {e}")
                is_speech = False
            
            if is_speech:
                self.speech_count += 1
                self.silence_count = max(0, self.silence_count - 2)
                if not self.speech_active and self.speech_count >= self.min_speech_frames:
                    self.speech_active = True
                    self.speech_buffer = bytearray()
                    logger.info(f"VAD: Speech started (frame {self.total_frames}, speech_count={self.speech_count})")
            else:
                self.silence_count += 1
                self.speech_count = max(0, self.speech_count - 1)
            
            if self.speech_active:
                self.speech_buffer.extend(frame)
                
                # Check silence
                if self.silence_count >= self.max_silence_frames:
                    utterance = bytes(self.speech_buffer)
                    duration_s = len(utterance) / (self.sample_rate * 2)
                    logger.info(
                        f"VAD: Speech ended by silence "
                        f"({duration_s:.1f}s, {self.total_frames} total frames, "
                        f"{len(self.speech_buffer)} bytes, "
                        f"silence_count={self.silence_count})"
                    )
                    self._reset()
                    break
                
                # Check max duration
                speech_frames = len(self.speech_buffer) // self.frame_bytes
                if speech_frames >= self.max_duration_frames:
                    utterance = bytes(self.speech_buffer)
                    duration_s = len(utterance) / (self.sample_rate * 2)
                    logger.info(
                        f"VAD: Speech ended by max duration "
                        f"({duration_s:.1f}s, {speech_frames} frames)"
                    )
                    self._reset()
                    break
        
        if frames_processed > 0 and not self.speech_active:
            logger.debug(
                f"VAD: Processed {frames_processed} frames, "
                f"total_audio={self.total_audio_bytes} bytes, "
                f"ring_buffer={len(self.ring_buffer)} bytes, "
                f"speech_count={self.speech_count}, silence_count={self.silence_count}"
            )
        
        return utterance
    
    def _reset(self):
        self.speech_active = False
        self.speech_buffer = bytearray()
        self.speech_count = 0
        self.silence_count = 0
    
    def flush(self) -> Optional[bytes]:
        """Return pending speech if any (e.g., on disconnect)."""
        if self.speech_active and len(self.speech_buffer) > self.frame_bytes * 5:
            utterance = bytes(self.speech_buffer)
            duration_s = len(utterance) / (self.sample_rate * 2)
            logger.info(f"VAD: Flushing pending speech ({duration_s:.1f}s)")
            self._reset()
            return utterance
        return None


# ── STT ─────────────────────────────────────────────────────

class STT:
    """Speech-to-Text using faster-whisper."""
    
    def __init__(self, model_size: str = "tiny", device: str = "cpu", compute_type: str = "int8"):
        self.available = False
        self.model = None
        
        try:
            from faster_whisper import WhisperModel
            logger.info(f"STT: Loading faster-whisper model '{model_size}' (device={device}, compute_type={compute_type})...")
            t0 = time.perf_counter()
            self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
            load_s = time.perf_counter() - t0
            self.available = True
            logger.info(f"STT: Model loaded in {load_s:.1f}s")
        except Exception as e:
            logger.error(f"STT: Failed to load model: {type(e).__name__}: {e}", exc_info=True)
    
    def transcribe(self, pcm_bytes: bytes) -> str:
        """Transcribe 16kHz mono PCM audio to text."""
        if not self.available or not self.model:
            logger.warning("STT: Model not available, skipping transcription")
            return ""
        
        # Convert PCM bytes to normalized float32 array
        audio_np = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        
        if len(audio_np) < 1600:  # < 100ms
            logger.debug(f"STT: Audio too short ({len(audio_np)/16:.0f}ms), skipping")
            return ""
        
        t0 = time.perf_counter()
        try:
            logger.info(f"STT: Starting transcription of {len(audio_np)/16000:.1f}s audio ({len(pcm_bytes)} bytes)")
            
            segments, info = self.model.transcribe(
                audio_np, 
                beam_size=5, 
                language="en",
                condition_on_previous_text=False,
                vad_filter=True,  # Use whisper's built-in VAD for better accuracy
                vad_parameters=dict(min_silence_duration_ms=500)
            )
            texts = []
            segment_count = 0
            for segment in segments:
                texts.append(segment.text.strip())
                segment_count += 1
            text = " ".join(texts).strip()
            proc_s = time.perf_counter() - t0
            duration_s = len(audio_np) / 16000
            
            logger.info(
                f"STT: Transcription complete in {proc_s:.2f}s "
                f"(duration={duration_s:.1f}s, segments={segment_count}, "
                f"text='{text[:120]}')"
            )
            return text
        except Exception as e:
            logger.error(f"STT: Transcription failed: {type(e).__name__}: {e}", exc_info=True)
            return ""


# ── TTS ─────────────────────────────────────────────────────

class TTS:
    """Text-to-Speech using piper-tts via subprocess CLI (primary) or Python API (fallback)."""
    
    def __init__(self, model_path: Optional[str] = None, config_path: Optional[str] = None):
        self.available = False
        self.voice = None
        self.sample_rate = 22050
        self.model_path = model_path
        self.config_path = config_path
        self._cli_available = False
        
        if not model_path or not config_path:
            logger.warning("TTS: No model/config paths provided, TTS disabled")
            return
        
        logger.info(f"TTS: Checking paths - model={model_path}, config={config_path}")
        
        if not os.path.exists(model_path):
            logger.error(f"TTS: Model file not found: {model_path}")
            return
        if not os.path.exists(config_path):
            logger.error(f"TTS: Config file not found: {config_path}")
            return
        
        # Check if piper CLI is available
        try:
            result = subprocess.run(
                ["piper", "--help"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                self._cli_available = True
                logger.info("TTS: piper CLI is available")
            else:
                logger.warning(f"TTS: piper CLI returned {result.returncode}: {result.stderr[:200]}")
        except Exception as e:
            logger.warning(f"TTS: piper CLI not found: {type(e).__name__}: {e}")
        
        # Try loading Python API voice as fallback
        try:
            from piper import PiperVoice
            logger.info(f"TTS: Loading piper voice from {model_path}...")
            t0 = time.perf_counter()
            self.voice = PiperVoice.load(model_path, config_path)
            self.sample_rate = getattr(self.voice.config, 'sample_rate', 22050)
            self.available = True
            load_s = time.perf_counter() - t0
            logger.info(f"TTS: Voice loaded in {load_s:.1f}s (sample_rate={self.sample_rate})")
            
            # Diagnostic using print() so it always shows in HF Spaces logs
            print(f"[TTS] === DIAGNOSTIC START ===", flush=True)
            self._diagnostic_synthesize()
            print(f"[TTS] === DIAGNOSTIC END ===", flush=True)
            
        except Exception as e:
            logger.error(f"TTS: Failed to load piper voice: {type(e).__name__}: {e}", exc_info=True)
            # If CLI is available, we can still function without Python API
            if self._cli_available:
                self.available = True
                logger.info("TTS: Using CLI-only mode (Python API failed)")
    
    def _diagnostic_synthesize(self):
        """Run quick tests with print() so logs always show."""
        test_text = "hello"
        
        # Test 1: CLI subprocess
        if self._cli_available:
            try:
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tf:
                    tmp_path = tf.name
                print(f"[TTS] DIAG: Testing CLI subprocess...", flush=True)
                result = subprocess.run(
                    ["piper", "--model", self.model_path, "--config", self.config_path, "--output_file", tmp_path],
                    input=test_text,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                if result.returncode == 0 and os.path.exists(tmp_path):
                    size = os.path.getsize(tmp_path)
                    print(f"[TTS] DIAG: CLI SUCCESS — {size} bytes", flush=True)
                    with open(tmp_path, 'rb') as f:
                        data = f.read()
                    os.unlink(tmp_path)
                    print(f"[TTS] DIAG: CLI header={data[:4]}", flush=True)
                else:
                    print(f"[TTS] DIAG: CLI FAILED rc={result.returncode} stderr={result.stderr[:200]}", flush=True)
            except Exception as e:
                print(f"[TTS] DIAG: CLI ERROR {type(e).__name__}: {e}", flush=True)
        
        # Test 2: Python file-based
        if self.voice:
            try:
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tf:
                    tmp_path = tf.name
                print(f"[TTS] DIAG: Testing Python file-based...", flush=True)
                self.voice.synthesize(test_text, open(tmp_path, 'wb'))
                size = os.path.getsize(tmp_path)
                print(f"[TTS] DIAG: Python file SUCCESS — {size} bytes", flush=True)
                with open(tmp_path, 'rb') as f:
                    data = f.read()
                os.unlink(tmp_path)
                print(f"[TTS] DIAG: Python header={data[:4]}", flush=True)
            except Exception as e:
                print(f"[TTS] DIAG: Python file FAILED {type(e).__name__}: {e}", flush=True)
            
            # Test 3: Python chunks
            try:
                print(f"[TTS] DIAG: Testing Python chunks...", flush=True)
                chunks = list(self.voice.synthesize(test_text))
                print(f"[TTS] DIAG: Got {len(chunks)} chunks", flush=True)
                if chunks:
                    chunk = chunks[0]
                    print(f"[TTS] DIAG: chunk type={type(chunk).__name__}", flush=True)
                    attrs = [a for a in dir(chunk) if not a.startswith('_') and not callable(getattr(chunk, a, None))]
                    print(f"[TTS] DIAG: chunk attrs={attrs[:10]}", flush=True)
            except Exception as e:
                print(f"[TTS] DIAG: Python chunks FAILED {type(e).__name__}: {e}", flush=True)

    def synthesize(self, text: str) -> Optional[bytes]:
        """Synthesize text to WAV bytes using CLI (primary) or Python API (fallback)."""
        if not self.available:
            logger.warning("TTS: Not available, skipping synthesis")
            return None
        
        if not text or not text.strip():
            return None
        
        t0 = time.perf_counter()
        
        # Strategy 1: Subprocess CLI (most reliable)
        if self._cli_available and self.model_path and self.config_path:
            try:
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tf:
                    tmp_path = tf.name
                
                logger.debug(f"TTS: Using CLI subprocess for '{text[:60]}...'")
                result = subprocess.run(
                    [
                        "piper",
                        "--model", self.model_path,
                        "--config", self.config_path,
                        "--output_file", tmp_path
                    ],
                    input=text,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result.returncode != 0:
                    logger.warning(f"TTS: CLI failed rc={result.returncode} stderr={result.stderr[:200]}")
                    # Clean up and fall through
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                elif os.path.exists(tmp_path):
                    with open(tmp_path, 'rb') as f:
                        audio_data = f.read()
                    os.unlink(tmp_path)
                    
                    proc_s = time.perf_counter() - t0
                    header_valid = audio_data[:4] == b'RIFF' if len(audio_data) >= 4 else False
                    logger.info(
                        f"TTS: '{text[:60]}...' → "
                        f"{len(audio_data)} bytes WAV via CLI ({proc_s:.2f}s, "
                        f"header_valid={header_valid})"
                    )
                    return audio_data
                    
            except Exception as e:
                logger.warning(f"TTS: CLI subprocess failed: {type(e).__name__}: {e}")
        
        # Strategy 2: Python file-based synthesis
        if self.voice:
            try:
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tf:
                    tmp_path = tf.name
                
                logger.debug(f"TTS: Using Python file-based for '{text[:60]}...'")
                file_obj = open(tmp_path, 'wb')
                try:
                    self.voice.synthesize(text, file_obj)
                finally:
                    file_obj.close()
                
                with open(tmp_path, 'rb') as f:
                    audio_data = f.read()
                os.unlink(tmp_path)
                
                proc_s = time.perf_counter() - t0
                header_valid = audio_data[:4] == b'RIFF' if len(audio_data) >= 4 else False
                logger.info(
                    f"TTS: '{text[:60]}...' → "
                    f"{len(audio_data)} bytes WAV via Python file ({proc_s:.2f}s, "
                    f"header_valid={header_valid})"
                )
                return audio_data
                
            except Exception as e:
                logger.warning(f"TTS: Python file-based failed: {type(e).__name__}: {e}")
        
        # Strategy 3: Python chunk-based (last resort)
        if self.voice and hasattr(self.voice, 'synthesize'):
            try:
                logger.debug(f"TTS: Using Python chunk-based for '{text[:60]}...'")
                audio_data = b''
                chunk_count = 0
                
                for chunk in self.voice.synthesize(text):
                    if isinstance(chunk, bytes):
                        audio_data += chunk
                        chunk_count += 1
                    elif hasattr(chunk, 'bytes'):
                        audio_data += chunk.bytes
                        chunk_count += 1
                    elif hasattr(chunk, 'data'):
                        audio_data += chunk.data
                        chunk_count += 1
                
                if audio_data:
                    wav_io = io.BytesIO()
                    with wave.open(wav_io, 'wb') as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(self.sample_rate)
                        wf.writeframes(audio_data)
                    
                    audio = wav_io.getvalue()
                    proc_s = time.perf_counter() - t0
                    logger.info(
                        f"TTS: '{text[:60]}...' → "
                        f"{len(audio)} bytes WAV via chunks ({proc_s:.2f}s, "
                        f"chunks={chunk_count})"
                    )
                    return audio
                    
            except Exception as e:
                logger.error(f"TTS: All strategies failed for '{text[:60]}...': {type(e).__name__}: {e}")
        
        return None


# ── Sentence splitting ──────────────────────────────────────

_SENTENCE_RE = re.compile(r'(?<=[.!?])\s+|\n+')

def split_sentences(text: str) -> List[str]:
    """Split text into sentences for incremental TTS."""
    sentences = _SENTENCE_RE.split(text)
    return [s.strip() for s in sentences if s.strip()]


# ── Orchestrator ────────────────────────────────────────────

class VoiceOrchestrator:
    """Orchestrates the full voice pipeline with streaming LLM and TTS."""
    
    def __init__(self, stt: STT, tts: TTS, llm_generate_fn: Callable):
        self.stt = stt
        self.tts = tts
        self.llm_generate = llm_generate_fn  # sync generator yielding tokens
        
    async def process_utterance(self, pcm_bytes: bytes) -> AsyncIterator[dict]:
        """
        Process a complete audio utterance through the full pipeline.
        Interleaved: LLM generates tokens in background, TTS runs on complete sentences
        as soon as they're detected, while LLM continues generating.
        
        Yields dicts with keys:
            type: "transcription" | "llm_text" | "audio" | "status" | "error"
            Additional fields depend on type.
        """
        pipeline_t0 = time.perf_counter()
        logger.info(
            f"Pipeline: Starting interleaved processing of {len(pcm_bytes)} bytes "
            f"(STT_available={self.stt.available if self.stt else False}, "
            f"TTS_available={self.tts.available if self.tts else False})"
        )
        
        # 1. STT
        yield {"type": "status", "message": "Transcribing speech..."}
        
        stt_t0 = time.perf_counter()
        text = self.stt.transcribe(pcm_bytes)
        stt_s = time.perf_counter() - stt_t0
        
        yield {"type": "transcription", "text": text, "stt_time_s": round(stt_s, 2)}
        logger.info(f"Pipeline: STT complete in {stt_s:.2f}s → '{text[:120]}'")
        
        if not text:
            yield {"type": "status", "message": "No speech detected", "done": True}
            logger.info("Pipeline: No speech detected, ending")
            return
        
        # 2+3. Interleaved LLM + TTS
        yield {"type": "status", "message": "Generating and synthesizing..."}
        
        loop = asyncio.get_event_loop()
        token_queue = asyncio.Queue()
        llm_done = threading.Event()
        
        # Start LLM generation in background thread
        def _generate_tokens():
            try:
                for token in self.llm_generate(text):
                    token_queue.put_nowait(token)
            except Exception as e:
                logger.error(f"Pipeline: LLM generation error: {e}")
            finally:
                llm_done.set()
        
        llm_thread = threading.Thread(target=_generate_tokens, daemon=True)
        llm_thread.start()
        
        # Accumulate tokens and detect sentences for interleaved TTS
        accumulated_text = ""
        sentence_index = 0
        audio_count = 0
        fail_count = 0
        llm_token_count = 0
        llm_t0 = time.perf_counter()
        
        def _extract_sentences(text):
            """Extract complete sentences from text, return (sentences, remaining_text)."""
            parts = _SENTENCE_RE.split(text)
            if len(parts) <= 1:
                return [], text
            
            # Check if text ends with sentence terminator (allowing trailing space)
            # A sentence is "complete" if it's followed by another non-empty part
            sentences = []
            remaining = ""
            
            for i, part in enumerate(parts):
                part = part.strip()
                if not part:
                    continue
                    
                # Check if this part is followed by whitespace in original text
                # (indicating it was split by the regex)
                if i < len(parts) - 1:
                    # This is a complete sentence
                    sentences.append(part)
                else:
                    # Last part — might be incomplete
                    remaining = part
            
            return sentences, remaining
        
        # Consume tokens and process sentences
        while not (llm_done.is_set() and token_queue.empty()):
            try:
                # Wait for tokens with timeout
                token = await asyncio.wait_for(token_queue.get(), timeout=0.1)
                accumulated_text += token
                llm_token_count += 1
                
                # Try to extract complete sentences
                sentences, remaining = _extract_sentences(accumulated_text)
                
                if sentences:
                    accumulated_text = remaining  # Keep incomplete part
                    
                    for sentence in sentences:
                        if not sentence.strip():
                            continue
                        
                        # Stream text to client
                        yield {"type": "llm_text", "text": sentence + " ", "index": sentence_index}
                        
                        # Run TTS in parallel executor
                        if self.tts.available:
                            def _synthesize(sent):
                                return self.tts.synthesize(sent)
                            
                            try:
                                audio = await loop.run_in_executor(None, _synthesize, sentence)
                                if audio and len(audio) > 100:
                                    audio_count += 1
                                    logger.info(
                                        f"Pipeline: Interleaved TTS sentence {sentence_index} "
                                        f"({len(audio)} bytes, LLM still running={not llm_done.is_set()})"
                                    )
                                    yield {"type": "audio", "data": audio, "index": sentence_index}
                                else:
                                    fail_count += 1
                                    logger.warning(
                                        f"Pipeline: TTS failed for sentence {sentence_index} "
                                        f"(audio={len(audio) if audio else 0} bytes)"
                                    )
                            except Exception as e:
                                fail_count += 1
                                logger.error(f"Pipeline: TTS error for sentence {sentence_index}: {e}")
                        else:
                            logger.debug(f"Pipeline: TTS unavailable, skipping sentence {sentence_index}")
                        
                        sentence_index += 1
                        
            except asyncio.TimeoutError:
                # No tokens available, check if LLM is done
                if llm_done.is_set() and token_queue.empty():
                    break
                continue
        
        llm_thread.join(timeout=1.0)
        llm_s = time.perf_counter() - llm_t0
        tps = llm_token_count / llm_s if llm_s > 0 else 0
        
        # Process any remaining accumulated text
        if accumulated_text.strip():
            yield {"type": "llm_text", "text": accumulated_text.strip() + " ", "index": sentence_index}
            
            if self.tts.available:
                try:
                    audio = await loop.run_in_executor(None, self.tts.synthesize, accumulated_text.strip())
                    if audio and len(audio) > 100:
                        audio_count += 1
                        yield {"type": "audio", "data": audio, "index": sentence_index}
                    else:
                        fail_count += 1
                except Exception as e:
                    fail_count += 1
                    logger.error(f"Pipeline: TTS error for final text: {e}")
            sentence_index += 1
        
        tts_s = time.perf_counter() - llm_t0 - llm_s if llm_t0 else 0
        total_s = time.perf_counter() - pipeline_t0
        
        if fail_count > 0 and audio_count == 0:
            logger.error(f"Pipeline: ALL {fail_count} TTS sentences failed")
            yield {
                "type": "error",
                "message": f"TTS failed for all {fail_count} sentences. Check server logs."
            }
        
        logger.info(
            f"Pipeline: COMPLETE (interleaved). Total={total_s:.1f}s "
            f"(STT={stt_s:.1f}s, LLM={llm_s:.1f}s [{tps:.1f} t/s], "
            f"sentences={sentence_index}, audio_sentences={audio_count}, tts_fails={fail_count})"
        )
        
        yield {
            "type": "status",
            "message": "Done",
            "done": True,
            "total_time_s": round(total_s, 2),
            "stt_time_s": round(stt_s, 2),
            "llm_time_s": round(llm_s, 2),
            "tts_time_s": round(tts_s, 2),
            "tokens_generated": llm_token_count,
            "tokens_per_second": round(tps, 1),
            "sentences": sentence_index,
            "audio_sentences": audio_count,
            "tts_fails": fail_count,
        }

    async def transcribe(self, pcm_bytes: bytes) -> AsyncIterator[dict]:
        """Dictation mode: STT only, returns transcription text."""
        yield {"type": "status", "message": "Transcribing speech..."}

        stt_t0 = time.perf_counter()
        text = self.stt.transcribe(pcm_bytes)
        stt_s = time.perf_counter() - stt_t0

        logger.info(f"Dictation: '{text[:120]}' in {stt_s:.2f}s")
        yield {"type": "dictation", "text": text, "stt_time_s": round(stt_s, 2)}

    async def chat(self, text: str, voice_output: bool = True) -> AsyncIterator[dict]:
        """Chat mode: LLM + optional TTS. Same interleaved logic as process_utterance but skips STT."""
        pipeline_t0 = time.perf_counter()
        logger.info(f"Chat: '{text[:120]}' (voice_output={voice_output})")

        # 1. LLM (streaming generation in background)
        yield {"type": "status", "message": "Generating response..."}

        loop = asyncio.get_event_loop()
        token_queue = asyncio.Queue()
        llm_done = threading.Event()

        def _generate_tokens():
            try:
                for token in self.llm_generate(text):
                    token_queue.put_nowait(token)
            except Exception as e:
                logger.error(f"Chat: LLM generation error: {e}")
            finally:
                llm_done.set()

        llm_thread = threading.Thread(target=_generate_tokens, daemon=True)
        llm_thread.start()

        accumulated_text = ""
        sentence_index = 0
        audio_count = 0
        fail_count = 0
        llm_token_count = 0
        llm_t0 = time.perf_counter()

        def _extract_sentences(text):
            parts = _SENTENCE_RE.split(text)
            if len(parts) <= 1:
                return [], text
            sentences = []
            remaining = ""
            for i, part in enumerate(parts):
                part = part.strip()
                if not part:
                    continue
                if i < len(parts) - 1:
                    sentences.append(part)
                else:
                    remaining = part
            return sentences, remaining

        while not (llm_done.is_set() and token_queue.empty()):
            try:
                token = await asyncio.wait_for(token_queue.get(), timeout=0.1)
                accumulated_text += token
                llm_token_count += 1

                sentences, remaining = _extract_sentences(accumulated_text)
                if sentences:
                    accumulated_text = remaining
                    for sentence in sentences:
                        if not sentence.strip():
                            continue
                        yield {"type": "llm_text", "text": sentence + " ", "index": sentence_index}
                        if voice_output and self.tts.available:
                            def _synthesize(sent):
                                return self.tts.synthesize(sent)
                            try:
                                audio = await loop.run_in_executor(None, _synthesize, sentence)
                                if audio and len(audio) > 100:
                                    audio_count += 1
                                    yield {"type": "audio", "data": audio, "index": sentence_index}
                                else:
                                    fail_count += 1
                            except Exception as e:
                                fail_count += 1
                                logger.error(f"Chat: TTS error sentence {sentence_index}: {e}")
                        sentence_index += 1
            except asyncio.TimeoutError:
                if llm_done.is_set() and token_queue.empty():
                    break
                continue

        llm_thread.join(timeout=1.0)
        llm_s = time.perf_counter() - llm_t0
        tps = llm_token_count / llm_s if llm_s > 0 else 0

        if accumulated_text.strip():
            yield {"type": "llm_text", "text": accumulated_text.strip() + " ", "index": sentence_index}
            if voice_output and self.tts.available:
                try:
                    audio = await loop.run_in_executor(None, self.tts.synthesize, accumulated_text.strip())
                    if audio and len(audio) > 100:
                        audio_count += 1
                        yield {"type": "audio", "data": audio, "index": sentence_index}
                    else:
                        fail_count += 1
                except Exception as e:
                    fail_count += 1
            sentence_index += 1

        total_s = time.perf_counter() - pipeline_t0
        logger.info(
            f"Chat: COMPLETE. Total={total_s:.1f}s "
            f"(LLM={llm_s:.1f}s [{tps:.1f} t/s], sentences={sentence_index}, audio={audio_count})"
        )

        yield {
            "type": "status",
            "message": "Done",
            "done": True,
            "total_time_s": round(total_s, 2),
            "llm_time_s": round(llm_s, 2),
            "tokens_generated": llm_token_count,
            "tokens_per_second": round(tps, 1),
            "sentences": sentence_index,
            "audio_sentences": audio_count,
            "tts_fails": fail_count,
        }

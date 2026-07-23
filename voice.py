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
import time
import wave
from collections import deque
from typing import AsyncIterator, Callable, List, Optional, Tuple

import numpy as np

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


def convert_audio_to_pcm(input_bytes: bytes, input_format: str = "webm", sample_rate: int = 16000) -> bytes:
    """Convert audio (WebM/Opus/etc) to 16kHz mono PCM s16le using ffmpeg."""
    t0 = time.perf_counter()
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-f", input_format,
        "-i", "pipe:0",
        "-ar", str(sample_rate), "-ac", "1",
        "-f", "s16le", "-acodec", "pcm_s16le",
        "pipe:1"
    ]
    try:
        result = subprocess.run(cmd, input=input_bytes, capture_output=True, timeout=30)
        if result.returncode != 0:
            err = result.stderr.decode('utf-8', errors='replace')[:300]
            logger.error(f"ffmpeg conversion failed: {err}")
            return b''
        proc_s = time.perf_counter() - t0
        logger.debug(f"ffmpeg: converted {len(input_bytes)} bytes → {len(result.stdout)} bytes PCM in {proc_s:.2f}s")
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
    """Text-to-Speech using piper-tts."""
    
    def __init__(self, model_path: Optional[str] = None, config_path: Optional[str] = None):
        self.available = False
        self.voice = None
        self.sample_rate = 22050
        self._api_type = "unknown"
        
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
        
        try:
            from piper import PiperVoice
            logger.info(f"TTS: Loading piper voice from {model_path}...")
            t0 = time.perf_counter()
            self.voice = PiperVoice.load(model_path, config_path)
            self.sample_rate = getattr(self.voice.config, 'sample_rate', 22050)
            
            # Log available methods for debugging
            methods = [m for m in dir(self.voice) if not m.startswith('_')]
            logger.info(f"TTS: PiperVoice methods available: {methods}")
            
            # Determine API type
            if hasattr(self.voice, 'synthesize_stream_raw'):
                self._api_type = "synthesize_stream_raw"
            elif hasattr(self.voice, 'synthesize'):
                self._api_type = "synthesize"
            else:
                logger.error(f"TTS: No suitable synthesize method found on PiperVoice")
                return
            
            self.available = True
            load_s = time.perf_counter() - t0
            logger.info(f"TTS: Voice loaded in {load_s:.1f}s (sample_rate={self.sample_rate}, api={self._api_type})")
            
            # Run diagnostic to determine chunk format
            self._diagnostic_synthesize()
            
        except Exception as e:
            logger.error(f"TTS: Failed to load piper voice: {type(e).__name__}: {e}", exc_info=True)
    
    @staticmethod
    def _extract_chunk_bytes(chunk, context=""):
        """Extract raw bytes from a TTS chunk with detailed logging."""
        chunk_type = type(chunk).__name__
        
        if isinstance(chunk, bytes):
            logger.debug(f"TTS: {context} chunk is bytes, len={len(chunk)}")
            return chunk
        if isinstance(chunk, bytearray):
            logger.debug(f"TTS: {context} chunk is bytearray, len={len(chunk)}")
            return bytes(chunk)
        
        # Try common attributes with logging
        for attr in ('data', 'bytes', 'raw', 'buffer', 'pcm', 'audio'):
            if hasattr(chunk, attr):
                val = getattr(chunk, attr)
                val_type = type(val).__name__
                if isinstance(val, (bytes, bytearray)):
                    logger.debug(f"TTS: {context} chunk.{attr} is {val_type}, len={len(val)}")
                    return bytes(val)
                if hasattr(val, 'tobytes'):
                    logger.debug(f"TTS: {context} chunk.{attr} has tobytes()")
                    return val.tobytes()
                if hasattr(val, '__len__'):
                    logger.debug(f"TTS: {context} chunk.{attr} is {val_type}, len={len(val)}")
        
        # Try numpy-like
        if hasattr(chunk, 'tobytes'):
            logger.debug(f"TTS: {context} chunk has tobytes()")
            return chunk.tobytes()
        if hasattr(chunk, 'numpy'):
            logger.debug(f"TTS: {context} chunk has numpy()")
            return chunk.numpy().tobytes()
        
        # Try buffer protocol
        try:
            result = bytes(memoryview(chunk))
            logger.debug(f"TTS: {context} chunk via memoryview, len={len(result)}")
            return result
        except Exception as e:
            logger.debug(f"TTS: {context} memoryview failed: {e}")
        
        # Try np.array
        try:
            import numpy as np
            arr = np.array(chunk)
            result = arr.tobytes()
            logger.debug(f"TTS: {context} chunk via np.array, dtype={arr.dtype}, shape={arr.shape}, len={len(result)}")
            return result
        except Exception as e:
            logger.debug(f"TTS: {context} np.array failed: {e}")
        
        # Log what we got
        attrs = [a for a in dir(chunk) if not a.startswith('_') and not callable(getattr(chunk, a, None))]
        logger.warning(f"TTS: Cannot extract bytes from chunk type={chunk_type} (context={context}), attrs={attrs[:20]}")
        return None

    def _diagnostic_synthesize(self):
        """Run a quick test synthesis to determine chunk format with aggressive logging."""
        try:
            logger.info("TTS: === DIAGNOSTIC SYNTHESIS START ===")
            
            # First, try file-based approach
            try:
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tf:
                    tmp_path = tf.name
                logger.info(f"TTS: DIAGNOSTIC trying file-based synthesis to {tmp_path}")
                t0 = time.perf_counter()
                self.voice.synthesize("hello", open(tmp_path, 'wb'))
                file_s = time.perf_counter() - t0
                file_size = os.path.getsize(tmp_path)
                logger.info(f"TTS: DIAGNOSTIC file-based SUCCESS — {file_size} bytes in {file_s:.2f}s")
                with open(tmp_path, 'rb') as f:
                    data = f.read()
                os.unlink(tmp_path)
                logger.info(f"TTS: DIAGNOSTIC file audio bytes={len(data)}, header_valid={data[:4]==b'RIFF'}")
                return  # File approach works, we're done
            except Exception as e:
                logger.warning(f"TTS: DIAGNOSTIC file-based failed: {type(e).__name__}: {e}")
            
            # Fall back to chunk inspection
            logger.info("TTS: DIAGNOSTIC trying chunk iteration...")
            chunks = list(self.voice.synthesize("hi"))
            logger.info(f"TTS: DIAGNOSTIC got {len(chunks)} chunks")
            
            if not chunks:
                logger.warning("TTS: DIAGNOSTIC synthesis returned no chunks")
                return
            
            chunk = chunks[0]
            chunk_type = type(chunk).__name__
            module = type(chunk).__module__
            logger.info(f"TTS: DIAGNOSTIC chunk type={chunk_type}, module={module}")
            
            # Log all public attributes and their types
            attrs = []
            for attr in dir(chunk):
                if attr.startswith('_'):
                    continue
                try:
                    val = getattr(chunk, attr)
                    if callable(val):
                        continue
                    val_type = type(val).__name__
                    val_info = str(val)[:60]
                    attrs.append(f"{attr}={val_type}:{val_info}")
                except Exception:
                    attrs.append(f"{attr}=ERROR")
            logger.info(f"TTS: DIAGNOSTIC chunk attrs: {attrs}")
            
            # Try __dict__ if available
            if hasattr(chunk, '__dict__'):
                logger.info(f"TTS: DIAGNOSTIC chunk __dict__={chunk.__dict__}")
            
            # Try to extract bytes with detailed logging
            extracted = self._extract_chunk_bytes(chunk)
            if extracted is not None:
                logger.info(f"TTS: DIAGNOSTIC extracted {len(extracted)} bytes from {chunk_type}")
            else:
                logger.error(f"TTS: DIAGNOSTIC cannot extract bytes from {chunk_type}")
                
            logger.info("TTS: === DIAGNOSTIC SYNTHESIS END ===")
        except Exception as e:
            logger.error(f"TTS: DIAGNOSTIC synthesis failed: {type(e).__name__}: {e}", exc_info=True)

    def synthesize(self, text: str) -> Optional[bytes]:
        """Synthesize text to WAV bytes."""
        if not self.available or not self.voice:
            logger.warning("TTS: Voice not available, skipping synthesis")
            return None
        
        if not text or not text.strip():
            return None
        
        t0 = time.perf_counter()
        
        # Strategy 1: File-based synthesis (most reliable)
        try:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tf:
                tmp_path = tf.name
            
            logger.debug(f"TTS: Trying file-based synthesis to {tmp_path}")
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
                f"{len(audio_data)} bytes WAV via file ({proc_s:.2f}s, "
                f"header_valid={header_valid})"
            )
            return audio_data
            
        except Exception as e:
            logger.warning(f"TTS: File-based synthesis failed: {type(e).__name__}: {e}")
        
        # Strategy 2: Chunk-based iteration with byte extraction
        try:
            logger.debug(f"TTS: Falling back to chunk-based synthesis for '{text[:60]}...'")
            audio_data = b''
            chunk_count = 0
            
            if self._api_type == "synthesize_stream_raw":
                logger.debug("TTS: Using synthesize_stream_raw")
                for chunk in self.voice.synthesize_stream_raw(text):
                    raw = self._extract_chunk_bytes(chunk, "stream_raw")
                    if raw:
                        audio_data += raw
                        chunk_count += 1
                        
            elif self._api_type == "synthesize":
                logger.debug("TTS: Using synthesize iterator")
                for chunk in self.voice.synthesize(text):
                    raw = self._extract_chunk_bytes(chunk, "synthesize")
                    if raw:
                        audio_data += raw
                        chunk_count += 1
            else:
                logger.error(f"TTS: Unknown API type '{self._api_type}'")
                return None
            
            if not audio_data:
                logger.error(f"TTS: No audio data extracted for '{text[:60]}...'")
                return None
            
            # Wrap raw PCM in WAV container
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
                f"chunks={chunk_count}, raw_audio={len(audio_data)} bytes)"
            )
            return audio
            
        except Exception as e:
            logger.error(
                f"TTS: All synthesis strategies failed for '{text[:60]}...': "
                f"{type(e).__name__}: {e}",
                exc_info=True
            )
            return None


# ── Sentence splitting ──────────────────────────────────────

_SENTENCE_RE = re.compile(r'(?<=[.!?])\s+|\n+')

def split_sentences(text: str) -> List[str]:
    """Split text into sentences for incremental TTS."""
    sentences = _SENTENCE_RE.split(text)
    return [s.strip() for s in sentences if s.strip()]


# ── Audio utilities ─────────────────────────────────────────

def combine_wavs(wav_bytes_list: List[bytes]) -> bytes:
    """Combine multiple WAV files into a single WAV file."""
    if not wav_bytes_list:
        return b''
    
    if len(wav_bytes_list) == 1:
        return wav_bytes_list[0]
    
    # Extract raw PCM from each WAV and combine
    combined_pcm = b''
    sample_rate = 22050
    sample_width = 2
    channels = 1
    
    for wav_bytes in wav_bytes_list:
        try:
            with io.BytesIO(wav_bytes) as wf_io:
                with wave.open(wf_io, 'rb') as wf:
                    sample_rate = wf.getframerate()
                    sample_width = wf.getsampwidth()
                    channels = wf.getnchannels()
                    combined_pcm += wf.readframes(wf.getnframes())
        except Exception as e:
            logger.warning(f"TTS: Error extracting PCM from WAV: {e}")
    
    if not combined_pcm:
        return b''
    
    # Write combined WAV
    out = io.BytesIO()
    with wave.open(out, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(combined_pcm)
    
    return out.getvalue()


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
        
        Yields dicts with keys:
            type: "transcription" | "llm_text" | "audio" | "status" | "error"
            Additional fields depend on type.
        """
        pipeline_t0 = time.perf_counter()
        logger.info(
            f"Pipeline: Starting processing of {len(pcm_bytes)} bytes "
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
        
        # 2. LLM (streaming generation)
        yield {"type": "status", "message": "Generating response..."}
        
        llm_t0 = time.perf_counter()
        
        # Run LLM generation in executor (since it's CPU-bound and blocking)
        loop = asyncio.get_event_loop()
        
        def _generate():
            return list(self.llm_generate(text))
        
        try:
            tokens = await loop.run_in_executor(None, _generate)
        except Exception as e:
            logger.error(f"Pipeline: LLM generation failed: {type(e).__name__}: {e}", exc_info=True)
            yield {"type": "error", "message": f"LLM generation failed: {e}"}
            return
        
        llm_s = time.perf_counter() - llm_t0
        tps = len(tokens) / llm_s if llm_s > 0 else 0
        full_text = "".join(tokens)
        logger.info(
            f"Pipeline: LLM generated {len(tokens)} tokens in {llm_s:.2f}s "
            f"({tps:.1f} t/s), text='{full_text[:120]}...'"
        )
        
        # 3. TTS (sentence-by-sentence, collect all audio)
        yield {"type": "status", "message": "Synthesizing speech..."}
        
        sentences = split_sentences(full_text)
        tts_t0 = time.perf_counter()
        
        logger.info(f"Pipeline: TTS processing {len(sentences)} sentences")
        
        sentence_audio_list = []  # Collect all sentence WAVs
        audio_count = 0
        fail_count = 0
        
        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue
            
            # Stream text to client for display
            yield {"type": "llm_text", "text": sentence + " ", "index": i}
            
            if self.tts.available:
                logger.debug(f"Pipeline: TTS sentence {i}: '{sentence[:60]}...'")
                audio = self.tts.synthesize(sentence)
                if audio and len(audio) > 100:  # Valid audio > 100 bytes
                    sentence_audio_list.append(audio)
                    audio_count += 1
                else:
                    fail_count += 1
                    if audio:
                        logger.warning(f"Pipeline: TTS sentence {i} produced only {len(audio)} bytes")
                    else:
                        logger.warning(f"Pipeline: TTS failed for sentence {i}")
            else:
                logger.debug(f"Pipeline: TTS unavailable, skipping sentence {i}")
        
        tts_s = time.perf_counter() - tts_t0
        
        # Combine all sentence audio into one WAV
        combined_audio = None
        if sentence_audio_list:
            combined_audio = combine_wavs(sentence_audio_list)
            logger.info(
                f"Pipeline: Combined {len(sentence_audio_list)} sentence WAVs → "
                f"{len(combined_audio)} bytes total audio"
            )
            if combined_audio:
                yield {"type": "audio_complete", "data": combined_audio}
        
        total_s = time.perf_counter() - pipeline_t0
        
        logger.info(
            f"Pipeline: COMPLETE. Total={total_s:.1f}s "
            f"(STT={stt_s:.1f}s, LLM={llm_s:.1f}s [{tps:.1f} t/s], TTS={tts_s:.1f}s, "
            f"sentences={len(sentences)}, audio_sentences={audio_count}, tts_fails={fail_count}, "
            f"combined_audio={len(combined_audio) if combined_audio else 0} bytes)"
        )
        
        yield {
            "type": "status", 
            "message": "Done", 
            "done": True,
            "total_time_s": round(total_s, 2),
            "stt_time_s": round(stt_s, 2),
            "llm_time_s": round(llm_s, 2),
            "tts_time_s": round(tts_s, 2),
            "tokens_generated": len(tokens),
            "tokens_per_second": round(tps, 1),
            "sentences": len(sentences),
            "audio_sentences": audio_count,
            "tts_fails": fail_count,
            "combined_audio_bytes": len(combined_audio) if combined_audio else 0,
        }

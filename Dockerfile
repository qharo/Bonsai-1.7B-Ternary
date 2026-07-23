FROM python:3.11-slim
WORKDIR /app

# Install system dependencies (build tools + ffmpeg for audio conversion)
RUN apt-get update && apt-get install -y \
    build-essential \
    clang \
    libomp-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy C source and build the shared library
COPY ml/ ./ml/
COPY Makefile .
RUN make inference.so

# Copy Python app, UI, and dependencies
COPY app.py voice.py voice.html index.html requirements.txt download_models.py ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download voice models to avoid startup timeout on HuggingFace Spaces
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV PIPER_MODEL_PATH=/app/piper_voice/en/en_US/lessac/medium/en_US-lessac-medium.onnx
ENV PIPER_CONFIG_PATH=/app/piper_voice/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json

RUN python download_models.py || echo "[WARN] Model pre-download failed, will retry at runtime"

# Copy the flattened model weights
COPY model_weights_repacked/ ./model_weights_repacked/

EXPOSE 7860

# Environment variables
ENV MODEL_DIR=/app/model_weights_repacked
ENV OMP_SCHEDULE=static
ENV OMP_WAIT_POLICY=passive
ENV OMP_PROC_BIND=close

CMD ["python", "app.py"]

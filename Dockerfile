FROM python:3.11-slim

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    clang \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy source files
COPY model_infer.c model_infer.h matmul_common.c matmul_common.h Makefile ./
COPY app.py requirements.txt ./
COPY static/ ./static/

# Build inference library — Makefile auto-detects platform & SIMD level
RUN make inference.so

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Model weights: copied from repo via Git LFS (HF Spaces) or overridden at runtime
# Local dev: docker run -v $(pwd)/models:/app/models cvp-app
COPY models/ ./models/

EXPOSE 7860

ENV MODEL_DIR=/app/models/Ternary-Bonsai-1.7B-unpacked
ENV OMP_SCHEDULE=static
ENV OMP_WAIT_POLICY=passive
ENV OMP_PROC_BIND=close

CMD ["python", "app.py"]

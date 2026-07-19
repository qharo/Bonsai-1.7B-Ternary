FROM python:3.11-slim
WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    clang \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy C source and build the shared library
COPY ml/ ./ml/
COPY Makefile .
RUN make inference.so

# Copy Python app, UI, and dependencies
COPY app.py index.html requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the flattened model weights
COPY model_weights_repacked/ ./model_weights_repacked/

EXPOSE 7860

# Environment variables
ENV MODEL_DIR=/app/model_weights_repacked
ENV OMP_SCHEDULE=static
ENV OMP_WAIT_POLICY=passive
ENV OMP_PROC_BIND=close

CMD ["python", "app.py"]

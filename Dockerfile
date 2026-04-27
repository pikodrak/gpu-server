FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Auto-download TinyLlama 1.1B Q4_K_M on first startup if no model is configured.
# Override GPU_SERVER_LLM_MODEL_PATH in .env to use a different model instead.
ENV GPU_SERVER_LLM_AUTO_DOWNLOAD_URL="https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-dev \
    python3-pip \
    build-essential \
    cmake \
    ninja-build \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch with CUDA 12.1
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install llama-cpp-python with CUDA — prefer prebuilt CUDA 12.1 wheel.
# LIBRARY_PATH points pip's source-build fallback at the libcuda.so stub,
# since libcuda is provided by the host driver at runtime, not in the toolkit image.
RUN LIBRARY_PATH=/usr/local/cuda/lib64/stubs \
    CMAKE_ARGS="-DGGML_CUDA=on" \
    pip install llama-cpp-python \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121

# Install Stable Diffusion stack
RUN pip install diffusers transformers accelerate Pillow

# Install server requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application
COPY server.py config.py auth.py gpu_backend.py ./

# Create directories and non-root user; grant ownership before volume mount
RUN mkdir -p /app/models /app/outputs && \
    useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

VOLUME ["/app/models", "/app/outputs"]

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["python3", "server.py"]

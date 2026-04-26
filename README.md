# GPU Server

HTTP REST server for GPU-accelerated ML inference on an RTX 4070 Ti (or any CUDA GPU).
Used by Paperclip agents to run LLM inference and generate game assets (textures, concept art) locally.

## Quick Start

### 1. Install dependencies

```bash
# Create virtualenv
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate   # Windows

# PyTorch with CUDA 12.1 (RTX 4070 Ti)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Core server
pip install fastapi "uvicorn[standard]" pydantic pydantic-settings pyyaml python-dotenv

# LLM inference (optional but recommended)
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python

# Stable Diffusion / image generation (optional)
pip install diffusers transformers accelerate Pillow
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env and set at minimum:
#   GPU_SERVER_API_KEYS=["your-secret-key"]
#   GPU_SERVER_LLM_MODEL_PATH=/path/to/model.gguf   (optional)
#   GPU_SERVER_SD_ENABLE=true                         (optional)
```

Or use `config.yaml` (copy `config.yaml.example` → `config.yaml`).

### 3. Download a model (for LLM inference)

```bash
# Example: Llama 3 8B Instruct (Q4_K_M quantization, ~4.7 GB)
mkdir -p models
wget -O models/llama-3-8b-instruct.Q4_K_M.gguf \
  https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf

# Then set in .env:
# GPU_SERVER_LLM_MODEL_PATH=./models/llama-3-8b-instruct.Q4_K_M.gguf
```

### 4. Run

```bash
python server.py
```

Server starts on `http://0.0.0.0:8000` by default.

---

## API Reference

All endpoints (except `/health`) require the header:
```
X-API-Key: your-secret-key
```

### `GET /health`
Public status check. Returns GPU info and server status.

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "gpu": {
    "name": "NVIDIA GeForce RTX 4070 Ti",
    "type": "cuda",
    "vram_total_gb": 12.0,
    "vram_free_gb": 11.2
  },
  "timestamp": 1714123456.789
}
```

### `GET /api/v1/gpu/info`
Detailed GPU and model status.

```bash
curl -H "X-API-Key: your-key" http://localhost:8000/api/v1/gpu/info
```

### `POST /api/v1/inference`
Run LLM text inference.

```bash
curl -X POST http://localhost:8000/api/v1/inference \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Write a short game design document for a roguelike dungeon crawler.",
    "model": "llama",
    "max_tokens": 512,
    "temperature": 0.7,
    "system_prompt": "You are a senior game designer."
  }'
```

**Response:**
```json
{
  "text": "...",
  "model": "llama",
  "tokens_generated": 312,
  "duration_ms": 4250.5,
  "device": "NVIDIA GeForce RTX 4070 Ti"
}
```

### `POST /api/v1/image/generate`
Generate images using Stable Diffusion (requires `GPU_SERVER_SD_ENABLE=true`).

```bash
curl -X POST http://localhost:8000/api/v1/image/generate \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "seamless stone dungeon texture, tileable, dark fantasy, high quality",
    "negative_prompt": "blurry, low quality, watermark",
    "width": 512,
    "height": 512,
    "steps": 30,
    "guidance_scale": 7.5,
    "output_format": "base64"
  }'
```

Returns a base64-encoded PNG (or a filepath if `output_format: "filepath"`).

> **Note:** The NSFW safety checker is disabled by default (`safety_checker=None`). This is intentional for a local developer tool — operators running this server in shared or production environments should be aware of this and take appropriate precautions.

### `POST /api/v1/torch`
Run a PyTorch operation on GPU. Useful for custom ML workloads and embeddings.

Allowed operations: `matmul`, `dot`, `norm`, `mean`, `std`, `softmax`, `sigmoid`, `relu`, `cosine_similarity`

```bash
curl -X POST http://localhost:8000/api/v1/torch \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{
    "operation": "softmax",
    "data": [1.0, 2.0, 3.0, 4.0]
  }'
```

---

## Docker

```bash
# Build and run with GPU passthrough
docker compose up --build

# Or manually
docker build -t gpu-server .
docker run --gpus all -p 8000:8000 \
  -v $(pwd)/models:/models \
  -v $(pwd)/.env:/app/.env:ro \
  gpu-server
```

Requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

---

## Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `GPU_SERVER_HOST` | `0.0.0.0` | Bind address |
| `GPU_SERVER_PORT` | `8000` | Port |
| `GPU_SERVER_API_KEYS` | `[]` | JSON array of valid API keys (required) |
| `GPU_SERVER_DEVICE` | `cuda` | `cuda`, `cpu`, or `mps` |
| `GPU_SERVER_LLM_MODEL_PATH` | `` | Path to GGUF model file |
| `GPU_SERVER_SD_ENABLE` | `false` | Enable Stable Diffusion |
| `GPU_SERVER_SD_MODEL_ID` | `runwayml/stable-diffusion-v1-5` | HuggingFace model ID |
| `GPU_SERVER_OUTPUT_DIR` | `./outputs` | Directory for saved images |

---

## Calling from a Paperclip Agent (Python)

```python
import httpx

GPU_SERVER_URL = "http://192.168.1.100:8000"
API_KEY = "your-secret-key"

headers = {"X-API-Key": API_KEY}

# LLM inference
resp = httpx.post(
    f"{GPU_SERVER_URL}/api/v1/inference",
    headers=headers,
    json={
        "prompt": "Generate 5 enemy names for a fantasy RPG",
        "max_tokens": 256,
        "temperature": 0.9,
    },
    timeout=120,
)
print(resp.json()["text"])

# Image generation
resp = httpx.post(
    f"{GPU_SERVER_URL}/api/v1/image/generate",
    headers=headers,
    json={
        "prompt": "game texture, mossy stone wall, seamless, 4k",
        "width": 512,
        "height": 512,
        "steps": 25,
    },
    timeout=300,
)
import base64, pathlib
img_bytes = base64.b64decode(resp.json()["image_data"])
pathlib.Path("texture.png").write_bytes(img_bytes)
```

---

## Windows Notes

Install PyTorch for CUDA 12.1 the same way. For llama-cpp-python on Windows:
```powershell
$env:CMAKE_ARGS="-DGGML_CUDA=on"
pip install llama-cpp-python
```

Requires Visual Studio Build Tools and CUDA Toolkit 12.x installed.

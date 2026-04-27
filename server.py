"""
GPU Server - FastAPI server for RTX 4070 Ti GPU acceleration.
Accepts REST requests from Paperclip agents and other clients.
"""
from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from auth import verify_api_key
from config import settings
from gpu_backend import GPUBackend

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

gpu_backend: GPUBackend | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global gpu_backend
    logger.info("Initializing GPU backend...")
    gpu_backend = GPUBackend()
    gpu_backend.initialize()
    logger.info(f"GPU backend ready. Device: {gpu_backend.device_info()['name']}")
    yield
    logger.info("Shutting down GPU backend...")
    if gpu_backend:
        gpu_backend.cleanup()


app = FastAPI(
    title="GPU Server",
    description="RTX 4070 Ti GPU acceleration server for Paperclip agents",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request/Response models ──────────────────────────────────────────────────

class InferenceRequest(BaseModel):
    prompt: str = Field(..., description="Input prompt for the model")
    model: str = Field(default="llama", description="Model to use: llama, phi, mistral")
    max_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    system_prompt: str | None = Field(default=None)


class InferenceResponse(BaseModel):
    text: str
    model: str
    tokens_generated: int
    duration_ms: float
    device: str


class ImageRequest(BaseModel):
    prompt: str = Field(..., description="Image generation prompt")
    negative_prompt: str = Field(default="", description="Negative prompt")
    width: int = Field(default=512, ge=64, le=1024)
    height: int = Field(default=512, ge=64, le=1024)
    steps: int = Field(default=20, ge=1, le=100)
    guidance_scale: float = Field(default=7.5, ge=1.0, le=20.0)
    seed: int | None = Field(default=None)
    output_format: str = Field(default="base64", description="base64 or filepath")


class ImageResponse(BaseModel):
    image_data: str
    format: str
    width: int
    height: int
    duration_ms: float
    seed: int


class TorchRequest(BaseModel):
    operation: str = Field(..., description="PyTorch operation name")
    data: Any = Field(..., description="Input data (JSON-serializable)")
    kwargs: dict[str, Any] = Field(default_factory=dict)


# ── OpenAI-compatible request/response models ────────────────────────────────

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = Field(default="llama")
    messages: list[ChatMessage]
    max_tokens: int | None = Field(default=None, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    stream: bool = Field(default=False)


# ── OpenAI Responses API models (used by opencode and newer OpenAI SDK clients) ─

class ResponsesInputContentPart(BaseModel):
    type: str
    text: str = ""


class ResponsesInputItem(BaseModel):
    role: str
    content: str | list[ResponsesInputContentPart] | list[dict]


class ResponsesRequest(BaseModel):
    model: str = Field(default="llama")
    input: list[ResponsesInputItem]
    max_output_tokens: int | None = Field(default=None, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    stream: bool = Field(default=False)


def _extract_text(content: str | list) -> str:
    if isinstance(content, str):
        return content
    return " ".join(
        p["text"] if isinstance(p, dict) else p.text
        for p in content
        if (isinstance(p, dict) and p.get("type") in ("input_text", "text", "output_text"))
        or (hasattr(p, "type") and p.type in ("input_text", "text", "output_text"))
    )


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Status endpoint - no auth required."""
    info = gpu_backend.device_info() if gpu_backend else {}
    return {
        "status": "ok" if gpu_backend else "degraded",
        "gpu": info,
        "timestamp": time.time(),
    }


@app.get("/api/v1/gpu/info", dependencies=[Depends(verify_api_key)])
async def gpu_info():
    """Detailed GPU information including VRAM usage."""
    if not gpu_backend:
        raise HTTPException(status_code=503, detail="GPU backend not initialized")
    return gpu_backend.detailed_info()


@app.post("/api/v1/inference", response_model=InferenceResponse, dependencies=[Depends(verify_api_key)])
async def run_inference(req: InferenceRequest):
    """
    Run LLM text inference on the GPU.
    Supports llama.cpp models loaded via llama-cpp-python.
    """
    if not gpu_backend:
        raise HTTPException(status_code=503, detail="GPU backend not initialized")

    t0 = time.monotonic()
    try:
        result = gpu_backend.run_inference(
            prompt=req.prompt,
            model_name=req.model,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            system_prompt=req.system_prompt,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail="Inference failed")

    duration_ms = (time.monotonic() - t0) * 1000
    return InferenceResponse(
        text=result["text"],
        model=req.model,
        tokens_generated=result.get("tokens_generated", 0),
        duration_ms=round(duration_ms, 2),
        device=gpu_backend.device_info()["name"],
    )


@app.post("/api/v1/image/generate", response_model=ImageResponse, dependencies=[Depends(verify_api_key)])
async def generate_image(req: ImageRequest):
    """
    Generate images using Stable Diffusion on GPU.
    Useful for game asset generation (textures, concept art).
    """
    if not gpu_backend:
        raise HTTPException(status_code=503, detail="GPU backend not initialized")

    t0 = time.monotonic()
    try:
        result = gpu_backend.generate_image(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            width=req.width,
            height=req.height,
            steps=req.steps,
            guidance_scale=req.guidance_scale,
            seed=req.seed,
            output_format=req.output_format,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        logger.error(f"Image generation error: {e}")
        raise HTTPException(status_code=500, detail="Image generation failed")

    duration_ms = (time.monotonic() - t0) * 1000
    return ImageResponse(
        image_data=result["image_data"],
        format=req.output_format,
        width=req.width,
        height=req.height,
        duration_ms=round(duration_ms, 2),
        seed=result["seed"],
    )


@app.post("/api/v1/torch", dependencies=[Depends(verify_api_key)])
async def run_torch_operation(req: TorchRequest):
    """
    Run arbitrary PyTorch operations on GPU.
    Useful for custom ML workloads and game AI.
    """
    if not gpu_backend:
        raise HTTPException(status_code=503, detail="GPU backend not initialized")

    try:
        result = gpu_backend.run_torch_operation(req.operation, req.data, **req.kwargs)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        logger.error(f"Torch operation error: {e}")
        raise HTTPException(status_code=500, detail="Operation failed")

    return {"result": result, "device": gpu_backend.device_info()["name"]}


@app.get("/v1/models", dependencies=[Depends(verify_api_key)])
async def list_models():
    """OpenAI-compatible model listing."""
    model_id = "llama"
    if gpu_backend and gpu_backend._llm is not None:
        from pathlib import Path
        raw = settings.llm_model_path
        model_id = Path(raw).stem if raw else "llama"
    return {
        "object": "list",
        "data": [{"id": model_id, "object": "model", "created": 0, "owned_by": "local"}],
    }


@app.post("/v1/chat/completions", dependencies=[Depends(verify_api_key)])
async def chat_completions(req: ChatCompletionRequest):
    """
    OpenAI-compatible chat completions endpoint.
    Wraps the llama-cpp-python backend so any OpenAI client can use this server.
    Streaming is not yet supported; stream=true is accepted but ignored.
    """
    if not gpu_backend:
        raise HTTPException(status_code=503, detail="GPU backend not initialized")

    system_prompt: str | None = None
    user_parts: list[str] = []
    for msg in req.messages:
        if msg.role == "system":
            system_prompt = msg.content
        elif msg.role == "user":
            user_parts.append(msg.content)
        elif msg.role == "assistant":
            user_parts.append(f"Assistant: {msg.content}")

    prompt = "\n\n".join(user_parts) if user_parts else ""
    max_tokens = req.max_tokens if req.max_tokens is not None else 512

    t0 = time.monotonic()
    try:
        result = gpu_backend.run_inference(
            prompt=prompt,
            model_name=req.model,
            max_tokens=max_tokens,
            temperature=req.temperature,
            system_prompt=system_prompt,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail="Inference failed")

    tokens_out = result.get("tokens_generated", 0)
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result["text"]},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": tokens_out,
            "total_tokens": tokens_out,
        },
    }


@app.post("/v1/responses", dependencies=[Depends(verify_api_key)])
async def responses(req: ResponsesRequest):
    """
    OpenAI Responses API endpoint (used by opencode and newer OpenAI SDK clients).
    Translates the Responses API input format to chat messages and delegates to the
    llama-cpp backend, then wraps the result in the Responses API output format.
    """
    if not gpu_backend:
        raise HTTPException(status_code=503, detail="GPU backend not initialized")

    system_prompt: str | None = None
    user_parts: list[str] = []
    for item in req.input:
        text = _extract_text(item.content)
        if item.role == "system" or item.role == "developer":
            system_prompt = text
        elif item.role == "user":
            if text:
                user_parts.append(text)
        elif item.role == "assistant":
            if text:
                user_parts.append(f"Assistant: {text}")

    prompt = "\n\n".join(user_parts) if user_parts else ""
    max_tokens = req.max_output_tokens if req.max_output_tokens is not None else 512

    try:
        result = gpu_backend.run_inference(
            prompt=prompt,
            model_name=req.model,
            max_tokens=max_tokens,
            temperature=req.temperature,
            system_prompt=system_prompt,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail="Inference failed")

    tokens_out = result.get("tokens_generated", 0)
    return {
        "id": f"resp-{uuid.uuid4().hex[:12]}",
        "object": "response",
        "created_at": int(time.time()),
        "model": req.model,
        "status": "completed",
        "output": [
            {
                "type": "message",
                "id": f"msg-{uuid.uuid4().hex[:12]}",
                "role": "assistant",
                "content": [{"type": "output_text", "text": result["text"], "annotations": []}],
                "status": "completed",
            }
        ],
        "usage": {
            "input_tokens": 0,
            "output_tokens": tokens_out,
            "total_tokens": tokens_out,
        },
    }


@app.exception_handler(Exception)
async def generic_error_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error on {request.url}: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"},
    )


if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info",
    )

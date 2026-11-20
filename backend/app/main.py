import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from threading import Lock

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from myllm_model import (
    GenerationConfig,
    generate_text,
    load_checkpoint,
)
from pydantic import BaseModel, Field

from app.settings import get_checkpoint_dir, get_cors_origins

logger = logging.getLogger(__name__)


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1)
    max_new_tokens: int = Field(default=100, ge=0, le=256, strict=True)
    temperature: float = Field(
        default=1.0,
        ge=0,
        allow_inf_nan=False,
    )


class GenerateResponse(BaseModel):
    generated_text: str


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.loaded_checkpoint = None
    app.state.generation_lock = Lock()

    checkpoint_dir = get_checkpoint_dir()
    if checkpoint_dir is None:
        logger.error(
            "MYLLM_CHECKPOINT_DIR is not configured; generation is unavailable"
        )
    else:
        try:
            app.state.loaded_checkpoint = load_checkpoint(checkpoint_dir)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            logger.error(
                "MyLLM checkpoint could not be loaded (%s)",
                type(error).__name__,
            )

    try:
        yield
    finally:
        app.state.loaded_checkpoint = None


app = FastAPI(
    title="MyLLM API",
    description="Backend API for MyLLM text generation.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health", tags=["health"])
def health_check() -> dict[str, str]:
    return {"status": "ok", "service": "myllm-backend"}


@app.post(
    "/generate",
    response_model=GenerateResponse,
    tags=["generation"],
)
def generate(request: GenerateRequest, http_request: Request) -> GenerateResponse:
    checkpoint = http_request.app.state.loaded_checkpoint
    generation_lock = http_request.app.state.generation_lock
    if checkpoint is None:
        raise HTTPException(
            status_code=503,
            detail="Generation model is unavailable",
        )

    config = GenerationConfig(
        max_new_tokens=request.max_new_tokens,
        temperature=request.temperature,
    )
    try:
        with generation_lock:
            generated_text = generate_text(
                checkpoint.model,
                checkpoint.tokenizer,
                request.prompt,
                config,
            )
    except (RuntimeError, TypeError, ValueError) as error:
        logger.exception("MyLLM text generation failed")
        raise HTTPException(
            status_code=500,
            detail="Text generation failed",
        ) from error

    return GenerateResponse(generated_text=generated_text)

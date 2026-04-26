import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, field_validator

from .settings import settings
from .translator import SUPPORTED_PAIRS, TranslatorRegistry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

registry = TranslatorRegistry(chunk_size=settings.chunk_size)
_inference_sem = asyncio.Semaphore(1)
_UI_HTML = (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("Service starting — models load lazily on first request per pair")
    yield


app = FastAPI(
    title="Local Translation Service",
    version="2.0.0",
    lifespan=lifespan,
)

_origins = (
    ["*"]
    if settings.allowed_origins.strip() == "*"
    else [o.strip() for o in settings.allowed_origins.split(",")]
)
# Starlette raises ValueError if allow_credentials=True is combined with
# allow_origins=["*"] (per CORS spec, browsers reject wildcard + credentials).
_credentials = _origins != ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ──────────────────────────────────────────────────


class TranslateRequest(BaseModel):
    text: str
    source_lang: str
    target_lang: str

    @field_validator("text")
    @classmethod
    def text_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be empty")
        return v


class TranslateResponse(BaseModel):
    translated_text: str
    model: str
    source_lang: str
    target_lang: str
    char_count: int


# ── Endpoints ──────────────────────────────────────────────────────────────────


@app.get("/ui", response_class=HTMLResponse, include_in_schema=False)
async def ui() -> str:
    return _UI_HTML


@app.get("/")
async def root() -> dict:
    return {
        "service": "Local Translation Service",
        "version": "2.0.0",
        "ui": "/ui",
        "supported_pairs": [f"{s}→{t}" for s, t in sorted(SUPPORTED_PAIRS)],
        "endpoints": [
            "GET  /ui",
            "GET  /",
            "GET  /health",
            "GET  /model-info",
            "POST /translate",
        ],
    }


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/model-info")
async def model_info() -> dict:
    return {
        "supported_pairs": [f"{s}→{t}" for s, t in sorted(SUPPORTED_PAIRS)],
        "loaded_models": registry.loaded_models,
    }


@app.post("/translate", response_model=TranslateResponse)
async def translate(request: TranslateRequest) -> TranslateResponse:
    if (request.source_lang, request.target_lang) not in SUPPORTED_PAIRS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unsupported language pair: {request.source_lang}→{request.target_lang}. "
                f"Supported: {', '.join(f'{s}→{t}' for s, t in sorted(SUPPORTED_PAIRS))}"
            ),
        )

    char_count = len(request.text)

    if char_count > settings.max_chars_per_request:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Text length {char_count} exceeds MAX_CHARS_PER_REQUEST "
                f"({settings.max_chars_per_request})."
            ),
        )

    logger.info(
        "Translate request | source=%s target=%s chars=%d",
        request.source_lang,
        request.target_lang,
        char_count,
    )
    if settings.log_text:
        logger.debug("Input text: %s", request.text)

    async with _inference_sem:
        translated_text, model_id = await asyncio.to_thread(
            registry.translate, request.source_lang, request.target_lang, request.text
        )

    return TranslateResponse(
        translated_text=translated_text,
        model=model_id,
        source_lang=request.source_lang,
        target_lang=request.target_lang,
        char_count=char_count,
    )


@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
    logger.error("Unhandled error: %s", exc, exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})

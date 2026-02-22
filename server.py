"""
D-MMCE Web Server
==================
FastAPI backend that serves the UI and exposes REST + SSE endpoints
for the D-MMCE pipeline.

Run with:
    python server.py
    # or: uvicorn server:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from d_mmce.observer import Event
from d_mmce.orchestrator import D_MMCE
from d_mmce.providers import ProviderFactory
from d_mmce.providers.ollama_provider import OllamaProvider

# ------------------------------------------------------------------ #
#  Logging
# ------------------------------------------------------------------ #
LOG_FILE = Path(__file__).parent / "d_mmce.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
logger = logging.getLogger("d_mmce.server")
logger.info("=" * 60)
logger.info("D-MMCE server starting — log file: %s", LOG_FILE)

# ------------------------------------------------------------------ #
#  FastAPI app
# ------------------------------------------------------------------ #
app = FastAPI(title="D-MMCE", version="1.0.0")
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

# ------------------------------------------------------------------ #
#  Pydantic request / response models
# ------------------------------------------------------------------ #

class QueryRequest(BaseModel):
    query: str
    providers: list[str] | None = None
    ollama_models: list[str] | None = None  # e.g. ["mistral", "codellama:13b"]
    review_provider: str = "auto"
    embedding_model: str = "all-MiniLM-L6-v2"
    stability_threshold: float = 0.85
    max_reruns: int = 3


class SettingsUpdate(BaseModel):
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    google_api_key: str | None = None
    ollama_base_url: str | None = None


class ProviderStatus(BaseModel):
    name: str
    available: bool


# ------------------------------------------------------------------ #
#  Serve the SPA
# ------------------------------------------------------------------ #

@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).parent / "static" / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


# ------------------------------------------------------------------ #
#  API endpoints
# ------------------------------------------------------------------ #

@app.get("/api/providers")
async def list_providers():
    """Return registered cloud providers with availability status.

    The base ``ollama`` entry is excluded because local models are
    managed via ``/api/ollama/models`` and the Local-LLM picker in the UI.
    """
    names = ProviderFactory.available_names()
    statuses = []
    for name in names:
        if name == "ollama":
            continue  # handled by /api/ollama/models
        try:
            p = ProviderFactory.create(name)
            avail = await p.is_available()
        except Exception:
            avail = False
        statuses.append({"name": name, "available": avail})
    return {"providers": statuses}


@app.get("/api/ollama/status")
async def ollama_status():
    """Check if the Ollama server is reachable."""
    reachable = await OllamaProvider.server_reachable()
    return {"reachable": reachable}


@app.get("/api/ollama/models")
async def list_ollama_models():
    """Return all models available on the local Ollama instance.

    Each model includes name, parameter count, quantization, and family.
    """
    reachable = await OllamaProvider.server_reachable()
    if not reachable:
        return {"reachable": False, "models": []}

    models = await OllamaProvider.list_local_models()
    return {"reachable": True, "models": models}


@app.get("/api/settings")
async def get_settings():
    """Return current env-var configuration (redacted)."""
    def _redact(val: str | None) -> str:
        if not val:
            return ""
        if len(val) <= 8:
            return "****"
        return val[:4] + "…" + val[-4:]

    return {
        "openai_api_key": _redact(os.getenv("OPENAI_API_KEY")),
        "anthropic_api_key": _redact(os.getenv("ANTHROPIC_API_KEY")),
        "google_api_key": _redact(os.getenv("GOOGLE_API_KEY")),
        "ollama_base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    }


@app.post("/api/settings")
async def update_settings(settings: SettingsUpdate):
    """Update environment variables at runtime (does NOT persist to .env)."""
    if settings.openai_api_key is not None:
        os.environ["OPENAI_API_KEY"] = settings.openai_api_key
    if settings.anthropic_api_key is not None:
        os.environ["ANTHROPIC_API_KEY"] = settings.anthropic_api_key
    if settings.google_api_key is not None:
        os.environ["GOOGLE_API_KEY"] = settings.google_api_key
    if settings.ollama_base_url is not None:
        os.environ["OLLAMA_BASE_URL"] = settings.ollama_base_url
    return {"status": "ok", "message": "Settings updated (runtime only)."}


@app.post("/api/run")
async def run_query(req: QueryRequest):
    """Run the full D-MMCE pipeline and stream events via SSE."""

    async def event_stream():
        # Collect events from the pipeline
        events_queue: asyncio.Queue[dict] = asyncio.Queue()

        def on_event(event: Event):
            event_data = {
                "type": event.event_type.name,
                "message": event.message,
                "payload": event.payload,
                "timestamp": time.time(),
            }
            logger.debug("Pipeline event: %s — %s", event.event_type.name, event.message)
            events_queue.put_nowait(event_data)

        logger.info("=== NEW RUN === query=%r providers=%s ollama_models=%s review=%s",
                     req.query[:80], req.providers, req.ollama_models, req.review_provider)

        # Build providers
        if req.providers:
            providers = []
            for name in req.providers:
                if name == "ollama":
                    logger.info("Skipping raw 'ollama' — local models handled via ollama_models")
                    continue
                try:
                    p = ProviderFactory.create(name)
                    logger.info("Created provider: %s (type=%s)", p.name, type(p).__name__)
                    providers.append(p)
                except KeyError:
                    logger.warning("Unknown provider name: %s", name)
        else:
            providers = await ProviderFactory.create_all_async()
            logger.info("Auto-discovered providers: %s", [p.name for p in providers])

        # Add explicitly-selected Ollama local models
        if req.ollama_models:
            for model_tag in req.ollama_models:
                tag = model_tag.strip()
                if tag:
                    p = OllamaProvider(model=tag)
                    logger.info("Added Ollama model: %s", p.name)
                    providers.append(p)

        # Filter to only available providers before creating the engine
        logger.info("Pre-filter provider list: %s", [p.name for p in providers])
        avail_checks = await asyncio.gather(
            *(p.is_available() for p in providers), return_exceptions=True
        )
        for p, ok in zip(providers, avail_checks):
            logger.info("  %s → is_available=%s", p.name, ok)
        skipped = [p.name for p, ok in zip(providers, avail_checks) if ok is not True]
        providers = [
            p for p, ok in zip(providers, avail_checks) if ok is True
        ]
        if skipped:
            logger.info("Skipped unavailable: %s", skipped)
        logger.info("Final provider list for engine: %s", [p.name for p in providers])

        if not providers:
            yield f"data: {json.dumps({'type': 'ERROR', 'message': 'No providers available.'})}\n\n"
            return

        engine = D_MMCE(
            providers=providers,
            review_provider_name=req.review_provider,
            embedding_model=req.embedding_model,
            stability_threshold=req.stability_threshold,
            max_stability_reruns=req.max_reruns,
            enable_logging_observer=False,
        )

        # Subscribe to all events
        engine.event_bus.subscribe_all(on_event)

        # Send initial event
        yield f"data: {json.dumps({'type': 'PIPELINE_STARTED', 'message': f'Starting D-MMCE with {len(providers)} providers...', 'payload': {'providers': [p.name for p in providers]}})}\n\n"

        # Run pipeline in background task
        result_holder: dict[str, Any] = {}

        async def _run():
            try:
                verdict = await engine.run(req.query)
                result_holder["verdict"] = {
                    "answer": verdict.answer,
                    "stability_score": verdict.stability_score,
                    "num_reruns": verdict.num_reruns,
                    "audit_trail": verdict.audit_trail,
                }
            except Exception as e:
                result_holder["error"] = str(e)

        task = asyncio.create_task(_run())

        # Stream events as they arrive
        while not task.done() or not events_queue.empty():
            try:
                event_data = await asyncio.wait_for(events_queue.get(), timeout=0.5)
                yield f"data: {json.dumps(event_data)}\n\n"
            except asyncio.TimeoutError:
                # Send heartbeat
                yield f"data: {json.dumps({'type': 'HEARTBEAT', 'message': '', 'payload': {}})}\n\n"

        # Drain remaining events
        while not events_queue.empty():
            event_data = events_queue.get_nowait()
            yield f"data: {json.dumps(event_data)}\n\n"

        # Send final result
        if "error" in result_holder:
            yield f"data: {json.dumps({'type': 'PIPELINE_ERROR', 'message': result_holder['error'], 'payload': {}})}\n\n"
        elif "verdict" in result_holder:
            yield f"data: {json.dumps({'type': 'PIPELINE_COMPLETE', 'message': 'Pipeline complete.', 'payload': result_holder['verdict']})}\n\n"

        yield f"data: {json.dumps({'type': 'STREAM_END', 'message': 'Done', 'payload': {}})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ------------------------------------------------------------------ #
#  Run
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)


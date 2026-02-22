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

# ------------------------------------------------------------------ #
#  Logging
# ------------------------------------------------------------------ #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("d_mmce.server")

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
    review_provider: str = "openai"
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
    """Return all registered providers with availability status."""
    names = ProviderFactory.available_names()
    statuses = []
    for name in names:
        try:
            p = ProviderFactory.create(name)
            avail = await p.is_available()
        except Exception:
            avail = False
        statuses.append({"name": name, "available": avail})
    return {"providers": statuses}


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
            events_queue.put_nowait({
                "type": event.event_type.name,
                "message": event.message,
                "payload": event.payload,
                "timestamp": time.time(),
            })

        # Build providers
        if req.providers:
            providers = []
            for name in req.providers:
                try:
                    providers.append(ProviderFactory.create(name))
                except KeyError:
                    pass
        else:
            providers = ProviderFactory.create_all()

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


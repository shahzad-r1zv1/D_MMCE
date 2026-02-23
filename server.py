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

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from d_mmce.observer import Event
from d_mmce.orchestrator import D_MMCE
from d_mmce.providers import ProviderFactory
from d_mmce.providers.ollama_provider import OllamaProvider
from d_mmce.history import RunHistoryDB

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
#  Security — settings endpoint protection
# ------------------------------------------------------------------ #
_PRODUCTION_MODE = os.getenv("D_MMCE_PRODUCTION_MODE", "").lower() in ("1", "true", "yes")
_API_AUTH_TOKEN = os.getenv("D_MMCE_API_AUTH_TOKEN", "")
_settings_audit_log = logging.getLogger("d_mmce.settings_audit")


def _verify_settings_auth(authorization: str | None) -> None:
    """Raise 403 if settings access is not authorised.

    In **production mode** (``D_MMCE_PRODUCTION_MODE=1``):
    * If ``D_MMCE_API_AUTH_TOKEN`` is set, the caller must supply it via
      ``Authorization: Bearer <token>``.
    * If no token is configured, the settings endpoints are **disabled**.
    """
    if not _PRODUCTION_MODE:
        return  # open access in development
    if not _API_AUTH_TOKEN:
        raise HTTPException(
            status_code=403,
            detail="Settings endpoint disabled in production mode "
                   "(set D_MMCE_API_AUTH_TOKEN to enable).",
        )
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token.")
    token = authorization[len("Bearer "):]
    if token != _API_AUTH_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid auth token.")

# ------------------------------------------------------------------ #
#  History database
# ------------------------------------------------------------------ #
history_db = RunHistoryDB()

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
    max_concurrent: int = 4
    max_retries: int = 2
    request_timeout: float = 120.0
    enable_streaming: bool = True


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
async def get_settings(authorization: str | None = Header(default=None)):
    """Return current env-var configuration (redacted)."""
    _verify_settings_auth(authorization)

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
async def update_settings(
    settings: SettingsUpdate,
    authorization: str | None = Header(default=None),
):
    """Update environment variables at runtime (does NOT persist to .env)."""
    _verify_settings_auth(authorization)

    changed: list[str] = []
    if settings.openai_api_key is not None:
        os.environ["OPENAI_API_KEY"] = settings.openai_api_key
        changed.append("OPENAI_API_KEY")
    if settings.anthropic_api_key is not None:
        os.environ["ANTHROPIC_API_KEY"] = settings.anthropic_api_key
        changed.append("ANTHROPIC_API_KEY")
    if settings.google_api_key is not None:
        os.environ["GOOGLE_API_KEY"] = settings.google_api_key
        changed.append("GOOGLE_API_KEY")
    if settings.ollama_base_url is not None:
        os.environ["OLLAMA_BASE_URL"] = settings.ollama_base_url
        changed.append("OLLAMA_BASE_URL")

    _settings_audit_log.info("Settings updated: %s", changed)
    return {"status": "ok", "message": "Settings updated (runtime only).", "changed": changed}


@app.post("/api/run")
async def run_query(req: QueryRequest):
    """Run the full D-MMCE pipeline and stream events via SSE."""

    async def event_stream():
        # Collect events from the pipeline
        events_queue: asyncio.Queue[dict] = asyncio.Queue()

        def on_event(event: Event):
            # Ensure payload is JSON-serializable (convert any non-standard types)
            payload = event.payload or {}
            try:
                json.dumps(payload)  # test serialization
            except (TypeError, ValueError):
                payload = {k: str(v) for k, v in payload.items()}

            event_data = {
                "type": event.event_type.name,
                "message": event.message,
                "payload": payload,
                "timestamp": time.time(),
                "run_id": event.run_id,
            }
            if event.event_type.name == "MODEL_RESPONSE":
                logger.info("MODEL_RESPONSE: provider=%s variant=%s text_len=%d",
                            payload.get("provider", "?"), payload.get("variant", "?"),
                            len(payload.get("text", "")))
            elif event.event_type.name == "FINAL_VERDICT":
                logger.info("FINAL_VERDICT: answer_len=%d stability=%.4f",
                            len(payload.get("answer", "")),
                            payload.get("stability_score", 0))
            elif event.event_type.name == "TOKEN_CHUNK":
                pass  # don't log individual tokens — too noisy
            else:
                logger.info("Event: %s — %s", event.event_type.name, event.message[:100])
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

        # Add explicitly-selected Ollama local models (skip duplicates)
        if req.ollama_models:
            existing_names = {p.name for p in providers}
            for model_tag in req.ollama_models:
                tag = model_tag.strip()
                if tag:
                    p = OllamaProvider(model=tag)
                    if p.name not in existing_names:
                        logger.info("Added Ollama model: %s", p.name)
                        providers.append(p)
                        existing_names.add(p.name)
                    else:
                        logger.debug("Skipping duplicate Ollama model: %s", p.name)

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
            max_concurrent_tasks=req.max_concurrent,
            max_retries=req.max_retries,
            request_timeout=req.request_timeout,
            enable_streaming=req.enable_streaming,
            history_db=history_db,
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
                    "confidence_score": verdict.confidence_score,
                    "num_reruns": verdict.num_reruns,
                    "audit_trail": verdict.audit_trail,
                    "run_id": verdict.run_id,
                    "stage_timings": verdict.stage_timings,
                }
                logger.info("Pipeline completed: stability=%.4f, reruns=%d, answer_len=%d",
                            verdict.stability_score, verdict.num_reruns, len(verdict.answer))
            except Exception as e:
                logger.error("Pipeline _run() failed: %s: %s", type(e).__name__, str(e), exc_info=True)
                result_holder["error"] = f"{type(e).__name__}: {str(e)}"

        task = asyncio.create_task(_run())

        # Stream events as they arrive
        while not task.done() or not events_queue.empty():
            try:
                event_data = await asyncio.wait_for(events_queue.get(), timeout=0.5)
                try:
                    yield f"data: {json.dumps(event_data)}\n\n"
                except (TypeError, ValueError) as json_err:
                    logger.warning("Failed to JSON-serialize event: %s", json_err)
                    # Send a simplified version
                    yield f"data: {json.dumps({'type': event_data.get('type','UNKNOWN'), 'message': str(event_data.get('message','')), 'payload': {{}}, 'timestamp': time.time()})}\n\n"
            except asyncio.TimeoutError:
                # Send heartbeat
                yield f"data: {json.dumps({'type': 'HEARTBEAT', 'message': '', 'payload': {}})}\n\n"

        # Drain remaining events
        while not events_queue.empty():
            event_data = events_queue.get_nowait()
            try:
                yield f"data: {json.dumps(event_data)}\n\n"
            except (TypeError, ValueError):
                pass

        # Send final result
        try:
            if "error" in result_holder:
                logger.info("Sending PIPELINE_ERROR: %s", result_holder['error'][:200])
                yield f"data: {json.dumps({'type': 'PIPELINE_ERROR', 'message': result_holder['error'], 'payload': {}})}\n\n"
            elif "verdict" in result_holder:
                logger.info("Sending PIPELINE_COMPLETE with %d char answer", len(result_holder['verdict'].get('answer', '')))
                yield f"data: {json.dumps({'type': 'PIPELINE_COMPLETE', 'message': 'Pipeline complete.', 'payload': result_holder['verdict']})}\n\n"
            else:
                logger.error("No verdict and no error in result_holder — keys: %s", list(result_holder.keys()))
                yield f"data: {json.dumps({'type': 'PIPELINE_ERROR', 'message': 'Pipeline finished without producing a result.', 'payload': {}})}\n\n"
        except Exception as final_err:
            logger.error("Failed to send final result: %s", final_err, exc_info=True)
            yield f"data: {json.dumps({'type': 'PIPELINE_ERROR', 'message': f'Internal error sending result: {final_err}', 'payload': {}})}\n\n"

        yield f"data: {json.dumps({'type': 'STREAM_END', 'message': 'Done', 'payload': {}})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ------------------------------------------------------------------ #
#  History API (Feature 5)
# ------------------------------------------------------------------ #

@app.get("/api/history")
async def list_history(limit: int = 50, offset: int = 0):
    """Return a paginated list of past runs (newest first)."""
    runs = await history_db.list_runs(limit=limit, offset=offset)
    total = await history_db.count_runs()
    return {"runs": runs, "total": total}


@app.get("/api/history/{run_id}")
async def get_history_run(run_id: str):
    """Return full details for a single historical run."""
    run = await history_db.get_run(run_id)
    if run is None:
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=404, content={"error": "Run not found"})
    return run


@app.delete("/api/history/{run_id}")
async def delete_history_run(run_id: str):
    """Delete a single historical run."""
    deleted = await history_db.delete_run(run_id)
    if not deleted:
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=404, content={"error": "Run not found"})
    return {"status": "ok", "message": f"Run {run_id} deleted."}


@app.delete("/api/history")
async def clear_history():
    """Delete all historical runs."""
    count = await history_db.clear_all()
    return {"status": "ok", "deleted": count}


# ------------------------------------------------------------------ #
#  Run
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)


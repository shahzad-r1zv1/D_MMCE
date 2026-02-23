"""
Run History – SQLite Persistence
=================================
Stores every D-MMCE pipeline run (query, verdict, model responses,
peer-review critiques) in a local SQLite database for later browsing,
comparison, and export.

All operations are async via ``aiosqlite``.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import aiosqlite

from d_mmce.schemas import Critique, FinalVerdict, ModelResponse

logger = logging.getLogger(__name__)

_DEFAULT_DB = Path(os.getenv("D_MMCE_HISTORY_DB", "d_mmce_history.db"))

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id          TEXT PRIMARY KEY,
    query       TEXT NOT NULL,
    answer      TEXT NOT NULL,
    stability   REAL NOT NULL,
    num_reruns  INTEGER NOT NULL,
    audit_trail TEXT NOT NULL,          -- JSON array
    provider_count INTEGER NOT NULL,
    response_count INTEGER NOT NULL,
    created_at  TEXT NOT NULL           -- ISO-8601
);

CREATE TABLE IF NOT EXISTS model_responses (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    provider    TEXT NOT NULL,
    variant     TEXT NOT NULL,
    text        TEXT NOT NULL,
    latency     REAL NOT NULL,
    metadata    TEXT NOT NULL           -- JSON object
);

CREATE TABLE IF NOT EXISTS critiques (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    reviewer    TEXT NOT NULL,
    reviewee    TEXT NOT NULL,
    critique_text TEXT NOT NULL,
    is_validated  INTEGER NOT NULL,     -- 0/1
    issues      TEXT NOT NULL           -- JSON array
);

CREATE INDEX IF NOT EXISTS idx_responses_run ON model_responses(run_id);
CREATE INDEX IF NOT EXISTS idx_critiques_run ON critiques(run_id);
"""


class RunHistoryDB:
    """Async SQLite store for D-MMCE pipeline runs.

    Parameters
    ----------
    db_path : str or Path
        Path to the SQLite file.  Created automatically on first use.
        Defaults to ``d_mmce_history.db`` in the working directory
        (overridable via ``D_MMCE_HISTORY_DB`` env var).
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._db_path = str(db_path or _DEFAULT_DB)
        self._initialised = False

    async def _ensure_schema(self) -> None:
        if self._initialised:
            return
        async with aiosqlite.connect(self._db_path) as db:
            await db.executescript(_SCHEMA)
            await db.commit()
        self._initialised = True
        logger.info("History DB ready: %s", self._db_path)

    # ------------------------------------------------------------------ #
    #  Write
    # ------------------------------------------------------------------ #

    async def save_run(
        self,
        query: str,
        verdict: FinalVerdict,
        responses: Sequence[ModelResponse],
        critiques: Sequence[Critique] | None = None,
    ) -> str:
        """Persist a complete pipeline run.  Returns the generated run ID."""
        await self._ensure_schema()
        run_id = uuid.uuid4().hex[:12]
        now = datetime.now(timezone.utc).isoformat()

        providers_used = list({r.provider_name for r in responses})

        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "INSERT INTO runs (id, query, answer, stability, num_reruns, "
                "audit_trail, provider_count, response_count, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    query,
                    verdict.answer,
                    verdict.stability_score,
                    verdict.num_reruns,
                    json.dumps(verdict.audit_trail),
                    len(providers_used),
                    len(responses),
                    now,
                ),
            )

            for r in responses:
                await db.execute(
                    "INSERT INTO model_responses "
                    "(run_id, provider, variant, text, latency, metadata) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        run_id,
                        r.provider_name,
                        r.prompt_variant,
                        r.text,
                        r.latency,
                        json.dumps(r.metadata, default=str),
                    ),
                )

            for c in (critiques or []):
                await db.execute(
                    "INSERT INTO critiques "
                    "(run_id, reviewer, reviewee, critique_text, is_validated, issues) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        run_id,
                        c.reviewer,
                        c.reviewee,
                        c.critique_text,
                        1 if c.is_validated else 0,
                        json.dumps(c.issues),
                    ),
                )

            await db.commit()

        logger.info("Saved run %s (%d responses, %d critiques)",
                     run_id, len(responses), len(critiques or []))
        return run_id

    # ------------------------------------------------------------------ #
    #  Read
    # ------------------------------------------------------------------ #

    async def list_runs(
        self, limit: int = 50, offset: int = 0
    ) -> list[dict[str, Any]]:
        """Return a paginated list of past runs (newest first)."""
        await self._ensure_schema()
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT id, query, stability, num_reruns, provider_count, "
                "response_count, created_at FROM runs "
                "ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            )
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]

    async def get_run(self, run_id: str) -> dict[str, Any] | None:
        """Return full details for a single run, including responses and critiques."""
        await self._ensure_schema()
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row

            cursor = await db.execute("SELECT * FROM runs WHERE id = ?", (run_id,))
            run_row = await cursor.fetchone()
            if not run_row:
                return None
            result = dict(run_row)
            result["audit_trail"] = json.loads(result["audit_trail"])

            cursor = await db.execute(
                "SELECT provider, variant, text, latency FROM model_responses "
                "WHERE run_id = ? ORDER BY id",
                (run_id,),
            )
            result["responses"] = [dict(r) for r in await cursor.fetchall()]

            cursor = await db.execute(
                "SELECT reviewer, reviewee, critique_text, is_validated, issues "
                "FROM critiques WHERE run_id = ? ORDER BY id",
                (run_id,),
            )
            crits = []
            for r in await cursor.fetchall():
                d = dict(r)
                d["issues"] = json.loads(d["issues"])
                d["is_validated"] = bool(d["is_validated"])
                crits.append(d)
            result["critiques"] = crits

        return result

    async def count_runs(self) -> int:
        """Return total number of stored runs."""
        await self._ensure_schema()
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute("SELECT COUNT(*) FROM runs")
            row = await cursor.fetchone()
            return row[0] if row else 0

    # ------------------------------------------------------------------ #
    #  Delete
    # ------------------------------------------------------------------ #

    async def delete_run(self, run_id: str) -> bool:
        """Delete a single run and its related data. Returns True if found."""
        await self._ensure_schema()
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute("DELETE FROM model_responses WHERE run_id = ?", (run_id,))
            await db.execute("DELETE FROM critiques WHERE run_id = ?", (run_id,))
            cursor = await db.execute("DELETE FROM runs WHERE id = ?", (run_id,))
            await db.commit()
            return cursor.rowcount > 0

    async def clear_all(self) -> int:
        """Delete all runs. Returns the number deleted."""
        await self._ensure_schema()
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute("SELECT COUNT(*) FROM runs")
            row = await cursor.fetchone()
            count = row[0] if row else 0
            await db.execute("DELETE FROM model_responses")
            await db.execute("DELETE FROM critiques")
            await db.execute("DELETE FROM runs")
            await db.commit()
            return count


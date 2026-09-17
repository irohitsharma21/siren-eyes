"""
In-process analysis job registry.

Analyses are CPU-bound and take tens of seconds, so the HTTP request that
starts one returns immediately and the client subscribes to a WebSocket for
results. This module holds the state in between.

Deliberately in-memory: the deployment target is a single container, and a
job outlives only the page that started it. Persisting to Redis would add an
operational dependency for no benefit at this scale.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.params import AnalysisParams

# queued -> running -> complete | failed | cancelled
TERMINAL = frozenset({"complete", "failed", "cancelled"})


@dataclass
class Job:
    """One analysis, from queued to finished."""

    job_id: str
    source: Path
    status: str = "queued"
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    frames: list[Any] = field(default_factory=list)
    summary: Any = None
    error: str | None = None
    hfov_deg: float | None = None
    params: AnalysisParams = field(default_factory=AnalysisParams)
    # True when the recorded demo analysis will be replayed rather than the
    # pipeline run; decided at creation so the client knows before it starts.
    precomputed: bool = False
    # Cooperative cancellation: the worker polls this between frames. A hard
    # kill is not an option - the worker is a thread inside the same process.
    _cancel: threading.Event = field(default_factory=threading.Event, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # ── transitions ───────────────────────────────────────────────────
    def add_frame(self, frame: Any) -> None:
        with self._lock:
            self.status = "running"
            self.frames.append(frame)

    def start(self) -> None:
        with self._lock:
            if self.status == "queued":
                self.status = "running"

    def complete(self, summary: Any) -> None:
        with self._lock:
            self.status = "complete"
            self.summary = summary

    def fail(self, error: str) -> None:
        with self._lock:
            self.status = "failed"
            self.error = error

    def cancel(self) -> str:
        """
        Ask the run to stop. Returns the status after the request:
        `cancelled` if nothing was running yet, `cancelling` while the worker
        winds down, or the terminal status if it had already finished.
        """
        with self._lock:
            if self.status in TERMINAL:
                return self.status
            self._cancel.set()
            if self.status == "queued":
                self.status = "cancelled"
                return "cancelled"
            return "cancelling"

    def mark_cancelled(self, summary: Any = None) -> None:
        with self._lock:
            self.status = "cancelled"
            if summary is not None:
                self.summary = summary

    @property
    def cancel_requested(self) -> bool:
        return self._cancel.is_set()

    # ── views ─────────────────────────────────────────────────────────
    def snapshot(self) -> dict:
        """JSON-ready view of the job as it stands."""
        # Live runs hold FrameResult dataclasses; replays hold the recorded
        # dicts as they were read from disk. Both must serialise.
        def plain(v: Any) -> Any:
            return asdict(v) if is_dataclass(v) and not isinstance(v, type) else v

        with self._lock:
            return {
                "job_id": self.job_id,
                "source": self.source.name,
                "status": self.status,
                "created_at": self.created_at,
                "precomputed": self.precomputed,
                "parameters": self.params.as_dict(),
                "frame_count": len(self.frames),
                "frames": [plain(f) for f in self.frames],
                "summary": plain(self.summary) if self.summary else None,
                "error": self.error,
            }


class JobStore:
    """Thread-safe job registry with a bounded LRU of finished jobs."""

    MAX_JOBS = 24

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(
        self,
        source: Path,
        hfov_deg: float | None = None,
        params: AnalysisParams | None = None,
        precomputed: bool = False,
    ) -> str:
        job_id = uuid.uuid4().hex[:12]
        with self._lock:
            self._jobs[job_id] = Job(
                job_id=job_id,
                source=Path(source),
                hfov_deg=hfov_deg,
                params=params or AnalysisParams(),
                precomputed=precomputed,
            )
            # Evict oldest once the cap is exceeded; frame lists are large.
            while len(self._jobs) > self.MAX_JOBS:
                self._jobs.pop(next(iter(self._jobs)))
        return job_id

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

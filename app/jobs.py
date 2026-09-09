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
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class Job:
    """One analysis, from queued to finished."""

    job_id: str
    source: Path
    status: str = "queued"          # queued | running | complete | failed
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    frames: list[Any] = field(default_factory=list)
    summary: Any = None
    error: str | None = None
    hfov_deg: float | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add_frame(self, frame: Any) -> None:
        with self._lock:
            self.status = "running"
            self.frames.append(frame)

    def complete(self, summary: Any) -> None:
        with self._lock:
            self.status = "complete"
            self.summary = summary

    def fail(self, error: str) -> None:
        with self._lock:
            self.status = "failed"
            self.error = error

    def snapshot(self) -> dict:
        """JSON-ready view of the job as it stands."""
        with self._lock:
            return {
                "job_id": self.job_id,
                "source": self.source.name,
                "status": self.status,
                "created_at": self.created_at,
                "frame_count": len(self.frames),
                "frames": [asdict(f) for f in self.frames],
                "summary": asdict(self.summary) if self.summary else None,
                "error": self.error,
            }


class JobStore:
    """Thread-safe job registry with a bounded LRU of finished jobs."""

    MAX_JOBS = 24

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self, source: Path, hfov_deg: float | None = None) -> str:
        job_id = uuid.uuid4().hex[:12]
        with self._lock:
            self._jobs[job_id] = Job(
                job_id=job_id, source=Path(source), hfov_deg=hfov_deg
            )
            # Evict oldest once the cap is exceeded; frame lists are large.
            while len(self._jobs) > self.MAX_JOBS:
                self._jobs.pop(next(iter(self._jobs)))
        return job_id

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

"""
Siren Eyes — FastAPI application.

Serves the analysis API, streams per-frame results over a WebSocket so the
dashboard can render an analysis as it happens, and hosts the built frontend
as static files so the whole system ships as one container.
"""

from __future__ import annotations

import asyncio
import json
import os
import platform
import subprocess
import threading
import uuid
from contextlib import asynccontextmanager, suppress
from dataclasses import asdict
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from fastapi import (
    FastAPI, File, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import DEMO_DIR, ROOT_DIR, RUNS_DIR, STATIC_DIR, settings
from app.core.params import BOUNDS, AnalysisParams
from app.core.pipeline import SirenEyesPipeline
from app.jobs import TERMINAL, JobStore

pipeline: SirenEyesPipeline | None = None
jobs = JobStore()

# The pipeline keeps per-run state on the instance, and the free tier has a
# tenth of a core anyway, so live analyses run one at a time. A second request
# waits its turn and is told so over its own stream.
live_lock = threading.Lock()

STARTED_AT = datetime.now(timezone.utc)
BUILD: dict = {}


# ── build provenance ──────────────────────────────────────────────────
def _git(*args: str) -> str | None:
    """Ask the local checkout, if there is one; the container ships without .git."""
    try:
        out = subprocess.run(
            ["git", *args], cwd=ROOT_DIR, capture_output=True, text=True, timeout=3
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def _build_info() -> dict:
    """
    Where this running copy came from.

    Resolution order for the commit: an explicit SIREN_GIT_SHA (stamped by the
    Dockerfile build arg), the RENDER_GIT_COMMIT that Render injects at
    runtime, then `git rev-parse` for a local checkout. Build time falls back
    to the dashboard bundle's modification time, which is when `npm run build`
    last ran - close enough to "when was this built" to be useful.
    """
    sha = (
        settings.GIT_SHA
        or os.environ.get("RENDER_GIT_COMMIT")
        or os.environ.get("SOURCE_COMMIT")
        or _git("rev-parse", "HEAD")
    )
    build_time = settings.BUILD_TIME
    if not build_time:
        index = STATIC_DIR / "index.html"
        if index.exists():
            build_time = datetime.fromtimestamp(
                index.stat().st_mtime, tz=timezone.utc
            ).isoformat(timespec="seconds")

    if os.environ.get("RENDER"):
        host = "render"
    elif os.environ.get("SPACE_ID"):
        host = "huggingface"
    else:
        host = "local"

    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "git_sha": sha,
        "git_sha_short": sha[:7] if sha else None,
        "git_branch": os.environ.get("RENDER_GIT_BRANCH")
        or _git("rev-parse", "--abbrev-ref", "HEAD"),
        "build_time": build_time,
        "python": platform.python_version(),
        "host": host,
        "environment": settings.ENVIRONMENT,
    }


@lru_cache(maxsize=1)
def _siren_description() -> dict:
    """
    The classifier checkpoint is static for the life of the process. Health is
    polled, and re-reading a torch checkpoint on every poll is wasted CPU on a
    host that has very little of it.
    """
    from app.core.audio import SirenClassifier

    return SirenClassifier.describe()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models once at startup; first-frame latency otherwise includes them."""
    global pipeline
    RUNS_DIR.mkdir(exist_ok=True)
    BUILD.update(_build_info())
    print(f"[siren-eyes] build {BUILD.get('git_sha_short') or 'unknown'} ({BUILD['host']})")
    print("[siren-eyes] loading models ...")
    pipeline = SirenEyesPipeline()
    await asyncio.get_running_loop().run_in_executor(None, pipeline.detector.warmup)
    print("[siren-eyes] ready")
    yield
    print("[siren-eyes] shutting down")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "Stereo-aware multimodal ambulance detection with safety-buffered "
        "traffic signal preemption."
    ),
    lifespan=lifespan,
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── system ────────────────────────────────────────────────────────────
@app.get("/api/health", tags=["system"])
async def health():
    return {
        "status": "healthy",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "models": {
            "detector": settings.YOLO_WEIGHTS.name,
            "detector_loaded": pipeline is not None,
            "siren_classifier": _siren_description(),
        },
        "build": {k: BUILD.get(k) for k in ("git_sha_short", "build_time", "host")},
        "time": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/version", tags=["system"])
async def version():
    """Build provenance: which commit is running, when it was built, where."""
    now = datetime.now(timezone.utc)
    return {
        **BUILD,
        "started_at": STARTED_AT.isoformat(timespec="seconds"),
        "uptime_s": round((now - STARTED_AT).total_seconds(), 1),
        "time": now.isoformat(timespec="seconds"),
    }


@app.get("/api/config", tags=["system"])
async def config():
    """Expose the paper's operating parameters so the UI can display them."""
    return {
        "vision_confidence_threshold": settings.VISION_CONF_THRESHOLD,
        "siren_threshold": settings.SIREN_THRESHOLD,
        "fusion": {
            "alpha_vision": settings.FUSION_ALPHA_VISION,
            "beta_audio": settings.FUSION_BETA_AUDIO,
            "trigger_threshold": settings.FUSION_TRIGGER_THRESHOLD,
        },
        "safety": {
            "buffer_s": settings.SAFETY_BUFFER_S,
            "ttc_threshold_s": settings.TTC_THRESHOLD_S,
            "approach_tolerance_deg": settings.APPROACH_ANGLE_TOLERANCE_DEG,
        },
        "geometry": {
            "ambulance_width_m": settings.AMBULANCE_WIDTH_M,
            "camera_hfov_deg": settings.CAMERA_HFOV_DEG,
        },
        "audio": {
            "sample_rate": settings.AUDIO_SAMPLE_RATE,
            "window_s": settings.AUDIO_WINDOW_SECONDS,
            "bandpass_hz": [settings.BANDPASS_LOW_HZ, settings.BANDPASS_HIGH_HZ],
            "mic_baseline_m": settings.MIC_BASELINE_M,
        },
        # What a client may override per run, with the ranges the server will
        # accept, so the UI's controls and the 422s agree by construction.
        "parameters": AnalysisParams.describe(),
        "upload": {
            "max_mb": settings.MAX_UPLOAD_MB,
            "extensions": settings.UPLOAD_EXTENSIONS,
        },
        "frame_sample_stride": settings.FRAME_SAMPLE_STRIDE,
    }


# ── demo clips ────────────────────────────────────────────────────────
@app.get("/api/demos", tags=["demo"])
async def list_demos():
    """Curated clips so a visitor never has to find their own footage."""
    manifest = DEMO_DIR / "manifest.json"
    if not manifest.exists():
        return {"demos": []}
    return json.loads(manifest.read_text())


@app.get("/api/demos/{name}/video", tags=["demo"])
async def demo_video(name: str):
    path = (DEMO_DIR / name).resolve()
    if not path.is_file() or DEMO_DIR.resolve() not in path.parents:
        raise HTTPException(404, "demo clip not found")
    return FileResponse(path, media_type="video/mp4")


# ── analysis ──────────────────────────────────────────────────────────
def _demo_hfov(name: str) -> float | None:
    """Horizontal field of view recorded for a bundled clip, if calibrated."""
    manifest = DEMO_DIR / "manifest.json"
    if not manifest.exists():
        return None
    for clip in json.loads(manifest.read_text()).get("demos", []):
        if clip.get("file") == name:
            return clip.get("camera_hfov_deg")
    return None


def _precomputed_for(source: Path, params: AnalysisParams) -> Path | None:
    """
    The recorded analysis for a bundled demo, if one shipped with it and the
    request asks for exactly the thresholds it was recorded with.

    Analysis is CPU-bound - roughly 111 ms per frame for YOLOv8 over 1080p on a
    desktop core. A free-tier container gets about 0.1 of a core, which turns a
    twenty-second clip into a ten-minute wait on an instance already close to
    its memory ceiling, so in practice the stream never finishes.

    The bundled clips are therefore analysed once by scripts/precompute_demos.py
    on hardware that can do the work, and the result ships in the repository.
    The figures are real output from this same pipeline, not fabrications, and
    the summary carries `precomputed: true` so the dashboard can say so.

    A recording is only valid for the parameters it was made with. Move any
    threshold off its default and the demo is analysed live like an upload:
    slow on the free tier, and the UI says so before the run starts.
    """
    if not params.is_default:
        return None
    if DEMO_DIR.resolve() not in source.resolve().parents:
        return None
    candidate = source.with_name(source.name + ".analysis.json")
    return candidate if candidate.is_file() else None


def _upload_extension_ok(filename: str | None) -> bool:
    return Path(filename or "").suffix.lower() in settings.UPLOAD_EXTENSIONS


def _param_query(name: str, description: str):
    lo, hi, _ = BOUNDS[name]
    default = getattr(AnalysisParams(), name)
    return Query(default=None, ge=lo, le=hi, description=f"{description} (default {default})")


@app.post("/api/analyses", tags=["analysis"], status_code=202)
async def create_analysis(
    file: UploadFile | None = File(default=None),
    demo: str | None = None,
    vision_threshold: float | None = _param_query(
        "vision_threshold", "YOLO confidence a box needs to count as an ambulance"
    ),
    fused_threshold: float | None = _param_query(
        "fused_threshold", "Fused 0.7*vision + 0.3*audio confidence that arms the safety gate"
    ),
    safety_buffer_s: float | None = _param_query(
        "safety_buffer_s", "Minimum ETA, in seconds, before a phase change is allowed"
    ),
    min_ttc_s: float | None = _param_query(
        "min_ttc_s", "Minimum time-to-collision for conflicting traffic, in seconds"
    ),
):
    """
    Queue an analysis of an uploaded clip or a bundled demo clip.

    Threshold overrides are optional query parameters; each is validated
    against the range in `app/core/params.py`, and omitted ones take the
    paper's defaults. Returns immediately with a job id; results stream over
    `/api/analyses/{job_id}/stream`.
    """
    params = AnalysisParams.build(
        vision_threshold=vision_threshold,
        fused_threshold=fused_threshold,
        safety_buffer_s=safety_buffer_s,
        min_ttc_s=min_ttc_s,
    )

    if demo:
        source = (DEMO_DIR / demo).resolve()
        if not source.is_file() or DEMO_DIR.resolve() not in source.parents:
            raise HTTPException(404, "demo clip not found")
    elif file is not None:
        # Browsers are inconsistent about MIME for video containers (.mkv often
        # arrives as application/octet-stream), so accept either signal.
        is_video_mime = (file.content_type or "").startswith("video/")
        if not is_video_mime and not _upload_extension_ok(file.filename):
            raise HTTPException(
                415,
                f"upload must be a video file ({', '.join(settings.UPLOAD_EXTENSIONS)})",
            )

        job_dir = RUNS_DIR / "uploads"
        job_dir.mkdir(parents=True, exist_ok=True)
        suffix = Path(file.filename or "").suffix.lower() or ".mp4"
        source = job_dir / f"{uuid.uuid4().hex}{suffix}"

        size = 0
        limit = settings.MAX_UPLOAD_MB * 1024 * 1024
        with source.open("wb") as fh:
            while chunk := await file.read(1 << 20):
                size += len(chunk)
                if size > limit:
                    fh.close()
                    source.unlink(missing_ok=True)
                    raise HTTPException(
                        413, f"file exceeds {settings.MAX_UPLOAD_MB} MB limit"
                    )
                fh.write(chunk)
        if size == 0:
            source.unlink(missing_ok=True)
            raise HTTPException(400, "uploaded file is empty")
    else:
        raise HTTPException(400, "provide either an uploaded file or ?demo=<name>")

    precomputed = _precomputed_for(source, params) is not None

    # Uploaded footage has no calibration; the global default applies and the
    # UI labels distances as uncalibrated.
    job_id = jobs.create(
        source,
        hfov_deg=_demo_hfov(source.name) if demo else None,
        params=params,
        precomputed=precomputed,
    )
    return {
        "job_id": job_id,
        "source": source.name,
        "parameters": params.as_dict(),
        "default_parameters": params.is_default,
        "precomputed": precomputed,
    }


@app.get("/api/analyses/{job_id}", tags=["analysis"])
async def get_analysis(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    return job.snapshot()


@app.delete("/api/analyses/{job_id}", tags=["analysis"])
async def cancel_analysis(job_id: str):
    """
    Stop a running analysis.

    Cancellation is cooperative: the worker thread checks between frames and
    stops with the frames it has, then the stream sends a partial summary and
    a `cancelled` event. A job that already finished is left as it is.
    """
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    status = job.cancel()
    if status in ("complete", "failed"):
        raise HTTPException(409, f"job already {status}")
    return {"job_id": job_id, "status": status}


async def _replay_precomputed(websocket: WebSocket, job, path: Path) -> None:
    """Stream a recorded analysis, paced roughly as the live run was."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    frames = payload["frames"]
    summary = payload["summary"]
    # Recordings made before the schema gained these carry the defaults.
    summary.setdefault("parameters", AnalysisParams().as_dict())
    summary.setdefault("cancelled", False)

    job.start()
    await websocket.send_json({"type": "started", "job_id": job.job_id, "precomputed": True})
    await websocket.send_json({"type": "status", "phase": "replaying recorded analysis"})

    for frame in frames:
        if job.cancel_requested:
            job.mark_cancelled()
            await websocket.send_json({"type": "cancelled", "frames": len(job.frames)})
            return
        # Pace with the latency actually measured for that frame, capped so a
        # slow frame does not stall the dashboard. Sending all 165 at once
        # would arrive as a single jump with no visible pipeline.
        delay = min(float(frame.get("latency_ms", 40)), 120.0) / 1000.0
        await asyncio.sleep(delay)
        job.frames.append(frame)
        await websocket.send_json({"type": "frame", "frame": frame})

    job.complete(summary)
    await websocket.send_json({"type": "summary", "summary": summary})
    await websocket.send_json({"type": "done"})


@app.websocket("/api/analyses/{job_id}/stream")
async def stream_analysis(websocket: WebSocket, job_id: str):
    """
    Stream per-frame results.

    Event order: `started`, zero or more `status` (phase) events, `frame`
    events, then `summary` followed by `done` - or `cancelled` after the
    partial summary if the client stopped the run, or `error`.

    The pipeline is synchronous and CPU-bound, so it runs on a worker thread
    and hands frames back through a queue; running it on the event loop would
    stall every other connection for the duration of the analysis.
    """
    await websocket.accept()

    job = jobs.get(job_id)
    if job is None:
        await websocket.send_json({"type": "error", "message": "unknown job"})
        await websocket.close()
        return

    if pipeline is None:
        await websocket.send_json({"type": "error", "message": "models still loading"})
        await websocket.close()
        return

    if job.status != "queued":
        await websocket.send_json(
            {"type": "error", "message": f"job is {job.status}; start a new analysis"}
        )
        await websocket.close()
        return

    recorded = _precomputed_for(Path(job.source), job.params)
    if recorded is not None:
        try:
            await _replay_precomputed(websocket, job, recorded)
        except WebSocketDisconnect:
            job.cancel()
        finally:
            with suppress(Exception):
                await websocket.close()
        return

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue(maxsize=256)
    SENTINEL = object()

    def push(item: object) -> None:
        asyncio.run_coroutine_threadsafe(queue.put(item), loop).result()

    def worker() -> None:
        if not live_lock.acquire(blocking=False):
            push({"__status__": "waiting for another analysis to finish"})
            live_lock.acquire()
        try:
            if job.cancel_requested:
                job.mark_cancelled()
                push({"__cancelled__": True})
                return
            job.start()
            for frame in pipeline.stream(
                job.source,
                hfov_deg=job.hfov_deg,
                params=job.params,
                on_phase=lambda phase: push({"__status__": phase}),
                should_stop=lambda: job.cancel_requested,
            ):
                job.add_frame(frame)
                push(asdict(frame))
            summary = pipeline.summarise()
            if summary.cancelled:
                job.mark_cancelled(summary)
                push({"__summary__": asdict(summary)})
                push({"__cancelled__": True})
            else:
                job.complete(summary)
                push({"__summary__": asdict(summary)})
        except Exception as exc:  # surfaced to the client, not swallowed
            job.fail(str(exc))
            push({"__error__": str(exc)})
        finally:
            live_lock.release()
            push(SENTINEL)

    task = loop.run_in_executor(None, worker)

    try:
        await websocket.send_json({"type": "started", "job_id": job_id, "precomputed": False})
        cancelled = False
        while True:
            item = await queue.get()
            if item is SENTINEL:
                break
            if "__summary__" in item:
                await websocket.send_json({"type": "summary", "summary": item["__summary__"]})
            elif "__status__" in item:
                await websocket.send_json({"type": "status", "phase": item["__status__"]})
            elif "__cancelled__" in item:
                cancelled = True
                await websocket.send_json({"type": "cancelled", "frames": len(job.frames)})
            elif "__error__" in item:
                await websocket.send_json({"type": "error", "message": item["__error__"]})
            else:
                await websocket.send_json({"type": "frame", "frame": item})
        if not cancelled and job.status == "complete":
            await websocket.send_json({"type": "done"})
    except WebSocketDisconnect:
        # Nobody is listening; do not spend the CPU finishing the run.
        job.cancel()
    except Exception:
        job.cancel()
    finally:
        if job.status not in TERMINAL:
            job.cancel()
        # Drain so the worker's blocking push() calls can complete and the
        # thread exits, then wait for it.
        while not task.done():
            with suppress(Exception):
                await asyncio.wait_for(queue.get(), timeout=0.25)
        try:
            await websocket.close()
        except Exception:
            pass


# ── static frontend ───────────────────────────────────────────────────
if STATIC_DIR.exists():
    app.mount("/assets", StaticFiles(directory=STATIC_DIR / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    async def spa(path: str):
        """Serve the SPA, letting the client router own unknown paths."""
        if path.startswith("api/"):
            raise HTTPException(404, "not found")
        candidate = (STATIC_DIR / path).resolve()
        if path and candidate.is_file() and STATIC_DIR.resolve() in candidate.parents:
            return FileResponse(candidate)
        return FileResponse(STATIC_DIR / "index.html")
else:

    @app.get("/", include_in_schema=False)
    async def no_frontend():
        return JSONResponse(
            {
                "message": f"{settings.APP_NAME} API is running.",
                "note": "Frontend not built. Run: cd web && npm install && npm run build",
                "docs": "/api/docs",
            }
        )

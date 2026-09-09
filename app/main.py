"""
Siren Eyes — FastAPI application.

Serves the analysis API, streams per-frame results over a WebSocket so the
dashboard can render an analysis as it happens, and hosts the built frontend
as static files so the whole system ships as one container.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import DEMO_DIR, RUNS_DIR, STATIC_DIR, settings
from app.core.pipeline import SirenEyesPipeline
from app.jobs import JobStore

pipeline: SirenEyesPipeline | None = None
jobs = JobStore()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models once at startup; first-frame latency otherwise includes them."""
    global pipeline
    RUNS_DIR.mkdir(exist_ok=True)
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
    from app.core.audio import SirenClassifier

    return {
        "status": "healthy",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "models": {
            "detector": settings.YOLO_WEIGHTS.name,
            "detector_loaded": pipeline is not None,
            "siren_classifier": SirenClassifier.describe(),
        },
        "time": datetime.now(timezone.utc).isoformat(),
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


@app.post("/api/analyses", tags=["analysis"], status_code=202)
async def create_analysis(
    file: UploadFile | None = File(default=None),
    demo: str | None = None,
):
    """
    Queue an analysis of an uploaded clip or a bundled demo clip.

    Returns immediately with a job id; results stream over
    `/api/analyses/{job_id}/stream`.
    """
    if demo:
        source = (DEMO_DIR / demo).resolve()
        if not source.is_file() or DEMO_DIR.resolve() not in source.parents:
            raise HTTPException(404, "demo clip not found")
    elif file is not None:
        if not (file.content_type or "").startswith("video/"):
            raise HTTPException(415, "upload must be a video file")

        job_dir = RUNS_DIR / "uploads"
        job_dir.mkdir(parents=True, exist_ok=True)
        source = job_dir / f"{uuid.uuid4().hex}.mp4"

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
    else:
        raise HTTPException(400, "provide either an uploaded file or ?demo=<name>")

    # Uploaded footage has no calibration; the global default applies and the
    # UI labels distances as uncalibrated.
    job_id = jobs.create(source, hfov_deg=_demo_hfov(source.name) if demo else None)
    return {"job_id": job_id, "source": source.name}


@app.get("/api/analyses/{job_id}", tags=["analysis"])
async def get_analysis(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    return job.snapshot()


@app.websocket("/api/analyses/{job_id}/stream")
async def stream_analysis(websocket: WebSocket, job_id: str):
    """
    Stream per-frame results.

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

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue(maxsize=256)
    SENTINEL = object()

    def worker() -> None:
        try:
            for frame in pipeline.stream(job.source, hfov_deg=job.hfov_deg):
                job.add_frame(frame)
                asyncio.run_coroutine_threadsafe(queue.put(asdict(frame)), loop).result()
            job.complete(pipeline.summarise())
            asyncio.run_coroutine_threadsafe(
                queue.put({"__summary__": asdict(job.summary)}), loop
            ).result()
        except Exception as exc:  # surfaced to the client, not swallowed
            job.fail(str(exc))
            asyncio.run_coroutine_threadsafe(
                queue.put({"__error__": str(exc)}), loop
            ).result()
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(SENTINEL), loop).result()

    task = loop.run_in_executor(None, worker)

    try:
        await websocket.send_json({"type": "started", "job_id": job_id})
        while True:
            item = await queue.get()
            if item is SENTINEL:
                break
            if "__summary__" in item:
                await websocket.send_json({"type": "summary", "summary": item["__summary__"]})
            elif "__error__" in item:
                await websocket.send_json({"type": "error", "message": item["__error__"]})
            else:
                await websocket.send_json({"type": "frame", "frame": item})
        await websocket.send_json({"type": "done"})
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        await task
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

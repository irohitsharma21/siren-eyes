"""
End-to-end analysis pipeline.

Realises the eight-stage flow of Paper section II-H:

    1. acquire video + stereo audio
    2. YOLOv8 visual detection
    3. CNN siren confirmation
    4. ITD/ILD directional fusion
    5. perspective-corrected distance / speed / ETA
    6. safety-aware signal decision (buffer + TTC)
    7. signal override, held until clearance
    8. local logging of every decision tick

Audio is analysed once up front and indexed by time, then joined onto the
video timeline. Scoring the audio lazily per frame would re-run the same
2.97 s window dozens of times.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

from app.config import settings
from app.core.audio import SirenClassifier
from app.core.direction import DirectionEstimate, estimate_direction
from app.core.geometry import BoxTracker, KinematicState
from app.core.preemption import (
    ConflictVehicle,
    Decision,
    FusionResult,
    PreemptionController,
    PreemptionEvent,
)
from app.core.vision import AmbulanceDetector, Detection
from app.media import AudioTrack, VideoInfo, extract_audio, iter_frames, probe_video


@dataclass
class AudioTimeline:
    """Siren confidence and bearing sampled on a regular grid."""

    times: list[float] = field(default_factory=list)
    siren_confidence: list[float] = field(default_factory=list)
    directions: list[DirectionEstimate | None] = field(default_factory=list)
    available: bool = True
    note: str = ""

    def at(self, t: float) -> tuple[float, DirectionEstimate | None]:
        """Nearest-neighbour lookup onto the audio grid."""
        if not self.times:
            return 0.0, None
        idx = int(np.argmin(np.abs(np.asarray(self.times) - t)))
        return self.siren_confidence[idx], self.directions[idx]


@dataclass
class FrameResult:
    """Per-sampled-frame record, streamed to the UI and written to the log."""

    frame_index: int
    timestamp_s: float
    detections: list[dict]
    vision_confidence: float
    audio_confidence: float
    fused_confidence: float
    distance_m: float | None
    speed_kmph: float | None
    eta_s: float | None
    direction_deg: float | None
    direction_bearing: str
    direction_available: bool
    signal_state: str
    decision: str
    reason: str
    latency_ms: float


@dataclass
class AnalysisSummary:
    """Aggregate outcome of one clip."""

    video: str
    duration_s: float
    fps: float
    resolution: str
    frames_analysed: int
    detection_frames: int
    peak_vision_confidence: float
    peak_siren_confidence: float
    peak_fused_confidence: float
    preemption_granted: bool
    first_detection_s: float | None
    grant_time_s: float | None
    response_latency_s: float | None
    audio_available: bool
    audio_stereo: bool
    audio_note: str
    mean_latency_ms: float
    p95_latency_ms: float
    stage_latency_ms: dict[str, float]
    blocked_reasons: dict[str, int]


class SirenEyesPipeline:
    """
    Orchestrates the full detection and preemption pipeline.

    Models are loaded once at construction so the API can hold a single warm
    instance across requests.
    """

    def __init__(self) -> None:
        self.detector = AmbulanceDetector()
        self.siren = SirenClassifier()

    # ── audio ─────────────────────────────────────────────────────────
    def analyse_audio(self, track: AudioTrack | None) -> AudioTimeline:
        """Score siren confidence and bearing across the whole recording."""
        if track is None:
            return AudioTimeline(available=False, note="no audio stream in container")
        if track.is_silent:
            return AudioTimeline(
                available=False, note="audio stream present but digitally silent"
            )

        sr = track.sample_rate
        hop = settings.AUDIO_HOP_SECONDS
        window = settings.AUDIO_WINDOW_SECONDS

        timeline = AudioTimeline(
            note="" if track.is_stereo else "mono/dual-mono source — direction unavailable"
        )

        mono = track.mono
        t = 0.0
        while t < track.duration_s:
            a = int(t * sr)
            b = min(a + int(window * sr), len(mono))
            if b - a < int(0.5 * sr):
                break

            timeline.times.append(t)
            timeline.siren_confidence.append(
                self.siren.predict_window(mono[a:b], sr)
            )

            left, right = track.window(t, window)
            timeline.directions.append(
                estimate_direction(left, right, sr, is_stereo=track.is_stereo)
            )
            t += hop

        return timeline

    # ── main loop ─────────────────────────────────────────────────────
    def stream(
        self,
        video_path: str | Path,
        conflicts: Callable[[float], list[ConflictVehicle]] | None = None,
        approach_angle_deg: float = 0.0,
        hfov_deg: float | None = None,
        progress: Callable[[float], None] | None = None,
    ) -> Iterator[FrameResult]:
        """
        Analyse a clip, yielding one FrameResult per sampled frame.

        Streaming rather than batching so the dashboard can render progress on
        long clips instead of waiting for the whole file.
        """
        video_path = Path(video_path)
        info = probe_video(video_path)

        track = extract_audio(video_path)
        audio = self.analyse_audio(track)

        tracker = BoxTracker()
        controller = PreemptionController(approach_angle_deg=approach_angle_deg)

        # Focal length in pixels for this clip. A per-clip horizontal field of
        # view (from the demo manifest) beats the global default, which assumes
        # a wide lens and therefore reads telephoto footage as far too close.
        import math

        if hfov_deg:
            focal_px = (info.width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
        else:
            focal_px = settings.focal_px(info.width)
        self._focal_px = focal_px
        self._hfov_deg = hfov_deg or settings.CAMERA_HFOV_DEG

        self._info = info
        self._audio_timeline = audio
        self._audio_track = track
        self._latencies: list[float] = []
        self._stage_ms = {"vision": 0.0, "audio": 0.0, "direction": 0.0, "decision": 0.0}
        self._events: list[PreemptionEvent] = []
        self._frames = 0
        self._detection_frames = 0
        self._peak_vision = 0.0
        self._first_detection: float | None = None

        for idx, ts, frame in iter_frames(video_path):
            t0 = time.perf_counter()

            # ── stage 2: vision ───────────────────────────────────────
            detections = self.detector.detect(frame, conf=0.20)
            t_vision = time.perf_counter()

            gated = [
                d for d in detections if d.confidence >= settings.VISION_CONF_THRESHOLD
            ]
            vision_conf = max((d.confidence for d in gated), default=0.0)

            # ── stages 3 + 4: audio, pre-computed ────────────────────
            audio_conf, direction = audio.at(ts)
            t_audio = time.perf_counter()

            # ── stage 5: kinematics ──────────────────────────────────
            kin: KinematicState | None = None
            if gated:
                tracks = tracker.update(
                    [(d.box, d.confidence) for d in gated], ts, focal_px
                )
                best = max(tracks, key=lambda t: t.peak_confidence, default=None)
                kin = best.state if best else None

                self._detection_frames += 1
                if self._first_detection is None:
                    self._first_detection = ts
            t_geom = time.perf_counter()

            # ── stages 6-8: fusion and decision ──────────────────────
            fusion = FusionResult.compute(vision_conf, audio_conf)
            event = controller.step(
                ts, fusion, kin, direction, conflicts(ts) if conflicts else []
            )
            t_end = time.perf_counter()

            self._stage_ms["vision"] += (t_vision - t0) * 1000
            self._stage_ms["audio"] += (t_audio - t_vision) * 1000
            self._stage_ms["direction"] += (t_geom - t_audio) * 1000
            self._stage_ms["decision"] += (t_end - t_geom) * 1000

            latency_ms = (t_end - t0) * 1000
            self._latencies.append(latency_ms)
            self._peak_vision = max(self._peak_vision, vision_conf)
            self._frames += 1
            self._events.append(event)

            if progress and info.frame_count:
                progress(min(idx / info.frame_count, 1.0))

            yield FrameResult(
                frame_index=idx,
                timestamp_s=round(ts, 3),
                detections=[
                    {
                        "box": [round(v, 1) for v in d.box],
                        "confidence": round(d.confidence, 4),
                        "label": d.label,
                    }
                    for d in detections
                ],
                vision_confidence=round(vision_conf, 4),
                audio_confidence=round(audio_conf, 4),
                fused_confidence=round(fusion.fused_confidence, 4),
                distance_m=round(kin.distance_m, 1) if kin else None,
                speed_kmph=round(kin.speed_kmph, 1) if kin and kin.speed_kmph else None,
                eta_s=round(kin.eta_s, 2) if kin and kin.eta_s else None,
                direction_deg=(
                    direction.angle_deg if direction and direction.available else None
                ),
                direction_bearing=direction.bearing if direction else "unknown",
                direction_available=bool(direction and direction.available),
                signal_state=event.signal_state.value,
                decision=event.decision.value,
                reason=event.reason,
                latency_ms=round(latency_ms, 2),
            )

    def summarise(self) -> AnalysisSummary:
        """Aggregate the run that `stream` just completed."""
        info: VideoInfo = self._info
        audio: AudioTimeline = self._audio_timeline
        track: AudioTrack | None = self._audio_track

        lat = np.asarray(self._latencies) if self._latencies else np.zeros(1)
        granted = [e for e in self._events if e.decision == Decision.GRANTED]

        blocked: dict[str, int] = {}
        for e in self._events:
            if e.decision.value.startswith("blocked"):
                blocked[e.decision.value] = blocked.get(e.decision.value, 0) + 1

        grant_t = granted[0].timestamp_s if granted else None
        n = max(self._frames, 1)

        return AnalysisSummary(
            video=info.path.name,
            duration_s=round(info.duration_s, 2),
            fps=round(info.fps, 2),
            resolution=f"{info.width}x{info.height}",
            frames_analysed=self._frames,
            detection_frames=self._detection_frames,
            peak_vision_confidence=round(self._peak_vision, 4),
            peak_siren_confidence=round(
                max(audio.siren_confidence, default=0.0), 4
            ),
            peak_fused_confidence=round(
                max((e.fused_confidence for e in self._events), default=0.0), 4
            ),
            preemption_granted=bool(granted),
            # `is not None`, not truthiness: a first detection at t=0.0 is
            # falsy and would be reported as "never detected".
            first_detection_s=(
                round(self._first_detection, 2)
                if self._first_detection is not None
                else None
            ),
            grant_time_s=round(grant_t, 2) if grant_t is not None else None,
            response_latency_s=(
                round(grant_t - self._first_detection, 2)
                if grant_t is not None and self._first_detection is not None
                else None
            ),
            audio_available=audio.available,
            audio_stereo=bool(track and track.is_stereo),
            audio_note=audio.note,
            mean_latency_ms=round(float(lat.mean()), 2),
            p95_latency_ms=round(float(np.percentile(lat, 95)), 2),
            stage_latency_ms={k: round(v / n, 2) for k, v in self._stage_ms.items()},
            blocked_reasons=blocked,
        )

    def analyse(self, video_path: str | Path, **kw) -> dict:
        """Convenience wrapper: run to completion and return a JSON-ready dict."""
        frames = [asdict(f) for f in self.stream(video_path, **kw)]
        return {"summary": asdict(self.summarise()), "frames": frames}

"""
Perspective-aware distance, speed and ETA estimation.

Implements Paper §III-D. Uses the pinhole camera model to recover metric
distance from the apparent width of a detected ambulance:

    f / Z = w / W        =>      Z = f * W / w

    v   = (Z_t+1 - Z_t) / dt
    ETA = Z / v

where f is focal length in pixels, W the true vehicle width (1.9 m), and
w the bounding-box width in pixels.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from app.config import settings


@dataclass
class KinematicState:
    """A single resolved kinematic estimate for one tracked vehicle."""

    distance_m: float
    speed_mps: float | None
    eta_s: float | None
    closing: bool

    @property
    def speed_kmph(self) -> float | None:
        return None if self.speed_mps is None else self.speed_mps * 3.6


def distance_from_box_width(box_width_px: float, focal_px: float) -> float:
    """
    Recover distance Z from apparent width using the pinhole model.

    Paper section III-D, equations (4)/(5).

    `focal_px` is supplied by the caller rather than derived from the frame
    width here, because focal length is a property of the *lens*, not the
    sensor resolution. Assuming a fixed 60-degree field of view for every clip
    puts a telephoto shot an order of magnitude too close — a vehicle 12 m away
    reads as 1.7 m. Per-clip calibration lives in the demo manifest.
    """
    if box_width_px <= 0:
        return float("inf")
    return (focal_px * settings.AMBULANCE_WIDTH_M) / box_width_px


class KinematicTracker:
    """
    Maintains a short history of distance samples for one track and derives
    speed and ETA by least-squares fitting dZ/dt.

    A regression over a window is used instead of a raw two-point difference
    because bounding-box width is noisy; the paper reports +-1.8 s ETA error,
    which naive differencing cannot reach.
    """

    WINDOW = 12  # samples retained for the velocity fit

    def __init__(self) -> None:
        self._samples: deque[tuple[float, float]] = deque(maxlen=self.WINDOW)

    def update(self, timestamp_s: float, distance_m: float) -> KinematicState:
        self._samples.append((timestamp_s, distance_m))

        speed = self._fit_speed()
        closing = speed is not None and speed > 0.5  # m/s, approaching camera

        eta: float | None = None
        if closing and speed:
            eta = max(distance_m / speed, 0.0)

        return KinematicState(
            distance_m=distance_m,
            speed_mps=speed,
            eta_s=eta,
            closing=closing,
        )

    def _fit_speed(self) -> float | None:
        """
        Least-squares slope of distance against time, sign-flipped so that a
        vehicle approaching the camera yields a positive closing speed.
        """
        if len(self._samples) < 3:
            return None

        ts = [t for t, _ in self._samples]
        zs = [z for _, z in self._samples]
        n = len(ts)

        mean_t = sum(ts) / n
        mean_z = sum(zs) / n

        denom = sum((t - mean_t) ** 2 for t in ts)
        if denom < 1e-9:
            return None

        slope = sum((t - mean_t) * (z - mean_z) for t, z in zip(ts, zs)) / denom
        return -slope  # closing speed is positive as distance decreases


@dataclass
class Track:
    """One tracked ambulance across frames."""

    track_id: int
    kinematics: KinematicTracker = field(default_factory=KinematicTracker)
    last_seen_s: float = 0.0
    last_box: tuple[float, float, float, float] = (0, 0, 0, 0)
    peak_confidence: float = 0.0
    state: KinematicState | None = None


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """Intersection over union of two xyxy boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class BoxTracker:
    """
    Minimal IoU-greedy multi-object tracker.

    Sufficient for the intersection scenario (few simultaneous ambulances,
    high frame rate) and adds no third-party dependency. Its only job is to
    give distance samples a stable identity so speed/ETA can be fitted.
    """

    def __init__(self, iou_threshold: float = 0.3, max_age_s: float = 1.5) -> None:
        self.iou_threshold = iou_threshold
        self.max_age_s = max_age_s
        self._tracks: dict[int, Track] = {}
        self._next_id = 1

    @property
    def tracks(self) -> dict[int, Track]:
        return self._tracks

    def update(
        self,
        detections: list[tuple[tuple[float, float, float, float], float]],
        timestamp_s: float,
        focal_px: float,
    ) -> list[Track]:
        """Associate detections to tracks and refresh their kinematics."""
        self._retire_stale(timestamp_s)

        unmatched = list(range(len(detections)))
        assigned: list[Track] = []

        # Greedy association, strongest overlap first.
        for track in sorted(
            self._tracks.values(), key=lambda t: t.peak_confidence, reverse=True
        ):
            best_idx, best_iou = None, self.iou_threshold
            for idx in unmatched:
                score = iou(track.last_box, detections[idx][0])
                if score >= best_iou:
                    best_idx, best_iou = idx, score

            if best_idx is not None:
                unmatched.remove(best_idx)
                assigned.append(self._refresh(track, detections[best_idx], timestamp_s, focal_px))

        # Anything left over starts a new track.
        for idx in unmatched:
            track = Track(track_id=self._next_id)
            self._next_id += 1
            self._tracks[track.track_id] = track
            assigned.append(self._refresh(track, detections[idx], timestamp_s, focal_px))

        return assigned

    def _refresh(
        self,
        track: Track,
        detection: tuple[tuple[float, float, float, float], float],
        timestamp_s: float,
        focal_px: float,
    ) -> Track:
        box, confidence = detection
        width_px = box[2] - box[0]

        track.last_box = box
        track.last_seen_s = timestamp_s
        track.peak_confidence = max(track.peak_confidence, confidence)
        track.state = track.kinematics.update(
            timestamp_s, distance_from_box_width(width_px, focal_px)
        )
        return track

    def _retire_stale(self, now_s: float) -> None:
        stale = [
            tid
            for tid, t in self._tracks.items()
            if now_s - t.last_seen_s > self.max_age_s
        ]
        for tid in stale:
            del self._tracks[tid]

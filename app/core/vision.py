"""
Visual ambulance detection.

Implements Paper section III-A. Wraps the fine-tuned YOLOv8 checkpoint
(single class: `ambulance`) and applies the publication's confidence gate.

The detector is loaded once and reused; Ultralytics keeps a warm inference
session, and reloading per request dominates latency on CPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.config import settings


@dataclass
class Detection:
    """One ambulance detection in one frame."""

    box: tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    label: str = "ambulance"

    @property
    def width_px(self) -> float:
        return self.box[2] - self.box[0]

    @property
    def centre(self) -> tuple[float, float]:
        return ((self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2)


class AmbulanceDetector:
    """YOLOv8 wrapper with lazy model loading."""

    def __init__(self, weights: Path | None = None, conf: float | None = None) -> None:
        self.weights = Path(weights or settings.YOLO_WEIGHTS)
        if not self.weights.exists():
            raise FileNotFoundError(f"YOLO weights not found: {self.weights}")
        # Gate applied when reporting; the raw model runs slightly lower so
        # near-threshold boxes are still available for the fusion stage.
        self.conf = conf if conf is not None else settings.VISION_CONF_THRESHOLD
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(str(self.weights))
        return self._model

    @property
    def class_names(self) -> dict[int, str]:
        return self.model.names

    def detect(self, frame: np.ndarray, conf: float | None = None) -> list[Detection]:
        """Run detection on a single BGR frame."""
        threshold = self.conf if conf is None else conf
        result = self.model.predict(
            frame,
            conf=threshold,
            imgsz=settings.YOLO_IMAGE_SIZE,
            verbose=False,
        )[0]

        names = self.model.names
        out: list[Detection] = []
        for box, score, cls in zip(
            result.boxes.xyxy.tolist(),
            result.boxes.conf.tolist(),
            result.boxes.cls.tolist(),
        ):
            out.append(
                Detection(
                    box=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                    confidence=float(score),
                    label=names.get(int(cls), str(int(cls))),
                )
            )
        out.sort(key=lambda d: d.confidence, reverse=True)
        return out

    def warmup(self, width: int = 640, height: int = 640) -> None:
        """Run one throwaway inference so the first real frame is not slow."""
        self.detect(np.zeros((height, width, 3), dtype=np.uint8), conf=0.99)

"""
Siren Eyes — system configuration.

Every constant here is traceable to the reference paper:

    "Stereo-Aware Multimodal Ambulance Detection and Real-Time Traffic
     Signal Preemption Using Edge AI"
    Dalal, Gupta & Sharma — USAR, GGSIPU

Section references are given inline so the implementation can be audited
against the publication.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
DEMO_DIR = ROOT_DIR / "demo"
STATIC_DIR = ROOT_DIR / "web" / "dist"
RUNS_DIR = ROOT_DIR / "runs"


class Settings(BaseSettings):
    """Runtime configuration. Every field is overridable via env or .env."""

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="SIREN_", extra="ignore"
    )

    # ── Application ────────────────────────────────────────────────────
    APP_NAME: str = "Siren Eyes"
    APP_VERSION: str = "2.1.0"
    ENVIRONMENT: str = "development"
    # Build provenance for GET /api/version. Both are optional: the Dockerfile
    # stamps them as build args, Render exposes the commit as RENDER_GIT_COMMIT,
    # and a local checkout falls back to asking git directly.
    GIT_SHA: str | None = None
    BUILD_TIME: str | None = None

    # ── Model artefacts ────────────────────────────────────────────────
    YOLO_WEIGHTS: Path = MODELS_DIR / "best.pt"
    SIREN_WEIGHTS: Path = MODELS_DIR / "ambulance_siren_model.h5"

    # ── Visual detection · Paper §III-A ────────────────────────────────
    # "Detection proceeds if c >= 0.65 and class = 'ambulance'"
    VISION_CONF_THRESHOLD: float = 0.65
    YOLO_IMAGE_SIZE: int = 640
    # Frames are sampled rather than exhaustively decoded. The paper reports
    # YOLOv8 running at 18 FPS on CPU; sampling keeps wall-clock analysis
    # near real time without changing detection semantics.
    FRAME_SAMPLE_STRIDE: int = 3

    # ── Siren classification · Paper §III-B ────────────────────────────
    # "A 5-second audio clip is converted to a 128x128 mel-spectrogram
    #  using a Hann window (window size 2048, hop length 512)"
    AUDIO_SAMPLE_RATE: int = 22_050
    AUDIO_WINDOW_SECONDS: float = 5.0
    AUDIO_HOP_SECONDS: float = 1.0
    MEL_BANDS: int = 128
    MEL_FRAMES: int = 128
    MEL_FMAX: int = 8_000
    STFT_WINDOW: int = 2_048
    STFT_HOP: int = 512
    SIREN_THRESHOLD: float = 0.5

    # ── Directional audio · Paper §III-C ───────────────────────────────
    # Bandpass isolates siren fundamentals before ITD/ILD estimation.
    BANDPASS_LOW_HZ: int = 300
    BANDPASS_HIGH_HZ: int = 3_000
    # "sigma values are empirically set to 15 deg for ILD and 10 deg for ITD"
    SIGMA_ILD_DEG: float = 15.0
    SIGMA_ITD_DEG: float = 10.0
    # "The angle theta is discretized into 5 deg bins from -90 to +90"
    ANGLE_MIN_DEG: float = -90.0
    ANGLE_MAX_DEG: float = 90.0
    ANGLE_BIN_DEG: float = 5.0
    # "If |theta_hat - theta_approach| > 45 deg, preemption is cancelled"
    APPROACH_ANGLE_TOLERANCE_DEG: float = 45.0
    # Microphone baseline (m) — stereo pair spacing at the intersection mast.
    MIC_BASELINE_M: float = 0.18
    SPEED_OF_SOUND_MPS: float = 343.0

    # ── Perspective geometry · Paper §III-D ────────────────────────────
    # "A survey of 30 ambulances in Delhi gave a range 1.75-2.15 m;
    #  a conservative average W = 1.9 m is used."
    AMBULANCE_WIDTH_M: float = 1.9
    # Focal length in pixels. Derived from horizontal FOV when uncalibrated.
    CAMERA_FOCAL_PX: float | None = None
    CAMERA_HFOV_DEG: float = 60.0
    # Distances beyond this are not trusted for ETA (paper notes ILD becomes
    # unreliable past ~80 m and siren range caps near 100 m).
    MAX_TRUSTED_DISTANCE_M: float = 100.0

    # ── Multi-modal fusion · Paper §III-E ──────────────────────────────
    # "Weights alpha = 0.7, beta = 0.3 were optimized via grid search"
    FUSION_ALPHA_VISION: float = 0.7
    FUSION_BETA_AUDIO: float = 0.3
    FUSION_TRIGGER_THRESHOLD: float = 0.60

    # ── Safety-aware preemption · Paper §III-F ─────────────────────────
    # "ETA > T_safe (fixed 5-second buffer ...)"
    SAFETY_BUFFER_S: float = 5.0
    # "No conflicting flow with Time-to-Collision (TTC) < 2 s"
    TTC_THRESHOLD_S: float = 2.0
    # Green is held until the ambulance clears, then timing reverts.
    CLEARANCE_DISTANCE_M: float = 8.0
    MIN_GREEN_HOLD_S: float = 3.0
    AMBER_DURATION_S: float = 3.0
    ALL_RED_DURATION_S: float = 2.0

    # ── API ────────────────────────────────────────────────────────────
    HOST: str = "0.0.0.0"
    PORT: int = 7860  # Hugging Face Spaces default
    ALLOWED_ORIGINS: list[str] = Field(
        default_factory=lambda: ["http://localhost:5173", "http://localhost:4173"]
    )
    MAX_UPLOAD_MB: int = 200
    # Container formats the ffmpeg + OpenCV pair is known to decode. Used for
    # the 415 check on upload and echoed to the UI for client-side validation.
    UPLOAD_EXTENSIONS: list[str] = Field(
        default_factory=lambda: [".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi", ".mpg", ".mpeg"]
    )

    @property
    def angle_bins(self) -> list[float]:
        """Discrete direction hypotheses, in degrees (paper §III-C)."""
        n = int((self.ANGLE_MAX_DEG - self.ANGLE_MIN_DEG) / self.ANGLE_BIN_DEG) + 1
        return [self.ANGLE_MIN_DEG + i * self.ANGLE_BIN_DEG for i in range(n)]

    def focal_px(self, frame_width: int) -> float:
        """
        Focal length in pixels.

        Uses an explicit calibration when supplied, otherwise derives it from
        the horizontal field of view:  f = (W_img / 2) / tan(HFOV / 2)
        """
        if self.CAMERA_FOCAL_PX:
            return self.CAMERA_FOCAL_PX
        import math

        return (frame_width / 2.0) / math.tan(math.radians(self.CAMERA_HFOV_DEG) / 2.0)


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

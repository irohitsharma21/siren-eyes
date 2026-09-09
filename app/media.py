"""
Media ingest: frame iteration and stereo audio extraction.

ffmpeg is invoked through the `imageio-ffmpeg` bundled binary so the project
has no system-level dependency; `moviepy` is deliberately avoided because its
`moviepy.editor` entry point moved in v2 and broke the original code path.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import soundfile as sf

from app.config import settings


@dataclass
class VideoInfo:
    path: Path
    fps: float
    frame_count: int
    width: int
    height: int

    @property
    def duration_s(self) -> float:
        return self.frame_count / self.fps if self.fps else 0.0


@dataclass
class AudioTrack:
    """Decoded stereo (or mono) audio aligned to the video timeline."""

    left: np.ndarray
    right: np.ndarray
    sample_rate: int
    is_stereo: bool
    is_silent: bool

    @property
    def mono(self) -> np.ndarray:
        return ((self.left + self.right) * 0.5).astype(np.float32)

    @property
    def duration_s(self) -> float:
        return len(self.left) / self.sample_rate if self.sample_rate else 0.0

    def window(self, start_s: float, length_s: float) -> tuple[np.ndarray, np.ndarray]:
        a = max(int(start_s * self.sample_rate), 0)
        b = min(a + int(length_s * self.sample_rate), len(self.left))
        return self.left[a:b], self.right[a:b]


def ffmpeg_exe() -> str:
    """Path to a usable ffmpeg binary."""
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def probe_video(path: str | Path) -> VideoInfo:
    path = Path(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    info = VideoInfo(
        path=path,
        fps=cap.get(cv2.CAP_PROP_FPS) or 25.0,
        frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    cap.release()
    return info


def iter_frames(
    path: str | Path, stride: int | None = None
) -> Iterator[tuple[int, float, np.ndarray]]:
    """
    Yield (frame_index, timestamp_seconds, frame_bgr).

    Frames are decoded sequentially and sampled by `stride`; seeking per frame
    is far slower on long clips and can drift on variable-GOP encodes.
    """
    stride = stride or settings.FRAME_SAMPLE_STRIDE
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                yield idx, idx / fps, frame
            idx += 1
    finally:
        cap.release()


def extract_audio(
    path: str | Path, sample_rate: int | None = None
) -> AudioTrack | None:
    """
    Decode the audio track to stereo PCM at the analysis sample rate.

    Returns None when the container carries no audio stream at all. A stream
    that exists but is digitally silent is returned with `is_silent=True` so
    callers can distinguish "no track" from "empty track" and report it.
    """
    sample_rate = sample_rate or settings.AUDIO_SAMPLE_RATE
    tmp = Path(tempfile.mkdtemp(prefix="siren_audio_"))
    wav = tmp / "track.wav"

    cmd = [
        ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(path),
        "-vn", "-ac", "2", "-ar", str(sample_rate),
        "-c:a", "pcm_s16le", str(wav),
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0 or not wav.exists():
            return None

        data, sr = sf.read(wav, always_2d=True, dtype="float32")
        if data.size == 0:
            return None

        left = np.ascontiguousarray(data[:, 0])
        right = np.ascontiguousarray(data[:, 1] if data.shape[1] > 1 else data[:, 0])

        peak = float(np.max(np.abs(data))) if data.size else 0.0
        identical = bool(np.allclose(left, right, atol=1e-6))

        return AudioTrack(
            left=left,
            right=right,
            sample_rate=sr,
            is_stereo=not identical,
            is_silent=peak < 1e-6,
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def mux_audio(video: Path, audio_source: Path, out: Path) -> Path:
    """
    Replace a clip's audio track with the audio from another file.

    Used to build the composited demo fixture when the footage and the usable
    siren recording come from different sources. The provenance of any clip
    produced this way is recorded in the demo manifest.
    """
    cmd = [
        ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(video), "-i", str(audio_source),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-shortest", str(out),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return out


def write_clip(
    src: Path, out: Path, start_s: float, duration_s: float, with_audio: bool = True
) -> Path:
    """Cut a segment out of a source video, re-encoding for frame accuracy."""
    cmd = [
        ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{start_s}", "-i", str(src), "-t", f"{duration_s}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
    ]
    cmd += ["-c:a", "aac", "-b:a", "192k"] if with_audio else ["-an"]
    cmd += [str(out)]
    subprocess.run(cmd, check=True, capture_output=True)
    return out

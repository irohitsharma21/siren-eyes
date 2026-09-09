"""
Siren classification from mel-spectrograms.

Implements Paper section III-B.

Two checkpoints are understood:

  siren_cnn.pt  (preferred)  The retrained classifier. Architecture follows the
                             publication: 32/64/128 convolutions into global
                             average pooling. Trained on ESC-50 with the fold
                             split the dataset ships, so its reported metrics
                             come from audio it never saw. Inputs are scaled
                             to [0, 1].

  ambulance_siren_model.h5   The original Keras artefact, kept for provenance.
  (legacy)                   It does not discriminate: swept across seven input
                             normalisations it scores non-siren audio above
                             siren audio every time (`scripts/diagnose_siren.py`).
                             Loaded only when the retrained checkpoint is
                             absent, and flagged as unreliable when it is.

TensorFlow is not a runtime dependency. The legacy weights are read straight
out of the HDF5 container and executed in PyTorch, which is already present
for YOLOv8; `scripts/validate_port.py` pins that port to Keras within 3e-08.
Three conversions make it exact:

  * Keras conv kernels are (kh, kw, in, out); PyTorch wants (out, in, kh, kw).
  * Keras is channels-last (NHWC); PyTorch is channels-first (NCHW).
  * Keras `Flatten` therefore serialises H-W-C, so the tensor is permuted back
    to NHWC before flattening; otherwise the 12544-wide dense kernel is applied
    to a permuted vector and the output is meaningless.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import librosa
import numpy as np
import torch
import torch.nn as nn

from app.config import MODELS_DIR, settings

# The legacy checkpoint was trained on 128 mel bands x 128 frames at hop 512
# and sr 22050, i.e. 128 * 512 / 22050 = 2.97 s of audio. The paper quotes a
# nominal 5 s buffer; the framing follows the artefact and the window is slid
# across the 5 s buffer instead.
FRAMES = 128
WINDOW_SAMPLES = FRAMES * 512

RETRAINED_PATH = MODELS_DIR / "siren_cnn.pt"


@dataclass
class SirenPrediction:
    """Siren classifier output for one analysis window."""

    confidence: float
    is_siren: bool
    start_s: float
    end_s: float


# ── architectures ─────────────────────────────────────────────────────
class SirenCNN(nn.Module):
    """PyTorch transcription of the legacy Keras classifier."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(12544, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.permute(0, 2, 3, 1).contiguous()  # NHWC, to match Keras flatten
        x = x.flatten(1)
        x = torch.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))


class SirenNet(nn.Module):
    """
    Retrained classifier, following the publication's section III-B stack.

    Global average pooling replaces the legacy 12544-wide flatten. With only
    40 positive source clips available, a dense layer that size memorises the
    training set; GAP cuts parameters ~50x and is what makes the held-out
    score meaningful. Returns logits — callers apply the sigmoid.
    """

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def _load_weights(model: SirenCNN, checkpoint: Path) -> SirenCNN:
    """Transfer legacy Keras HDF5 weights into the PyTorch module."""
    with h5py.File(checkpoint, "r") as f:
        root = f["model_weights"]

        def get(layer: str, kind: str) -> np.ndarray:
            return np.asarray(root[layer]["sequential"][layer][kind])

        with torch.no_grad():
            for name, module in (
                ("conv2d", model.conv1),
                ("conv2d_1", model.conv2),
                ("conv2d_2", model.conv3),
            ):
                kernel = get(name, "kernel")           # (kh, kw, in, out)
                module.weight.copy_(torch.from_numpy(kernel.transpose(3, 2, 0, 1).copy()))
                module.bias.copy_(torch.from_numpy(get(name, "bias").copy()))

            for name, module in (("dense", model.fc1), ("dense_1", model.fc2)):
                kernel = get(name, "kernel")           # (in, out)
                module.weight.copy_(torch.from_numpy(kernel.transpose(1, 0).copy()))
                module.bias.copy_(torch.from_numpy(get(name, "bias").copy()))

    return model.eval()


# ── classifier ────────────────────────────────────────────────────────
class SirenClassifier:
    """Loads the best available checkpoint and scores audio windows."""

    def __init__(self, checkpoint: Path | None = None) -> None:
        if checkpoint is not None:
            self._load_explicit(Path(checkpoint))
        elif RETRAINED_PATH.exists():
            self._load_retrained(RETRAINED_PATH)
        elif Path(settings.SIREN_WEIGHTS).exists():
            self._load_legacy(Path(settings.SIREN_WEIGHTS))
        else:
            raise FileNotFoundError(
                f"No siren checkpoint found. Expected {RETRAINED_PATH} "
                f"or {settings.SIREN_WEIGHTS}."
            )

    def _load_explicit(self, path: Path) -> None:
        if path.suffix == ".pt":
            self._load_retrained(path)
        else:
            self._load_legacy(path)

    def _load_retrained(self, path: Path) -> None:
        blob = torch.load(path, map_location="cpu", weights_only=False)
        model = SirenNet()
        model.load_state_dict(blob["state_dict"])

        self.model = model.eval()
        self.kind = "retrained"
        self.reliable = True
        self.threshold = float(blob.get("threshold", settings.SIREN_THRESHOLD))
        self.normalise = lambda d: (d + 80.0) / 80.0
        self.applies_sigmoid = True
        self.metrics = blob.get("test_metrics", {})
        self.checkpoint = path.name

    def _load_legacy(self, path: Path) -> None:
        self.model = _load_weights(SirenCNN(), path)
        self.kind = "legacy"
        self.reliable = False
        self.threshold = settings.SIREN_THRESHOLD
        self.normalise = lambda d: d          # raw dB, as originally fed
        self.applies_sigmoid = False           # the module already sigmoids
        self.metrics = {}
        self.checkpoint = path.name
        print(
            "[siren] WARNING: using the legacy checkpoint, which does not "
            "discriminate siren from non-siren audio. Run "
            "scripts/train_siren.py to produce models/siren_cnn.pt."
        )

    @classmethod
    def describe(cls) -> dict:
        """Static description of what would be loaded, for /api/health."""
        if RETRAINED_PATH.exists():
            blob = torch.load(RETRAINED_PATH, map_location="cpu", weights_only=False)
            return {
                "checkpoint": RETRAINED_PATH.name,
                "kind": "retrained",
                "reliable": True,
                "threshold": round(float(blob.get("threshold", 0.5)), 3),
                "test_metrics": {
                    k: round(v, 4)
                    for k, v in blob.get("test_metrics", {}).items()
                    if isinstance(v, (int, float))
                },
            }
        return {
            "checkpoint": Path(settings.SIREN_WEIGHTS).name,
            "kind": "legacy",
            "reliable": False,
            "note": "legacy checkpoint does not discriminate; retrain required",
        }

    # ── features ──────────────────────────────────────────────────────
    @staticmethod
    def melspectrogram(samples: np.ndarray, sr: int) -> np.ndarray:
        """
        Log-mel spectrogram, 128 bands x 128 frames.

        librosa defaults (n_fft=2048, hop_length=512) are retained because both
        checkpoints were trained against them.
        """
        mel = librosa.feature.melspectrogram(
            y=samples,
            sr=sr,
            n_mels=settings.MEL_BANDS,
            fmax=settings.MEL_FMAX,
            n_fft=settings.STFT_WINDOW,
            hop_length=settings.STFT_HOP,
        )
        log_mel = librosa.power_to_db(mel, ref=np.max)

        frames = log_mel.shape[1]
        if frames < FRAMES:
            log_mel = np.pad(
                log_mel, ((0, 0), (0, FRAMES - frames)),
                mode="constant", constant_values=log_mel.min(),
            )
        elif frames > FRAMES:
            log_mel = log_mel[:, :FRAMES]

        return log_mel.astype(np.float32)

    def predict_window(self, samples: np.ndarray, sr: int) -> float:
        """Siren probability for a single audio window."""
        if samples.size == 0:
            return 0.0

        spec = self.normalise(self.melspectrogram(samples, sr))
        tensor = torch.from_numpy(np.ascontiguousarray(spec, dtype=np.float32))[None, None]

        with torch.no_grad():
            out = self.model(tensor)
            if self.applies_sigmoid:
                out = torch.sigmoid(out)
            return float(out.item())

    def scan(
        self,
        samples: np.ndarray,
        sr: int,
        hop_s: float | None = None,
    ) -> list[SirenPrediction]:
        """Slide the classifier across a recording, one prediction per hop."""
        hop_s = hop_s or settings.AUDIO_HOP_SECONDS
        hop_samples = max(int(hop_s * sr), 1)
        window = int(WINDOW_SAMPLES * (sr / 22_050))

        if samples.size == 0:
            return []

        results: list[SirenPrediction] = []
        for start in range(0, max(len(samples) - window, 0) + 1, hop_samples):
            chunk = samples[start : start + window]
            if chunk.size < window // 2:
                break
            confidence = self.predict_window(chunk, sr)
            results.append(
                SirenPrediction(
                    confidence=confidence,
                    is_siren=confidence >= self.threshold,
                    start_s=start / sr,
                    end_s=(start + len(chunk)) / sr,
                )
            )
        return results

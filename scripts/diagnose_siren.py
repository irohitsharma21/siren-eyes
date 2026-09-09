"""
Diagnose the shipped siren checkpoint.

The original inference path fed raw `power_to_db` output (roughly -80..0 dB)
straight into the network. If training used a different normalisation, every
inference ever run against this checkpoint was wrong. This sweeps the
plausible normalisations and reports which, if any, separates siren audio
from non-siren audio.
"""

import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.core.audio import SirenCNN, _load_weights  # noqa: E402

CKPT = ROOT / "models" / "ambulance_siren_model.h5"
SR = 22050
model = _load_weights(SirenCNN(), CKPT)


def logmel(y: np.ndarray) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_mels=128, fmax=8000, n_fft=2048, hop_length=512
    )
    d = librosa.power_to_db(mel, ref=np.max)
    if d.shape[1] < 128:
        d = np.pad(d, ((0, 0), (0, 128 - d.shape[1])), constant_values=d.min())
    return d[:, :128].astype(np.float32)


NORMS = {
    "raw_db (original code)": lambda d: d,
    "(db+80)/80 -> [0,1]": lambda d: (d + 80.0) / 80.0,
    "db/80 -> [-1,0]": lambda d: d / 80.0,
    "minmax per-sample": lambda d: (d - d.min()) / (np.ptp(d) + 1e-9),
    "standardised": lambda d: (d - d.mean()) / (d.std() + 1e-9),
    "abs(db)/80": lambda d: np.abs(d) / 80.0,
    "1 - (db+80)/80 (inverted)": lambda d: 1.0 - (d + 80.0) / 80.0,
}


def score(d: np.ndarray, fn) -> float:
    x = torch.from_numpy(np.ascontiguousarray(fn(d), dtype=np.float32))[None, None]
    with torch.no_grad():
        return float(model(x).item())


def clips() -> dict[str, np.ndarray]:
    """Positive and negative fixtures."""
    out: dict[str, np.ndarray] = {}

    amb = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if amb and amb.exists():
        y, sr = sf.read(amb, always_2d=True)
        y = librosa.resample(y.mean(axis=1).astype(np.float32), orig_sr=sr, target_sr=SR)
        n = 128 * 512
        for i, s in enumerate(range(0, max(len(y) - n, 0) + 1, n)):
            out[f"REAL ambulance clip #{i}"] = y[s : s + n]

    t = np.linspace(0, 2.97, 128 * 512, endpoint=False)
    # Classic two-tone wail sweeping 700-1600 Hz.
    wail = 0.6 * np.sin(2 * np.pi * (1150 + 450 * np.sin(2 * np.pi * 0.4 * t)) * t)
    out["synthetic wail 700-1600Hz"] = wail.astype(np.float32)
    # Yelp: faster sweep.
    out["synthetic yelp"] = (
        0.6 * np.sin(2 * np.pi * (1150 + 450 * np.sin(2 * np.pi * 3.0 * t)) * t)
    ).astype(np.float32)
    out["white noise (negative)"] = (0.1 * np.random.default_rng(0).standard_normal(len(t))).astype(np.float32)
    out["440Hz tone (negative)"] = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    out["silence (negative)"] = np.zeros(len(t), dtype=np.float32)
    return out


fixtures = clips()
names = list(NORMS)

print(f"{'clip':<32}" + "".join(f"{n[:20]:>22}" for n in names))
print("-" * (32 + 22 * len(names)))
rows = {}
for label, y in fixtures.items():
    d = logmel(y)
    vals = [score(d, NORMS[n]) for n in names]
    rows[label] = vals
    print(f"{label:<32}" + "".join(f"{v:>22.4f}" for v in vals))

print()
print("separation (mean REAL+synthetic siren  minus  mean negatives):")
pos = [k for k in rows if "REAL" in k or "synthetic w" in k or "yelp" in k]
neg = [k for k in rows if "negative" in k]
for i, n in enumerate(names):
    p = np.mean([rows[k][i] for k in pos]) if pos else float("nan")
    q = np.mean([rows[k][i] for k in neg]) if neg else float("nan")
    print(f"  {n:<28} pos={p:.4f}  neg={q:.4f}  delta={p - q:+.4f}")

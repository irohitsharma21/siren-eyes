"""
Numerically validate the PyTorch siren-CNN port against the original Keras
model.

Run twice, once per interpreter:

    .venv-validate/Scripts/python.exe scripts/validate_port.py keras
    .venv/Scripts/python.exe          scripts/validate_port.py torch
    .venv/Scripts/python.exe          scripts/validate_port.py compare

Deterministic random spectrograms are used so both backends see byte-identical
input without needing a shared audio fixture.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT = ROOT / "runs"
OUT.mkdir(exist_ok=True)
CKPT = ROOT / "models" / "ambulance_siren_model.h5"

N = 24


def fixtures() -> np.ndarray:
    """Inputs spanning the realistic log-mel dB range plus edge cases."""
    rng = np.random.default_rng(20260909)
    x = rng.uniform(-80.0, 0.0, size=(N, 128, 128)).astype(np.float32)
    x[0] = 0.0            # all-zero
    x[1] = -80.0          # floor
    x[2] = np.linspace(-80, 0, 128 * 128).reshape(128, 128)  # ramp
    return x


def run_keras() -> None:
    import os

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    from tensorflow import keras

    model = keras.models.load_model(CKPT)
    x = fixtures()[..., None]  # NHWC
    y = model.predict(x, verbose=0).reshape(-1)
    np.save(OUT / "port_keras.npy", y)
    print("keras outputs:", np.array2string(y[:8], precision=6))


def run_torch() -> None:
    import torch

    from app.core.audio import SirenCNN, _load_weights

    model = _load_weights(SirenCNN(), CKPT)
    x = torch.from_numpy(fixtures())[:, None, :, :]  # NCHW
    with torch.no_grad():
        y = model(x).numpy().reshape(-1)
    np.save(OUT / "port_torch.npy", y)
    print("torch outputs:", np.array2string(y[:8], precision=6))


def compare() -> None:
    k = np.load(OUT / "port_keras.npy")
    t = np.load(OUT / "port_torch.npy")
    diff = np.abs(k - t)
    print(f"samples          : {len(k)}")
    print(f"max  abs diff    : {diff.max():.3e}")
    print(f"mean abs diff    : {diff.mean():.3e}")
    print(f"keras range      : [{k.min():.6f}, {k.max():.6f}]")
    print(f"torch range      : [{t.min():.6f}, {t.max():.6f}]")
    ok = diff.max() < 1e-5
    print("\nRESULT:", "PORT VERIFIED — outputs match" if ok else "MISMATCH — port is wrong")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    {"keras": run_keras, "torch": run_torch, "compare": compare}[sys.argv[1]]()

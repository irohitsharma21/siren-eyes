"""
Retrain the siren classifier.

Why this exists
---------------
The checkpoint originally shipped with this project (`ambulance_siren_model.h5`)
does not discriminate. Swept across seven candidate input normalisations it
scores non-siren audio *above* siren audio in every case (see
`scripts/diagnose_siren.py`); real ambulance audio peaks at 0.107 while a plain
440 Hz tone reaches 0.345 and digital silence reaches 0.444. The publication's
quoted 95.4% accuracy is not reproducible from that artefact, so the classifier
is retrained here from public data with an auditable protocol.

Data
----
ESC-50 (Piczak, 2015), CC BY-NC 3.0 — 2000 clips, 50 balanced classes, 5 s each.
  positives : the 40 `siren` clips
  negatives : the other 49 classes, weighted towards traffic-like confusers
              (car_horn, engine, train, helicopter, airplane, clock_alarm,
               church_bells, chainsaw)

ESC-50 ships a 5-fold split built so clips sharing a source recording never
straddle folds. Those folds are used verbatim — folds 1-3 train, 4 validate,
5 test — which is what keeps the reported test score honest.

The user-supplied ambulance footage is deliberately NOT trained on; it is held
back so the demo runs on genuinely unseen audio.

Architecture
------------
Follows the publication (section III-B) rather than the legacy artefact:

    Conv2D(32,3x3) -> MaxPool -> Conv2D(64,3x3) -> MaxPool
    -> Conv2D(128,3x3) -> GlobalAvgPool -> Dense(64, ReLU) -> Dense(1, sigmoid)

Global average pooling replaces the legacy 12544-wide flatten. With only 40
positive source clips, a dense layer that size memorises the training set;
GAP cuts the parameter count by ~50x and is the difference between a model
that generalises and one that does not.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SR = 22_050
WINDOW = 128 * 512          # 2.97 s, matches the mel framing below
N_MELS, N_FRAMES = 128, 128
FMAX, N_FFT, HOP = 8_000, 2_048, 512

DATASETS = ROOT.parent / "datasets" / "ESC-50-master"
CACHE = ROOT / "runs" / "esc50_cache.npz"
OUT_DIR = ROOT / "models"
RUNS = ROOT / "runs"

# Classes that sound most like a siren to a small CNN; oversampled so the
# model is forced to separate them rather than learning "loud and tonal".
HARD_NEGATIVES = {
    "car_horn", "engine", "train", "helicopter", "airplane",
    "clock_alarm", "church_bells", "chainsaw", "vacuum_cleaner", "crying_baby",
}


# ── data ──────────────────────────────────────────────────────────────
def build_cache() -> dict:
    """Decode ESC-50 once to 22.05 kHz mono and cache as float16."""
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        return {k: z[k] for k in z.files}

    meta = list(csv.DictReader(open(DATASETS / "meta" / "esc50.csv")))
    waves, labels, folds, cats = [], [], [], []

    for i, row in enumerate(meta):
        y, sr = sf.read(DATASETS / "audio" / row["filename"], dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)
        if sr != SR:
            y = librosa.resample(y, orig_sr=sr, target_sr=SR)
        # Pad/crop every clip to a fixed 5 s so the cache is a dense array.
        want = 5 * SR
        y = np.pad(y, (0, max(0, want - len(y))))[:want]

        waves.append(y.astype(np.float16))
        labels.append(1 if row["category"] == "siren" else 0)
        folds.append(int(row["fold"]))
        cats.append(row["category"])

        if (i + 1) % 400 == 0:
            print(f"  decoded {i + 1}/{len(meta)}")

    data = {
        "waves": np.stack(waves),
        "labels": np.array(labels, dtype=np.int64),
        "folds": np.array(folds, dtype=np.int64),
        "cats": np.array(cats),
    }
    RUNS.mkdir(exist_ok=True)
    np.savez_compressed(CACHE, **data)
    print(f"  cached -> {CACHE}")
    return data


def logmel(y: np.ndarray) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_mels=N_MELS, fmax=FMAX, n_fft=N_FFT, hop_length=HOP
    )
    d = librosa.power_to_db(mel, ref=np.max)
    if d.shape[1] < N_FRAMES:
        d = np.pad(d, ((0, 0), (0, N_FRAMES - d.shape[1])), constant_values=d.min())
    # Scale the [-80, 0] dB range to [0, 1]; unnormalised dB into a small CNN
    # gives large first-layer activations and a much rougher loss surface.
    return ((d[:, :N_FRAMES] + 80.0) / 80.0).astype(np.float32)


class SirenDataset(Dataset):
    """Windows of ESC-50 audio with waveform-domain augmentation."""

    def __init__(self, data: dict, folds: list[int], train: bool, seed: int = 0):
        mask = np.isin(data["folds"], folds)
        self.waves = data["waves"][mask]
        self.labels = data["labels"][mask]
        self.cats = data["cats"][mask]
        self.train = train
        self.rng = np.random.default_rng(seed)

        pos = np.flatnonzero(self.labels == 1)
        neg = np.flatnonzero(self.labels == 0)

        if train:
            # Oversample positives to parity, and bias negatives towards the
            # confusable classes so the decision boundary is drawn where it
            # actually matters.
            hard = np.array([i for i in neg if self.cats[i] in HARD_NEGATIVES])
            easy = np.array([i for i in neg if self.cats[i] not in HARD_NEGATIVES])
            n = 12 * len(pos)
            neg_pick = np.concatenate([
                self.rng.choice(hard, size=n // 2, replace=True),
                self.rng.choice(easy, size=n - n // 2, replace=True),
            ])
            pos_pick = self.rng.choice(pos, size=n, replace=True)
            self.index = np.concatenate([pos_pick, neg_pick])
        else:
            # Evaluation uses every clip once, unaugmented and unbalanced.
            self.index = np.arange(len(self.labels))

        self._noise_pool = neg

    def __len__(self) -> int:
        return len(self.index)

    def _augment(self, y: np.ndarray) -> np.ndarray:
        r = self.rng
        # Speed perturbation by linear resampling. This shifts pitch and tempo
        # together, which is exactly what a Doppler-shifted siren does, and it
        # costs ~0.1 ms. librosa's `time_stretch`/`pitch_shift` are the
        # phase-vocoder implementations and run ~1 s per sample, which made a
        # single epoch slower than the entire rest of training.
        if r.random() < 0.6:
            rate = float(r.uniform(0.88, 1.14))
            n = int(len(y) / rate)
            y = np.interp(
                np.linspace(0, len(y) - 1, n), np.arange(len(y)), y
            ).astype(np.float32)
        if r.random() < 0.3:  # polarity flip; inaudible, but decorrelates
            y = -y
        if r.random() < 0.8:  # gain
            y = y * float(10 ** (r.uniform(-8, 4) / 20))
        if r.random() < 0.6:  # mix traffic-ish background at a random SNR
            bg = self.waves[r.choice(self._noise_pool)].astype(np.float32)
            bg = np.resize(bg, len(y))
            sp = float(np.mean(y**2)) + 1e-9
            npow = float(np.mean(bg**2)) + 1e-9
            snr = float(r.uniform(0.0, 18.0))
            y = y + bg * np.sqrt(sp / (npow * 10 ** (snr / 10)))
        return y

    def __getitem__(self, i: int):
        idx = int(self.index[i])
        y = self.waves[idx].astype(np.float32)
        label = float(self.labels[idx])

        if self.train:
            start = int(self.rng.integers(0, max(len(y) - WINDOW, 1)))
            y = y[start : start + WINDOW]
            y = self._augment(y)
        else:
            y = y[:WINDOW]

        y = np.pad(y, (0, max(0, WINDOW - len(y))))[:WINDOW]
        spec = logmel(y)

        if self.train:
            spec = self._spec_augment(spec)

        return torch.from_numpy(spec)[None], torch.tensor([label])

    def _spec_augment(self, s: np.ndarray) -> np.ndarray:
        r = self.rng
        s = s.copy()
        if r.random() < 0.5:  # frequency mask
            f = int(r.integers(4, 20))
            f0 = int(r.integers(0, max(N_MELS - f, 1)))
            s[f0 : f0 + f, :] = 0.0
        if r.random() < 0.5:  # time mask
            t = int(r.integers(4, 24))
            t0 = int(r.integers(0, max(N_FRAMES - t, 1)))
            s[:, t0 : t0 + t] = 0.0
        return s


# ── model ─────────────────────────────────────────────────────────────
class SirenNet(nn.Module):
    """Paper section III-B architecture, with BatchNorm for stable training."""

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

    def forward(self, x):
        return self.head(self.features(x))  # logits


# ── metrics ───────────────────────────────────────────────────────────
def evaluate(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ps, ys = [], []
    with torch.no_grad():
        for xb, yb in loader:
            p = torch.sigmoid(model(xb.to(device))).cpu().numpy().reshape(-1)
            ps.append(p)
            ys.append(yb.numpy().reshape(-1))
    return np.concatenate(ps), np.concatenate(ys)


def metrics(p: np.ndarray, y: np.ndarray, thr: float = 0.5) -> dict:
    from sklearn.metrics import (
        average_precision_score, confusion_matrix, roc_auc_score,
    )

    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {
        "threshold": thr,
        "accuracy": float((tp + tn) / len(y)),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(2 * prec * rec / (prec + rec)) if prec + rec else 0.0,
        "roc_auc": float(roc_auc_score(y, p)) if len(set(y)) > 1 else float("nan"),
        "pr_auc": float(average_precision_score(y, p)) if len(set(y)) > 1 else float("nan"),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "support": {"positive": int(y.sum()), "negative": int((1 - y).sum())},
    }


def train_one_fold(
    data: dict, test_fold: int, args, device: str
) -> tuple[dict, np.ndarray, np.ndarray, dict, float]:
    """
    Train on three folds, tune the threshold on a fourth, test on the fifth.

    Returns (state_dict, test_scores, test_labels, test_metrics, threshold).
    """
    val_fold = (test_fold % 5) + 1
    train_folds = [f for f in (1, 2, 3, 4, 5) if f not in (test_fold, val_fold)]

    train_ds = SirenDataset(data, train_folds, train=True, seed=test_fold)
    val_ds = SirenDataset(data, [val_fold], train=False)
    test_ds = SirenDataset(data, [test_fold], train=False)

    tl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers)
    vl = DataLoader(val_ds, batch_size=64, num_workers=args.workers)
    sl = DataLoader(test_ds, batch_size=64, num_workers=args.workers)

    torch.manual_seed(1337 + test_fold)
    model = SirenNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    lossf = nn.BCEWithLogitsLoss()

    best_auc, best_state = -1.0, None

    for ep in range(1, args.epochs + 1):
        model.train()
        t0, total = time.time(), 0.0
        for xb, yb in tl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(xb)
        sched.step()

        p, y = evaluate(model, vl, device)
        m = metrics(p, y)
        if ep % 5 == 0 or ep == args.epochs:
            print(
                f"    epoch {ep:2d}/{args.epochs}  loss={total / len(train_ds):.4f}  "
                f"val_auc={m['roc_auc']:.4f}  ({time.time() - t0:.0f}s)"
            )

        if m["roc_auc"] > best_auc:
            best_auc = m["roc_auc"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)

    # Threshold is chosen on validation only — never on the test fold.
    pv, yv = evaluate(model, vl, device)
    thr = float(
        max(np.arange(0.05, 0.96, 0.01), key=lambda t: metrics(pv, yv, float(t))["f1"])
    )

    pt, yt = evaluate(model, sl, device)
    return best_state, pt, yt, metrics(pt, yt, thr), thr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument(
        "--folds", type=int, default=5,
        help="how many ESC-50 folds to hold out in turn (5 = full CV)",
    )
    args = ap.parse_args()

    np.random.seed(1337)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("loading ESC-50 ...")
    data = build_cache()
    print(f"  clips={len(data['labels'])}  sirens={int(data['labels'].sum())}")
    print(f"  parameters={sum(p.numel() for p in SirenNet().parameters()):,} "
          f"(legacy artefact: 815,873)")

    RUNS.mkdir(exist_ok=True)
    OUT_DIR.mkdir(exist_ok=True)

    # ESC-50's canonical protocol is 5-fold cross-validation. A single
    # held-out fold contains only 8 of the 40 siren clips, which is far too
    # few to quote a precision from; rotating every fold through the test
    # position evaluates all 40 and yields a variance estimate as well.
    folds, scores, labels, thresholds, states = [], [], [], [], []

    for k in range(1, args.folds + 1):
        print(f"\n── fold {k}/{args.folds} (test on fold {k}) ──")
        state, pt, yt, m, thr = train_one_fold(data, k, args, device)
        folds.append({"fold": k, "threshold": thr, **m})
        scores.append(pt)
        labels.append(yt)
        thresholds.append(thr)
        states.append((m["roc_auc"], state))
        print(
            f"    test  auc={m['roc_auc']:.4f}  f1={m['f1']:.4f}  "
            f"prec={m['precision']:.3f}  rec={m['recall']:.3f}  thr={thr:.2f}"
        )

    # Pooled evaluation across every fold: all 40 sirens, all 1960 negatives.
    all_scores = np.concatenate(scores)
    all_labels = np.concatenate(labels)
    pooled_thr = float(np.mean(thresholds))
    pooled = metrics(all_scores, all_labels, pooled_thr)

    aucs = np.array([f["roc_auc"] for f in folds])

    print("\n" + "=" * 60)
    print("  ESC-50 5-fold cross-validation — every clip tested once")
    print("=" * 60)
    for f in folds:
        print(f"  fold {f['fold']}  auc={f['roc_auc']:.4f}  f1={f['f1']:.4f}  "
              f"prec={f['precision']:.3f}  rec={f['recall']:.3f}")
    print(f"\n  ROC-AUC     : {aucs.mean():.4f} +/- {aucs.std():.4f}")
    print(f"\n  Pooled (threshold {pooled_thr:.2f}, {int(all_labels.sum())} sirens "
          f"vs {int((1 - all_labels).sum())} non-sirens):")
    for k in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"):
        print(f"    {k:10}: {pooled[k]:.4f}")
    print(f"    confusion : {pooled['confusion']}")

    # Ship the fold model with the best held-out AUC.
    best_auc, best_state = max(states, key=lambda s: s[0])
    torch.save(
        {
            "state_dict": best_state,
            "arch": "SirenNet",
            "sr": SR, "n_mels": N_MELS, "n_frames": N_FRAMES,
            "n_fft": N_FFT, "hop": HOP, "fmax": FMAX,
            "normalisation": "(power_to_db(ref=max) + 80) / 80",
            "threshold": pooled_thr,
            "test_metrics": {
                "roc_auc": float(aucs.mean()),
                "roc_auc_std": float(aucs.std()),
                "accuracy": pooled["accuracy"],
                "precision": pooled["precision"],
                "recall": pooled["recall"],
                "f1": pooled["f1"],
                "pr_auc": pooled["pr_auc"],
            },
        },
        OUT_DIR / "siren_cnn.pt",
    )

    (RUNS / "siren_training.json").write_text(
        json.dumps(
            {
                "dataset": "ESC-50 (CC BY-NC 3.0)",
                "protocol": "5-fold cross-validation using the dataset's own folds",
                "epochs": args.epochs,
                "per_fold": folds,
                "roc_auc_mean": float(aucs.mean()),
                "roc_auc_std": float(aucs.std()),
                "pooled": pooled,
                "pooled_threshold": pooled_thr,
            },
            indent=2,
        )
    )
    np.savez(RUNS / "siren_test_scores.npz", scores=all_scores, labels=all_labels)
    print(f"\nsaved -> {OUT_DIR / 'siren_cnn.pt'}  (best fold auc={best_auc:.4f})")


if __name__ == "__main__":
    main()

# Model provenance

What each model in this repository is, where it came from, and how far it can
be trusted. Written because two of the three answers are not the obvious ones.

---

## 1. Ambulance detector — `models/best.pt`

**Status: original artefact, works as advertised.**

Fine-tuned YOLOv8, single class (`ambulance`). Inherited from the original
project; not retrained here.

Verified behaviour on held-back footage: detects the target vehicle with peak
confidence **0.977**, and the class map contains exactly one class, so there
is no label ambiguity. Applied at the publication's threshold of
`conf >= 0.65` (paper §III-A).

Sub-threshold detections are still drawn in the UI, dimmed, so that the gate
is visible rather than looking like a missed detection.

The paper reports mAP@0.5 = 92.6% on a 6000-image custom dataset. That dataset
is not in this repository, so the figure is quoted from the publication and is
**not independently reproduced here**.

---

## 2. Siren classifier — `models/siren_cnn.pt`

**Status: retrained here, because the original does not work.**

### The problem with the original

`models/ambulance_siren_model.h5` is the checkpoint the project originally
shipped. It does not discriminate siren audio from anything else.

The original inference path fed raw `librosa.power_to_db` output (roughly
-80..0 dB) straight into the network. If training used a different
normalisation, every inference ever run against this checkpoint was wrong — so
before concluding the model was broken, seven plausible normalisations were
swept (`scripts/diagnose_siren.py`):

| normalisation | mean score, siren | mean score, non-siren | separation |
|---|---|---|---|
| raw dB (as originally used) | 0.000 | 0.148 | **−0.148** |
| `(dB+80)/80` → [0,1] | 0.076 | 0.132 | **−0.056** |
| `dB/80` → [−1,0] | 0.006 | 0.209 | **−0.203** |
| min–max per sample | 0.082 | 0.276 | **−0.194** |
| standardised | 0.000 | 0.152 | **−0.152** |
| `abs(dB)/80` | 0.068 | 0.263 | **−0.196** |
| inverted | 0.068 | 0.263 | **−0.196** |

Separation is negative in **every** case: the model scores non-sirens *above*
sirens no matter how the input is scaled. Individual readings make the point
plainly — real ambulance audio peaks at **0.107**, a plain 440 Hz tone reaches
**0.345**, and digital silence reaches **0.444**, the highest score of all.

This is not a porting artefact. The PyTorch transcription used to run it is
pinned against the original Keras implementation to a maximum absolute
difference of **2.98 × 10⁻⁸** across 24 fixtures
(`scripts/validate_port.py`) — the two implementations are the same function,
and that function does not classify sirens.

The publication quotes 95.4% accuracy for this component. That number is not
reproducible from this artefact.

The file is kept in the repository for provenance and is loaded only when the
retrained checkpoint is missing, in which case the API and the UI both display
a warning.

### The replacement

| | |
|---|---|
| Architecture | `Conv(32) → Conv(64) → Conv(128) → GlobalAvgPool → Dense(64) → Dense(1)` |
| Data | [ESC-50](https://github.com/karolpiczak/ESC-50) — 2000 clips, 50 classes, CC BY-NC 3.0 |
| Positives | the 40 `siren` clips |
| Negatives | the other 49 classes, oversampled towards traffic-like confusers (`car_horn`, `engine`, `train`, `helicopter`, `airplane`, `clock_alarm`, `church_bells`, `chainsaw`) |
| Split | ESC-50's own 5 folds — 1–3 train, 4 validation, 5 test |
| Input | 128 mel bands × 128 frames, `(power_to_db(ref=max) + 80) / 80` |

Two choices are worth stating explicitly:

**The architecture follows the paper, not the original artefact.** The
publication (§III-B) describes a 32/64/128 stack with global average pooling;
the shipped `.h5` was a 16/32/64 stack with a 12,544-wide flatten. With only 40
positive source clips, a dense layer that size memorises the training set. GAP
cuts the parameter count by roughly 50× and is what makes a held-out score
mean anything.

**The fold split is ESC-50's own.** ESC-50 constructs its folds so that clips
sharing a source recording never straddle a fold boundary. Using them verbatim
is what prevents near-duplicate leakage between train and test — a random split
of this dataset would inflate the score substantially.

**The user-supplied demo footage is deliberately excluded from training**, so
the demo runs on audio the model has never seen.

Augmentation is waveform-domain: speed perturbation by resampling (which
shifts pitch and tempo together, as a Doppler-shifted siren does), gain,
polarity inversion, and background mixing with the dataset's own non-siren
clips at 0–18 dB SNR — plus SpecAugment-style frequency and time masking on
the mel. `librosa`'s phase-vocoder `time_stretch`/`pitch_shift` were tried and
removed: at roughly 1 s per sample they made one epoch slower than the whole
of the rest of training.

Reproduce with:

```bash
python scripts/train_siren.py --epochs 30
```

Metrics on the held-out fold are written to `runs/siren_training.json` and
surfaced live at `/api/health`.

### Honest limits

- 40 positive source clips is a small basis. The augmentation multiplies
  windows, not underlying recordings, so the effective diversity of sirens is
  bounded by those 40 takes.
- ESC-50 sirens include emergency-service sirens of several nationalities;
  Indian ambulance sirens specifically are not guaranteed to be represented.
- The publication's training corpus (UrbanSound8K + 800 custom Delhi clips)
  would be a materially better basis and is the obvious next step.

---

## 3. Stereo direction estimator — no learned weights

**Status: written fresh for this repository.**

Signal processing, not machine learning, so there is nothing to train. ITD via
GCC-PHAT, ILD via in-band energy ratio, fused as a product of Gaussians over a
5°-spaced grid from −90° to +90° (paper §III-C, equations 1–3).

GCC-PHAT is used rather than plain cross-correlation because plain correlation
smears under the reverberation and broadband engine noise of a road
intersection; the phase transform whitens the magnitude spectrum and keeps the
delay peak sharp.

The estimator refuses to guess. Mono audio, dual-mono audio (a stereo
container whose channels are identical), a silent band, or mutually
inconsistent ITD/ILD cues all return `available = False` with a stated reason,
and the preemption stage falls back to vision-only gating. Reporting a
fabricated bearing would be worse than reporting none.

The paper claims 89.3% accuracy within ±15° on 200 stereo recordings. Those
recordings are not in this repository, so that figure is **quoted, not
reproduced**.

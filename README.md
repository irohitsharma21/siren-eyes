---
title: Siren Eyes
emoji: 🚑
colorFrom: red
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
short_description: Multimodal ambulance detection with signal preemption
---

# Siren Eyes

**Stereo-aware multimodal ambulance detection with safety-buffered traffic
signal preemption.**

An ambulance approaching an intersection is detected visually, confirmed
acoustically, located directionally from a stereo microphone pair, and tracked
in metric space — and only then, if a set of explicit safety conditions all
hold, is the signal preempted.

Implements *Stereo-Aware Multimodal Ambulance Detection and Real-Time Traffic
Signal Preemption Using Edge AI* (Dalal, Gupta & Sharma — ICACIS 2026).

---

## The idea

Existing emergency-vehicle preemption is GPS- or RFID-based. Both need
equipment inside the ambulance, both degrade in urban canyons and tunnels, and
neither can tell an ambulance *responding to a call* from one parked outside a
hospital. This system sits on the infrastructure instead: a camera and a
£15 stereo microphone on the signal mast.

The interesting part is not the detection. It is the **refusal to act**:

> Seeing an ambulance is not sufficient reason to change a traffic light.

A green phase granted too late is a phase change on top of moving cross-traffic.
So detection feeds a safety gate, and the gate says no unless the vehicle is
genuinely approaching, is far enough out that the change is not abrupt, is
coming from the direction the signal serves, and no conflicting vehicle is
inside its time-to-collision margin.

---

## Pipeline

```
video ─┬─► YOLOv8 detection ────────────► conf ≥ 0.65 ─┐
       │                                               │
       └─► box width ─► pinhole model ─► distance ──┐  │
                                        speed, ETA  │  ├─► fused = 0.7·V + 0.3·A
audio ─┬─► 128×128 log-mel ─► CNN ─► siren conf ────┼──┘
       │                                            │
       └─► GCC-PHAT (ITD) ─┐                        │
           in-band ILD ────┴─► N(μ,σ²) product ─► bearing θ
                                                    │
                                                    ▼
                                       ┌────────────────────────┐
                                       │  SAFETY GATE           │
                                       │   approaching?         │
                                       │   ETA > 5 s buffer?    │
                                       │   |θ − θ_approach|<45°?│
                                       │   min TTC ≥ 2 s?       │
                                       └───────────┬────────────┘
                                                   ▼
                            preempt green ─► hold until clear ─► amber ─► all-red
```

Every stage is inspectable in the UI as it runs, including the stage that
blocks and why.

| Stage | Implementation | Paper |
|---|---|---|
| Visual detection | YOLOv8, single class, `conf ≥ 0.65` | §III-A |
| Siren confirmation | 128×128 log-mel → CNN | §III-B |
| Direction of arrival | GCC-PHAT ITD + in-band ILD, fused as a Gaussian product over a 5° grid | §III-C |
| Distance / speed / ETA | Pinhole model, `Z = f·W/w`, W = 1.9 m; velocity by least-squares fit | §III-D |
| Fusion | `0.7·vision + 0.3·audio` | §III-E |
| Safety gate | 5 s buffer, TTC ≥ 2 s, ±45° approach axis | §III-F |

---

## Quick start

```bash
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

cd web && npm install && npm run build && cd ..

python -m uvicorn app.main:app --port 7860
```

Open <http://localhost:7860>, pick a bundled demo clip from the app bar and
press **Run** (or press `R`; `?` lists every shortcut). A shareable deep link
starts a clip on load: `/?clip=ambulance_departure.mp4&autorun=1`.

From the repository root, `./run.ps1 start siren` (or `./run.sh start siren`)
does the same in the background.

No API keys. No network calls at inference. No GPU.

---

## What you will see

The dashboard is a single page: an app bar (clip selector, Run/Stop,
parameters, server health, theme), a main stage with the video and its overlay,
and an instrument column that scrolls on its own. On a tablet or phone the
panels stack with the video on top. It streams one record per analysed frame
over a WebSocket and renders:

- **Detection overlay** — corner-bracket reticles on confirmed detections with
  a fading motion trail of the tracked box, a distance / speed / ETA / bearing
  readout beside it, and a bearing dial that follows the fused direction.
  Sub-threshold boxes stay visible but dimmed, so the gate is legible rather
  than looking like a missed detection. A **Compare** toggle shows the raw
  detector output (everything above 0.20) against the gated view. HUD chips
  carry frame index, timestamp, per-frame latency and signal state.
- **Timeline** — three confidence lanes (vision, audio, fused) each with its
  threshold drawn in, a signal-state band, decision markers, and a hover
  tooltip with every value at that instant. Click or drag to seek; `←` `→`
  step a frame (`Shift` for ten); `Space` plays. A decision-event list
  ("first detection 0.60 s", "preemption granted 2.28 s", "blocked: not
  approaching") jumps the playhead.
- **Signal decision, confidence meters, kinematics, direction compass,
  pipeline** — the instrument panels, all driven by the frame under the
  playhead, all stating their own units and thresholds.
- **Results** — verdict, peak confidences, detection-to-grant latency,
  per-stage latency bars, a breakdown of *why preemption was withheld* by gate,
  and buttons to export the run as JSON or per-frame CSV, or copy a plain-text
  summary to the clipboard.
- **Recent runs** — the last twelve runs in this browser, restored across
  reloads; click one to bring its frames and results back.

---

## Features

| | |
|---|---|
| **Stop a run** | `DELETE /api/analyses/{id}` asks the worker to stop between frames; the stream returns a partial summary and a `cancelled` event, and the dashboard keeps what it has. Closing the page does the same, so an abandoned upload does not burn the CPU. |
| **Per-run parameters** | Vision threshold, fused trigger, safety buffer and minimum TTC can be changed for one run from the *Parameters* popover. The server validates the ranges (`app/core/params.py`) and records the values in the summary. A demo clip only replays its recording at the defaults; move any value and it is analysed live, and the UI says so beforehand. |
| **Snapshot** | Renders the current video frame plus overlay at native resolution to a PNG, with a caption strip naming the clip, time, confidences and decision. |
| **Export & share** | JSON (run metadata, summary, every frame), CSV (one row per frame), and a clipboard summary suitable for a message or an issue. |
| **Run history** | Last twelve runs kept in memory and mirrored to `localStorage` under a 3 MB budget (frames are shed from the oldest first). Demo footage reloads from the server; an upload's video lives only as long as the page, and the UI says so. |
| **Upload onboarding** | Drag-and-drop onto the stage or the Source panel, type and size validation with the server's real limits (`GET /api/config` → `upload`), and upload progress. |
| **Live status** | Phase events (`probing video`, `scoring audio`, `analysing frames`) so the seconds before the first frame do not read as a hang; a wake-up notice while a sleeping free instance starts. |
| **Build provenance** | `GET /api/version` reports commit, build time, host and uptime; shown in the footer and in the health chip's tooltip. |
| **Keyboard** | `Space`, `←`/`→`, `Shift`+arrows, `Home`/`End`, `R` run, `Esc` stop, `S` snapshot, `C` compare, `P` parameters, `U` upload, `M` mute, `?` help. |
| **Themes** | Light, dark, or follow the system; no flash of the wrong theme on load. |

---

## Performance

Measured on the bundled clip, CPU only (no GPU), 1920×1080 at 25 fps:

| | |
|---|---|
| Mean end-to-end latency | ~320 ms/frame |
| Dominated by | YOLOv8 inference at full 1080p |
| Audio, direction, geometry, decision | < 1 ms combined |

The paper reports 322 ms; this implementation lands in the same place, though
its budget is distributed differently — the audio window here is scored once
up front and indexed by time rather than recomputed per frame.

`GET /api/health` reports the live model metrics; `GET /api/config` returns
every operating threshold, so the numbers in this README can be checked against
the running system.

---

## Models

Read [`docs/model-provenance.md`](docs/model-provenance.md) before quoting any
number from this project. Summary:

| Model | Status |
|---|---|
| `models/best.pt` — YOLOv8 detector | Original artefact. Works; peak confidence 0.977 on held-back footage. |
| `models/siren_cnn.pt` — siren classifier | **Retrained here.** The original scores silence (0.444) above real ambulance audio (0.107). Replacement: **ROC-AUC 0.9857 ± 0.0082** over ESC-50 5-fold cross-validation; pooled accuracy 0.972, recall 0.800, precision 0.400. |
| Direction estimator | No learned weights; pure signal processing. |

The original Keras checkpoint is retained for provenance and loaded only if the
retrained model is missing, in which case the API and UI both warn.

---

## Honest limitations

Stated here rather than buried, because a demo that overclaims is worse than
one that is modest:

- **The siren classifier is trained on 40 source recordings.** ESC-50 has 40
  `siren` clips. Augmentation multiplies windows, not underlying takes. The
  publication's corpus (UrbanSound8K + 800 custom Delhi recordings) would be a
  materially better basis.
- **Its precision is 0.40, and that number should not be glossed over.** AUC is
  0.986, so ranking is reliable, but at the 0.80-recall operating point it
  flags 48 of 1960 negatives. This is survivable only because fusion means
  audio alone cannot trigger preemption — `0.3 × 1.0 < 0.6`. On its own the
  siren channel is not trustworthy enough to act on.
- **Detector mAP is quoted, not reproduced.** The 6000-image training set is
  not in this repository.
- **Direction accuracy is quoted, not reproduced.** The paper's 200 stereo
  recordings are not in this repository.
- **TTC conflict vehicles are injected, not perceived.** The safety gate
  consumes a list of conflicting vehicles; in the paper these come from SUMO
  simulation. This repository does not detect cross-traffic from the camera, so
  the TTC gate is exercised with supplied values.
- **The bundled footage is street-level, not an intersection approach.** It is
  a close-range pass on a narrow residential road, which is what was available.
  Every stage runs on it, but it is not the deployment geometry.
- **Camera calibration is per-clip and approximate.** Focal length is recovered
  by solving the pinhole model against a hand-measured reference (see
  `camera_hfov_deg` in the demo manifest), not from a checkerboard calibration.
  Absolute distances therefore carry real error; relative motion — which is what
  the safety gate actually uses — is far more robust than the absolute figure.
  Uploaded clips have no calibration at all and fall back to a 60° assumption.
- **The paper's safety gate has no minimum approach speed.** Running the
  bundled clip surfaces this: a vehicle 4.3 m out closing at 1.9 km/h produces
  an ETA of 8.2 s, clears the 5 s buffer, and is granted preemption — the rules
  are satisfied, but a vehicle crawling at walking pace is not an emergency
  approach. The implementation follows the publication rather than silently
  diverging from it; a `MIN_APPROACH_SPEED` gate is the obvious next revision.

---

## Layout

```
app/
  config.py          every constant, annotated with its paper section
  main.py            FastAPI app, WebSocket streaming, static hosting
  jobs.py            in-process analysis job registry, cooperative cancellation
  media.py           ffmpeg-backed video and stereo audio ingest
  core/
    vision.py        YOLOv8 wrapper
    audio.py         mel features + siren CNN (legacy port and retrained net)
    direction.py     GCC-PHAT ITD, in-band ILD, Gaussian-product fusion
    geometry.py      pinhole distance, IoU tracker, least-squares velocity
    params.py        per-run threshold overrides and their bounds
    preemption.py    fusion + the safety state machine
    pipeline.py      orchestration
scripts/
  train_siren.py     retrain the classifier (5-fold CV)
  validate_port.py   pin the PyTorch port against Keras
  diagnose_siren.py  normalisation sweep used to condemn the original model
  build_demos.py     regenerate the demo manifest
web/                 React dashboard
  src/App.tsx        layout, run lifecycle, keyboard shortcuts
  src/components/    VideoStage (overlay + snapshot), Timeline (lanes + events),
                     Results, Parameters, History, Onboarding, Panels
  src/lib/           api client, analysis helpers, export, history, theme
```

---

## API

| | |
|---|---|
| `GET /api/health` | model status, live metrics, build summary |
| `GET /api/version` | commit, build time, host, uptime |
| `GET /api/config` | all operating thresholds, per-run parameter bounds, upload limits |
| `GET /api/demos` | bundled demo clips |
| `POST /api/analyses` | queue an analysis (upload or `?demo=`); optional `vision_threshold`, `fused_threshold`, `safety_buffer_s`, `min_ttc_s` |
| `GET /api/analyses/{id}` | job snapshot |
| `DELETE /api/analyses/{id}` | stop a running analysis |
| `WS /api/analyses/{id}/stream` | `started` → `status`* → `frame`* → `summary` → `done` \| `cancelled` \| `error` |
| `GET /api/docs` | OpenAPI / Swagger |

---

## Licence

Code MIT. The siren classifier is trained on
[ESC-50](https://github.com/karolpiczak/ESC-50) (CC BY-NC 3.0) — **the trained
weights therefore inherit a non-commercial restriction**. Demo footage is
author-supplied and not released under an open licence.

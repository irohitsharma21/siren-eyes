# Deployment

Siren Eyes ships as a single container: FastAPI serves both the API and the
built dashboard, so there is nothing to wire together and no CORS to configure
in production.

---

## Hugging Face Spaces (recommended)

Spaces is the right host for this: the free CPU tier gives 16 GB of RAM and 2
vCPU, which comfortably fits a ~1.5 GB PyTorch image, and it does not sleep the
way a free web dyno does. Render and Railway free tiers cap at 512 MB and
cannot load the model at all.

### 1. Create the Space

<https://huggingface.co/new-space> → **Docker** → **Blank** → CPU basic (free).

### 2. Push

```bash
git init
git remote add space https://huggingface.co/spaces/<username>/siren-eyes
git add -A && git commit -m "Siren Eyes"
git push space main
```

Model weights total ~6 MB, so plain git is fine — Git LFS is not needed.

### 3. Space configuration

Spaces reads configuration from a YAML header in the repository `README.md`.
Add this at the very top of `README.md` before pushing:

```yaml
---
title: Siren Eyes
emoji: 🚑
colorFrom: red
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
---
```

The first build takes 8–12 minutes (PyTorch download dominates). Subsequent
pushes reuse the cached layers, so only the `COPY` steps re-run.

### Notes

- `app_port: 7860` must match the port in the `CMD`. Spaces routes to it.
- The container runs as UID 1000 with `$HOME=/home/user`. `YOLO_CONFIG_DIR`
  and `MPLCONFIGDIR` are set explicitly because Ultralytics and matplotlib
  both try to write to a config directory at import time and fail noisily on
  a read-only root.
- Uploads land in `runs/uploads`, which is ephemeral. That is intentional —
  visitor-uploaded video should not persist.

---

## Local Docker

```bash
docker build -t siren-eyes .
docker run --rm -p 7860:7860 siren-eyes
```

---

## Resource profile

| | |
|---|---|
| Image size | ~1.5 GB (CPU-only torch) |
| Idle memory | ~450 MB |
| Peak memory during analysis | ~1.2 GB |
| Cold start | 15–25 s (model load + warmup) |
| Analysis throughput | ~3 frames/s at 1080p on 2 vCPU |

The models are loaded once in the FastAPI lifespan and a warmup inference is
run at startup, so the first user request is not the one that pays for lazy
initialisation.

Analyses run on a worker thread rather than the event loop. The pipeline is
synchronous and CPU-bound; running it inline would block every other
connection — including the WebSocket it is trying to stream to — for the
length of the analysis.

---

## Scaling notes

For anything beyond a demo:

- **Frame sampling** is the main lever. `FRAME_SAMPLE_STRIDE` defaults to 3;
  raising it trades temporal resolution for throughput linearly.
- **Downscale before inference.** The detector runs at `imgsz=640`, but
  decoding 1080p dominates. Feeding 720p halves wall-clock time with little
  accuracy cost at the distances that matter.
- **The real deployment target is edge hardware**, not a web host. The paper
  specifies a Raspberry Pi 4 per intersection at under $150, with MQTT between
  intersections. This container is a demonstration and evaluation harness, not
  the production artefact.

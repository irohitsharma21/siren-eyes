# Deployment

Siren Eyes ships as a single container: FastAPI serves both the API and the
built dashboard, so there is nothing to wire together and no CORS to configure
in production.

The image is CPU-only by construction. Nothing in the pipeline benefits from a
GPU at the scale this demonstration runs at, and the CUDA wheels quadruple the
image for the privilege.

---

## Before you start: what a Docker Space costs now

Hugging Face's own documentation states that Gradio and Docker Spaces run on
compute and **require a paid plan to create** — PRO for a personal account,
Team or Enterprise for an organisation. Free personal accounts are limited to
two Gradio Spaces on ZeroGPU, which this project cannot use because it is a
Docker Space.

This is a change from how Spaces used to work and it is the first thing to
check, because everything below assumes the account can create a Docker Space
at all. If it cannot, the container still runs anywhere that accepts a
Dockerfile; only the README front matter is Spaces-specific.

On free CPU Basic hardware a Space is also put to sleep after a period without
traffic and has to cold-start on the next visit. Budget for the model load and
warmup described under *Resource profile* when that happens.

---

## Hugging Face Spaces

### 1. Hardware

CPU Basic gives 2 vCPU, 16 GB RAM and 50 GB of **non-persistent** disk. That
is comfortable for a roughly 1.5 GB image and a pipeline that peaks near
1.2 GB resident. The disk being non-persistent is why `runs/uploads` is
treated as scratch: visitor-uploaded video is not meant to survive a restart.

Outbound traffic from a Space is restricted to ports 80, 443 and 8080. The
container is built so that this never matters — every weight it loads is on
disk and it makes no network request at startup — but see the note on
`YOLO_OFFLINE` below, which is what makes that true rather than merely
intended.

### 2. Space configuration

Spaces reads its configuration from a YAML block at the very top of the
repository `README.md`. That block is now committed, so there is nothing to
add before pushing:

```yaml
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
```

`app_port` must match the port the container listens on, because Spaces routes
to that port rather than probing for one. `colorFrom` and `colorTo` are
restricted to the set `red, yellow, green, blue, indigo, purple, pink, gray`.

There is deliberately no `license:` key. The licensing here is mixed — the code
is MIT, the siren classifier inherits ESC-50's CC BY-NC restriction, and the
demo footage is author-supplied under no open licence — and a single SPDX
identifier would misstate at least two of those. The README's Licence section
says it properly. Add `license: other` with `license_name` and `license_link`
if the Space page needs something in that field.

### 3. Large files: this blocks the push if ignored

Hugging Face's repository guide says files larger than 10 MB should be tracked
with `git-xet` (the successor to Git LFS on the Hub; Git LFS still works
through the compatibility bridge). Two files in this repository are over that
line:

| File | Size | Over 10 MB |
|---|---|---|
| `demo/ambulance_departure.mp4` | 17.7 MB | yes |
| `demo/ambulance_inbound.mp4` | 16.1 MB | yes |
| `models/ambulance_siren_model.h5` | 9.5 MB | no |
| `models/best.pt` | 5.2 MB | no |
| `models/siren_cnn.pt` | 0.4 MB | no |

The model weights are genuinely fine over plain git. An earlier version of this
document said so and was right about the weights, but it did not account for
the demo clips, which are larger and were added later.

Both clips are already committed as ordinary git blobs. Adding a
`.gitattributes` now affects future commits only; the existing objects still
travel as plain blobs on push, so the file has to be added *and* the history
rewritten, or a fresh history pushed. Either works:

```bash
# Option A - rewrite this repository's history so the clips become LFS objects
git lfs install
git lfs track "*.mp4" "*.pt" "*.h5"
git add .gitattributes && git commit -m "Track large artefacts with Git LFS"
git lfs migrate import --include="*.mp4,*.pt,*.h5" --everything
```

```bash
# Option B - push one fresh commit to the Space, leaving local history alone
git clone https://huggingface.co/spaces/<username>/siren-eyes space
cd space
git lfs install
git lfs track "*.mp4" "*.pt" "*.h5"
git add .gitattributes && git commit -m "Track large artefacts with Git LFS"
cp -r ../app ../models ../demo ../docs ../scripts .
cp ../Dockerfile ../requirements.txt ../README.md ../.dockerignore ../.gitignore .
# web/ is copied file by file on purpose: a plain `cp -r ../web .` would drag
# in node_modules and a stale dist, neither of which belongs in the push.
mkdir -p web && cp -r ../web/src ../web/index.html ../web/package.json \
    ../web/package-lock.json ../web/tsconfig.json ../web/vite.config.ts web/
git add -A && git commit -m "Siren Eyes" && git push
```

Option B is the less destructive of the two and is what to reach for if the
local history matters. In both cases `git lfs track` must run and
`.gitattributes` must be committed *before* the large files are added, or the
same problem recurs.

A third option is not to ship the clips at all. `demo/manifest.json` drives the
demo list and `app/main.py` returns an empty list when the manifest is absent,
so a Space without `demo/` degrades to upload-only rather than breaking. Drop
the `COPY --chown=user demo/ ./demo/` line from the Dockerfile if you go that
way.

`.gitattributes` is not committed in this repository, deliberately. Adding it
while the large files are already tracked as plain blobs makes git report all
five as modified without actually converting anything, which is confusing
rather than helpful. It belongs in the same commit as the migration above.

### 4. Push

```bash
git remote add space https://huggingface.co/spaces/<username>/siren-eyes
git push space main
```

The first build takes on the order of ten minutes; the PyTorch CPU wheels
dominate it. Subsequent pushes reuse the cached layers, so only the `COPY`
steps and the build checks re-run. The Space's startup timeout defaults to 30
minutes, which the build fits under comfortably.

---

## What the Dockerfile does, and why

### CPU-only torch is installed first, from a separate index

```dockerfile
RUN pip install --no-cache-dir --user \
        torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir --user -r requirements.txt
```

The ordering is the point. `ultralytics` depends on torch, so if
`requirements.txt` were installed first pip would resolve torch from PyPI and
pull the entire CUDA runtime with it — the difference between roughly 1.5 GB
and over 6 GB, none of it reachable on a CPU Space. Installing the CPU build
first leaves the version bounds in `requirements.txt` already satisfied, and
pip never goes looking.

### TensorFlow is no longer installed

`requirements.txt` used to list `tensorflow-cpu` and `tf-keras`. Nothing under
`app/` imports either. The only consumer is `scripts/validate_port.py`, which
compares the PyTorch transcription of the legacy classifier against the
original Keras graph and documents its own separate virtualenv for exactly that
reason. `app/core/audio.py` reads the `.h5` weights with `h5py` and runs them in
PyTorch, so the container has no TensorFlow code path at all.

This has a consequence worth stating. `h5py` used to arrive as a TensorFlow
transitive dependency, but it is a direct module-scope import in
`app/core/audio.py`. Removing TensorFlow without declaring `h5py` explicitly
would have produced an image that builds and then fails on the first request.
It is now listed in `requirements.txt` in its own right.

`jinja2` was removed for the same kind of reason in reverse: it was declared but
nothing imports it. FastAPI's Swagger and OpenAPI routes build their HTML from
plain strings, and the SPA is served by `StaticFiles` and `FileResponse`.

Version bounds were added throughout. A Space rebuilt six months after it was
pushed resolves `requirements.txt` from scratch, and an unbounded `>=` is how an
image that worked at push time stops working at rebuild time.

### No system ffmpeg package

`app/media.py` shells out to ffmpeg through `imageio_ffmpeg.get_ffmpeg_exe()`.
The `imageio-ffmpeg` manylinux wheel bundles a static ffmpeg binary — 76 MB,
ffmpeg 7.0.2, in the 0.6.0 wheel — so the apt `ffmpeg` package would be a second
copy of the same tool. `libgl1` and `libglib2.0-0` are still installed, because
`ultralytics` declares `opencv-python` rather than the headless build and an
OpenCV that links against libGL can win the install regardless of what
`requirements.txt` asks for.

If a future `imageio-ffmpeg` release stops bundling a binary, the build check
below fails loudly rather than the first analysis failing quietly. The fix in
that case is to put `ffmpeg` back on the apt line: `get_ffmpeg_exe()` falls back
to whatever is on `PATH`.

### `YOLO_OFFLINE=1`

Ultralytics runs `is_online()` at import, which resolves `one.one.one.one` and
`dns.google`, and uses the result to gate a PyPI update check and a usage ping.
Every weight this application loads is on disk, so none of that is needed, and
on a network-restricted host the resolution is a delay in exchange for nothing.
`YOLO_OFFLINE` is read by `is_online()` directly and short-circuits it.

`YOLO_CONFIG_DIR` and `MPLCONFIGDIR` are set for a related reason. Ultralytics
writes a settings file and matplotlib a font cache, both at import time, and
both default to a location an unprivileged user cannot write.

### Directories are created at build time, not at startup

`app/main.py` calls `RUNS_DIR.mkdir()` in its lifespan handler. That only works
if the working directory is writable by the UID the platform actually runs, so
`runs/uploads` is created in the image as `user` rather than left to chance.

### The build asserts what it cannot otherwise detect

```dockerfile
RUN test "$(stat -c%s models/best.pt)"      -gt 1000000 \
 && test "$(stat -c%s models/siren_cnn.pt)" -gt 100000 \
 && test -f web/dist/index.html \
 && python -c "import os, imageio_ffmpeg as f; p = f.get_ffmpeg_exe(); assert os.path.exists(p), p" \
 && python -c "import ultralytics, cv2, h5py, librosa, app.main"
```

The size tests exist because an unfetched Git LFS pointer is a valid ~130-byte
text file. `test -f` passes on one, the image builds, and the failure surfaces
when a visitor opens the Space. Size is the cheapest signal that distinguishes a
pointer from a checkpoint.

The import line proves that libGL, the audio stack and `h5py` all resolve inside
the image. That is precisely the failure mode dropping TensorFlow could have
introduced, and it is worth catching at build time rather than at request time.

### `.dockerignore`

Added, because there was none. Two entries matter more than the rest:

- `**/node_modules/` — stage 1 runs `npm ci` itself, and copying the host tree
  in would clobber that install with modules built for the host platform. This
  is how a local `docker build` fails on a source tree that builds fine
  natively.
- `runs/` — `runs/esc50_cache.npz` alone is 365 MB. It is a feature cache for
  `scripts/train_siren.py`; nothing the API serves ever reads it.

`web/dist/` is excluded too, so a stale local build cannot silently ship in
place of the one stage 1 produces.

---

## Local Docker

```bash
docker build -t siren-eyes .
docker run --rm -p 7860:7860 siren-eyes
```

---

## What has been verified, and what has not

Stated plainly, because "it should work" and "it was run" are different claims.

Verified by running it, on Windows, against the project virtualenv:

- `app/` and `scripts/` compile, and `import app.main` succeeds.
- The FastAPI lifespan loads both models and completes the detector warmup in
  under four seconds on CPU with no GPU present. `GET /api/health` returns 200
  and reports `detector_loaded: true` with the retrained `siren_cnn.pt`
  checkpoint, its 0.898 threshold and its held-out metrics.
- The same holds with `YOLO_OFFLINE=1` set, which is the container's
  configuration: `ultralytics.utils.ONLINE` is `False` and startup is otherwise
  unaffected.
- A real `uvicorn app.main:app --host 0.0.0.0 --port <p>` process serves
  `/api/health`, serves the SPA at `/`, and the `HEALTHCHECK` command from the
  Dockerfile exits 0 against it. The server was stopped afterwards.
- `/api/docs`, `/api/openapi.json`, `/api/config`, `/api/demos` and `/` all
  return 200 with `jinja2` and `markupsafe` blocked at import, which is what
  justifies removing `jinja2` from `requirements.txt`.
- `tsc -b && vite build` completes: 1515 modules, 171 kB of JS, 13 kB of CSS.
- The `imageio-ffmpeg` 0.6.0 manylinux2014_x86_64 wheel was downloaded and its
  contents listed. It contains `binaries/ffmpeg-linux-x86_64-v7.0.2` at 76 MB,
  which is what the decision to drop the apt `ffmpeg` package rests on.
- The shell half of the build check runs and passes against the real files.
- The README front matter parses, and its `colorFrom` and `colorTo` values are
  in the set Spaces accepts.

**Not verified: the image has never been built.** Docker is not installed on the
machine this was prepared on. The Dockerfile was checked structurally — every
instruction parses, both stages resolve, the line continuations are intact — and
every command inside it was run in isolation wherever that was possible. But no
claim is made here that `docker build` succeeds end to end. The first build is
where an apt package name, or a dependency with no manylinux wheel, would
surface. Run it locally once before pushing if you can.

---

## Resource profile

| | |
|---|---|
| Image size | ~1.5 GB (CPU-only torch), not measured |
| Model load and warmup | 3-4 s, measured on a desktop CPU |
| Idle memory | ~450 MB |
| Peak memory during analysis | ~1.2 GB |
| Analysis throughput | ~3 frames/s at 1080p on 2 vCPU |

The model load figure is measured. The memory and throughput figures are carried
over from earlier work on this project and were not re-measured for this
deployment pass.

The models are loaded once in the FastAPI lifespan and a warmup inference is run
at startup, so the first user request is not the one that pays for lazy
initialisation.

Analyses run on a worker thread rather than the event loop. The pipeline is
synchronous and CPU-bound; running it inline would block every other
connection — including the WebSocket it is trying to stream to — for the length
of the analysis.

---

## Scaling notes

For anything beyond a demo:

- **Frame sampling** is the main lever. `FRAME_SAMPLE_STRIDE` defaults to 3;
  raising it trades temporal resolution for throughput linearly.
- **Downscale before inference.** The detector runs at `imgsz=640`, but decoding
  1080p dominates. Feeding 720p halves wall-clock time with little accuracy cost
  at the distances that matter.
- **The real deployment target is edge hardware**, not a web host. The paper
  specifies a Raspberry Pi 4 per intersection at under $150, with MQTT between
  intersections. This container is a demonstration and evaluation harness, not
  the production artefact.

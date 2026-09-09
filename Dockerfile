# ── Stage 1: build the dashboard ──────────────────────────────────────
FROM node:20-slim AS web

WORKDIR /web
# The manifests are copied on their own so the npm layer is keyed to them and
# survives every edit to web/src. `npm ci` is by far the slowest step here.
COPY web/package.json web/package-lock.json ./
RUN npm ci --no-audit --no-fund

COPY web/ ./
RUN npm run build


# ── Stage 2: runtime ──────────────────────────────────────────────────
FROM python:3.11-slim

# Ultralytics declares opencv-python rather than the headless build, so an
# OpenCV that wants libGL can end up winning the install regardless of what
# requirements.txt asks for. These two are what it links against.
#
# ffmpeg is deliberately *not* installed. The imageio-ffmpeg manylinux wheel
# bundles a static ffmpeg binary and app/media.py invokes it through
# imageio_ffmpeg.get_ffmpeg_exe(); the apt package would be a second copy of
# the same thing. The build check further down asserts the bundled binary is
# really present, so this fails at build time rather than at first analysis.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces runs the container as UID 1000 with $HOME=/home/user.
RUN useradd -m -u 1000 user
USER user

# YOLO_CONFIG_DIR / MPLCONFIGDIR: Ultralytics writes a settings file and
# matplotlib a font cache, both at import time. Left to their defaults they
# aim at a directory under / and fail noisily as an unprivileged user.
#
# YOLO_OFFLINE: skips the DNS probe, PyPI update check and usage ping that
# Ultralytics performs on import. Every weight this app loads is on disk, so
# the container needs no egress at all; without this the import blocks on a
# name resolution that will not succeed on a network-restricted host.
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    YOLO_CONFIG_DIR=/home/user/.config/Ultralytics \
    MPLCONFIGDIR=/home/user/.cache/matplotlib \
    YOLO_OFFLINE=1

# Created here, as `user`, rather than at startup: app/main.py calls
# RUNS_DIR.mkdir() in its lifespan, and that only works if the working
# directory is writable by whichever UID the platform actually runs.
RUN mkdir -p $HOME/app/runs/uploads \
             $HOME/.config/Ultralytics \
             $HOME/.cache/matplotlib

WORKDIR $HOME/app

# CPU-only torch first. The default PyPI wheel drags in the whole CUDA
# runtime and takes the image from roughly 1.5 GB to over 6 GB for no benefit
# on a CPU Space. Installing it ahead of requirements.txt means the torch
# bounds in that file are already satisfied and pip never reaches for the
# CUDA build while resolving ultralytics.
COPY --chown=user requirements.txt ./
RUN pip install --no-cache-dir --user \
        torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user app/    ./app/
COPY --chown=user models/ ./models/
COPY --chown=user demo/   ./demo/
COPY --chown=user --from=web /web/dist ./web/dist

# Fail the build, not the first request.
#
# The size tests catch a Git LFS pointer that was never fetched: that is a
# ~130-byte text file, so `test -f` passes and the model only fails to load
# once someone opens the Space. The import line proves that libGL, the audio
# stack and h5py all resolve inside the image, which is the failure mode that
# dropping TensorFlow from requirements.txt would otherwise have introduced
# silently.
RUN test "$(stat -c%s models/best.pt)"      -gt 1000000 \
 && test "$(stat -c%s models/siren_cnn.pt)" -gt 100000 \
 && test -f web/dist/index.html \
 && python -c "import os, imageio_ffmpeg as f; p = f.get_ffmpeg_exe(); assert os.path.exists(p), p; print('bundled ffmpeg:', p)" \
 && python -c "import ultralytics, cv2, h5py, librosa, app.main; print('module graph ok')"

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/api/health')"

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]

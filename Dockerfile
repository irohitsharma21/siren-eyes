# ── Stage 1: build the dashboard ──────────────────────────────────────
FROM node:20-slim AS web

WORKDIR /web
COPY web/package*.json ./
RUN npm ci --no-audit --no-fund

COPY web/ ./
RUN npm run build


# ── Stage 2: runtime ──────────────────────────────────────────────────
FROM python:3.11-slim

# libGL and libglib are OpenCV's runtime deps; ffmpeg decodes the video and
# demuxes stereo audio. Everything else stays out of the image.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces runs as a non-root user with $HOME=/home/user.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Ultralytics writes a settings file and would otherwise try to use /
    YOLO_CONFIG_DIR=/home/user/.config/Ultralytics \
    MPLCONFIGDIR=/home/user/.cache/matplotlib

WORKDIR $HOME/app

# CPU-only torch first: the default wheel pulls the full CUDA stack and
# inflates the image from roughly 1.5 GB to over 6 GB for no benefit here.
COPY --chown=user requirements.txt ./
RUN pip install --no-cache-dir --user \
        torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user app/    ./app/
COPY --chown=user models/ ./models/
COPY --chown=user demo/   ./demo/
COPY --chown=user --from=web /web/dist ./web/dist

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/api/health')"

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]

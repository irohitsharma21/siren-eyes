"""
Run the real pipeline over each bundled demo and record the result.

Why this exists
---------------
The analysis is CPU-bound: YOLOv8 over a 1080p clip costs about 111 ms per
frame on a desktop CPU. A free-tier container is throttled to roughly 0.15 of
a core, which turns that into seconds per frame - a twenty-second clip becomes
a ten-minute wait, on an instance already sitting at 99% of its memory limit.
The demo either never finishes or the container is killed part way through.

So the bundled clips are analysed once, here, on a machine that can actually
do the work, and the recorded output ships with the repository. Nothing is
fabricated: every number in these files came out of the same pipeline the
upload path runs, and the server replays them frame by frame. Uploaded videos
are still analysed live, because there is nothing to precompute for them.

Run from the repository root:

    python scripts/precompute_demos.py
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import DEMO_DIR  # noqa: E402
from app.core.pipeline import SirenEyesPipeline  # noqa: E402


def precompute(pipeline: SirenEyesPipeline, clip: dict) -> Path:
    name = clip["file"]
    source = DEMO_DIR / name
    out = DEMO_DIR / f"{name}.analysis.json"

    print(f"\n=== {name} ===")
    started = time.time()

    frames = []
    for frame in pipeline.stream(source, hfov_deg=clip.get("camera_hfov_deg")):
        frames.append(asdict(frame))
        if len(frames) % 25 == 0:
            print(f"  {len(frames)} frames", flush=True)

    summary = asdict(pipeline.summarise())
    summary["precomputed"] = True

    payload = {
        # Stamped so a stale cache is identifiable rather than silently served
        # against a clip that has since been replaced.
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_bytes": source.stat().st_size,
        "summary": summary,
        "frames": frames,
    }
    out.write_text(json.dumps(payload), encoding="utf-8")

    elapsed = time.time() - started
    print(
        f"  wrote {out.name}: {len(frames)} frames, {out.stat().st_size / 1e6:.1f} MB, "
        f"computed in {elapsed:.1f}s"
    )
    print(
        f"  peak vision {summary['peak_vision_confidence']:.3f} | "
        f"peak siren {summary['peak_siren_confidence']:.3f} | "
        f"audio {'stereo' if summary['audio_stereo'] else 'mono/none'}"
    )
    return out


def main() -> int:
    manifest_path = DEMO_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    pipeline = SirenEyesPipeline()
    for clip in manifest["demos"]:
        source = DEMO_DIR / clip["file"]
        if not source.exists():
            print(f"skipping {clip['file']}: not present")
            continue
        precompute(pipeline, clip)

    print("\nDone. Commit the .analysis.json files alongside the clips.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

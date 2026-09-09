"""Quick integration smoke test: run the pipeline over the first N frames."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.core.pipeline import SirenEyesPipeline  # noqa: E402

video = sys.argv[1]
limit = int(sys.argv[2]) if len(sys.argv) > 2 else 20

pipe = SirenEyesPipeline()
print(f"analysing {video} (first {limit} sampled frames)\n")

hdr = f"{'t(s)':>6} {'vis':>6} {'aud':>6} {'fuse':>6} {'dist':>7} {'km/h':>6} {'ETA':>6} {'dir':>6} {'signal':>13} {'decision':>26} {'ms':>7}"
print(hdr)
print("-" * len(hdr))

n = 0
for f in pipe.stream(video):
    d = f"{f.distance_m:7.1f}" if f.distance_m is not None else f"{'-':>7}"
    v = f"{f.speed_kmph:6.1f}" if f.speed_kmph is not None else f"{'-':>6}"
    e = f"{f.eta_s:6.2f}" if f.eta_s is not None else f"{'-':>6}"
    g = f"{f.direction_deg:+6.0f}" if f.direction_deg is not None else f"{'-':>6}"
    print(
        f"{f.timestamp_s:6.2f} {f.vision_confidence:6.3f} {f.audio_confidence:6.3f} "
        f"{f.fused_confidence:6.3f} {d} {v} {e} {g} {f.signal_state:>13} "
        f"{f.decision:>26} {f.latency_ms:7.1f}"
    )
    n += 1
    if n >= limit:
        break

print("\n--- summary ---")
s = pipe.summarise()
for k, val in vars(s).items():
    print(f"  {k:24}: {val}")

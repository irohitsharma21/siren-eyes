"""
Run every bundled demo clip through the live API and print the outcome.

Acts as an end-to-end smoke test: it exercises the HTTP job endpoint, the
WebSocket stream, and the full pipeline, and prints the decision trace so the
safety behaviour can be checked at a glance rather than inferred.
"""

from __future__ import annotations

import asyncio
import json
import sys
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:7860"
WS = BASE.replace("http://", "ws://").replace("https://", "wss://")


def demos() -> list[dict]:
    with urllib.request.urlopen(f"{BASE}/api/demos", timeout=30) as r:
        return json.load(r)["demos"]


def start(name: str) -> str:
    req = urllib.request.Request(
        f"{BASE}/api/analyses?demo={name}", method="POST", data=b""
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)["job_id"]


async def run(job_id: str):
    import websockets

    frames, summary = [], None
    async with websockets.connect(
        f"{WS}/api/analyses/{job_id}/stream", max_size=None, ping_timeout=None
    ) as ws:
        while True:
            msg = json.loads(await ws.recv())
            if msg["type"] == "frame":
                frames.append(msg["frame"])
            elif msg["type"] == "summary":
                summary = msg["summary"]
            elif msg["type"] == "error":
                raise RuntimeError(msg["message"])
            elif msg["type"] == "done":
                break
    return frames, summary


async def main() -> None:
    clips = demos()
    print(f"{len(clips)} demo clip(s)\n")
    failures = 0

    for clip in clips:
        print("=" * 72)
        print(f"  {clip['title']}")
        print(f"  {clip['file']} · {clip['duration_s']}s · "
              f"{'stereo' if clip['stereo'] else 'mono'} audio · "
              f"HFOV {clip.get('camera_hfov_deg') or 'default'}")
        print("=" * 72)

        frames, summary = await run(start(clip["file"]))
        if summary is None:
            print("  NO SUMMARY RETURNED\n")
            failures += 1
            continue

        print(f"  frames               : {summary['frames_analysed']} "
              f"({summary['detection_frames']} with detections)")
        print(f"  peak vision / siren  : {summary['peak_vision_confidence']:.3f} / "
              f"{summary['peak_siren_confidence']:.3f}")
        print(f"  peak fused           : {summary['peak_fused_confidence']:.3f}")
        print(f"  first detection      : {summary['first_detection_s']}s")
        print(f"  preemption granted   : {summary['preemption_granted']}"
              + (f" at {summary['grant_time_s']}s "
                 f"(+{summary['response_latency_s']}s after detection)"
                 if summary["preemption_granted"] else ""))
        print(f"  latency mean / p95   : {summary['mean_latency_ms']:.0f} / "
              f"{summary['p95_latency_ms']:.0f} ms")
        print(f"  blocked reasons      : {summary['blocked_reasons']}")

        # Distance sanity: the pinhole estimate should land in a plausible
        # range for street footage once the clip is calibrated.
        dists = [f["distance_m"] for f in frames if f["distance_m"] is not None]
        if dists:
            print(f"  distance range       : {min(dists):.1f} – {max(dists):.1f} m")

        states = []
        for f in frames:
            if not states or states[-1][1] != f["signal_state"]:
                states.append((f["timestamp_s"], f["signal_state"]))
        print(f"  signal transitions   : "
              + " → ".join(f"{s}@{t:.1f}s" for t, s in states))
        print()

    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    asyncio.run(main())

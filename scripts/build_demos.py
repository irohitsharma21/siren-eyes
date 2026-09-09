"""
Regenerate `demo/manifest.json`.

Probes every clip in `demo/` for duration, resolution and — importantly —
whether its audio track carries real content. A stereo container whose two
channels are identical, or whose samples are all zero, is reported honestly
so the UI can explain why the siren and direction stages contributed nothing
rather than silently showing 0.00 confidence.

Descriptions live in DESCRIPTIONS below; everything else is measured.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.media import extract_audio, probe_video  # noqa: E402

DEMO_DIR = ROOT / "demo"

DESCRIPTIONS: dict[str, dict[str, str]] = {
    "ambulance_departure.mp4": {
        "title": "Street pass — forward",
        "description": (
            "Municipal ambulance manoeuvring on a narrow residential street. "
            "Vision and siren both fire strongly (0.99 / 0.98) and the clip "
            "exercises the whole decision path: the vehicle drifts toward the "
            "camera early on, preemption is granted once ETA clears the 5 s "
            "buffer, green is held until it clears, then amber and all-red "
            "return the signal to normal timing. Once it starts pulling away "
            "the gate reports blocked_not_approaching for the rest of the clip."
        ),
        "source": "Supplied by the project author",
        "license": "Author-supplied; not redistributed under an open licence",
        # Telephoto framing. Estimated by solving the pinhole model against a
        # hand-measured reference: the ambulance (1.9 m wide) spans ~700 px at
        # roughly 12 m, giving f = Z*w/W ~= 4400 px and HFOV ~= 25 deg. The
        # 60 deg default would place the same vehicle at 1.7 m.
        "camera_hfov_deg": 25.0,
    },
    "ambulance_inbound.mp4": {
        "title": "Street pass — time-reversed",
        "description": (
            "The same take played backwards, stated plainly because it is a "
            "constructed fixture rather than new footage. Reversing inverts "
            "every closing speed, so the approach and recede phases swap "
            "places and the safety gate fires on a different part of the "
            "timeline — a cheap way to check the kinematics are driven by "
            "measured motion and not by anything baked into the clip."
        ),
        "source": "Time-reversed from ambulance_departure.mp4",
        "license": "Author-supplied; not redistributed under an open licence",
        "camera_hfov_deg": 25.0,
    },
}


def main() -> None:
    demos = []

    for path in sorted(DEMO_DIR.glob("*.mp4")):
        info = probe_video(path)
        track = extract_audio(path)

        has_audio = track is not None and not track.is_silent
        stereo = bool(track and track.is_stereo and not track.is_silent)

        meta = DESCRIPTIONS.get(
            path.name,
            {
                "title": path.stem.replace("_", " ").title(),
                "description": "",
                "source": "unknown",
                "license": "unknown",
            },
        )

        note = ""
        if track is None:
            note = "no audio stream in container"
        elif track.is_silent:
            note = "audio stream present but digitally silent"
        elif not track.is_stereo:
            note = "dual-mono audio — direction estimation unavailable"

        demos.append(
            {
                "name": path.name,
                "file": path.name,
                "title": meta["title"],
                "description": meta["description"],
                "duration_s": round(info.duration_s, 2),
                "resolution": f"{info.width}x{info.height}",
                "fps": round(info.fps, 2),
                "size_mb": round(path.stat().st_size / 1024 / 1024, 1),
                "has_audio": has_audio,
                "stereo": stereo,
                "audio_note": note,
                "source": meta["source"],
                "license": meta["license"],
                "camera_hfov_deg": meta.get("camera_hfov_deg"),
            }
        )

        print(
            f"  {path.name:32} {info.duration_s:6.1f}s  {info.width}x{info.height}  "
            f"audio={'stereo' if stereo else 'mono' if has_audio else 'none':6}  {note}"
        )

    out = DEMO_DIR / "manifest.json"
    out.write_text(json.dumps({"demos": demos}, indent=2))
    print(f"\nwrote {out} ({len(demos)} clips)")


if __name__ == "__main__":
    main()

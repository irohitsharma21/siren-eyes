"""
Per-run analysis parameters.

The paper fixes its thresholds (vision 0.65, fused 0.60, 5 s safety buffer,
2 s TTC), and those remain the defaults. A visitor can move them for one run to
see how the safety gate responds - a lower buffer grants earlier, a higher
fused threshold withholds longer - without touching the server configuration.

Bounds are deliberately wide enough to be instructive and narrow enough that
nothing degenerate happens: a vision threshold below the detector's own 0.20
floor would gate nothing, and a negative buffer has no meaning.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields

from app.config import settings

# name -> (minimum, maximum, step). The step is advisory, for the UI control.
BOUNDS: dict[str, tuple[float, float, float]] = {
    "vision_threshold": (0.20, 0.95, 0.01),
    "fused_threshold": (0.20, 0.95, 0.01),
    "safety_buffer_s": (0.0, 15.0, 0.5),
    "min_ttc_s": (0.0, 10.0, 0.5),
}


@dataclass(frozen=True)
class AnalysisParams:
    """The four operating thresholds a client may override for one run."""

    vision_threshold: float = settings.VISION_CONF_THRESHOLD
    fused_threshold: float = settings.FUSION_TRIGGER_THRESHOLD
    safety_buffer_s: float = settings.SAFETY_BUFFER_S
    min_ttc_s: float = settings.TTC_THRESHOLD_S

    @classmethod
    def defaults(cls) -> "AnalysisParams":
        return cls()

    @classmethod
    def build(cls, **overrides: float | None) -> "AnalysisParams":
        """
        Construct from partial overrides, validating each against BOUNDS.

        `None` means "use the default", so query parameters that were simply
        not supplied fall through cleanly.
        """
        values: dict[str, float] = {}
        for f in fields(cls):
            raw = overrides.get(f.name)
            if raw is None:
                continue
            lo, hi, _ = BOUNDS[f.name]
            value = float(raw)
            if not (lo <= value <= hi):
                raise ValueError(f"{f.name} must be between {lo} and {hi}, got {value}")
            values[f.name] = round(value, 4)
        return cls(**values)

    @property
    def is_default(self) -> bool:
        """True when every value equals the configured default.

        Recorded demo analyses were produced with the defaults, so this is what
        decides whether a demo run can be replayed or must be computed live.
        """
        return self == AnalysisParams()

    def as_dict(self) -> dict[str, float]:
        return asdict(self)

    @staticmethod
    def describe() -> dict:
        """Defaults and bounds, for the UI's parameter controls."""
        d = AnalysisParams()
        return {
            name: {"default": getattr(d, name), "min": lo, "max": hi, "step": step}
            for name, (lo, hi, step) in BOUNDS.items()
        }

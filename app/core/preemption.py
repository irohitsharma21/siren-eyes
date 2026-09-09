"""
Multi-modal fusion and safety-aware traffic signal preemption.

Implements Paper sections III-E and III-F.

Fusion (equation 6):

    Conf_final = alpha * Conf_vision + beta * Conf_audio,   alpha=0.7, beta=0.3

Preemption is granted only when every safety precondition holds:

  * fused confidence clears the trigger threshold
  * the ambulance is actually approaching (closing speed > 0)
  * ETA exceeds the fixed 5 s safety buffer, so the phase change is not abrupt
  * no conflicting movement has a Time-to-Collision below 2 s
  * the direction of arrival agrees with the approach axis within 45 degrees

Green is then held until the vehicle clears the stop line, after which normal
timing resumes through a proper amber and all-red interval rather than
snapping back.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from app.config import settings
from app.core.direction import DirectionEstimate
from app.core.geometry import KinematicState


class SignalState(str, Enum):
    RED = "red"
    GREEN = "green"
    AMBER = "amber"
    ALL_RED = "all_red"
    PREEMPT_GREEN = "preempt_green"


class Decision(str, Enum):
    IDLE = "idle"
    GRANTED = "granted"
    HOLDING = "holding"
    REVERTING = "reverting"
    BLOCKED_CONFIDENCE = "blocked_confidence"
    BLOCKED_NOT_APPROACHING = "blocked_not_approaching"
    BLOCKED_SAFETY_BUFFER = "blocked_safety_buffer"
    BLOCKED_TTC = "blocked_ttc"
    BLOCKED_DIRECTION = "blocked_direction"


@dataclass
class FusionResult:
    """Weighted multi-modal confidence, equation (6)."""

    vision_confidence: float
    audio_confidence: float
    fused_confidence: float
    triggered: bool

    @classmethod
    def compute(cls, vision: float, audio: float) -> "FusionResult":
        fused = (
            settings.FUSION_ALPHA_VISION * vision
            + settings.FUSION_BETA_AUDIO * audio
        )
        return cls(
            vision_confidence=vision,
            audio_confidence=audio,
            fused_confidence=fused,
            triggered=fused >= settings.FUSION_TRIGGER_THRESHOLD,
        )


@dataclass
class ConflictVehicle:
    """A vehicle on a conflicting approach, used for the TTC gate."""

    distance_m: float
    speed_mps: float

    @property
    def ttc_s(self) -> float:
        if self.speed_mps <= 0.01:
            return float("inf")
        return self.distance_m / self.speed_mps


@dataclass
class PreemptionEvent:
    """One decision tick, suitable for logging and timeline replay."""

    timestamp_s: float
    decision: Decision
    signal_state: SignalState
    fused_confidence: float
    vision_confidence: float
    audio_confidence: float
    distance_m: float | None
    eta_s: float | None
    speed_kmph: float | None
    direction_deg: float | None
    direction_available: bool
    min_ttc_s: float | None
    reason: str


@dataclass
class PreemptionController:
    """
    Safety-buffered signal controller.

    Deliberately a small explicit state machine rather than a set of ad-hoc
    booleans: the sequencing (grant -> hold -> amber -> all-red -> revert) is
    the part that keeps the intersection safe, so it needs to be auditable.
    """

    approach_angle_deg: float = 0.0
    state: SignalState = SignalState.RED
    _granted_at: float | None = field(default=None, init=False)
    _revert_at: float | None = field(default=None, init=False)
    _cleared: bool = field(default=False, init=False)
    events: list[PreemptionEvent] = field(default_factory=list, init=False)

    def reset(self) -> None:
        self.state = SignalState.RED
        self._granted_at = None
        self._revert_at = None
        self._cleared = False
        self.events.clear()

    def step(
        self,
        timestamp_s: float,
        fusion: FusionResult,
        kinematics: KinematicState | None,
        direction: DirectionEstimate | None,
        conflicts: list[ConflictVehicle] | None = None,
    ) -> PreemptionEvent:
        """Advance the controller by one observation."""
        conflicts = conflicts or []
        min_ttc = min((c.ttc_s for c in conflicts), default=float("inf"))

        decision, reason = self._decide(timestamp_s, fusion, kinematics, direction, min_ttc)

        event = PreemptionEvent(
            timestamp_s=timestamp_s,
            decision=decision,
            signal_state=self.state,
            fused_confidence=fusion.fused_confidence,
            vision_confidence=fusion.vision_confidence,
            audio_confidence=fusion.audio_confidence,
            distance_m=kinematics.distance_m if kinematics else None,
            eta_s=kinematics.eta_s if kinematics else None,
            speed_kmph=kinematics.speed_kmph if kinematics else None,
            direction_deg=direction.angle_deg if direction and direction.available else None,
            direction_available=bool(direction and direction.available),
            min_ttc_s=None if min_ttc == float("inf") else min_ttc,
            reason=reason,
        )
        self.events.append(event)
        return event

    # ── decision logic ────────────────────────────────────────────────
    def _decide(
        self,
        now: float,
        fusion: FusionResult,
        kin: KinematicState | None,
        direction: DirectionEstimate | None,
        min_ttc: float,
    ) -> tuple[Decision, str]:
        # Already preempting: hold green until the vehicle clears, then revert.
        if self.state == SignalState.PREEMPT_GREEN:
            return self._hold_or_revert(now, kin)

        # Sequencing back to normal timing after a preemption.
        if self.state in (SignalState.AMBER, SignalState.ALL_RED):
            return self._sequence_back(now)

        if not fusion.triggered:
            return (
                Decision.BLOCKED_CONFIDENCE,
                f"fused confidence {fusion.fused_confidence:.2f} below "
                f"{settings.FUSION_TRIGGER_THRESHOLD:.2f}",
            )

        if kin is None or not kin.closing:
            return (
                Decision.BLOCKED_NOT_APPROACHING,
                "vehicle is not closing on the intersection",
            )

        # Direction gate: paper section III-C.
        if direction and direction.available:
            offset = abs(direction.angle_deg - self.approach_angle_deg)
            if offset > settings.APPROACH_ANGLE_TOLERANCE_DEG:
                return (
                    Decision.BLOCKED_DIRECTION,
                    f"bearing {direction.angle_deg:+.0f} deg is {offset:.0f} deg off the "
                    f"approach axis (limit {settings.APPROACH_ANGLE_TOLERANCE_DEG:.0f})",
                )

        # TTC gate: paper section III-F.
        if min_ttc < settings.TTC_THRESHOLD_S:
            return (
                Decision.BLOCKED_TTC,
                f"conflicting movement at TTC {min_ttc:.1f}s "
                f"(< {settings.TTC_THRESHOLD_S:.0f}s)",
            )

        # Safety buffer: never cut a phase shorter than the clearance time.
        if kin.eta_s is None or kin.eta_s <= settings.SAFETY_BUFFER_S:
            eta_txt = "unknown" if kin.eta_s is None else f"{kin.eta_s:.1f}s"
            return (
                Decision.BLOCKED_SAFETY_BUFFER,
                f"ETA {eta_txt} does not clear the "
                f"{settings.SAFETY_BUFFER_S:.0f}s safety buffer",
            )

        self.state = SignalState.PREEMPT_GREEN
        self._granted_at = now
        self._cleared = False
        return (
            Decision.GRANTED,
            f"preemption granted — ETA {kin.eta_s:.1f}s, "
            f"{kin.distance_m:.0f}m out at {kin.speed_kmph:.0f} km/h",
        )

    def _hold_or_revert(
        self, now: float, kin: KinematicState | None
    ) -> tuple[Decision, str]:
        held = now - (self._granted_at or now)

        cleared = kin is None or kin.distance_m <= settings.CLEARANCE_DISTANCE_M
        if cleared and held >= settings.MIN_GREEN_HOLD_S:
            self.state = SignalState.AMBER
            self._revert_at = now
            self._cleared = True
            return Decision.REVERTING, "ambulance cleared — reverting to normal timing"

        where = f"{kin.distance_m:.0f}m" if kin else "clear"
        return Decision.HOLDING, f"holding green for {held:.1f}s, ambulance at {where}"

    def _sequence_back(self, now: float) -> tuple[Decision, str]:
        elapsed = now - (self._revert_at or now)

        if self.state == SignalState.AMBER:
            if elapsed >= settings.AMBER_DURATION_S:
                self.state = SignalState.ALL_RED
                self._revert_at = now
            return Decision.REVERTING, "amber interval"

        if elapsed >= settings.ALL_RED_DURATION_S:
            self.state = SignalState.RED
            self._granted_at = None
            self._revert_at = None
            return Decision.IDLE, "normal timing restored"
        return Decision.REVERTING, "all-red clearance interval"

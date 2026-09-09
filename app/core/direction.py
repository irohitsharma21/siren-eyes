"""
Probabilistic stereo direction-of-arrival estimation.

Implements Paper section III-C. Two independent binaural cues are extracted
from the stereo microphone pair and fused as a product of Gaussians over a
discrete angle grid:

    ILD(f)   = 20 * log10( |L(f)| / |R(f)| )                          (1)
    ITD(t)   = argmax_tau  integral L(t) R(t + tau) dt                (2)
    P(theta) = N(mu_ILD, sigma_ILD^2) . N(mu_ITD, sigma_ITD^2)        (3)
    theta_hat = argmax_theta P(theta)

ITD is computed with GCC-PHAT rather than raw cross-correlation. Plain
correlation smears badly under the reverberation and broadband engine noise
present at a road intersection; PHAT whitening flattens the magnitude
spectrum so the delay peak stays sharp, which is what keeps the +/-15 degree
accuracy claim (89.3%) reachable in practice.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, sosfiltfilt

from app.config import settings


@dataclass
class DirectionEstimate:
    """Resolved direction of arrival for one audio window."""

    angle_deg: float
    confidence: float
    itd_angle_deg: float | None
    ild_angle_deg: float | None
    itd_seconds: float | None
    ild_db: float | None
    available: bool
    reason: str = ""

    @property
    def bearing(self) -> str:
        """Human-readable bearing; negative is left of the camera axis."""
        if not self.available:
            return "unknown"
        a = self.angle_deg
        if a <= -60:
            return "far left"
        if a <= -20:
            return "left"
        if a < 20:
            return "centre"
        if a < 60:
            return "right"
        return "far right"


def _bandpass(x: np.ndarray, sr: int) -> np.ndarray:
    """
    Isolate siren fundamentals before cue extraction.

    Paper section II-I: a 300 Hz - 3 kHz bandpass is applied to isolate siren
    frequencies. This removes tyre roar and wind rumble that would otherwise
    dominate the correlation.
    """
    nyq = sr / 2.0
    low = max(settings.BANDPASS_LOW_HZ / nyq, 1e-4)
    high = min(settings.BANDPASS_HIGH_HZ / nyq, 0.99)
    if low >= high:
        return x
    sos = butter(4, [low, high], btype="band", output="sos")
    return sosfiltfilt(sos, x)


def _gcc_phat(left: np.ndarray, right: np.ndarray, sr: int, max_tau: float):
    """
    Generalised cross-correlation with phase transform.

    Returns (delay_seconds, peak_sharpness). A positive delay means the signal
    reached the LEFT microphone first, so the source sits left of the axis.
    """
    n = 1
    while n < len(left) + len(right):
        n <<= 1

    L = np.fft.rfft(left, n=n)
    R = np.fft.rfft(right, n=n)

    cross = L * np.conj(R)
    magnitude = np.abs(cross)
    # PHAT weighting: retain phase, discard magnitude.
    cross = np.divide(
        cross, magnitude, out=np.zeros_like(cross), where=magnitude > 1e-12
    )

    corr = np.fft.irfft(cross, n=n)
    corr = np.concatenate((corr[-(n // 2) :], corr[: n // 2 + 1]))

    max_shift = int(min(n // 2, math.ceil(max_tau * sr)))
    centre = len(corr) // 2
    window = corr[centre - max_shift : centre + max_shift + 1]
    if window.size == 0:
        return None, 0.0

    peak = int(np.argmax(np.abs(window)))
    delay = (peak - max_shift) / float(sr)

    # Sharpness = peak height over RMS of the search window. A diffuse or
    # ambiguous sound field gives a low ratio and should not be trusted.
    rms = float(np.sqrt(np.mean(window**2))) or 1e-12
    sharpness = float(np.abs(window[peak]) / rms)

    return delay, sharpness


def _itd_to_angle(delay_s: float) -> float:
    """
    Convert an inter-aural delay to an incident angle.

    Far-field plane-wave model across a baseline d:
        tau = (d / c) * sin(theta)  =>  theta = asin(tau * c / d)
    """
    max_tau = settings.MIC_BASELINE_M / settings.SPEED_OF_SOUND_MPS
    ratio = float(np.clip(delay_s / max_tau, -1.0, 1.0))
    return math.degrees(math.asin(ratio))


def _ild_db(left: np.ndarray, right: np.ndarray, sr: int) -> float:
    """
    Broadband inter-aural level difference, equation (1).

    Energy is measured only inside the siren passband so that low-frequency
    wind loading on one capsule cannot bias the result.
    """
    n = min(len(left), len(right))
    if n < 32:
        return 0.0

    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    band = (freqs >= settings.BANDPASS_LOW_HZ) & (freqs <= settings.BANDPASS_HIGH_HZ)
    if not band.any():
        return 0.0

    L = np.abs(np.fft.rfft(left[:n]))[band]
    R = np.abs(np.fft.rfft(right[:n]))[band]

    el = float(np.sqrt(np.mean(L**2))) + 1e-12
    er = float(np.sqrt(np.mean(R**2))) + 1e-12
    return 20.0 * math.log10(el / er)


def _ild_to_angle(ild: float) -> float:
    """
    Map an ILD in dB onto an incident angle.

    A compact two-capsule array on a mast produces a near-linear level
    gradient across the frontal arc; saturation is handled by clamping at the
    reference maximum rather than letting the estimate run past +/-90 degrees.
    """
    ILD_AT_90_DEG = 9.0  # dB, empirical saturation point for the array
    ratio = float(np.clip(ild / ILD_AT_90_DEG, -1.0, 1.0))
    return ratio * 90.0


def _gaussian(grid: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * ((grid - mu) / sigma) ** 2)


def estimate_direction(
    left: np.ndarray,
    right: np.ndarray,
    sr: int,
    is_stereo: bool = True,
) -> DirectionEstimate:
    """
    Fuse ITD and ILD into a single direction estimate, equation (3).

    A mono source, or a stereo file whose channels are identical, carries no
    spatial information at all. Rather than emit a fabricated bearing, the
    estimator reports available=False and the preemption stage falls back to
    vision-only gating.
    """
    grid = np.array(settings.angle_bins, dtype=float)

    def unavailable(reason: str, **kw) -> DirectionEstimate:
        base = dict(
            angle_deg=0.0,
            confidence=0.0,
            itd_angle_deg=None,
            ild_angle_deg=None,
            itd_seconds=None,
            ild_db=None,
            available=False,
            reason=reason,
        )
        base.update(kw)
        return DirectionEstimate(**base)

    if not is_stereo:
        return unavailable("mono audio source - no binaural cues present")

    # Identical channels mean the file is stereo only in container terms.
    if np.allclose(left, right, atol=1e-6):
        return unavailable("dual-mono audio - channels are identical")

    left_f = _bandpass(np.asarray(left, dtype=float), sr)
    right_f = _bandpass(np.asarray(right, dtype=float), sr)

    # Silence guard: a window with no siren energy has no direction.
    if float(np.sqrt(np.mean(left_f**2) + np.mean(right_f**2))) < 1e-5:
        return unavailable("no signal energy in the 300 Hz - 3 kHz siren band")

    max_tau = settings.MIC_BASELINE_M / settings.SPEED_OF_SOUND_MPS
    delay, sharpness = _gcc_phat(left_f, right_f, sr, max_tau)

    ild = _ild_db(left_f, right_f, sr)
    ild_angle = _ild_to_angle(ild)
    itd_angle = _itd_to_angle(delay) if delay is not None else None

    posterior = np.ones_like(grid)
    if itd_angle is not None:
        posterior *= _gaussian(grid, itd_angle, settings.SIGMA_ITD_DEG)
    posterior *= _gaussian(grid, ild_angle, settings.SIGMA_ILD_DEG)

    total = float(posterior.sum())
    if total <= 1e-12:
        return unavailable(
            "ITD and ILD cues are mutually inconsistent",
            itd_angle_deg=itd_angle,
            ild_angle_deg=ild_angle,
            itd_seconds=delay,
            ild_db=ild,
        )

    posterior /= total
    best = int(np.argmax(posterior))

    # Confidence blends how peaked the posterior is against how sharp the
    # correlation was, so an acoustically mushy window cannot score high.
    peakedness = float(posterior[best] * len(grid))  # 1.0 == uniform
    peak_term = float(np.clip((peakedness - 1.0) / 8.0, 0.0, 1.0))
    sharp_term = float(np.clip((sharpness - 1.0) / 4.0, 0.0, 1.0))
    confidence = float(np.clip(0.5 * peak_term + 0.5 * sharp_term, 0.0, 1.0))

    return DirectionEstimate(
        angle_deg=float(grid[best]),
        confidence=confidence,
        itd_angle_deg=itd_angle,
        ild_angle_deg=ild_angle,
        itd_seconds=delay,
        ild_db=ild,
        available=True,
    )

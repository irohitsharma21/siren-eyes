/**
 * Pure helpers over a run's frames: lookup by time, decision events, trails,
 * and a client-side summary for runs that never received a server one.
 */

import type { AnalysisParams, AnalysisSummary, Detection, FrameResult } from './api'

export const DECISION_LABEL: Record<string, string> = {
  idle: 'Normal timing',
  granted: 'Preemption granted',
  holding: 'Holding green',
  reverting: 'Reverting to normal timing',
  blocked_confidence: 'Blocked: confidence below trigger',
  blocked_not_approaching: 'Blocked: not approaching',
  blocked_safety_buffer: 'Blocked: inside safety buffer',
  blocked_ttc: 'Blocked: conflicting traffic (TTC)',
  blocked_direction: 'Blocked: off the approach axis',
}

export const SIGNAL_LABEL: Record<string, string> = {
  red: 'Red — normal timing',
  green: 'Green',
  amber: 'Amber — reverting',
  all_red: 'All-red clearance',
  preempt_green: 'Preemption granted',
}

export function shortDecision(decision: string): string {
  return DECISION_LABEL[decision] ?? decision.replace(/_/g, ' ')
}

export function blockedReasonLabel(reason: string): string {
  return (DECISION_LABEL[reason] ?? reason.replace(/_/g, ' ')).replace(/^Blocked: /, '')
}

/** Index of the last frame at or before `t` (binary search; frames are time-ordered). */
export function indexAt(frames: FrameResult[], t: number): number {
  if (!frames.length) return -1
  let lo = 0
  let hi = frames.length - 1
  if (frames[0].timestamp_s > t) return 0
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1
    if (frames[mid].timestamp_s <= t) lo = mid
    else hi = mid - 1
  }
  return lo
}

export function frameAt(frames: FrameResult[], t: number): FrameResult | null {
  const i = indexAt(frames, t)
  return i < 0 ? null : frames[i]
}

/** Strongest detection at or above the vision threshold, if any. */
export function bestBox(frame: FrameResult, threshold: number): Detection | null {
  let best: Detection | null = null
  for (const d of frame.detections) {
    if (d.confidence >= threshold && (!best || d.confidence > best.confidence)) best = d
  }
  return best
}

export function boxCentre(box: [number, number, number, number]): [number, number] {
  return [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
}

/**
 * Recent centroids of the tracked box leading up to `index`, oldest first.
 * Stops at the first gap so a trail never bridges two separate passes.
 */
export function trailFor(
  frames: FrameResult[],
  index: number,
  threshold: number,
  length = 14,
): [number, number][] {
  const pts: [number, number][] = []
  for (let i = index; i >= 0 && pts.length < length; i--) {
    const b = bestBox(frames[i], threshold)
    if (!b) break
    pts.push(boxCentre(b.box))
  }
  return pts.reverse()
}

export type EventKind = 'detect' | 'trigger' | 'grant' | 'hold' | 'revert' | 'restore' | 'blocked'

export interface DecisionEvent {
  t: number
  index: number
  kind: EventKind
  label: string
  detail: string
}

/**
 * Turn per-frame decisions into a short list of moments worth jumping to.
 * Consecutive identical decisions collapse; blocked-reason flapping within
 * half a second of the same reason is suppressed so the list stays readable.
 */
export function deriveEvents(frames: FrameResult[], params: AnalysisParams | null): DecisionEvent[] {
  const out: DecisionEvent[] = []
  if (!frames.length) return out
  const vThr = params?.vision_threshold ?? 0.65
  const fThr = params?.fused_threshold ?? 0.6

  let detected = false
  let triggered = false
  let prev = ''
  const lastBlocked: Record<string, number> = {}

  frames.forEach((f, i) => {
    if (!detected && f.vision_confidence >= vThr) {
      detected = true
      out.push({
        t: f.timestamp_s, index: i, kind: 'detect', label: 'First detection',
        detail: `vision ${f.vision_confidence.toFixed(2)}`,
      })
    }
    if (!triggered && f.fused_confidence >= fThr) {
      triggered = true
      out.push({
        t: f.timestamp_s, index: i, kind: 'trigger', label: 'Fused trigger armed',
        detail: `fused ${f.fused_confidence.toFixed(2)} ≥ ${fThr.toFixed(2)}`,
      })
    }
    const d = f.decision
    if (d !== prev) {
      if (d === 'granted') {
        out.push({ t: f.timestamp_s, index: i, kind: 'grant', label: 'Preemption granted', detail: f.reason })
      } else if (d === 'holding') {
        out.push({ t: f.timestamp_s, index: i, kind: 'hold', label: 'Holding green', detail: f.reason })
      } else if (d === 'reverting') {
        if (prev !== 'reverting') {
          out.push({ t: f.timestamp_s, index: i, kind: 'revert', label: 'Ambulance cleared — reverting', detail: f.reason })
        }
      } else if (d === 'idle' && prev === 'reverting') {
        out.push({ t: f.timestamp_s, index: i, kind: 'restore', label: 'Normal timing restored', detail: f.reason })
      } else if (d.startsWith('blocked')) {
        const last = lastBlocked[d]
        if (last === undefined || f.timestamp_s - last > 0.5) {
          out.push({ t: f.timestamp_s, index: i, kind: 'blocked', label: shortDecision(d), detail: f.reason })
        }
        lastBlocked[d] = f.timestamp_s
      }
      prev = d
    } else if (d.startsWith('blocked')) {
      lastBlocked[d] = f.timestamp_s
    }
  })
  return out
}

/**
 * What can be said about a run from its frames alone. Used when the server
 * summary never arrived (a cancelled replay) and as a cross-check otherwise.
 */
export function summariseFrames(
  frames: FrameResult[],
  params: AnalysisParams,
): Pick<
  AnalysisSummary,
  | 'frames_analysed' | 'detection_frames' | 'peak_vision_confidence' | 'peak_siren_confidence'
  | 'peak_fused_confidence' | 'preemption_granted' | 'first_detection_s' | 'grant_time_s'
  | 'response_latency_s' | 'mean_latency_ms' | 'p95_latency_ms' | 'blocked_reasons' | 'duration_s'
> {
  let peakV = 0
  let peakA = 0
  let peakF = 0
  let detection = 0
  let first: number | null = null
  let grant: number | null = null
  const blocked: Record<string, number> = {}
  const lat: number[] = []
  for (const f of frames) {
    peakV = Math.max(peakV, f.vision_confidence)
    peakA = Math.max(peakA, f.audio_confidence)
    peakF = Math.max(peakF, f.fused_confidence)
    if (f.vision_confidence >= params.vision_threshold) {
      detection++
      if (first === null) first = f.timestamp_s
    }
    if (grant === null && f.decision === 'granted') grant = f.timestamp_s
    if (f.decision.startsWith('blocked')) blocked[f.decision] = (blocked[f.decision] ?? 0) + 1
    lat.push(f.latency_ms)
  }
  const sorted = [...lat].sort((a, b) => a - b)
  const mean = lat.length ? lat.reduce((a, b) => a + b, 0) / lat.length : 0
  const p95 = sorted.length ? sorted[Math.min(sorted.length - 1, Math.floor(0.95 * (sorted.length - 1)))] : 0
  return {
    frames_analysed: frames.length,
    detection_frames: detection,
    peak_vision_confidence: peakV,
    peak_siren_confidence: peakA,
    peak_fused_confidence: peakF,
    preemption_granted: grant !== null,
    first_detection_s: first,
    grant_time_s: grant,
    response_latency_s: grant !== null && first !== null ? grant - first : null,
    mean_latency_ms: mean,
    p95_latency_ms: p95,
    blocked_reasons: blocked,
    duration_s: frames.length ? frames[frames.length - 1].timestamp_s : 0,
  }
}

export function paramsEqual(a: AnalysisParams | null, b: AnalysisParams | null): boolean {
  if (!a || !b) return a === b
  return (Object.keys(a) as (keyof AnalysisParams)[]).every((k) => Math.abs(a[k] - b[k]) < 1e-9)
}

export function fmtTime(t: number): string {
  return `${t.toFixed(2)} s`
}

export function fmtClock(t: number): string {
  const m = Math.floor(t / 60)
  const s = t - m * 60
  return `${m}:${s.toFixed(1).padStart(4, '0')}`
}

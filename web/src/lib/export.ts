/**
 * Export a run as JSON, as per-frame CSV, or as a shareable text summary.
 */

import type { AnalysisParams, AnalysisSummary, FrameResult } from './api'
import { blockedReasonLabel, summariseFrames } from './analysis'

export interface ExportableRun {
  id: string
  createdAt: string
  clipLabel: string
  params: AnalysisParams
  status: string
  precomputed: boolean
  frames: FrameResult[]
  summary: AnalysisSummary | null
}

export function download(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  a.remove()
  // Revoke on the next tick so the click has a chance to start the download.
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}

export function safeName(label: string): string {
  return label.replace(/\.[a-z0-9]+$/i, '').replace(/[^a-z0-9_-]+/gi, '_').slice(0, 60) || 'run'
}

export function runToJSON(run: ExportableRun): Blob {
  const payload = {
    exported_at: new Date().toISOString(),
    app: 'Siren Eyes',
    run: {
      id: run.id,
      created_at: run.createdAt,
      clip: run.clipLabel,
      status: run.status,
      precomputed: run.precomputed,
      parameters: run.params,
    },
    summary: run.summary,
    frames: run.frames,
  }
  return new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' })
}

const CSV_COLUMNS = [
  'frame_index', 'timestamp_s', 'vision_confidence', 'audio_confidence', 'fused_confidence',
  'distance_m', 'speed_kmph', 'eta_s', 'direction_deg', 'direction_bearing',
  'signal_state', 'decision', 'reason', 'latency_ms', 'detections', 'best_box',
] as const

function csvCell(v: unknown): string {
  if (v === null || v === undefined) return ''
  const s = String(v)
  return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s
}

export function runToCSV(frames: FrameResult[]): Blob {
  const lines = [CSV_COLUMNS.join(',')]
  for (const f of frames) {
    const best = f.detections.reduce<FrameResult['detections'][number] | null>(
      (acc, d) => (!acc || d.confidence > acc.confidence ? d : acc), null,
    )
    const row = [
      f.frame_index, f.timestamp_s, f.vision_confidence, f.audio_confidence, f.fused_confidence,
      f.distance_m, f.speed_kmph, f.eta_s, f.direction_deg, f.direction_bearing,
      f.signal_state, f.decision, f.reason, f.latency_ms, f.detections.length,
      best ? best.box.map((v) => v.toFixed(0)).join(';') : '',
    ]
    lines.push(row.map(csvCell).join(','))
  }
  return new Blob([lines.join('\r\n')], { type: 'text/csv' })
}

export function shareText(run: ExportableRun, appUrl = location.origin): string {
  const s = run.summary
  const d = summariseFrames(run.frames, run.params)
  const peakV = s?.peak_vision_confidence ?? d.peak_vision_confidence
  const peakA = s?.peak_siren_confidence ?? d.peak_siren_confidence
  const peakF = s?.peak_fused_confidence ?? d.peak_fused_confidence
  const granted = s?.preemption_granted ?? d.preemption_granted
  const grantT = s?.grant_time_s ?? d.grant_time_s
  const firstT = s?.first_detection_s ?? d.first_detection_s
  const resp = s?.response_latency_s ?? d.response_latency_s
  const frames = s?.frames_analysed ?? d.frames_analysed
  const detFrames = s?.detection_frames ?? d.detection_frames
  const mean = s?.mean_latency_ms ?? d.mean_latency_ms
  const p95 = s?.p95_latency_ms ?? d.p95_latency_ms
  const blocked = Object.entries(s?.blocked_reasons ?? d.blocked_reasons)
  const p = run.params
  const defaults = p.vision_threshold === 0.65 && p.fused_threshold === 0.6 && p.safety_buffer_s === 5 && p.min_ttc_s === 2

  const clipLine = s
    ? `${run.clipLabel} (${s.resolution}, ${s.fps} fps, ${s.duration_s.toFixed(1)} s)`
    : run.clipLabel
  const lines = [
    `Siren Eyes — ${clipLine}`,
    granted
      ? `Preemption: GRANTED at ${grantT?.toFixed(2)} s` +
        (firstT != null ? ` (first detection ${firstT.toFixed(2)} s, response ${resp?.toFixed(2)} s)` : '')
      : `Preemption: withheld` + (firstT != null ? ` (first detection ${firstT.toFixed(2)} s)` : ' (no detection)'),
    `Peak confidence — vision ${peakV.toFixed(3)} · siren ${peakA.toFixed(3)} · fused ${peakF.toFixed(3)}`,
    `Frames: ${frames} analysed, ${detFrames} with detections · latency ${mean.toFixed(0)} ms mean, ${p95.toFixed(0)} ms p95`,
  ]
  if (blocked.length) {
    lines.push('Withheld: ' + blocked.map(([r, n]) => `${blockedReasonLabel(r)} ×${n}`).join(', '))
  }
  lines.push(
    `Parameters: vision ≥ ${p.vision_threshold}, fused ≥ ${p.fused_threshold}, ` +
      `buffer ${p.safety_buffer_s} s, TTC ≥ ${p.min_ttc_s} s${defaults ? ' (paper defaults)' : ' (modified)'}`,
  )
  if (run.status === 'cancelled') lines.push('Run stopped early by the operator.')
  if (run.precomputed) lines.push('Recorded run: computed once by the same pipeline, replayed here.')
  lines.push(appUrl)
  return lines.join('\n')
}

export async function copyText(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text)
    return true
  } catch {
    // Insecure context or denied permission: fall back to a hidden textarea.
    try {
      const ta = document.createElement('textarea')
      ta.value = text
      ta.setAttribute('readonly', '')
      ta.style.position = 'fixed'
      ta.style.opacity = '0'
      document.body.appendChild(ta)
      ta.select()
      const ok = document.execCommand('copy')
      ta.remove()
      return ok
    } catch {
      return false
    }
  }
}

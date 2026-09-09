/**
 * API types and client.
 *
 * Types mirror the dataclasses in `app/core/pipeline.py`; keep them in step
 * with that file.
 */

export interface Detection {
  box: [number, number, number, number]
  confidence: number
  label: string
}

export interface FrameResult {
  frame_index: number
  timestamp_s: number
  detections: Detection[]
  vision_confidence: number
  audio_confidence: number
  fused_confidence: number
  distance_m: number | null
  speed_kmph: number | null
  eta_s: number | null
  direction_deg: number | null
  direction_bearing: string
  direction_available: boolean
  signal_state: string
  decision: string
  reason: string
  latency_ms: number
}

export interface AnalysisSummary {
  video: string
  duration_s: number
  fps: number
  resolution: string
  frames_analysed: number
  detection_frames: number
  peak_vision_confidence: number
  peak_siren_confidence: number
  peak_fused_confidence: number
  preemption_granted: boolean
  first_detection_s: number | null
  grant_time_s: number | null
  response_latency_s: number | null
  audio_available: boolean
  audio_stereo: boolean
  audio_note: string
  mean_latency_ms: number
  p95_latency_ms: number
  stage_latency_ms: Record<string, number>
  blocked_reasons: Record<string, number>
}

export interface DemoClip {
  name: string
  file: string
  title: string
  description: string
  duration_s: number
  has_audio: boolean
  stereo: boolean
  source: string
  license: string
}

export interface SystemConfig {
  vision_confidence_threshold: number
  siren_threshold: number
  fusion: { alpha_vision: number; beta_audio: number; trigger_threshold: number }
  safety: { buffer_s: number; ttc_threshold_s: number; approach_tolerance_deg: number }
  geometry: { ambulance_width_m: number; camera_hfov_deg: number }
  audio: {
    sample_rate: number
    window_s: number
    bandpass_hz: [number, number]
    mic_baseline_m: number
  }
}

export interface HealthInfo {
  status: string
  app: string
  version: string
  models: {
    detector: string
    detector_loaded: boolean
    siren_classifier: {
      checkpoint: string
      kind: 'retrained' | 'legacy'
      reliable: boolean
      threshold?: number
      test_metrics?: Record<string, number> & { roc_auc?: number; roc_auc_std?: number }
      note?: string
    }
  }
}

const json = async <T,>(r: Response): Promise<T> => {
  if (!r.ok) throw new Error((await r.text()) || `HTTP ${r.status}`)
  return r.json() as Promise<T>
}

export const api = {
  health: () => fetch('/api/health').then(json<HealthInfo>),
  config: () => fetch('/api/config').then(json<SystemConfig>),
  demos: () => fetch('/api/demos').then(json<{ demos: DemoClip[] }>),

  analyseDemo: (name: string) =>
    fetch(`/api/analyses?demo=${encodeURIComponent(name)}`, { method: 'POST' }).then(
      json<{ job_id: string; source: string }>,
    ),

  analyseUpload: (file: File) => {
    const body = new FormData()
    body.append('file', file)
    return fetch('/api/analyses', { method: 'POST', body }).then(
      json<{ job_id: string; source: string }>,
    )
  },

  demoVideoUrl: (file: string) => `/api/demos/${encodeURIComponent(file)}/video`,
}

type StreamHandlers = {
  onFrame: (f: FrameResult) => void
  onSummary: (s: AnalysisSummary) => void
  onError: (message: string) => void
  onDone: () => void
}

/**
 * Subscribe to a running analysis.
 *
 * Returns a disposer; call it on unmount so an abandoned analysis does not
 * keep pushing frames into a dead component.
 */
export function streamAnalysis(jobId: string, h: StreamHandlers): () => void {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws'
  const ws = new WebSocket(`${proto}://${location.host}/api/analyses/${jobId}/stream`)

  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data)
    switch (msg.type) {
      case 'frame':
        h.onFrame(msg.frame as FrameResult)
        break
      case 'summary':
        h.onSummary(msg.summary as AnalysisSummary)
        break
      case 'error':
        h.onError(msg.message)
        break
      case 'done':
        h.onDone()
        break
    }
  }

  ws.onerror = () => h.onError('connection to the analysis stream failed')
  ws.onclose = (e) => {
    if (!e.wasClean && e.code !== 1000) h.onDone()
  }

  return () => {
    if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) {
      ws.close(1000)
    }
  }
}

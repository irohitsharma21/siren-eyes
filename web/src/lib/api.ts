/**
 * API types and client.
 *
 * Types mirror the dataclasses in `app/core/pipeline.py` and the response
 * shapes in `app/main.py`; keep them in step with those files.
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

/** The four thresholds a run may override; see `app/core/params.py`. */
export interface AnalysisParams {
  vision_threshold: number
  fused_threshold: number
  safety_buffer_s: number
  min_ttc_s: number
}

export type ParamName = keyof AnalysisParams

export interface ParamBound {
  default: number
  min: number
  max: number
  step: number
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
  /** Thresholds the run was gated with. */
  parameters: AnalysisParams
  /** True when the client stopped the run early; aggregates cover the frames analysed. */
  cancelled: boolean
  /** True when this result was recorded by scripts/precompute_demos.py rather
   *  than computed for this request. Only ever set for bundled demo clips. */
  precomputed?: boolean
}

export interface DemoClip {
  name: string
  file: string
  title: string
  description: string
  duration_s: number
  resolution?: string
  fps?: number
  size_mb?: number
  has_audio: boolean
  stereo: boolean
  audio_note?: string
  source: string
  license: string
  camera_hfov_deg?: number
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
  parameters: Record<ParamName, ParamBound>
  upload: { max_mb: number; extensions: string[] }
  frame_sample_stride: number
}

export interface BuildInfo {
  git_sha_short: string | null
  build_time: string | null
  host: string
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
  build: BuildInfo
  time: string
}

export interface VersionInfo extends BuildInfo {
  app: string
  version: string
  git_sha: string | null
  git_branch: string | null
  python: string
  environment: string
  started_at: string
  uptime_s: number
  time: string
}

export interface CreateJobResponse {
  job_id: string
  source: string
  parameters: AnalysisParams
  default_parameters: boolean
  precomputed: boolean
}

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message)
  }
}

/** Pull a readable message out of a FastAPI error body. */
function describeError(status: number, text: string): string {
  try {
    const body = JSON.parse(text)
    const detail = body?.detail
    if (typeof detail === 'string') return detail
    if (Array.isArray(detail)) {
      return detail
        .map((d) => `${(d.loc ?? []).slice(-1)[0] ?? ''}: ${d.msg}`.replace(/^: /, ''))
        .join('; ')
    }
  } catch {
    /* not JSON */
  }
  return text || `HTTP ${status}`
}

const json = async <T,>(r: Response): Promise<T> => {
  if (!r.ok) throw new ApiError(r.status, describeError(r.status, await r.text()))
  return r.json() as Promise<T>
}

export function paramsQuery(params: AnalysisParams | null): string {
  if (!params) return ''
  const q = new URLSearchParams()
  for (const [k, v] of Object.entries(params)) q.set(k, String(v))
  return q.toString()
}

export const api = {
  health: () => fetch('/api/health').then(json<HealthInfo>),
  version: () => fetch('/api/version').then(json<VersionInfo>),
  config: () => fetch('/api/config').then(json<SystemConfig>),
  demos: () => fetch('/api/demos').then(json<{ demos: DemoClip[] }>),

  analyseDemo: (name: string, params: AnalysisParams | null) => {
    const q = paramsQuery(params)
    return fetch(`/api/analyses?demo=${encodeURIComponent(name)}${q ? `&${q}` : ''}`, {
      method: 'POST',
    }).then(json<CreateJobResponse>)
  },

  /**
   * Upload with progress. `fetch` cannot report upload progress, so this one
   * call uses XMLHttpRequest; the returned `abort` cancels it mid-flight.
   */
  analyseUpload: (
    file: File,
    params: AnalysisParams | null,
    onProgress?: (fraction: number) => void,
  ): { promise: Promise<CreateJobResponse>; abort: () => void } => {
    const xhr = new XMLHttpRequest()
    const q = paramsQuery(params)
    const promise = new Promise<CreateJobResponse>((resolve, reject) => {
      xhr.open('POST', `/api/analyses${q ? `?${q}` : ''}`)
      xhr.responseType = 'text'
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable && onProgress) onProgress(e.loaded / e.total)
      }
      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          try {
            resolve(JSON.parse(xhr.responseText) as CreateJobResponse)
          } catch {
            reject(new ApiError(xhr.status, 'malformed response from server'))
          }
        } else {
          reject(new ApiError(xhr.status, describeError(xhr.status, xhr.responseText)))
        }
      }
      xhr.onerror = () => reject(new ApiError(0, 'upload failed — network error'))
      xhr.onabort = () => reject(new ApiError(0, 'upload cancelled'))
      const body = new FormData()
      body.append('file', file)
      xhr.send(body)
    })
    return { promise, abort: () => xhr.abort() }
  },

  cancel: (jobId: string) =>
    fetch(`/api/analyses/${jobId}`, { method: 'DELETE' }).then(
      json<{ job_id: string; status: string }>,
    ),

  demoVideoUrl: (file: string) => `/api/demos/${encodeURIComponent(file)}/video`,
}

export type StreamHandlers = {
  onStarted?: (precomputed: boolean) => void
  onStatus?: (phase: string) => void
  onFrame: (f: FrameResult) => void
  onSummary: (s: AnalysisSummary) => void
  onCancelled?: (frames: number) => void
  onError: (message: string) => void
  onDone: () => void
}

/**
 * Subscribe to a running analysis.
 *
 * Returns a disposer; call it on unmount so an abandoned analysis does not
 * keep pushing frames into a dead component. The server treats a closed
 * socket as a cancellation, so disposing also stops the CPU work.
 */
export function streamAnalysis(jobId: string, h: StreamHandlers): () => void {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws'
  const ws = new WebSocket(`${proto}://${location.host}/api/analyses/${jobId}/stream`)
  let finished = false

  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data)
    switch (msg.type) {
      case 'started':
        h.onStarted?.(Boolean(msg.precomputed))
        break
      case 'status':
        h.onStatus?.(String(msg.phase ?? ''))
        break
      case 'frame':
        h.onFrame(msg.frame as FrameResult)
        break
      case 'summary':
        h.onSummary(msg.summary as AnalysisSummary)
        break
      case 'cancelled':
        finished = true
        h.onCancelled?.(Number(msg.frames ?? 0))
        break
      case 'error':
        finished = true
        h.onError(msg.message)
        break
      case 'done':
        finished = true
        h.onDone()
        break
    }
  }

  ws.onerror = () => {
    if (!finished) h.onError('connection to the analysis stream failed')
  }
  ws.onclose = (e) => {
    // A close without a terminal event means the server went away mid-run
    // (free instances get recycled); report it rather than hanging.
    if (!finished && !e.wasClean && e.code !== 1000) h.onError('analysis stream closed unexpectedly')
    else if (!finished) h.onDone()
  }

  return () => {
    finished = true
    if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) {
      ws.close(1000)
    }
  }
}

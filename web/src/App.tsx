import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  Activity, AlertTriangle, ChevronDown, Clock, Compass, Film, Gauge, HelpCircle,
  History as HistoryIcon, Info, Layers, Loader2, Play, RefreshCw, ShieldCheck, Siren, Square, Upload,
} from 'lucide-react'

import {
  api, streamAnalysis, ApiError,
  type AnalysisParams, type AnalysisSummary, type DemoClip, type FrameResult,
  type HealthInfo, type SystemConfig, type VersionInfo,
} from './lib/api'
import { deriveEvents, indexAt, paramsEqual } from './lib/analysis'
import { download, safeName, type ExportableRun } from './lib/export'
import {
  clearHistory, loadHistory, rememberSourceUrl, removeRun, saveRun, sourceUrlFor,
  type RunRecord, type RunStatus,
} from './lib/history'
import { ThemeToggle } from './lib/theme'
import { VideoStage, renderSnapshot, type OverlayMode, type StageStatus } from './components/VideoStage'
import { Timeline } from './components/Timeline'
import { Results } from './components/Results'
import { ParametersButton, defaultsFrom } from './components/Parameters'
import { History } from './components/History'
import { Explainer, ShortcutSheet, UploadZone, useExplainer, validateUpload } from './components/Onboarding'
import { Compass as CompassDial, Kinematics, Meters, Notice, Pipeline, SignalHead } from './components/Panels'

/* Paper defaults; replaced by the server's `parameters` block once /api/config loads. */
const PAPER_DEFAULTS: AnalysisParams = {
  vision_threshold: 0.65, fused_threshold: 0.6, safety_buffer_s: 5, min_ttc_s: 2,
}

type Source = {
  url: string
  label: string
  kind: 'demo' | 'upload'
  demoFile?: string
}

/** What the stage, timeline and results are currently showing. */
interface ViewRun {
  id: string
  createdAt: string
  clipLabel: string
  clipKind: 'demo' | 'upload'
  demoFile?: string
  params: AnalysisParams
  precomputed: boolean
}

const isTyping = (t: EventTarget | null) => {
  const el = t as HTMLElement | null
  if (!el) return false
  const tag = el.tagName
  return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || el.isContentEditable
}

export default function App() {
  // ── server state ───────────────────────────────────────────────────
  const [health, setHealth] = useState<HealthInfo | null>(null)
  const [version, setVersion] = useState<VersionInfo | null>(null)
  const [backendDown, setBackendDown] = useState(false)
  const [wakeAttempts, setWakeAttempts] = useState(0)
  const [config, setConfig] = useState<SystemConfig | null>(null)
  const [demos, setDemos] = useState<DemoClip[]>([])

  // ── clip selection ─────────────────────────────────────────────────
  const [source, setSource] = useState<Source | null>(null)
  const [selectedDemo, setSelectedDemo] = useState<DemoClip | null>(null)
  const [pendingFile, setPendingFile] = useState<File | null>(null)
  const [uploadError, setUploadError] = useState<string | null>(null)
  const [missingVideo, setMissingVideo] = useState(false)
  const [sourceSize, setSourceSize] = useState<{ w: number; h: number } | null>(null)
  const [videoDuration, setVideoDuration] = useState(0)

  // ── run state ──────────────────────────────────────────────────────
  const [params, setParams] = useState<AnalysisParams>(PAPER_DEFAULTS)
  const [status, setStatus] = useState<StageStatus>('idle')
  const [phase, setPhase] = useState('')
  const [uploadProgress, setUploadProgress] = useState<number | null>(null)
  const [precomputed, setPrecomputed] = useState(false)
  const [frames, setFrames] = useState<FrameResult[]>([])
  const [summary, setSummary] = useState<AnalysisSummary | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [view, setView] = useState<ViewRun | null>(null)

  // ── playback / overlay ─────────────────────────────────────────────
  const [playhead, setPlayhead] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [muted, setMuted] = useState(true)
  const [mode, setMode] = useState<OverlayMode>('gated')

  // ── history + chrome ───────────────────────────────────────────────
  const [runs, setRuns] = useState<RunRecord[]>(() => loadHistory())
  const [activeRunId, setActiveRunId] = useState<string | null>(null)
  const [showShortcuts, setShowShortcuts] = useState(false)
  const [paramsOpen, setParamsOpen] = useState(false)
  const [clipMenuOpen, setClipMenuOpen] = useState(false)
  const [toast, setToast] = useState<string | null>(null)
  const explainer = useExplainer()

  const videoRef = useRef<HTMLVideoElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const clipMenuRef = useRef<HTMLDivElement>(null)
  const disposeRef = useRef<(() => void) | null>(null)
  const abortUploadRef = useRef<(() => void) | null>(null)
  const jobIdRef = useRef<string | null>(null)
  const precomputedRef = useRef(false)
  const framesRef = useRef<FrameResult[]>([])
  const summaryRef = useRef<AnalysisSummary | null>(null)
  const viewRef = useRef<ViewRun | null>(null)
  const sourceSizeRef = useRef<{ w: number; h: number } | null>(null)
  const pendingRef = useRef<FrameResult[]>([])
  const flushRafRef = useRef<number | null>(null)
  const stopTimerRef = useRef<number | null>(null)
  const runsRef = useRef<RunRecord[]>(runs)
  const autorunRef = useRef(false)
  runsRef.current = runs
  sourceSizeRef.current = sourceSize

  const busy = status === 'uploading' || status === 'connecting' || status === 'analysing'

  const showToast = useCallback((message: string) => {
    setToast(message)
    window.setTimeout(() => setToast((t) => (t === message ? null : t)), 2400)
  }, [])

  // ── bootstrap ──────────────────────────────────────────────────────
  const selectDemo = useCallback((clip: DemoClip) => {
    disposeRef.current?.()
    setSelectedDemo(clip)
    setPendingFile(null)
    setUploadError(null)
    setMissingVideo(false)
    setSource({ url: api.demoVideoUrl(clip.file), label: clip.title, kind: 'demo', demoFile: clip.file })
    setSourceSize(null)
    setFrames([]); framesRef.current = []
    setSummary(null); summaryRef.current = null
    setView(null); viewRef.current = null
    setActiveRunId(null)
    setError(null)
    setStatus('idle')
    setPlayhead(0)
    setPlaying(false)
  }, [])

  useEffect(() => {
    let cancelled = false
    let timer: number | undefined
    const connect = async () => {
      try {
        const h = await api.health()
        if (cancelled) return
        setHealth(h)
        setBackendDown(false)
        const [cfg, d, v] = await Promise.all([
          api.config().catch(() => null),
          api.demos().catch(() => ({ demos: [] as DemoClip[] })),
          api.version().catch(() => null),
        ])
        if (cancelled) return
        if (cfg) {
          setConfig(cfg)
          // Adopt the server's defaults unless the visitor already moved a slider.
          setParams((p) => (paramsEqual(p, PAPER_DEFAULTS) ? defaultsFrom(cfg.parameters) : p))
        }
        if (v) setVersion(v)
        setDemos(d.demos)
        // Deep link: ?clip=<file>&autorun=1 opens a clip and starts it, so a
        // shared URL lands on a running analysis rather than a blank stage.
        const q = new URLSearchParams(location.search)
        const wanted = q.get('clip')
        const clip = d.demos.find((x) => x.file === wanted || x.name === wanted) ?? d.demos[0]
        if (clip) selectDemo(clip)
        if (q.get('autorun') === '1') autorunRef.current = true
      } catch {
        if (cancelled) return
        setBackendDown(true)
        setWakeAttempts((n) => n + 1)
        timer = window.setTimeout(connect, 4000)
      }
    }
    connect()
    return () => {
      cancelled = true
      if (timer) window.clearTimeout(timer)
      disposeRef.current?.()
    }
  }, [selectDemo])

  // Close the clip menu on outside click / Escape.
  useEffect(() => {
    if (!clipMenuOpen) return
    const onDoc = (e: MouseEvent) => {
      if (clipMenuRef.current && !clipMenuRef.current.contains(e.target as Node)) setClipMenuOpen(false)
    }
    document.addEventListener('mousedown', onDoc)
    return () => document.removeEventListener('mousedown', onDoc)
  }, [clipMenuOpen])

  const selectFile = useCallback((file: File) => {
    const problem = validateUpload(file, config)
    if (problem) {
      setUploadError(problem)
      showToast('That file cannot be analysed')
      return
    }
    disposeRef.current?.()
    setUploadError(null)
    setPendingFile(file)
    setSelectedDemo(null)
    setMissingVideo(false)
    setSource({ url: URL.createObjectURL(file), label: file.name, kind: 'upload' })
    setSourceSize(null)
    setFrames([]); framesRef.current = []
    setSummary(null); summaryRef.current = null
    setView(null); viewRef.current = null
    setActiveRunId(null)
    setError(null)
    setStatus('idle')
    setPlayhead(0)
    setPlaying(false)
  }, [config, showToast])

  // ── frame ingest ───────────────────────────────────────────────────
  const syncVideo = useCallback((ts: number) => {
    const v = videoRef.current
    if (!v) return
    if (precomputedRef.current) {
      // Replays arrive near real time: let the video play and nudge it.
      const drift = v.currentTime - ts
      if (drift > 0.3) { v.pause(); v.currentTime = ts }
      else if (drift < -0.5) v.currentTime = ts
      if (v.paused && drift <= 0.3 && !v.ended) v.play().catch(() => {})
    } else {
      // Live CPU analysis is far slower than real time: step the picture to
      // each analysed frame rather than letting it run ahead.
      if (!v.paused) v.pause()
      v.currentTime = ts
    }
  }, [])

  const flush = useCallback(() => {
    flushRafRef.current = null
    const batch = pendingRef.current.splice(0)
    if (!batch.length) return
    framesRef.current = framesRef.current.concat(batch)
    setFrames(framesRef.current)
    const last = batch[batch.length - 1]
    setPlayhead(last.timestamp_s)
    syncVideo(last.timestamp_s)
  }, [syncVideo])

  const enqueue = useCallback((f: FrameResult) => {
    pendingRef.current.push(f)
    if (flushRafRef.current === null) flushRafRef.current = requestAnimationFrame(flush)
  }, [flush])

  const persistRun = useCallback((finalStatus: RunStatus, err?: string) => {
    const meta = viewRef.current
    if (!meta) return
    if (!framesRef.current.length && finalStatus !== 'error') return
    const record: RunRecord = {
      id: meta.id,
      createdAt: meta.createdAt,
      clip: { label: meta.clipLabel, kind: meta.clipKind, demoFile: meta.demoFile },
      params: meta.params,
      status: finalStatus,
      precomputed: meta.precomputed,
      frames: framesRef.current,
      summary: summaryRef.current,
      sourceSize: sourceSizeRef.current,
      error: err,
    }
    setRuns(saveRun(runsRef.current, record))
    setActiveRunId(meta.id)
  }, [])

  const finish = useCallback((finalStatus: RunStatus, err?: string) => {
    if (flushRafRef.current !== null) cancelAnimationFrame(flushRafRef.current)
    flush()
    if (stopTimerRef.current) { window.clearTimeout(stopTimerRef.current); stopTimerRef.current = null }
    disposeRef.current?.()
    disposeRef.current = null
    jobIdRef.current = null
    setPhase('')
    setStatus(finalStatus)
    if (err) setError(err)
    const v = videoRef.current
    if (v && !v.paused) v.pause()
    persistRun(finalStatus, err)
  }, [flush, persistRun])

  // ── run / stop ─────────────────────────────────────────────────────
  const run = useCallback(async () => {
    if (busy || !source || missingVideo) return
    disposeRef.current?.()
    setFrames([]); framesRef.current = []
    setSummary(null); summaryRef.current = null
    setError(null)
    setPlayhead(0)
    setPlaying(false)
    setActiveRunId(null)
    setPrecomputed(false); precomputedRef.current = false
    setClipMenuOpen(false)
    setParamsOpen(false)

    const meta: ViewRun = {
      id: `${Date.now().toString(36)}${Math.random().toString(36).slice(2, 6)}`,
      createdAt: new Date().toISOString(),
      clipLabel: source.label,
      clipKind: source.kind,
      demoFile: source.demoFile,
      params: { ...params },
      precomputed: false,
    }
    setView(meta); viewRef.current = meta
    if (source.kind === 'upload') rememberSourceUrl(meta.id, source.url)

    try {
      let job
      if (source.kind === 'upload') {
        if (!pendingFile) throw new Error('Choose a video first.')
        setStatus('uploading')
        setUploadProgress(0)
        const { promise, abort } = api.analyseUpload(pendingFile, params, setUploadProgress)
        abortUploadRef.current = abort
        job = await promise
        abortUploadRef.current = null
        setUploadProgress(null)
      } else {
        setStatus('connecting')
        job = await api.analyseDemo(source.demoFile!, params)
      }

      jobIdRef.current = job.job_id
      setPrecomputed(job.precomputed); precomputedRef.current = job.precomputed
      const withPre = { ...meta, precomputed: job.precomputed }
      setView(withPre); viewRef.current = withPre
      setStatus('connecting')
      setPhase(job.precomputed ? 'replaying recorded analysis' : 'queuing')

      const v = videoRef.current
      if (v) { v.pause(); v.currentTime = 0 }

      disposeRef.current = streamAnalysis(job.job_id, {
        onStarted: (pre) => {
          setStatus('analysing')
          setPrecomputed(pre); precomputedRef.current = pre
        },
        onStatus: (p) => setPhase(p),
        onFrame: enqueue,
        onSummary: (s) => { summaryRef.current = s; setSummary(s) },
        onCancelled: () => finish('cancelled'),
        onError: (m) => finish('error', m),
        onDone: () => finish('done'),
      })
    } catch (e) {
      abortUploadRef.current = null
      setUploadProgress(null)
      if (e instanceof ApiError && e.message === 'upload cancelled') {
        setStatus('idle')
        return
      }
      const msg = e instanceof Error ? e.message : String(e)
      setError(msg)
      setStatus('error')
    }
  }, [busy, source, missingVideo, params, pendingFile, enqueue, finish])

  const stop = useCallback(() => {
    if (status === 'uploading') {
      abortUploadRef.current?.()
      return
    }
    const id = jobIdRef.current
    if (!id) return
    setPhase('stopping')
    api.cancel(id).catch(() => { /* the socket close below is the fallback */ })
    // If the server never acknowledges (connection lost), wind down locally.
    stopTimerRef.current = window.setTimeout(() => finish('cancelled'), 6000)
  }, [status, finish])

  useEffect(() => {
    if (autorunRef.current && source && config && status === 'idle') {
      autorunRef.current = false
      run()
    }
  }, [source, config, status, run])

  // ── playback ───────────────────────────────────────────────────────
  useEffect(() => {
    if (!playing) return
    const v = videoRef.current
    if (!v) { setPlaying(false); return }
    if (v.ended || (videoDuration && v.currentTime >= videoDuration - 0.05)) v.currentTime = 0
    v.play().catch(() => setPlaying(false))
    let raf = 0
    const tick = () => {
      setPlayhead(v.currentTime)
      if (v.ended) { setPlaying(false); return }
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => { cancelAnimationFrame(raf); v.pause() }
  }, [playing, videoDuration])

  const duration = summary?.duration_s || videoDuration || (frames.length ? frames[frames.length - 1].timestamp_s : 0)

  const seek = useCallback((t: number) => {
    if (busy) return
    const clamped = Math.min(Math.max(t, 0), duration || t)
    setPlayhead(clamped)
    const v = videoRef.current
    if (v && Math.abs(v.currentTime - clamped) > 0.02) v.currentTime = clamped
  }, [busy, duration])

  const step = useCallback((n: number) => {
    if (!frames.length || busy) return
    setPlaying(false)
    const i = indexAt(frames, playhead)
    const j = Math.min(Math.max(i + n, 0), frames.length - 1)
    seek(frames[j].timestamp_s)
  }, [frames, busy, playhead, seek])

  const togglePlay = useCallback(() => {
    if (busy || !frames.length || missingVideo) return
    setPlaying((p) => !p)
  }, [busy, frames.length, missingVideo])

  const frameIndex = useMemo(() => indexAt(frames, playhead), [frames, playhead])
  const currentFrame = frameIndex >= 0 ? frames[frameIndex] : null
  const viewParams = view?.params ?? params
  const events = useMemo(() => deriveEvents(frames, viewParams), [frames, viewParams])

  // ── snapshot ───────────────────────────────────────────────────────
  const snapshot = useCallback(async () => {
    const v = videoRef.current
    if (!v || frameIndex < 0 || missingVideo) return
    const blob = await renderSnapshot(v, frames, frameIndex, viewParams.vision_threshold, mode, view?.clipLabel ?? source?.label ?? 'clip')
    if (!blob) { showToast('Could not capture the frame'); return }
    const t = frames[frameIndex].timestamp_s.toFixed(2).replace('.', '_')
    download(blob, `siren-eyes_${safeName(view?.clipLabel ?? source?.label ?? 'clip')}_t${t}s.png`)
    showToast('Snapshot downloaded')
  }, [frameIndex, frames, missingVideo, mode, view, source, viewParams, showToast])

  // ── history ────────────────────────────────────────────────────────
  const loadRun = useCallback((r: RunRecord) => {
    if (busy) return
    disposeRef.current?.()
    setActiveRunId(r.id)
    setFrames(r.frames); framesRef.current = r.frames
    setSummary(r.summary); summaryRef.current = r.summary
    const meta: ViewRun = {
      id: r.id, createdAt: r.createdAt, clipLabel: r.clip.label, clipKind: r.clip.kind,
      demoFile: r.clip.demoFile, params: r.params, precomputed: r.precomputed,
    }
    setView(meta); viewRef.current = meta
    setPrecomputed(r.precomputed); precomputedRef.current = r.precomputed
    setError(r.error ?? null)
    setStatus(r.status)
    setPlaying(false)
    setPlayhead(0)
    setPendingFile(null)
    setUploadError(null)

    if (r.clip.kind === 'demo' && r.clip.demoFile) {
      const clip = demos.find((d) => d.file === r.clip.demoFile) ?? null
      setSelectedDemo(clip)
      setMissingVideo(false)
      setSource({ url: api.demoVideoUrl(r.clip.demoFile), label: r.clip.label, kind: 'demo', demoFile: r.clip.demoFile })
    } else {
      setSelectedDemo(null)
      const url = sourceUrlFor(r.id)
      setMissingVideo(!url)
      setSource({ url: url ?? '', label: r.clip.label, kind: 'upload' })
    }
    if (r.sourceSize) setSourceSize(r.sourceSize)
    const v = videoRef.current
    if (v) { v.pause(); v.currentTime = 0 }
  }, [busy, demos])

  // ── keyboard ───────────────────────────────────────────────────────
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.ctrlKey || e.metaKey || e.altKey) return
      if (e.defaultPrevented) return
      const typing = isTyping(e.target)
      const onButton = (e.target as HTMLElement | null)?.tagName === 'BUTTON'

      if (e.key === '?' && !typing) { e.preventDefault(); setShowShortcuts((s) => !s); return }
      if (e.key === 'Escape') {
        if (showShortcuts) { setShowShortcuts(false); return }
        if (clipMenuOpen) { setClipMenuOpen(false); return }
        if (paramsOpen) return // the popover handles its own Escape
        if (busy) { e.preventDefault(); stop() }
        return
      }
      if (typing) return

      switch (e.key) {
        case ' ':
          if (onButton) return
          e.preventDefault(); togglePlay(); break
        case 'ArrowLeft':
          e.preventDefault(); step(e.shiftKey ? -10 : -1); break
        case 'ArrowRight':
          e.preventDefault(); step(e.shiftKey ? 10 : 1); break
        case 'Home':
          if (frames.length) { e.preventDefault(); setPlaying(false); seek(0) }
          break
        case 'End':
          if (frames.length) { e.preventDefault(); setPlaying(false); seek(frames[frames.length - 1].timestamp_s) }
          break
        case 'r': case 'R':
          if (!busy) { e.preventDefault(); run() }
          break
        case 's': case 'S':
          e.preventDefault(); snapshot(); break
        case 'c': case 'C':
          if (frames.length) { e.preventDefault(); setMode((m) => (m === 'raw' ? 'gated' : 'raw')) }
          break
        case 'p': case 'P':
          e.preventDefault(); setParamsOpen((o) => !o); break
        case 'u': case 'U':
          if (!busy) { e.preventDefault(); fileInputRef.current?.click() }
          break
        case 'm': case 'M':
          e.preventDefault(); setMuted((m) => !m); break
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [busy, clipMenuOpen, frames, paramsOpen, run, seek, showShortcuts, snapshot, step, stop, togglePlay])

  // ── derived UI bits ────────────────────────────────────────────────
  const progress = duration > 0 ? Math.min(playhead / duration, 1) : 0
  const legacyModel = health?.models.siren_classifier.kind === 'legacy'
  const serverDefaults = config ? defaultsFrom(config.parameters) : PAPER_DEFAULTS
  const paramsModified = !paramsEqual(params, serverDefaults)
  const willRunLive = source?.kind === 'upload' || paramsModified
  const canRun = Boolean(source) && !busy && !missingVideo && !backendDown
  const hasResults = (status === 'done' || status === 'cancelled') && frames.length > 0

  const exportable: ExportableRun | null = view
    ? {
      id: view.id, createdAt: view.createdAt, clipLabel: view.clipLabel, params: view.params,
      status, precomputed: view.precomputed, frames, summary,
    }
    : null

  const statusLine = (() => {
    switch (status) {
      case 'uploading': return uploadProgress != null ? `Uploading ${(uploadProgress * 100).toFixed(0)}%` : 'Uploading'
      case 'connecting': return phase ? `${phase}…` : 'Starting…'
      case 'analysing': return phase && !frames.length ? `${phase}…` : precomputed ? 'Replaying recorded analysis' : 'Analysing live on CPU'
      case 'done': return precomputed ? 'Recorded run' : 'Analysis complete'
      case 'cancelled': return 'Stopped'
      case 'error': return 'Failed'
      default: return source ? 'Ready' : 'No clip'
    }
  })()

  return (
    <div className="app">
      {/* ── App bar ──────────────────────────────────────────────── */}
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark"><Siren size={15} /></span>
          <span className="brand-text">
            Siren Eyes
            <span className="brand-sub">stereo-aware preemption</span>
          </span>
        </div>

        <div className="topbar-center">
          <div className="popover-anchor" ref={clipMenuRef}>
            <button
              className="btn btn-sm clip-select"
              onClick={() => setClipMenuOpen((o) => !o)}
              disabled={busy}
              aria-haspopup="menu"
              aria-expanded={clipMenuOpen}
              title="Choose a clip"
            >
              {source?.kind === 'upload' ? <Upload size={13} /> : <Film size={13} />}
              <span className="clip-select-label">{source?.label ?? 'Choose a clip'}</span>
              <ChevronDown size={13} />
            </button>
            {clipMenuOpen && (
              <div className="menu" role="menu">
                <div className="menu-section">Bundled demos</div>
                {demos.length === 0 && <div className="menu-empty">No demo clips on this server</div>}
                {demos.map((d) => (
                  <button
                    key={d.name}
                    role="menuitemradio"
                    aria-checked={selectedDemo?.name === d.name}
                    className="menu-item"
                    data-selected={selectedDemo?.name === d.name}
                    onClick={() => { selectDemo(d); setClipMenuOpen(false) }}
                  >
                    <Film size={13} />
                    <span className="menu-item-text">
                      <span>{d.title}</span>
                      <span className="menu-item-meta">
                        {d.duration_s.toFixed(0)} s {'·'} {d.has_audio ? (d.stereo ? 'stereo' : 'mono') : 'no audio'} {'·'} recorded result
                      </span>
                    </span>
                  </button>
                ))}
                <div className="menu-sep" />
                <button
                  role="menuitem"
                  className="menu-item"
                  onClick={() => { setClipMenuOpen(false); fileInputRef.current?.click() }}
                >
                  <Upload size={13} />
                  <span className="menu-item-text">
                    <span>Upload a video{'…'}</span>
                    <span className="menu-item-meta">analysed live {'·'} up to {config?.upload.max_mb ?? 200} MB</span>
                  </span>
                </button>
              </div>
            )}
          </div>

          {busy ? (
            <button className="btn btn-danger btn-sm" onClick={stop} title="Stop (Esc)">
              <Square size={12} /> Stop
            </button>
          ) : (
            <button className="btn btn-primary btn-sm" onClick={run} disabled={!canRun} title="Run analysis (R)">
              <Play size={12} /> Run
            </button>
          )}

          <ParametersButton
            params={params}
            bounds={config?.parameters ?? null}
            disabled={busy}
            recordedAvailable={source?.kind === 'demo'}
            open={paramsOpen}
            onOpenChange={setParamsOpen}
            onChange={setParams}
          />
        </div>

        <div className="topbar-right">
          <button className="btn btn-ghost btn-sm icon-only" onClick={() => setShowShortcuts(true)} title="Keyboard shortcuts (?)" aria-label="Keyboard shortcuts">
            <HelpCircle size={15} />
          </button>
          <HealthChip health={health} version={version} down={backendDown} />
          <ThemeToggle />
        </div>
      </header>

      <div className="shell">
        {/* ── Stage column ───────────────────────────────────────── */}
        <main className="stage-col">
          {explainer.visible && (
            <Explainer
              onDismiss={explainer.dismiss}
              onRunDemo={() => { explainer.dismiss(); run() }}
              canRun={canRun}
            />
          )}

          {error && (
            <div className="banner banner-error" role="alert">
              <AlertTriangle size={14} />
              <div className="banner-body">
                <strong>{status === 'error' ? 'Analysis failed' : 'Problem'}</strong> {'—'} {error}
              </div>
              <button className="btn btn-ghost btn-sm" onClick={() => setError(null)} aria-label="Dismiss">{'×'}</button>
            </div>
          )}

          <VideoStage
            src={source?.url ?? null}
            frames={frames}
            frameIndex={frameIndex}
            sourceSize={sourceSize}
            status={status}
            phase={phase}
            precomputed={precomputed}
            threshold={viewParams.vision_threshold}
            mode={mode}
            playing={playing}
            muted={muted}
            missingVideo={missingVideo}
            clipLabel={source?.label ?? ''}
            videoRef={videoRef}
            onMetadata={(size, dur) => { setSourceSize(size); setVideoDuration(dur) }}
            onPickFile={() => fileInputRef.current?.click()}
            onDropFile={selectFile}
            onTogglePlay={togglePlay}
            onToggleMute={() => setMuted((m) => !m)}
            onToggleMode={() => setMode((m) => (m === 'raw' ? 'gated' : 'raw'))}
            onSnapshot={snapshot}
          />

          <div className="transport">
            <span className="transport-status" data-busy={busy}>
              {busy && <Loader2 size={12} className="spin" />}
              {statusLine}
            </span>
            <div className="progress transport-bar">
              <div className="progress-fill" style={{ width: `${(status === 'uploading' ? (uploadProgress ?? 0) : progress) * 100}%` }} />
            </div>
            <span className="mono transport-time">
              {playhead.toFixed(2)} s / {duration ? duration.toFixed(2) : '—'} s
            </span>
            <span className="mono transport-frames">{frames.length} frames</span>
          </div>

          <div className="panel">
            <div className="panel-head">
              <span className="panel-title"><Activity size={12} /> Timeline</span>
              {summary ? (
                <span className="panel-meta mono">
                  {summary.mean_latency_ms.toFixed(0)} ms mean {'·'} p95 {summary.p95_latency_ms.toFixed(0)} ms
                </span>
              ) : busy && frames.length ? (
                <span className="panel-meta mono">frame {frames[frames.length - 1].frame_index}</span>
              ) : null}
            </div>
            <div className="panel-body">
              <Timeline
                frames={frames}
                duration={duration}
                current={playhead}
                params={viewParams}
                config={config}
                events={events}
                onSeek={(t) => { setPlaying(false); seek(t) }}
              />
            </div>
          </div>

          {hasResults && exportable && (
            <div className="panel">
              <div className="panel-head">
                <span className="panel-title"><ShieldCheck size={12} /> Results</span>
                <span className="panel-meta">{new Date(exportable.createdAt).toLocaleString()}</span>
              </div>
              <div className="panel-body">
                <Results run={exportable} config={config} onSeek={(t) => { setPlaying(false); seek(t) }} />
              </div>
            </div>
          )}
        </main>

        {/* ── Instrument column ──────────────────────────────────── */}
        <aside className="instruments">
          {backendDown && (
            <Notice kind="warn">
              <strong>Waking the server</strong> {'—'} free instances sleep after 15 minutes idle
              and take about a minute to start. Retrying automatically ({wakeAttempts}).
            </Notice>
          )}
          {legacyModel && (
            <Notice kind="warn">
              Running the legacy siren checkpoint, which does not discriminate
              siren from non-siren audio. Run <code>scripts/train_siren.py</code>{' '}
              to produce a working classifier.
            </Notice>
          )}
          {!busy && source && willRunLive && !missingVideo && (
            <Notice kind={source.kind === 'demo' ? 'warn' : 'info'}>
              {source.kind === 'demo' ? (
                <>
                  <strong>Parameters differ from the defaults</strong>, so this demo will be analysed
                  live on the CPU instead of replaying its recording. On the free tier that is
                  minutes, not seconds; reset the parameters to replay instantly.
                </>
              ) : (
                <>
                  <strong>Uploads are analysed live on the CPU.</strong> Expect roughly 1{'–'}3 s per
                  analysed frame on the free tier; the run can be stopped at any point and keeps what
                  it has.
                </>
              )}
            </Notice>
          )}

          <Panel title="Signal decision" Icon={ShieldCheck}>
            <SignalHead frame={currentFrame} />
          </Panel>

          <Panel title="Detection confidence" Icon={Gauge}>
            <Meters frame={currentFrame} config={config} params={viewParams} />
          </Panel>

          <Panel title="Kinematics" Icon={Clock} meta={source?.kind === 'upload' || (view?.clipKind === 'upload') ? 'uncalibrated' : selectedDemo?.camera_hfov_deg ? `HFOV ${selectedDemo.camera_hfov_deg}°` : undefined}>
            <Kinematics frame={currentFrame} />
          </Panel>

          <Panel title="Direction of arrival" Icon={Compass}>
            <CompassDial frame={currentFrame} />
          </Panel>

          <Panel title="Pipeline" Icon={Layers}>
            <Pipeline frame={currentFrame} config={config} params={viewParams} />
          </Panel>

          <Panel title="Source" Icon={Film}>
            {selectedDemo && source?.kind === 'demo' && (
              <div className="source-desc">
                <p>{selectedDemo.description}</p>
                <p className="muted">{selectedDemo.source}. {selectedDemo.license}.</p>
              </div>
            )}
            <UploadZone
              config={config}
              pendingFile={pendingFile}
              error={uploadError}
              disabled={busy}
              progress={status === 'uploading' ? uploadProgress : null}
              inputRef={fileInputRef}
              onFile={selectFile}
            />
          </Panel>

          <Panel title="Recent runs" Icon={HistoryIcon} meta={runs.length ? `${runs.length}` : undefined}>
            <History
              runs={runs}
              activeId={activeRunId}
              busy={busy}
              onSelect={loadRun}
              onRemove={(id) => setRuns(removeRun(runsRef.current, id))}
              onClear={() => setRuns(clearHistory())}
            />
          </Panel>

          {!explainer.visible && (
            <button className="btn btn-ghost btn-sm" onClick={explainer.show} style={{ alignSelf: 'flex-start' }}>
              <Info size={12} /> How it works
            </button>
          )}
        </aside>
      </div>

      <footer className="footer">
        <div>
          Implements <em>Stereo-Aware Multimodal Ambulance Detection and Real-Time
          Traffic Signal Preemption Using Edge AI</em> {'—'} Dalal, Gupta &amp; Sharma.
          {health?.models.siren_classifier.test_metrics?.roc_auc != null && (
            <> Siren classifier ROC-AUC{' '}
              {health.models.siren_classifier.test_metrics.roc_auc.toFixed(3)}
              {health.models.siren_classifier.test_metrics.roc_auc_std != null && (
                <> &plusmn; {health.models.siren_classifier.test_metrics.roc_auc_std.toFixed(3)}</>
              )}
              {' '}over ESC-50 5-fold cross-validation.</>
          )}
        </div>
        <div className="footer-build mono">
          {health ? `v${health.version}` : 'v—'}
          {version?.git_sha_short && <> {'·'} build <a href={`#${version.git_sha}`} title={version.git_sha ?? ''}>{version.git_sha_short}</a></>}
          {version?.build_time && <> {'·'} built {new Date(version.build_time).toLocaleString()}</>}
          {version?.host && <> {'·'} {version.host}</>}
          {' · '}<a href="/api/docs" target="_blank" rel="noreferrer">API docs</a>
        </div>
      </footer>

      <ShortcutSheet open={showShortcuts} onClose={() => setShowShortcuts(false)} />
      {toast && <div className="toast" role="status">{toast}</div>}
    </div>
  )
}

/* ── Small pieces ────────────────────────────────────────────────── */

function Panel({
  title, Icon, meta, children,
}: { title: string; Icon: typeof Gauge; meta?: string; children: React.ReactNode }) {
  return (
    <div className="panel">
      <div className="panel-head">
        <span className="panel-title"><Icon size={12} /> {title}</span>
        {meta && <span className="panel-meta mono">{meta}</span>}
      </div>
      <div className="panel-body">{children}</div>
    </div>
  )
}

function HealthChip({ health, version, down }: { health: HealthInfo | null; version: VersionInfo | null; down: boolean }) {
  const ok = Boolean(health?.models.detector_loaded) && !down
  const siren = health?.models.siren_classifier
  return (
    <div className="popover-anchor health-anchor">
      <button className="status-chip" aria-describedby="health-tip" aria-label="Server status">
        {down ? <RefreshCw size={10} className="spin" /> : <span className="status-dot" data-ok={ok} />}
        {down ? 'waking' : health ? `v${health.version}` : 'connecting'}
      </button>
      <div className="tooltip" id="health-tip" role="tooltip">
        <div className="tip-grid">
          <span>Status</span><b>{down ? 'unreachable — retrying' : health?.status ?? 'connecting'}</b>
          <span>Detector</span><b>{health?.models.detector ?? '—'}{health ? (health.models.detector_loaded ? ' (loaded)' : ' (loading)') : ''}</b>
          <span>Siren model</span>
          <b>
            {siren ? `${siren.checkpoint} · ${siren.kind}` : '—'}
            {siren?.test_metrics?.roc_auc != null && ` · AUC ${siren.test_metrics.roc_auc.toFixed(3)}`}
          </b>
          <span>Build</span><b className="mono">{version?.git_sha_short ?? 'unknown'}{version?.git_branch ? ` (${version.git_branch})` : ''}</b>
          <span>Built</span><b>{version?.build_time ? new Date(version.build_time).toLocaleString() : '—'}</b>
          <span>Host</span><b>{version ? `${version.host} · python ${version.python}` : '—'}</b>
          <span>Uptime</span><b>{version ? formatUptime(version.uptime_s) : '—'}</b>
        </div>
      </div>
    </div>
  )
}

function formatUptime(s: number): string {
  if (s < 90) return `${Math.round(s)} s`
  const m = Math.round(s / 60)
  if (m < 90) return `${m} min`
  const h = Math.floor(m / 60)
  return `${h} h ${m - h * 60} min`
}

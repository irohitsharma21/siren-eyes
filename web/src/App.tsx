import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  Activity, AlertTriangle, Clock, Compass, Film, Gauge, Layers, Loader2,
  Play, ShieldCheck, Siren, Upload,
} from 'lucide-react'

import {
  api, streamAnalysis,
  type AnalysisSummary, type DemoClip, type FrameResult,
  type HealthInfo, type SystemConfig,
} from './lib/api'
import { VideoStage } from './components/VideoStage'
import { Timeline } from './components/Timeline'
import { Compass as CompassDial, Kinematics, Meters, Notice, Pipeline, SignalHead } from './components/Panels'

export default function App() {
  const [health, setHealth] = useState<HealthInfo | null>(null)
  const [config, setConfig] = useState<SystemConfig | null>(null)
  const [demos, setDemos] = useState<DemoClip[]>([])

  const [source, setSource] = useState<{ url: string; label: string } | null>(null)
  const [selectedDemo, setSelectedDemo] = useState<string | null>(null)
  const [pendingFile, setPendingFile] = useState<File | null>(null)

  const [frames, setFrames] = useState<FrameResult[]>([])
  const [summary, setSummary] = useState<AnalysisSummary | null>(null)
  const [analysing, setAnalysing] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [playhead, setPlayhead] = useState(0)
  const [sourceSize, setSourceSize] = useState<{ w: number; h: number } | null>(null)

  const videoRef = useRef<HTMLVideoElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const disposeRef = useRef<(() => void) | null>(null)

  // ── bootstrap ──────────────────────────────────────────────────────
  useEffect(() => {
    api.health().then(setHealth).catch(() => setError('Backend unreachable'))
    api.config().then(setConfig).catch(() => {})
    api.demos()
      .then((d) => {
        setDemos(d.demos)
        if (d.demos.length) selectDemo(d.demos[0])
      })
      .catch(() => {})
    return () => disposeRef.current?.()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const selectDemo = (clip: DemoClip) => {
    disposeRef.current?.()
    setSelectedDemo(clip.name)
    setPendingFile(null)
    setSource({ url: api.demoVideoUrl(clip.file), label: clip.title })
    setFrames([])
    setSummary(null)
    setError(null)
    setPlayhead(0)
  }

  const selectFile = (file: File) => {
    disposeRef.current?.()
    setPendingFile(file)
    setSelectedDemo(null)
    setSource({ url: URL.createObjectURL(file), label: file.name })
    setFrames([])
    setSummary(null)
    setError(null)
    setPlayhead(0)
  }

  // ── analysis ───────────────────────────────────────────────────────
  const run = useCallback(async () => {
    if (analysing) return
    setAnalysing(true)
    setFrames([])
    setSummary(null)
    setError(null)

    try {
      const job = pendingFile
        ? await api.analyseUpload(pendingFile)
        : selectedDemo
          ? await api.analyseDemo(selectedDemo)
          : null

      if (!job) {
        setError('Select a demo clip or upload a video first.')
        setAnalysing(false)
        return
      }

      const video = videoRef.current
      if (video) {
        video.currentTime = 0
        video.play().catch(() => {})
      }

      disposeRef.current = streamAnalysis(job.job_id, {
        onFrame: (f) => {
          setFrames((prev) => [...prev, f])
          setPlayhead(f.timestamp_s)
        },
        onSummary: setSummary,
        onError: (m) => { setError(m); setAnalysing(false) },
        onDone: () => setAnalysing(false),
      })
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      setAnalysing(false)
    }
  }, [analysing, pendingFile, selectedDemo])

  // Keep the video roughly in step with the frame being analysed. The
  // pipeline is slower than real time on CPU, so the video is driven by the
  // analysis rather than the other way round.
  useEffect(() => {
    const video = videoRef.current
    if (!video || !analysing) return
    if (Math.abs(video.currentTime - playhead) > 0.4) video.currentTime = playhead
  }, [playhead, analysing])

  const currentFrame = useMemo(() => {
    if (!frames.length) return null
    let best = frames[0]
    for (const f of frames) {
      if (f.timestamp_s <= playhead) best = f
      else break
    }
    return best
  }, [frames, playhead])

  const duration = summary?.duration_s ?? videoRef.current?.duration ?? 0
  const progress = duration > 0 ? Math.min(playhead / duration, 1) : 0
  const legacyModel = health?.models.siren_classifier.kind === 'legacy'

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark"><Siren size={15} /></span>
          <span>
            Siren Eyes
            <span className="brand-sub" style={{ marginLeft: 8 }}>
              stereo-aware preemption
            </span>
          </span>
        </div>

        <div style={{ flex: 1 }} />

        {health && (
          <span className="hud-chip" title={`Detector: ${health.models.detector}`}>
            <span
              className="status-dot"
              style={{
                width: 6, height: 6, borderRadius: '50%',
                background: health.models.detector_loaded ? 'var(--go)' : 'var(--warn)',
              }}
            />
            v{health.version}
          </span>
        )}
      </header>

      <div className="shell">
        {/* ── Left column ─────────────────────────────────────────── */}
        <main style={{ minWidth: 0 }}>
          <VideoStage
            src={source?.url ?? null}
            frame={currentFrame}
            sourceSize={sourceSize}
            analysing={analysing}
            onPickFile={() => fileInputRef.current?.click()}
            videoRef={videoRef}
          />

          {/* hidden metadata reader keeps the overlay mapping honest */}
          <video
            src={source?.url}
            style={{ display: 'none' }}
            onLoadedMetadata={(e) =>
              setSourceSize({
                w: e.currentTarget.videoWidth,
                h: e.currentTarget.videoHeight,
              })
            }
          />

          <div style={{ marginTop: 12, display: 'flex', alignItems: 'center', gap: 10 }}>
            <button className="btn btn-primary" onClick={run} disabled={analysing || !source}>
              {analysing ? <Loader2 size={14} className="spin" /> : <Play size={14} />}
              {analysing ? 'Analysing…' : 'Run analysis'}
            </button>

            <div style={{ flex: 1 }}>
              <div className="progress">
                <div className="progress-fill" style={{ width: `${progress * 100}%` }} />
              </div>
            </div>

            <span className="mono" style={{ fontSize: '0.75rem', color: 'var(--text-mute)' }}>
              {frames.length} frames
            </span>
          </div>

          <div className="panel" style={{ marginTop: 12 }}>
            <div className="panel-head">
              <span className="panel-title"><Activity size={12} /> Confidence timeline</span>
              {summary && (
                <span className="mono" style={{ fontSize: '0.6875rem', color: 'var(--text-mute)' }}>
                  {summary.mean_latency_ms.toFixed(0)} ms mean · p95 {summary.p95_latency_ms.toFixed(0)} ms
                </span>
              )}
            </div>
            <div className="panel-body">
              <Timeline
                frames={frames}
                duration={duration}
                current={playhead}
                onSeek={(t) => {
                  if (analysing) return
                  setPlayhead(t)
                  if (videoRef.current) videoRef.current.currentTime = t
                }}
              />
            </div>
          </div>

          {summary && <SummaryPanel summary={summary} />}
        </main>

        {/* ── Right column ────────────────────────────────────────── */}
        <aside style={{ minWidth: 0 }}>
          {error && (
            <div style={{ marginBottom: 12 }}>
              <Notice kind="warn">{error}</Notice>
            </div>
          )}

          {legacyModel && (
            <div style={{ marginBottom: 12 }}>
              <Notice kind="warn">
                Running the legacy siren checkpoint, which does not discriminate
                siren from non-siren audio. Run <code>scripts/train_siren.py</code>{' '}
                to produce a working classifier.
              </Notice>
            </div>
          )}

          <div className="panel">
            <div className="panel-head">
              <span className="panel-title"><ShieldCheck size={12} /> Signal decision</span>
            </div>
            <div className="panel-body">
              <SignalHead frame={currentFrame} />
            </div>
          </div>

          <div className="panel">
            <div className="panel-head">
              <span className="panel-title"><Gauge size={12} /> Detection confidence</span>
            </div>
            <div className="panel-body">
              <Meters frame={currentFrame} config={config} />
            </div>
          </div>

          <div className="panel">
            <div className="panel-head">
              <span className="panel-title"><Clock size={12} /> Kinematics</span>
            </div>
            <div className="panel-body">
              <Kinematics frame={currentFrame} />
            </div>
          </div>

          <div className="panel">
            <div className="panel-head">
              <span className="panel-title"><Compass size={12} /> Direction of arrival</span>
            </div>
            <div className="panel-body">
              <CompassDial frame={currentFrame} />
            </div>
          </div>

          <div className="panel">
            <div className="panel-head">
              <span className="panel-title"><Layers size={12} /> Pipeline</span>
            </div>
            <div className="panel-body">
              <Pipeline frame={currentFrame} config={config} />
            </div>
          </div>

          <div className="panel">
            <div className="panel-head">
              <span className="panel-title"><Film size={12} /> Source</span>
            </div>
            <div className="panel-body">
              <div className="demo-list">
                {demos.map((d) => (
                  <button
                    key={d.name}
                    className="demo-card"
                    data-selected={selectedDemo === d.name}
                    onClick={() => selectDemo(d)}
                    disabled={analysing}
                  >
                    <span className="demo-thumb"><Film size={14} /></span>
                    <span style={{ minWidth: 0, flex: 1 }}>
                      <span className="demo-name" style={{ display: 'block' }}>{d.title}</span>
                      <span className="demo-meta">
                        {d.duration_s.toFixed(0)}s ·{' '}
                        {d.has_audio ? (d.stereo ? 'stereo audio' : 'mono audio') : 'no audio'}
                      </span>
                    </span>
                  </button>
                ))}

                <label
                  className="dropzone"
                  onDragOver={(e) => e.preventDefault()}
                  onDrop={(e) => {
                    e.preventDefault()
                    const f = e.dataTransfer.files?.[0]
                    if (f) selectFile(f)
                  }}
                >
                  <Upload size={16} />
                  <span>{pendingFile ? pendingFile.name : 'Drop a video, or click to browse'}</span>
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="video/*"
                    style={{ display: 'none' }}
                    onChange={(e) => {
                      const f = e.target.files?.[0]
                      if (f) selectFile(f)
                    }}
                  />
                </label>
              </div>
            </div>
          </div>
        </aside>
      </div>

      <footer className="footer">
        Implements <em>Stereo-Aware Multimodal Ambulance Detection and Real-Time
        Traffic Signal Preemption Using Edge AI</em> — Dalal, Gupta &amp; Sharma.
        {health?.models.siren_classifier.test_metrics?.roc_auc != null && (
          <> Siren classifier ROC-AUC{' '}
            {health.models.siren_classifier.test_metrics.roc_auc.toFixed(3)}
            {health.models.siren_classifier.test_metrics.roc_auc_std != null && (
              <> &plusmn; {health.models.siren_classifier.test_metrics.roc_auc_std.toFixed(3)}</>
            )}
            {' '}over ESC-50 5-fold cross-validation.</>
        )}
      </footer>
    </div>
  )
}

/* ── Summary ─────────────────────────────────────────────────────── */

function SummaryPanel({ summary }: { summary: AnalysisSummary }) {
  const rows: [string, string][] = [
    ['Clip', `${summary.video} · ${summary.resolution} · ${summary.fps} fps`],
    ['Duration', `${summary.duration_s.toFixed(1)} s`],
    ['Frames analysed', `${summary.frames_analysed} (${summary.detection_frames} with detections)`],
    ['Peak vision confidence', summary.peak_vision_confidence.toFixed(3)],
    ['Peak siren confidence', summary.peak_siren_confidence.toFixed(3)],
    ['Peak fused confidence', summary.peak_fused_confidence.toFixed(3)],
    ['First detection', summary.first_detection_s != null ? `${summary.first_detection_s.toFixed(2)} s` : '—'],
    ['Preemption granted', summary.preemption_granted ? `yes, at ${summary.grant_time_s?.toFixed(2)} s` : 'no'],
    ['Detection → grant', summary.response_latency_s != null ? `${summary.response_latency_s.toFixed(2)} s` : '—'],
    ['Mean latency', `${summary.mean_latency_ms.toFixed(1)} ms`],
    ['p95 latency', `${summary.p95_latency_ms.toFixed(1)} ms`],
  ]

  const blocked = Object.entries(summary.blocked_reasons)

  return (
    <div className="panel" style={{ marginTop: 12 }}>
      <div className="panel-head">
        <span className="panel-title"><Activity size={12} /> Analysis summary</span>
      </div>
      <div className="panel-body">
        {!summary.audio_available && (
          <div style={{ marginBottom: 12 }}>
            <Notice kind="warn">
              <strong>Audio unavailable</strong> — {summary.audio_note}. The siren
              and direction stages could not contribute; the decision used vision
              only.
            </Notice>
          </div>
        )}

        <div className="kv">
          {rows.map(([k, v]) => (
            <div className="kv-row" key={k}>
              <span className="kv-key">{k}</span>
              <span className="kv-val">{v}</span>
            </div>
          ))}
        </div>

        {blocked.length > 0 && (
          <>
            <div className="panel-title" style={{ marginTop: 16, marginBottom: 8 }}>
              <AlertTriangle size={12} /> Why preemption was withheld
            </div>
            <div className="kv">
              {blocked.map(([reason, count]) => (
                <div className="kv-row" key={reason}>
                  <span className="kv-key">{reason.replace(/_/g, ' ').replace('blocked ', '')}</span>
                  <span className="kv-val">{count} frames</span>
                </div>
              ))}
            </div>
          </>
        )}

        <div style={{ marginTop: 14 }}>
          <div className="panel-title" style={{ marginBottom: 8 }}>
            Stage latency (mean per frame)
          </div>
          <div className="stat-grid">
            {Object.entries(summary.stage_latency_ms).map(([stage, ms]) => (
              <div className="stat" key={stage}>
                <div className="stat-label">{stage}</div>
                <div className="stat-value">
                  {ms.toFixed(1)}<span className="stat-unit">ms</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}

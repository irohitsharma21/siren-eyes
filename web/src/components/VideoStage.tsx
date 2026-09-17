import { useEffect, useRef, useState } from 'react'
import type { DragEvent } from 'react'
import { Camera, Eye, EyeOff, Pause, Play, Radio, Upload, VideoOff, Volume2, VolumeX } from 'lucide-react'
import type { FrameResult } from '../lib/api'
import { bestBox, trailFor } from '../lib/analysis'

export type OverlayMode = 'gated' | 'raw'

export type StageStatus =
  | 'idle' | 'uploading' | 'connecting' | 'analysing' | 'done' | 'cancelled' | 'error'

interface Props {
  src: string | null
  frames: FrameResult[]
  /** Index into `frames` of the frame under the playhead, or -1. */
  frameIndex: number
  /** Native pixel size of the source, needed to map boxes onto the canvas. */
  sourceSize: { w: number; h: number } | null
  status: StageStatus
  phase: string
  precomputed: boolean
  /** Vision threshold the run was gated with; drives the reticle split. */
  threshold: number
  mode: OverlayMode
  playing: boolean
  /** A history run whose footage is no longer available. */
  missingVideo: boolean
  clipLabel: string
  videoRef: React.RefObject<HTMLVideoElement>
  muted: boolean
  onToggleMute: () => void
  onMetadata: (size: { w: number; h: number }, duration: number) => void
  onPickFile: () => void
  onDropFile: (file: File) => void
  onTogglePlay: () => void
  onToggleMode: () => void
  onSnapshot: () => void
}

/**
 * Video surface with a detection overlay.
 *
 * The overlay is a canvas rather than positioned DOM nodes: at 30 fps with
 * several boxes plus labels, DOM churn drops frames, and a canvas lets the
 * box, its label, the trail and the readout be drawn as one unit. The same
 * drawing routine renders the downloadable snapshot at native resolution.
 */
export function VideoStage({
  src, frames, frameIndex, sourceSize, status, phase, precomputed, threshold, mode,
  playing, missingVideo, clipLabel, videoRef, muted, onToggleMute, onMetadata,
  onPickFile, onDropFile, onTogglePlay, onToggleMode, onSnapshot,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [drag, setDrag] = useState(false)
  const [, bump] = useState(0)

  // Redraw when the stage resizes; the mapping from source pixels to canvas
  // pixels depends on the displayed size.
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || typeof ResizeObserver === 'undefined') return
    const ro = new ResizeObserver(() => bump((n) => n + 1))
    ro.observe(canvas)
    return () => ro.disconnect()
  }, [src])

  const frame = frameIndex >= 0 ? frames[frameIndex] : null

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const rect = canvas.getBoundingClientRect()
    const dpr = window.devicePixelRatio || 1
    if (canvas.width !== Math.round(rect.width * dpr) || canvas.height !== Math.round(rect.height * dpr)) {
      canvas.width = Math.round(rect.width * dpr)
      canvas.height = Math.round(rect.height * dpr)
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.clearRect(0, 0, rect.width, rect.height)

    if (!frame || !sourceSize) return

    // `object-fit: contain` letterboxes the video; replicate that mapping so
    // boxes land on the vehicle rather than on the black bars.
    const scale = Math.min(rect.width / sourceSize.w, rect.height / sourceSize.h)
    const drawW = sourceSize.w * scale
    const drawH = sourceSize.h * scale
    drawOverlay(ctx, {
      frames, index: frameIndex, scale,
      offX: (rect.width - drawW) / 2, offY: (rect.height - drawH) / 2,
      width: rect.width, height: rect.height, threshold, mode, fontScale: 1,
    })
  })

  const onDragOver = (e: DragEvent) => {
    e.preventDefault()
    if (!drag) setDrag(true)
  }
  const onDrop = (e: DragEvent) => {
    e.preventDefault()
    setDrag(false)
    const f = e.dataTransfer.files?.[0]
    if (f) onDropFile(f)
  }

  const busy = status === 'analysing' || status === 'connecting' || status === 'uploading'
  const statusChip = status === 'uploading'
    ? 'UPLOADING'
    : status === 'connecting'
      ? 'STARTING'
      : status === 'analysing'
        ? (precomputed ? 'REPLAYING' : 'ANALYSING')
        : status === 'cancelled'
          ? 'STOPPED'
          : status === 'done'
            ? (precomputed ? 'RECORDED' : 'LIVE RESULT')
            : null

  return (
    <div
      className="stage"
      data-drag={drag}
      onDragOver={onDragOver}
      onDragLeave={() => setDrag(false)}
      onDrop={onDrop}
    >
      {src && !missingVideo ? (
        <>
          <video
            ref={videoRef}
            src={src}
            playsInline
            muted={muted}
            preload="auto"
            onLoadedMetadata={(e) => {
              const v = e.currentTarget
              onMetadata({ w: v.videoWidth, h: v.videoHeight }, Number.isFinite(v.duration) ? v.duration : 0)
            }}
          />
          <canvas ref={canvasRef} aria-hidden="true" />
        </>
      ) : missingVideo ? (
        <>
          <div className="stage-empty">
            <VideoOff size={26} />
            <div className="stage-empty-title">Footage not retained</div>
            <div className="stage-empty-copy">
              This run came from an uploaded file, which only lives as long as
              the page did. The timeline and results below are intact; upload
              the clip again to see the overlay.
            </div>
            <button className="btn btn-sm" onClick={onPickFile}>
              <Upload size={13} /> Upload a video
            </button>
          </div>
          <canvas ref={canvasRef} aria-hidden="true" style={{ display: 'none' }} />
        </>
      ) : (
        <div className="stage-empty">
          <Radio size={26} />
          <div className="stage-empty-title">No clip loaded</div>
          <div className="stage-empty-copy">
            Pick a bundled demo from the top bar, or drop a video here.
          </div>
          <button className="btn btn-sm" onClick={onPickFile}>
            <Upload size={13} /> Choose a video
          </button>
        </div>
      )}

      {drag && (
        <div className="stage-drop">
          <Upload size={22} />
          <span>Drop to load {'—'} analysis starts when you press Run</span>
        </div>
      )}

      {src && !missingVideo && (
        <>
          <div className="stage-hud">
            <div className="hud-group">
              {statusChip && (
                <span className="hud-chip" data-tone={busy ? 'live' : status}>
                  {busy && <span className="live-dot" />} {statusChip}
                </span>
              )}
              {frame && (
                <>
                  <span className="hud-chip mono" title="Source frame index">#{frame.frame_index}</span>
                  <span className="hud-chip mono" title="Timestamp">t={frame.timestamp_s.toFixed(2)}s</span>
                </>
              )}
              {busy && !frame && phase && (
                <span className="hud-chip">{phase}{'…'}</span>
              )}
            </div>
            <div className="hud-group">
              {frame && frame.latency_ms > 0 && (
                <span className="hud-chip mono" title="End-to-end pipeline latency for this frame">
                  {frame.latency_ms.toFixed(0)} ms
                </span>
              )}
              {frame && (
                <span className="hud-chip" data-signal={frame.signal_state} title="Signal state">
                  {frame.signal_state.replace('_', ' ')}
                </span>
              )}
              {mode === 'raw' && (
                <span className="hud-chip" data-tone="raw">raw detector {'·'} gate {threshold.toFixed(2)}</span>
              )}
            </div>
          </div>

          <div className="stage-controls">
            <button
              className="stage-btn"
              onClick={onTogglePlay}
              disabled={busy || !frames.length}
              title={playing ? 'Pause (Space)' : 'Play (Space)'}
              aria-label={playing ? 'Pause' : 'Play'}
            >
              {playing ? <Pause size={14} /> : <Play size={14} />}
            </button>
            <button
              className="stage-btn"
              onClick={onToggleMute}
              title={muted ? 'Unmute (M)' : 'Mute (M)'}
              aria-label={muted ? 'Unmute' : 'Mute'}
              aria-pressed={!muted}
            >
              {muted ? <VolumeX size={14} /> : <Volume2 size={14} />}
            </button>
            <button
              className="stage-btn"
              data-active={mode === 'raw'}
              onClick={onToggleMode}
              disabled={!frames.length}
              title={mode === 'raw' ? 'Showing raw detector output — click for gated view (C)' : 'Compare: show raw detector boxes (C)'}
              aria-pressed={mode === 'raw'}
            >
              {mode === 'raw' ? <EyeOff size={14} /> : <Eye size={14} />}
              <span>{mode === 'raw' ? 'Raw' : 'Compare'}</span>
            </button>
            <button
              className="stage-btn"
              onClick={onSnapshot}
              disabled={!frame}
              title="Download this frame with its overlay as PNG (S)"
              aria-label="Snapshot"
            >
              <Camera size={14} />
              <span>Snapshot</span>
            </button>
          </div>
          <span className="stage-caption" title={clipLabel}>{clipLabel}</span>
        </>
      )}
    </div>
  )
}

/* ── Overlay renderer ─────────────────────────────────────────────── */

export interface OverlayOptions {
  frames: FrameResult[]
  index: number
  scale: number
  offX: number
  offY: number
  width: number
  height: number
  threshold: number
  mode: OverlayMode
  /** Multiplier for text and stroke sizes; the snapshot renders at native resolution. */
  fontScale: number
}

const C = {
  strong: '#4c8dff',
  strongText: '#dce8ff',
  readout: '#8fb4ff',
  weak: 'rgba(255,255,255,0.34)',
  raw: '#f5a524',
  trail: '#4c8dff',
  audio: '#38bdb4',
  panel: 'rgba(10,12,14,0.86)',
}

export function drawOverlay(ctx: CanvasRenderingContext2D, o: OverlayOptions) {
  const frame = o.frames[o.index]
  if (!frame) return

  ctx.save()
  ctx.lineJoin = 'round'
  ctx.lineCap = 'round'

  if (o.mode === 'raw') {
    // Every box the detector produced (it runs at 0.20), styled alike, so the
    // effect of the confidence gate is visible by contrast with gated view.
    for (const d of frame.detections) {
      const gated = d.confidence >= o.threshold
      drawRawBox(ctx, d, o, gated)
    }
  } else {
    const strong = frame.detections.filter((d) => d.confidence >= o.threshold)
    const weak = frame.detections.filter((d) => d.confidence < o.threshold)
    // Sub-threshold detections are shown, dimmed, so the gate is legible
    // rather than looking like a missed detection.
    for (const d of weak) drawBox(ctx, d, o, false, frame)

    const trail = trailFor(o.frames, o.index, o.threshold)
    if (trail.length > 1) drawTrail(ctx, trail, o)

    const best = bestBox(frame, o.threshold)
    for (const d of strong) drawBox(ctx, d, o, true, frame, d === best)
  }

  // Bearing arrow, bottom-left of the picture area.
  if (frame.direction_available && frame.direction_deg != null) {
    drawBearing(ctx, frame, o)
  }

  ctx.restore()
}

function mapBox(d: { box: [number, number, number, number] }, o: OverlayOptions) {
  const [x1, y1, x2, y2] = d.box
  return {
    x: x1 * o.scale + o.offX,
    y: y1 * o.scale + o.offY,
    w: (x2 - x1) * o.scale,
    h: (y2 - y1) * o.scale,
  }
}

function drawTrail(ctx: CanvasRenderingContext2D, pts: [number, number][], o: OverlayOptions) {
  const k = o.fontScale
  const n = pts.length
  for (let i = 1; i < n; i++) {
    const a = pts[i - 1]
    const b = pts[i]
    const alpha = 0.15 + 0.75 * (i / n)
    ctx.strokeStyle = C.trail
    ctx.globalAlpha = alpha
    ctx.lineWidth = (1 + 1.6 * (i / n)) * k
    ctx.beginPath()
    ctx.moveTo(a[0] * o.scale + o.offX, a[1] * o.scale + o.offY)
    ctx.lineTo(b[0] * o.scale + o.offX, b[1] * o.scale + o.offY)
    ctx.stroke()
  }
  ctx.globalAlpha = 1
  for (let i = 0; i < n - 1; i++) {
    const p = pts[i]
    ctx.fillStyle = C.trail
    ctx.globalAlpha = 0.25 + 0.6 * (i / n)
    ctx.beginPath()
    ctx.arc(p[0] * o.scale + o.offX, p[1] * o.scale + o.offY, 2 * k, 0, Math.PI * 2)
    ctx.fill()
  }
  ctx.globalAlpha = 1
}

function drawRawBox(
  ctx: CanvasRenderingContext2D,
  d: { box: [number, number, number, number]; confidence: number },
  o: OverlayOptions,
  gated: boolean,
) {
  const k = o.fontScale
  const { x, y, w, h } = mapBox(d, o)
  ctx.strokeStyle = gated ? C.strong : C.raw
  ctx.lineWidth = 1.5 * k
  ctx.setLineDash([])
  ctx.strokeRect(x, y, w, h)

  const label = `${(d.confidence * 100).toFixed(1)}%`
  ctx.font = `600 ${11 * k}px "JetBrains Mono", ui-monospace, monospace`
  const tw = ctx.measureText(label).width
  const boxH = 16 * k
  const ly = y - boxH - 2 * k < 0 ? y + 2 * k : y - boxH - 2 * k
  ctx.fillStyle = gated ? C.strong : C.raw
  ctx.beginPath()
  ctx.roundRect(x, ly, tw + 8 * k, boxH, 3 * k)
  ctx.fill()
  ctx.fillStyle = '#0b0d0f'
  ctx.fillText(label, x + 4 * k, ly + 12 * k)
}

function drawBox(
  ctx: CanvasRenderingContext2D,
  d: { box: [number, number, number, number]; confidence: number },
  o: OverlayOptions,
  strong: boolean,
  frame: FrameResult,
  withReadout = false,
) {
  const k = o.fontScale
  const { x, y, w, h } = mapBox(d, o)
  const colour = strong ? C.strong : C.weak

  ctx.save()
  ctx.strokeStyle = colour
  ctx.lineWidth = (strong ? 2 : 1) * k
  if (!strong) ctx.setLineDash([4 * k, 4 * k])

  // Corner brackets read as a targeting reticle and stay legible over a
  // busy scene in a way a full rectangle does not.
  if (strong) {
    const c = Math.min(18 * k, w * 0.28, h * 0.28)
    ctx.beginPath()
    ctx.moveTo(x, y + c); ctx.lineTo(x, y); ctx.lineTo(x + c, y)
    ctx.moveTo(x + w - c, y); ctx.lineTo(x + w, y); ctx.lineTo(x + w, y + c)
    ctx.moveTo(x + w, y + h - c); ctx.lineTo(x + w, y + h); ctx.lineTo(x + w - c, y + h)
    ctx.moveTo(x + c, y + h); ctx.lineTo(x, y + h); ctx.lineTo(x, y + h - c)
    ctx.stroke()

    ctx.globalAlpha = 0.45
    ctx.lineWidth = 1 * k
    ctx.strokeRect(x, y, w, h)
    ctx.globalAlpha = 1

    // Centre mark
    const cx = x + w / 2
    const cy = y + h / 2
    ctx.globalAlpha = 0.9
    ctx.beginPath()
    ctx.moveTo(cx - 5 * k, cy); ctx.lineTo(cx + 5 * k, cy)
    ctx.moveTo(cx, cy - 5 * k); ctx.lineTo(cx, cy + 5 * k)
    ctx.stroke()
    ctx.globalAlpha = 1
  } else {
    ctx.strokeRect(x, y, w, h)
  }
  ctx.setLineDash([])

  // Label above (or below when clipped at the top).
  const label = `AMBULANCE ${(d.confidence * 100).toFixed(1)}%`
  ctx.font = `600 ${11 * k}px "JetBrains Mono", ui-monospace, monospace`
  const tw = ctx.measureText(label).width
  const boxW = tw + 12 * k
  const boxH = 18 * k
  const ly = y - boxH - 4 * k < 0 ? y + h + 4 * k : y - boxH - 4 * k

  ctx.fillStyle = strong ? C.panel : 'rgba(10,12,14,0.6)'
  ctx.beginPath()
  ctx.roundRect(x, ly, boxW, boxH, 4 * k)
  ctx.fill()
  ctx.strokeStyle = strong ? 'rgba(76,141,255,0.5)' : 'rgba(255,255,255,0.16)'
  ctx.lineWidth = 1 * k
  ctx.stroke()
  ctx.fillStyle = strong ? C.strongText : 'rgba(255,255,255,0.6)'
  ctx.fillText(label, x + 6 * k, ly + 13 * k)

  // Kinematic readout beside the tracked box: distance, speed, ETA, bearing.
  if (withReadout && (frame.distance_m != null || frame.eta_s != null)) {
    const lines: [string, string][] = []
    if (frame.distance_m != null) lines.push(['DIST', `${frame.distance_m.toFixed(0)} m`])
    if (frame.speed_kmph != null) lines.push(['SPEED', `${frame.speed_kmph.toFixed(0)} km/h`])
    lines.push(['ETA', frame.eta_s != null ? `${frame.eta_s.toFixed(1)} s` : '—'])
    if (frame.direction_available && frame.direction_deg != null) {
      lines.push(['BEARING', `${frame.direction_deg > 0 ? '+' : ''}${frame.direction_deg.toFixed(0)}°`])
    }
    ctx.font = `500 ${10.5 * k}px "JetBrains Mono", ui-monospace, monospace`
    const keyW = Math.max(...lines.map(([a]) => ctx.measureText(a).width))
    const valW = Math.max(...lines.map(([, b]) => ctx.measureText(b).width))
    const rw = keyW + valW + 22 * k
    const rh = lines.length * 14 * k + 10 * k
    // Prefer the right side of the box; flip left when it would clip.
    let rx = x + w + 8 * k
    if (rx + rw > o.width - 4 * k) rx = x - rw - 8 * k
    if (rx < 4 * k) rx = Math.min(x, o.width - rw - 4 * k)
    let ry = y
    if (ry + rh > o.height - 4 * k) ry = Math.max(4 * k, o.height - rh - 4 * k)

    // Leader line from box edge to readout.
    ctx.strokeStyle = 'rgba(76,141,255,0.55)'
    ctx.lineWidth = 1 * k
    ctx.beginPath()
    const fromX = rx > x ? x + w : x
    ctx.moveTo(fromX, y + Math.min(h, 12 * k))
    ctx.lineTo(rx > x ? rx : rx + rw, ry + 8 * k)
    ctx.stroke()

    ctx.fillStyle = C.panel
    ctx.beginPath()
    ctx.roundRect(rx, ry, rw, rh, 4 * k)
    ctx.fill()
    ctx.strokeStyle = 'rgba(76,141,255,0.35)'
    ctx.stroke()
    lines.forEach(([key, val], i) => {
      const ty = ry + 12 * k + i * 14 * k
      ctx.fillStyle = 'rgba(255,255,255,0.55)'
      ctx.fillText(key, rx + 7 * k, ty)
      ctx.fillStyle = C.readout
      ctx.fillText(val, rx + rw - 7 * k - ctx.measureText(val).width, ty)
    })
  }

  ctx.restore()
}

function drawBearing(ctx: CanvasRenderingContext2D, frame: FrameResult, o: OverlayOptions) {
  const k = o.fontScale
  const angle = frame.direction_deg ?? 0
  const r = 20 * k
  const cx = o.offX + r + 14 * k
  const cy = o.offY + (o.height - 2 * o.offY) - r - 14 * k

  ctx.save()
  ctx.fillStyle = 'rgba(10,12,14,0.72)'
  ctx.beginPath()
  ctx.arc(cx, cy, r + 4 * k, 0, Math.PI * 2)
  ctx.fill()
  ctx.strokeStyle = 'rgba(255,255,255,0.18)'
  ctx.lineWidth = 1 * k
  ctx.stroke()

  // Tick marks at -90, -45, 0, 45, 90 on the forward half.
  for (const t of [-90, -45, 0, 45, 90]) {
    const a = ((t - 90) * Math.PI) / 180
    ctx.beginPath()
    ctx.moveTo(cx + (r - 5 * k) * Math.cos(a), cy + (r - 5 * k) * Math.sin(a))
    ctx.lineTo(cx + r * Math.cos(a), cy + r * Math.sin(a))
    ctx.strokeStyle = t === 0 ? 'rgba(255,255,255,0.7)' : 'rgba(255,255,255,0.3)'
    ctx.stroke()
  }

  // Arrow: 0 deg is straight ahead (up), positive to the right.
  const a = ((angle - 90) * Math.PI) / 180
  const tipX = cx + (r - 3 * k) * Math.cos(a)
  const tipY = cy + (r - 3 * k) * Math.sin(a)
  ctx.strokeStyle = C.audio
  ctx.fillStyle = C.audio
  ctx.lineWidth = 2 * k
  ctx.beginPath()
  ctx.moveTo(cx, cy)
  ctx.lineTo(tipX, tipY)
  ctx.stroke()
  const headLen = 6 * k
  ctx.beginPath()
  ctx.moveTo(tipX, tipY)
  ctx.lineTo(tipX - headLen * Math.cos(a - 0.5), tipY - headLen * Math.sin(a - 0.5))
  ctx.lineTo(tipX - headLen * Math.cos(a + 0.5), tipY - headLen * Math.sin(a + 0.5))
  ctx.closePath()
  ctx.fill()
  ctx.beginPath()
  ctx.arc(cx, cy, 2 * k, 0, Math.PI * 2)
  ctx.fill()

  ctx.font = `600 ${10 * k}px "JetBrains Mono", ui-monospace, monospace`
  ctx.fillStyle = '#e8fbf9'
  const label = `${angle > 0 ? '+' : ''}${angle.toFixed(0)}° ${frame.direction_bearing}`
  ctx.fillText(label, cx + r + 10 * k, cy + 4 * k)
  ctx.restore()
}

/**
 * Render the current video frame plus overlay at native resolution.
 * Returns null when the video has no decodable frame yet.
 */
export async function renderSnapshot(
  video: HTMLVideoElement,
  frames: FrameResult[],
  index: number,
  threshold: number,
  mode: OverlayMode,
  caption: string,
): Promise<Blob | null> {
  const w = video.videoWidth
  const h = video.videoHeight
  if (!w || !h) return null
  const canvas = document.createElement('canvas')
  canvas.width = w
  canvas.height = h
  const ctx = canvas.getContext('2d')
  if (!ctx) return null
  try {
    ctx.drawImage(video, 0, 0, w, h)
  } catch {
    return null
  }
  const k = Math.max(1, w / 960)
  drawOverlay(ctx, {
    frames, index, scale: 1, offX: 0, offY: 0, width: w, height: h, threshold, mode, fontScale: k,
  })

  // Caption strip along the bottom so the PNG is self-describing.
  const frame = frames[index]
  if (frame) {
    const text =
      `Siren Eyes · ${caption} · t=${frame.timestamp_s.toFixed(2)} s · frame ${frame.frame_index} · ` +
      `V ${frame.vision_confidence.toFixed(2)} A ${frame.audio_confidence.toFixed(2)} F ${frame.fused_confidence.toFixed(2)} · ` +
      `${frame.signal_state.replace('_', ' ')} · ${frame.decision.replace(/_/g, ' ')}`
    ctx.font = `500 ${12 * k}px "JetBrains Mono", ui-monospace, monospace`
    const tw = ctx.measureText(text).width
    const pad = 8 * k
    const bh = 22 * k
    ctx.fillStyle = 'rgba(10,12,14,0.78)'
    ctx.fillRect(w - tw - pad * 2 - 10 * k, h - bh - 10 * k, tw + pad * 2, bh)
    ctx.fillStyle = '#f2f5f9'
    ctx.fillText(text, w - tw - pad - 10 * k, h - 10 * k - 7 * k)
  }

  return new Promise((resolve) => canvas.toBlob((b) => resolve(b), 'image/png'))
}

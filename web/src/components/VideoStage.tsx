import { useEffect, useRef } from 'react'
import { Radio, Upload } from 'lucide-react'
import type { FrameResult } from '../lib/api'

interface Props {
  src: string | null
  frame: FrameResult | null
  /** Native pixel size of the source, needed to map boxes onto the canvas. */
  sourceSize: { w: number; h: number } | null
  analysing: boolean
  onPickFile: () => void
  videoRef: React.RefObject<HTMLVideoElement>
}

/**
 * Video surface with a detection overlay.
 *
 * The overlay is a canvas rather than positioned DOM nodes: at 30 fps with
 * several boxes plus labels, DOM churn drops frames, and a canvas lets the
 * box, its label and the distance readout be drawn as one unit.
 */
export function VideoStage({
  src,
  frame,
  sourceSize,
  analysing,
  onPickFile,
  videoRef,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    const video = videoRef.current
    if (!canvas || !video) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Match the backing store to the displayed size, accounting for DPR.
    const rect = canvas.getBoundingClientRect()
    const dpr = window.devicePixelRatio || 1
    if (canvas.width !== rect.width * dpr || canvas.height !== rect.height * dpr) {
      canvas.width = rect.width * dpr
      canvas.height = rect.height * dpr
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.clearRect(0, 0, rect.width, rect.height)

    if (!frame || !sourceSize) return

    // `object-fit: contain` letterboxes the video; replicate that mapping so
    // boxes land on the vehicle rather than on the black bars.
    const scale = Math.min(rect.width / sourceSize.w, rect.height / sourceSize.h)
    const drawW = sourceSize.w * scale
    const drawH = sourceSize.h * scale
    const offX = (rect.width - drawW) / 2
    const offY = (rect.height - drawH) / 2

    const strong = frame.detections.filter((d) => d.confidence >= 0.65)
    const weak = frame.detections.filter((d) => d.confidence < 0.65)

    // Sub-threshold detections are shown, dimmed, so the 0.65 gate is legible
    // rather than looking like a missed detection.
    for (const d of weak) drawBox(ctx, d, scale, offX, offY, false, frame)
    for (const d of strong) drawBox(ctx, d, scale, offX, offY, true, frame)
  }, [frame, sourceSize, videoRef])

  return (
    <div className="stage">
      {src ? (
        <>
          <video ref={videoRef} src={src} playsInline muted preload="auto" />
          <canvas ref={canvasRef} />
          <div className="stage-hud">
            <div style={{ display: 'flex', gap: 6 }}>
              {analysing && (
                <span className="hud-chip">
                  <span className="live-dot" /> ANALYSING
                </span>
              )}
              {frame && (
                <span className="hud-chip mono">
                  t={frame.timestamp_s.toFixed(2)}s
                </span>
              )}
            </div>
            {frame && frame.latency_ms > 0 && (
              <span className="hud-chip mono" title="End-to-end pipeline latency for this frame">
                {frame.latency_ms.toFixed(0)} ms
              </span>
            )}
          </div>
        </>
      ) : (
        <div className="stage-empty">
          <Radio size={26} />
          <div style={{ fontSize: '0.875rem', color: 'var(--text-dim)' }}>
            No clip loaded
          </div>
          <button className="btn btn-sm" onClick={onPickFile}>
            <Upload size={13} /> Choose a video
          </button>
        </div>
      )}
    </div>
  )
}

function drawBox(
  ctx: CanvasRenderingContext2D,
  d: { box: [number, number, number, number]; confidence: number },
  scale: number,
  offX: number,
  offY: number,
  strong: boolean,
  frame: FrameResult,
) {
  const [x1, y1, x2, y2] = d.box
  const x = x1 * scale + offX
  const y = y1 * scale + offY
  const w = (x2 - x1) * scale
  const h = (y2 - y1) * scale

  const colour = strong ? '#4c8dff' : 'rgba(255,255,255,0.32)'

  ctx.save()
  ctx.strokeStyle = colour
  ctx.lineWidth = strong ? 2 : 1
  if (!strong) ctx.setLineDash([4, 4])

  // Corner brackets read as a targeting reticle and stay legible over a
  // busy scene in a way a full rectangle does not.
  if (strong) {
    const c = Math.min(18, w * 0.28, h * 0.28)
    ctx.beginPath()
    ctx.moveTo(x, y + c); ctx.lineTo(x, y); ctx.lineTo(x + c, y)
    ctx.moveTo(x + w - c, y); ctx.lineTo(x + w, y); ctx.lineTo(x + w, y + c)
    ctx.moveTo(x + w, y + h - c); ctx.lineTo(x + w, y + h); ctx.lineTo(x + w - c, y + h)
    ctx.moveTo(x + c, y + h); ctx.lineTo(x, y + h); ctx.lineTo(x, y + h - c)
    ctx.stroke()

    ctx.globalAlpha = 0.5
    ctx.lineWidth = 1
    ctx.strokeRect(x, y, w, h)
    ctx.globalAlpha = 1
  } else {
    ctx.strokeRect(x, y, w, h)
  }
  ctx.setLineDash([])

  // Label
  const label = `AMBULANCE ${(d.confidence * 100).toFixed(1)}%`
  const sub =
    strong && frame.distance_m != null
      ? `${frame.distance_m.toFixed(0)} m` +
        (frame.eta_s != null ? ` · ETA ${frame.eta_s.toFixed(1)} s` : '')
      : null

  ctx.font = '600 11px "JetBrains Mono", ui-monospace, monospace'
  const tw = ctx.measureText(label).width
  const sw = sub ? ctx.measureText(sub).width : 0
  const boxW = Math.max(tw, sw) + 12
  const boxH = sub ? 32 : 18

  const ly = y - boxH - 4 < 0 ? y + h + 4 : y - boxH - 4

  ctx.fillStyle = strong ? 'rgba(10,12,14,0.88)' : 'rgba(10,12,14,0.6)'
  ctx.beginPath()
  ctx.roundRect(x, ly, boxW, boxH, 4)
  ctx.fill()
  ctx.strokeStyle = strong ? 'rgba(76,141,255,0.5)' : 'rgba(255,255,255,0.16)'
  ctx.lineWidth = 1
  ctx.stroke()

  ctx.fillStyle = strong ? '#dce8ff' : 'rgba(255,255,255,0.6)'
  ctx.fillText(label, x + 6, ly + 13)
  if (sub) {
    ctx.fillStyle = '#8fb4ff'
    ctx.fillText(sub, x + 6, ly + 27)
  }

  ctx.restore()
}

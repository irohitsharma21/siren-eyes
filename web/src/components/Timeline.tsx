import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { PointerEvent as ReactPointerEvent } from 'react'
import { ListOrdered } from 'lucide-react'
import type { AnalysisParams, FrameResult, SystemConfig } from '../lib/api'
import { indexAt, shortDecision, type DecisionEvent } from '../lib/analysis'

interface Props {
  frames: FrameResult[]
  duration: number
  current: number
  params: AnalysisParams | null
  config: SystemConfig | null
  events: DecisionEvent[]
  onSeek: (t: number) => void
}

const COLOUR = {
  vision: '#4c8dff',
  audio: '#38bdb4',
  fused: '#a78bfa',
  go: '#34c759',
  warn: '#f5a524',
  stop: '#ff453a',
  text: '#8b939d',
  textStrong: '#d7dce2',
  grid: 'rgba(255,255,255,0.06)',
  hair: 'rgba(255,255,255,0.35)',
  bg: '#0c0e10',
}

const EVENT_COLOUR: Record<DecisionEvent['kind'], string> = {
  detect: COLOUR.vision,
  trigger: COLOUR.fused,
  grant: COLOUR.go,
  hold: COLOUR.go,
  revert: COLOUR.warn,
  restore: COLOUR.textStrong,
  blocked: COLOUR.text,
}

// Vertical layout (CSS px). Lanes are fixed height so labels stay legible.
const PAD_L = 8
const PAD_R = 8
const BAND_H = 8          // signal state band
const MARK_H = 12         // decision markers
const LANE_H = 34
const LANE_GAP = 4
const AXIS_H = 16
export const TIMELINE_HEIGHT = BAND_H + 2 + MARK_H + 2 + LANE_H * 3 + LANE_GAP * 2 + 4 + AXIS_H

function signalColour(state: string): string | null {
  switch (state) {
    case 'preempt_green':
    case 'green':
      return COLOUR.go
    case 'amber':
      return COLOUR.warn
    case 'all_red':
      return 'rgba(255,69,58,0.55)'
    default:
      return null
  }
}

function tickStep(duration: number): number {
  if (duration <= 12) return 1
  if (duration <= 30) return 2
  if (duration <= 90) return 5
  if (duration <= 300) return 15
  return 60
}

/**
 * Multi-lane confidence chart with a signal-state band and decision markers.
 *
 * Drawn to a canvas because a full analysis produces hundreds of samples and
 * three series; as SVG that is thousands of nodes for something that is
 * redrawn on every frame. Interaction (hover, scrub) is handled on the
 * wrapper so the tooltip can be ordinary DOM.
 */
export function Timeline({ frames, duration, current, params, config, events, onSeek }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const wrapRef = useRef<HTMLDivElement>(null)
  const [hover, setHover] = useState<{ x: number; index: number } | null>(null)
  const [width, setWidth] = useState(0)
  const dragging = useRef(false)

  useEffect(() => {
    const el = wrapRef.current
    if (!el || typeof ResizeObserver === 'undefined') return
    const ro = new ResizeObserver(() => setWidth(el.getBoundingClientRect().width))
    ro.observe(el)
    setWidth(el.getBoundingClientRect().width)
    return () => ro.disconnect()
  }, [])

  const vThr = params?.vision_threshold ?? config?.vision_confidence_threshold ?? 0.65
  const aThr = config?.siren_threshold ?? 0.5
  const fThr = params?.fused_threshold ?? config?.fusion.trigger_threshold ?? 0.6

  const lanes = useMemo(
    () => [
      { key: 'vision_confidence' as const, label: 'VISION', colour: COLOUR.vision, threshold: vThr },
      { key: 'audio_confidence' as const, label: 'AUDIO', colour: COLOUR.audio, threshold: aThr },
      { key: 'fused_confidence' as const, label: 'FUSED', colour: COLOUR.fused, threshold: fThr },
    ],
    [vThr, aThr, fThr],
  )

  const plotW = Math.max(0, width - PAD_L - PAD_R)
  const xOf = useCallback(
    (t: number) => PAD_L + (duration > 0 ? (t / duration) * plotW : 0),
    [duration, plotW],
  )
  const tOf = useCallback(
    (x: number) => (plotW > 0 ? Math.min(Math.max((x - PAD_L) / plotW, 0), 1) * duration : 0),
    [duration, plotW],
  )

  // ── draw ───────────────────────────────────────────────────────────
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || width === 0) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const H = TIMELINE_HEIGHT
    const dpr = window.devicePixelRatio || 1
    if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(H * dpr)) {
      canvas.width = Math.round(width * dpr)
      canvas.height = Math.round(H * dpr)
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.fillStyle = COLOUR.bg
    ctx.fillRect(0, 0, width, H)

    const laneTop = (i: number) => BAND_H + 2 + MARK_H + 2 + i * (LANE_H + LANE_GAP)
    const axisY = H - AXIS_H

    if (!frames.length || duration <= 0) {
      ctx.fillStyle = COLOUR.text
      ctx.font = '500 11px Inter, system-ui, sans-serif'
      ctx.textAlign = 'center'
      ctx.fillText('Confidence traces appear here as frames arrive', width / 2, H / 2 + 4)
      ctx.textAlign = 'left'
      return
    }

    // Time grid + axis labels
    const step = tickStep(duration)
    ctx.font = '500 9.5px "JetBrains Mono", ui-monospace, monospace'
    for (let t = 0; t <= duration + 1e-6; t += step) {
      const x = Math.round(xOf(t)) + 0.5
      ctx.strokeStyle = COLOUR.grid
      ctx.lineWidth = 1
      ctx.beginPath()
      ctx.moveTo(x, 0)
      ctx.lineTo(x, axisY)
      ctx.stroke()
      ctx.fillStyle = COLOUR.text
      ctx.textAlign = t === 0 ? 'left' : 'center'
      ctx.fillText(`${t}s`, t === 0 ? x + 2 : x, H - 4)
    }
    ctx.textAlign = 'left'

    // Signal-state band: one cell per frame, coloured when not plain red.
    const half = frames.length > 1 ? Math.max((plotW / frames.length) / 2, 1) : 2
    for (const f of frames) {
      const c = signalColour(f.signal_state)
      if (!c) continue
      ctx.fillStyle = c
      const x = xOf(f.timestamp_s)
      ctx.fillRect(x - half, 0, half * 2 + 0.5, BAND_H)
    }
    ctx.fillStyle = 'rgba(255,255,255,0.05)'
    ctx.fillRect(PAD_L, 0, plotW, BAND_H)

    // Preemption shading down through the lanes so the granted window reads
    // against the traces, not only in the band.
    ctx.fillStyle = 'rgba(52,199,89,0.08)'
    for (const f of frames) {
      if (f.signal_state === 'preempt_green') {
        ctx.fillRect(xOf(f.timestamp_s) - half, laneTop(0), half * 2 + 0.5, laneTop(2) + LANE_H - laneTop(0))
      }
    }

    // Lanes
    lanes.forEach((lane, i) => {
      const top = laneTop(i)
      const yOf = (v: number) => top + LANE_H - 2 - Math.min(Math.max(v, 0), 1) * (LANE_H - 6)

      ctx.fillStyle = 'rgba(255,255,255,0.025)'
      ctx.fillRect(PAD_L, top, plotW, LANE_H)

      // Threshold
      const ty = Math.round(yOf(lane.threshold)) + 0.5
      ctx.strokeStyle = lane.colour
      ctx.globalAlpha = 0.45
      ctx.setLineDash([3, 3])
      ctx.lineWidth = 1
      ctx.beginPath()
      ctx.moveTo(PAD_L, ty)
      ctx.lineTo(PAD_L + plotW, ty)
      ctx.stroke()
      ctx.setLineDash([])
      ctx.globalAlpha = 1

      // Area + line
      ctx.beginPath()
      frames.forEach((f, j) => {
        const x = xOf(f.timestamp_s)
        const y = yOf(f[lane.key])
        if (j === 0) ctx.moveTo(x, y)
        else ctx.lineTo(x, y)
      })
      const lastX = xOf(frames[frames.length - 1].timestamp_s)
      const firstX = xOf(frames[0].timestamp_s)
      ctx.lineTo(lastX, top + LANE_H - 2)
      ctx.lineTo(firstX, top + LANE_H - 2)
      ctx.closePath()
      ctx.fillStyle = lane.colour
      ctx.globalAlpha = 0.14
      ctx.fill()
      ctx.globalAlpha = 1

      ctx.beginPath()
      ctx.strokeStyle = lane.colour
      ctx.lineWidth = 1.5
      ctx.lineJoin = 'round'
      frames.forEach((f, j) => {
        const x = xOf(f.timestamp_s)
        const y = yOf(f[lane.key])
        if (j === 0) ctx.moveTo(x, y)
        else ctx.lineTo(x, y)
      })
      ctx.stroke()

      // Lane label + threshold value
      ctx.font = '600 9px Inter, system-ui, sans-serif'
      ctx.fillStyle = lane.colour
      ctx.globalAlpha = 0.9
      ctx.fillText(lane.label, PAD_L + 5, top + 10)
      ctx.globalAlpha = 0.6
      ctx.font = '500 9px "JetBrains Mono", ui-monospace, monospace'
      const thrText = `≥${lane.threshold.toFixed(2)}`
      ctx.fillText(thrText, PAD_L + plotW - ctx.measureText(thrText).width - 5, top + 10)
      ctx.globalAlpha = 1
    })

    // Decision markers
    const markY = BAND_H + 2
    for (const ev of events) {
      const x = xOf(ev.t)
      ctx.fillStyle = EVENT_COLOUR[ev.kind]
      ctx.beginPath()
      if (ev.kind === 'blocked') {
        ctx.globalAlpha = 0.55
        ctx.arc(x, markY + MARK_H / 2, 2, 0, Math.PI * 2)
      } else {
        ctx.moveTo(x, markY + MARK_H - 1)
        ctx.lineTo(x - 4.5, markY + 1)
        ctx.lineTo(x + 4.5, markY + 1)
        ctx.closePath()
      }
      ctx.fill()
      ctx.globalAlpha = 1
    }

    // Hover hairline
    if (hover) {
      const hx = Math.round(hover.x) + 0.5
      ctx.strokeStyle = COLOUR.hair
      ctx.lineWidth = 1
      ctx.setLineDash([2, 3])
      ctx.beginPath()
      ctx.moveTo(hx, 0)
      ctx.lineTo(hx, axisY)
      ctx.stroke()
      ctx.setLineDash([])
      const f = frames[hover.index]
      if (f) {
        lanes.forEach((lane, i) => {
          const top = laneTop(i)
          const y = top + LANE_H - 2 - Math.min(Math.max(f[lane.key], 0), 1) * (LANE_H - 6)
          ctx.fillStyle = lane.colour
          ctx.beginPath()
          ctx.arc(xOf(f.timestamp_s), y, 3, 0, Math.PI * 2)
          ctx.fill()
        })
      }
    }

    // Playhead
    const px = Math.round(xOf(current)) + 0.5
    ctx.strokeStyle = 'rgba(255,255,255,0.85)'
    ctx.lineWidth = 1.25
    ctx.beginPath()
    ctx.moveTo(px, 0)
    ctx.lineTo(px, axisY)
    ctx.stroke()
    ctx.fillStyle = '#ffffff'
    ctx.beginPath()
    ctx.moveTo(px - 4, 0)
    ctx.lineTo(px + 4, 0)
    ctx.lineTo(px, 5)
    ctx.closePath()
    ctx.fill()
  }, [frames, duration, current, width, plotW, lanes, events, hover, xOf])

  // ── interaction ────────────────────────────────────────────────────
  const locate = (e: ReactPointerEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect()
    const x = e.clientX - rect.left
    return { x, t: tOf(x) }
  }

  const onPointerMove = (e: ReactPointerEvent<HTMLDivElement>) => {
    if (!frames.length || duration <= 0) return
    const { x, t } = locate(e)
    setHover({ x, index: indexAt(frames, t) })
    if (dragging.current) onSeek(t)
  }
  const onPointerDown = (e: ReactPointerEvent<HTMLDivElement>) => {
    if (!frames.length || duration <= 0) return
    dragging.current = true
    e.currentTarget.setPointerCapture(e.pointerId)
    onSeek(locate(e).t)
  }
  const onPointerUp = (e: ReactPointerEvent<HTMLDivElement>) => {
    dragging.current = false
    if (e.currentTarget.hasPointerCapture(e.pointerId)) e.currentTarget.releasePointerCapture(e.pointerId)
  }

  const hovered = hover ? frames[hover.index] : null
  const tooltipLeft = hover ? Math.min(Math.max(hover.x, 90), Math.max(width - 90, 90)) : 0

  return (
    <div className="timeline-wrap">
      <div
        ref={wrapRef}
        className="timeline"
        style={{ height: TIMELINE_HEIGHT, cursor: duration > 0 ? 'crosshair' : 'default' }}
        role="slider"
        tabIndex={0}
        aria-label="Playhead"
        aria-valuemin={0}
        aria-valuemax={Math.max(duration, 0)}
        aria-valuenow={current}
        aria-valuetext={`${current.toFixed(2)} seconds`}
        onPointerMove={onPointerMove}
        onPointerLeave={() => { setHover(null); dragging.current = false }}
        onPointerDown={onPointerDown}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
      >
        <canvas ref={canvasRef} style={{ width: '100%', height: TIMELINE_HEIGHT }} />
        {hovered && (
          <div className="timeline-tip" style={{ left: tooltipLeft }} role="tooltip">
            <div className="tip-head mono">t = {hovered.timestamp_s.toFixed(2)} s <span>frame {hovered.frame_index}</span></div>
            <div className="tip-grid">
              <span style={{ color: COLOUR.vision }}>Vision</span><b className="mono">{hovered.vision_confidence.toFixed(3)}</b>
              <span style={{ color: COLOUR.audio }}>Audio</span><b className="mono">{hovered.audio_confidence.toFixed(3)}</b>
              <span style={{ color: COLOUR.fused }}>Fused</span><b className="mono">{hovered.fused_confidence.toFixed(3)}</b>
              <span>Distance</span><b className="mono">{hovered.distance_m != null ? `${hovered.distance_m.toFixed(1)} m` : '—'}</b>
              <span>Speed</span><b className="mono">{hovered.speed_kmph != null ? `${hovered.speed_kmph.toFixed(0)} km/h` : '—'}</b>
              <span>ETA</span><b className="mono">{hovered.eta_s != null ? `${hovered.eta_s.toFixed(1)} s` : '—'}</b>
              <span>Bearing</span><b className="mono">{hovered.direction_available && hovered.direction_deg != null ? `${hovered.direction_deg > 0 ? '+' : ''}${hovered.direction_deg.toFixed(0)}° ${hovered.direction_bearing}` : 'n/a'}</b>
              <span>Signal</span><b>{hovered.signal_state.replace('_', ' ')}</b>
              <span>Latency</span><b className="mono">{hovered.latency_ms.toFixed(0)} ms</b>
            </div>
            <div className="tip-decision">{shortDecision(hovered.decision)}</div>
          </div>
        )}
      </div>

      <div className="timeline-foot">
        <div className="timeline-legend">
          <LegendItem colour={COLOUR.vision} label="Vision" />
          <LegendItem colour={COLOUR.audio} label="Audio" />
          <LegendItem colour={COLOUR.fused} label="Fused" />
          <LegendItem colour={COLOUR.go} label="Preemption" block />
          <LegendItem colour={COLOUR.warn} label="Amber" block />
          <span className="legend-item legend-hint">dashed = threshold {'·'} drag to scrub {'·'} {'←'} {'→'} step</span>
        </div>
      </div>

      {events.length > 0 && (
        <div className="events">
          <div className="events-head">
            <ListOrdered size={12} /> Decision events <span className="mono">{events.length}</span>
          </div>
          <ul className="events-list">
            {events.map((ev, i) => (
              <li key={`${ev.kind}-${ev.index}-${i}`}>
                <button
                  className="event-row"
                  data-kind={ev.kind}
                  data-current={Math.abs(ev.t - current) < 0.001}
                  onClick={() => onSeek(ev.t)}
                  title={ev.detail}
                >
                  <span className="event-dot" style={{ background: EVENT_COLOUR[ev.kind] }} />
                  <span className="event-time mono">{ev.t.toFixed(2)} s</span>
                  <span className="event-label">{ev.label}</span>
                  <span className="event-detail">{ev.detail}</span>
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}

function LegendItem({ colour, label, block }: { colour: string; label: string; block?: boolean }) {
  return (
    <span className="legend-item">
      <span className="legend-swatch" style={{ background: colour, height: block ? 8 : 3, opacity: block ? 0.7 : 1 }} />
      {label}
    </span>
  )
}

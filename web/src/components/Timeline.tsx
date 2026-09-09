import { useEffect, useRef } from 'react'
import type { FrameResult } from '../lib/api'

interface Props {
  frames: FrameResult[]
  duration: number
  current: number
  onSeek: (t: number) => void
}

/**
 * Confidence-over-time strip with preemption bands.
 *
 * Drawn to a canvas because a full analysis produces hundreds of samples and
 * three overlaid series; as SVG that is thousands of nodes for something that
 * is redrawn on every frame.
 */
export function Timeline({ frames, duration, current, onSeek }: Props) {
  const ref = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = ref.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const rect = canvas.getBoundingClientRect()
    const dpr = window.devicePixelRatio || 1
    canvas.width = rect.width * dpr
    canvas.height = rect.height * dpr
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)

    const W = rect.width
    const H = rect.height
    ctx.clearRect(0, 0, W, H)

    // Ground
    ctx.fillStyle = '#0c0e10'
    ctx.fillRect(0, 0, W, H)

    if (!frames.length || duration <= 0) {
      ctx.fillStyle = '#5f666e'
      ctx.font = '11px Inter, system-ui, sans-serif'
      ctx.fillText('No analysis yet', 10, H / 2 + 4)
      return
    }

    const x = (t: number) => (t / duration) * W
    const y = (v: number) => H - 4 - v * (H - 10)

    // Preemption bands first, so the traces read on top of them.
    ctx.fillStyle = 'rgba(52,199,89,0.16)'
    for (const f of frames) {
      if (f.signal_state === 'preempt_green') {
        ctx.fillRect(x(f.timestamp_s) - 1, 0, 3, H)
      }
    }

    const series: [keyof FrameResult, string, number][] = [
      ['vision_confidence', '#4c8dff', 1.4],
      ['audio_confidence', '#38bdb4', 1.4],
      ['fused_confidence', '#a78bfa', 1.8],
    ]

    for (const [key, colour, width] of series) {
      ctx.beginPath()
      ctx.strokeStyle = colour
      ctx.lineWidth = width
      ctx.lineJoin = 'round'
      frames.forEach((f, i) => {
        const vx = x(f.timestamp_s)
        const vy = y(f[key] as number)
        if (i === 0) ctx.moveTo(vx, vy)
        else ctx.lineTo(vx, vy)
      })
      ctx.stroke()
    }

    // Playhead
    const px = x(current)
    ctx.strokeStyle = 'rgba(255,255,255,0.7)'
    ctx.lineWidth = 1
    ctx.beginPath()
    ctx.moveTo(px, 0)
    ctx.lineTo(px, H)
    ctx.stroke()
  }, [frames, duration, current])

  return (
    <div>
      <div className="timeline">
        <canvas
          ref={ref}
          onClick={(e) => {
            const rect = e.currentTarget.getBoundingClientRect()
            onSeek(((e.clientX - rect.left) / rect.width) * duration)
          }}
          style={{ cursor: duration > 0 ? 'pointer' : 'default' }}
        />
      </div>
      <div className="timeline-legend">
        <LegendItem colour="#4c8dff" label="Vision" />
        <LegendItem colour="#38bdb4" label="Audio" />
        <LegendItem colour="#a78bfa" label="Fused" />
        <LegendItem colour="rgba(52,199,89,0.6)" label="Preemption active" />
      </div>
    </div>
  )
}

function LegendItem({ colour, label }: { colour: string; label: string }) {
  return (
    <span className="legend-item">
      <span className="legend-swatch" style={{ background: colour }} />
      {label}
    </span>
  )
}

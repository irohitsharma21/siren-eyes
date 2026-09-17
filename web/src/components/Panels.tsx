import {
  AlertTriangle, Compass as CompassIcon, Gauge, Info, Radio, ShieldCheck,
} from 'lucide-react'
import type { AnalysisParams, FrameResult, SystemConfig } from '../lib/api'

/* ── Confidence meters ─────────────────────────────────────────────── */

export function Meters({
  frame, config, params,
}: { frame: FrameResult | null; config: SystemConfig | null; params: AnalysisParams | null }) {
  const v = frame?.vision_confidence ?? 0
  const a = frame?.audio_confidence ?? 0
  const f = frame?.fused_confidence ?? 0
  const trigger = params?.fused_threshold ?? config?.fusion.trigger_threshold ?? 0.6
  const visionGate = params?.vision_threshold ?? config?.vision_confidence_threshold ?? 0.65

  return (
    <div>
      <Meter name="Vision (YOLOv8)" value={v} kind="vision" threshold={visionGate} />
      <Meter name="Audio (siren CNN)" value={a} kind="audio" threshold={config?.siren_threshold ?? 0.5} />
      <Meter
        name={`Fused  ${config ? `${config.fusion.alpha_vision}·V + ${config.fusion.beta_audio}·A` : ''}`}
        value={f}
        kind="fused"
        threshold={trigger}
      />
    </div>
  )
}

function Meter({
  name, value, kind, threshold,
}: {
  name: string
  value: number
  kind: 'vision' | 'audio' | 'fused'
  threshold?: number
}) {
  return (
    <div className="meter">
      <div className="meter-top">
        <span className="meter-name">{name}</span>
        <span className="meter-value">{value.toFixed(3)}</span>
      </div>
      <div className="meter-track">
        <div
          className="meter-fill"
          data-kind={kind}
          style={{ width: `${Math.min(value, 1) * 100}%` }}
        />
        {threshold !== undefined && (
          <div
            className="meter-threshold"
            style={{ left: `${threshold * 100}%` }}
            title={`threshold ${threshold}`}
          />
        )}
      </div>
    </div>
  )
}

/* ── Signal head ───────────────────────────────────────────────────── */

const SIGNAL_LABEL: Record<string, string> = {
  red: 'Red — normal timing',
  green: 'Green',
  amber: 'Amber — reverting',
  all_red: 'All-red clearance',
  preempt_green: 'PREEMPTION GRANTED',
}

export function SignalHead({ frame }: { frame: FrameResult | null }) {
  const state = frame?.signal_state ?? 'red'
  const on = (which: string) =>
    state === 'preempt_green' || state === 'green'
      ? which === 'green'
      : state === 'amber'
        ? which === 'amber'
        : which === 'red'

  return (
    <div className="signal-head" data-state={state}>
      <div className="lamps">
        <div className={`lamp ${on('red') ? 'on-red' : ''}`} />
        <div className={`lamp ${on('amber') ? 'on-amber' : ''}`} />
        <div className={`lamp ${on('green') ? 'on-green' : ''}`} />
      </div>
      <div style={{ minWidth: 0 }}>
        <div className="signal-label">{SIGNAL_LABEL[state] ?? state}</div>
        <div className="signal-reason">{frame?.reason ?? 'Awaiting analysis'}</div>
      </div>
    </div>
  )
}

/* ── Kinematics ────────────────────────────────────────────────────── */

export function Kinematics({ frame }: { frame: FrameResult | null }) {
  const fmt = (n: number | null | undefined, digits = 0) =>
    n === null || n === undefined ? '—' : n.toFixed(digits)

  return (
    <div className="stat-grid">
      <Stat label="Distance" value={fmt(frame?.distance_m)} unit="m" />
      <Stat label="Closing speed" value={fmt(frame?.speed_kmph)} unit="km/h" />
      <Stat label="ETA" value={fmt(frame?.eta_s, 1)} unit="s" />
      <Stat label="Latency" value={fmt(frame?.latency_ms)} unit="ms" />
    </div>
  )
}

function Stat({ label, value, unit }: { label: string; value: string; unit: string }) {
  return (
    <div className="stat">
      <div className="stat-label">{label}</div>
      <div className="stat-value">
        {value}
        <span className="stat-unit">{unit}</span>
      </div>
    </div>
  )
}

/* ── Direction compass ─────────────────────────────────────────────── */

export function Compass({ frame }: { frame: FrameResult | null }) {
  const available = frame?.direction_available ?? false
  const angle = frame?.direction_deg ?? 0

  // -90..+90 maps onto a 180-degree arc drawn facing the camera.
  const R = 34
  const rad = ((angle - 90) * Math.PI) / 180
  const px = 44 + R * Math.cos(rad)
  const py = 44 + R * Math.sin(rad)

  return (
    <div className="compass">
      <div className="compass-dial">
        <svg width="88" height="52" viewBox="0 0 88 52" aria-hidden="true">
          <path
            d="M 10 44 A 34 34 0 0 1 78 44"
            fill="none"
            stroke="var(--border-strong)"
            strokeWidth="1.5"
          />
          {[-90, -45, 0, 45, 90].map((t) => {
            const r = ((t - 90) * Math.PI) / 180
            return (
              <line
                key={t}
                x1={44 + 29 * Math.cos(r)}
                y1={44 + 29 * Math.sin(r)}
                x2={44 + 34 * Math.cos(r)}
                y2={44 + 34 * Math.sin(r)}
                stroke="var(--text-mute)"
                strokeWidth="1"
              />
            )
          })}
          {available && (
            <>
              <line
                x1="44" y1="44" x2={px} y2={py}
                stroke="var(--audio)" strokeWidth="2.5" strokeLinecap="round"
              />
              <circle cx={px} cy={py} r="3.5" fill="var(--audio)" />
            </>
          )}
          <circle cx="44" cy="44" r="2.5" fill="var(--text-mute)" />
        </svg>
      </div>

      <div className="compass-readout">
        {available ? (
          <>
            <div className="compass-angle">
              {angle > 0 ? '+' : ''}
              {angle.toFixed(0)}°
            </div>
            <div className="compass-bearing">{frame?.direction_bearing}</div>
            <div className="compass-cue">ITD + ILD fusion</div>
          </>
        ) : (
          <div className="compass-unavailable">
            Direction unavailable — requires a true stereo source. Vision-only
            gating is applied.
          </div>
        )}
      </div>
    </div>
  )
}

/* ── Pipeline stages ───────────────────────────────────────────────── */

type StageStatus = 'idle' | 'pass' | 'block'

export function Pipeline({
  frame, config, params,
}: { frame: FrameResult | null; config: SystemConfig | null; params: AnalysisParams | null }) {
  const visionOk = (frame?.vision_confidence ?? 0) >= (params?.vision_threshold ?? config?.vision_confidence_threshold ?? 0.65)
  const audioOk = (frame?.audio_confidence ?? 0) >= (config?.siren_threshold ?? 0.5)
  const fusedOk = (frame?.fused_confidence ?? 0) >= (params?.fused_threshold ?? config?.fusion.trigger_threshold ?? 0.6)
  const decision = frame?.decision ?? 'idle'

  const stages: { name: string; status: StageStatus; value: string }[] = [
    {
      name: 'Visual detection',
      status: visionOk ? 'pass' : frame?.detections.length ? 'block' : 'idle',
      value: (frame?.vision_confidence ?? 0).toFixed(2),
    },
    {
      name: 'Siren confirmation',
      status: audioOk ? 'pass' : frame ? 'block' : 'idle',
      value: (frame?.audio_confidence ?? 0).toFixed(2),
    },
    {
      name: 'Direction (ITD/ILD)',
      status: frame?.direction_available ? 'pass' : 'idle',
      value: frame?.direction_available ? `${frame.direction_deg?.toFixed(0)}°` : 'n/a',
    },
    {
      name: 'Distance & ETA',
      status: frame?.eta_s != null ? 'pass' : frame?.distance_m != null ? 'block' : 'idle',
      value: frame?.eta_s != null ? `${frame.eta_s.toFixed(1)}s` : '—',
    },
    {
      name: 'Multi-modal fusion',
      status: fusedOk ? 'pass' : frame ? 'block' : 'idle',
      value: (frame?.fused_confidence ?? 0).toFixed(2),
    },
    {
      name: 'Safety gate (TTC + buffer)',
      status:
        decision === 'granted' || decision === 'holding'
          ? 'pass'
          : decision.startsWith('blocked')
            ? 'block'
            : 'idle',
      value: decision.startsWith('blocked') ? 'blocked' : decision === 'idle' ? '—' : 'clear',
    },
    {
      name: 'Signal override',
      status:
        frame?.signal_state === 'preempt_green'
          ? 'pass'
          : frame?.signal_state === 'amber' || frame?.signal_state === 'all_red'
            ? 'block'
            : 'idle',
      value: frame?.signal_state ?? '—',
    },
  ]

  return (
    <div className="pipeline">
      {stages.map((s, i) => (
        <div
          key={s.name}
          className="stage-row"
          data-status={s.status}
          data-active={s.status !== 'idle'}
        >
          <span className="stage-dot" />
          <span className="stage-name">
            <span className="mono" style={{ color: 'var(--text-mute)', marginRight: 6 }}>
              {i + 1}
            </span>
            {s.name}
          </span>
          <span className="stage-val">{s.value}</span>
        </div>
      ))}
    </div>
  )
}

/* ── Notices ───────────────────────────────────────────────────────── */

export function Notice({
  kind = 'info',
  children,
}: {
  kind?: 'info' | 'warn'
  children: React.ReactNode
}) {
  return (
    <div className={`notice notice-${kind}`}>
      {kind === 'warn' ? <AlertTriangle size={14} /> : <Info size={14} />}
      <div>{children}</div>
    </div>
  )
}

export const PanelIcons = { Gauge, CompassIcon, ShieldCheck, Radio }

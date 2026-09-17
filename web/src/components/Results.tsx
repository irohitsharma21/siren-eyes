import { useEffect, useState } from 'react'
import { Check, ClipboardCopy, Download, FileJson, FileSpreadsheet, ShieldCheck, ShieldOff, Square } from 'lucide-react'
import type { AnalysisSummary, SystemConfig } from '../lib/api'
import { blockedReasonLabel, paramsEqual, summariseFrames } from '../lib/analysis'
import { copyText, download, runToCSV, runToJSON, safeName, shareText, type ExportableRun } from '../lib/export'
import { Notice } from './Panels'

interface Props {
  run: ExportableRun
  config: SystemConfig | null
  onSeek: (t: number) => void
}

const STAGE_LABEL: Record<string, string> = {
  vision: 'YOLOv8 detection',
  audio: 'Siren lookup',
  direction: 'Geometry + tracking',
  decision: 'Fusion + safety gate',
}

/**
 * Post-run results: verdict, peaks, latencies, withheld reasons and export.
 * Works from the server summary when there is one, and from the frames alone
 * when the run was stopped before a summary arrived.
 */
export function Results({ run, config, onSeek }: Props) {
  const { frames, summary, params } = run
  const derived = summariseFrames(frames, params)
  const s: Partial<AnalysisSummary> = summary ?? {}

  const peakV = s.peak_vision_confidence ?? derived.peak_vision_confidence
  const peakA = s.peak_siren_confidence ?? derived.peak_siren_confidence
  const peakF = s.peak_fused_confidence ?? derived.peak_fused_confidence
  const granted = s.preemption_granted ?? derived.preemption_granted
  const grantT = s.grant_time_s ?? derived.grant_time_s
  const firstT = s.first_detection_s ?? derived.first_detection_s
  const response = s.response_latency_s ?? derived.response_latency_s
  const framesN = s.frames_analysed ?? derived.frames_analysed
  const detN = s.detection_frames ?? derived.detection_frames
  const meanMs = s.mean_latency_ms ?? derived.mean_latency_ms
  const p95Ms = s.p95_latency_ms ?? derived.p95_latency_ms
  const blocked = Object.entries(s.blocked_reasons ?? derived.blocked_reasons).sort((a, b) => b[1] - a[1])
  const blockedTotal = blocked.reduce((n, [, c]) => n + c, 0)
  const stages = Object.entries(s.stage_latency_ms ?? {})
  const stageMax = Math.max(1e-6, ...stages.map(([, v]) => v))

  const defaults = config
    ? paramsEqual(params, {
      vision_threshold: config.parameters.vision_threshold.default,
      fused_threshold: config.parameters.fused_threshold.default,
      safety_buffer_s: config.parameters.safety_buffer_s.default,
      min_ttc_s: config.parameters.min_ttc_s.default,
    })
    : true

  const [copied, setCopied] = useState(false)
  useEffect(() => {
    if (!copied) return
    const t = window.setTimeout(() => setCopied(false), 1800)
    return () => window.clearTimeout(t)
  }, [copied])

  const base = safeName(run.clipLabel)
  const stamp = run.createdAt.replace(/[:.]/g, '-').slice(0, 19)

  return (
    <div className="results">
      <div className={`verdict ${granted ? 'verdict-go' : 'verdict-hold'}`}>
        <span className="verdict-icon">
          {run.status === 'cancelled' ? <Square size={18} /> : granted ? <ShieldCheck size={20} /> : <ShieldOff size={20} />}
        </span>
        <div className="verdict-body">
          <div className="verdict-title">
            {run.status === 'cancelled'
              ? `Stopped after ${framesN} frames`
              : granted
                ? 'Preemption granted'
                : 'Preemption withheld'}
          </div>
          <div className="verdict-sub">
            {granted && grantT != null ? (
              <>
                Granted at <button className="linkish mono" onClick={() => onSeek(grantT)}>{grantT.toFixed(2)} s</button>
                {firstT != null && (
                  <>
                    , {response?.toFixed(2)} s after first detection at{' '}
                    <button className="linkish mono" onClick={() => onSeek(firstT)}>{firstT.toFixed(2)} s</button>
                  </>
                )}
                .
              </>
            ) : firstT != null ? (
              <>
                Ambulance first detected at{' '}
                <button className="linkish mono" onClick={() => onSeek(firstT)}>{firstT.toFixed(2)} s</button>
                {blocked.length ? `; the gate withheld on ${blocked.length} ground${blocked.length > 1 ? 's' : ''}.` : '.'}
              </>
            ) : (
              'No detection cleared the vision threshold.'
            )}
          </div>
        </div>
      </div>

      {run.precomputed && (
        <Notice kind="info">
          <strong>Recorded run</strong> {'—'} real figures from this pipeline, computed once
          on an unthrottled machine and replayed here. The hosted container has about a
          tenth of a CPU core, which is not enough for YOLOv8 over 1080p in reasonable time.
          Uploads and non-default parameters are always analysed live.
        </Notice>
      )}
      {summary && !summary.audio_available && (
        <Notice kind="warn">
          <strong>Audio unavailable</strong> {'—'} {summary.audio_note}. The siren and
          direction stages could not contribute; the decision used vision only.
        </Notice>
      )}
      {run.status === 'cancelled' && (
        <Notice kind="info">
          Figures cover the {framesN} frames analysed before the run was stopped.
        </Notice>
      )}

      <div className="kpi-grid">
        <Kpi label="Peak vision" value={peakV.toFixed(3)} tone="vision" />
        <Kpi label="Peak siren" value={peakA.toFixed(3)} tone="audio" />
        <Kpi label="Peak fused" value={peakF.toFixed(3)} tone="fused" />
        <Kpi label="Detect → grant" value={response != null ? response.toFixed(2) : '—'} unit={response != null ? 's' : ''} />
        <Kpi label="Frames" value={String(framesN)} unit={detN ? `${detN} det.` : ''} />
        <Kpi label="Latency" value={meanMs.toFixed(0)} unit={`ms · p95 ${p95Ms.toFixed(0)}`} />
      </div>

      <div className="results-cols">
        <section>
          <h4 className="section-title">Stage latency <span>mean per frame</span></h4>
          {stages.length ? (
            <div className="bars">
              {stages.map(([stage, ms]) => (
                <div className="bar-row" key={stage}>
                  <span className="bar-label">{STAGE_LABEL[stage] ?? stage}</span>
                  <span className="bar-track">
                    <span className="bar-fill" style={{ width: `${Math.max(1.5, (ms / stageMax) * 100)}%` }} />
                  </span>
                  <span className="bar-val mono">{ms < 1 ? ms.toFixed(2) : ms.toFixed(1)} ms</span>
                </div>
              ))}
            </div>
          ) : (
            <p className="muted">Per-stage figures arrive with the server summary.</p>
          )}
        </section>

        <section>
          <h4 className="section-title">Why preemption was withheld <span>{blockedTotal} frames</span></h4>
          {blocked.length ? (
            <div className="bars">
              {blocked.map(([reason, n]) => (
                <div className="bar-row" key={reason}>
                  <span className="bar-label">{blockedReasonLabel(reason)}</span>
                  <span className="bar-track">
                    <span className="bar-fill bar-fill-warn" style={{ width: `${(n / Math.max(blockedTotal, 1)) * 100}%` }} />
                  </span>
                  <span className="bar-val mono">{n}</span>
                </div>
              ))}
            </div>
          ) : (
            <p className="muted">The gate never withheld: every triggered frame passed.</p>
          )}
        </section>
      </div>

      <div className="results-meta">
        {summary && (
          <span>{summary.video} {'·'} {summary.resolution} {'·'} {summary.fps} fps {'·'} {summary.duration_s.toFixed(1)} s</span>
        )}
        <span>
          Parameters: V {'≥'} {params.vision_threshold} {'·'} F {'≥'} {params.fused_threshold} {'·'} buffer {params.safety_buffer_s} s {'·'} TTC {'≥'} {params.min_ttc_s} s
          {' '}<em className={defaults ? 'tag' : 'tag tag-warn'}>{defaults ? 'paper defaults' : 'modified'}</em>
        </span>
      </div>

      <div className="results-actions">
        <button className="btn btn-sm" onClick={() => download(runToJSON(run), `siren-eyes_${base}_${stamp}.json`)}>
          <FileJson size={13} /> Export JSON
        </button>
        <button className="btn btn-sm" onClick={() => download(runToCSV(frames), `siren-eyes_${base}_${stamp}.csv`)}>
          <FileSpreadsheet size={13} /> Export CSV
        </button>
        <button
          className="btn btn-sm"
          onClick={async () => { if (await copyText(shareText(run))) setCopied(true) }}
        >
          {copied ? <Check size={13} /> : <ClipboardCopy size={13} />} {copied ? 'Copied' : 'Copy summary'}
        </button>
        <span className="results-hint"><Download size={11} /> {frames.length} frames per file</span>
      </div>
    </div>
  )
}

function Kpi({ label, value, unit, tone }: { label: string; value: string; unit?: string; tone?: string }) {
  return (
    <div className="kpi" data-tone={tone}>
      <div className="kpi-label">{label}</div>
      <div className="kpi-value">{value}{unit ? <span className="kpi-unit">{unit}</span> : null}</div>
    </div>
  )
}

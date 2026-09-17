import { useEffect, useRef } from 'react'
import { RotateCcw, SlidersHorizontal } from 'lucide-react'
import type { AnalysisParams, ParamBound, ParamName } from '../lib/api'
import { paramsEqual } from '../lib/analysis'

interface Props {
  params: AnalysisParams
  bounds: Record<ParamName, ParamBound> | null
  disabled: boolean
  /** True when the selected clip has a recorded result that defaults would replay. */
  recordedAvailable: boolean
  open: boolean
  onOpenChange: (open: boolean) => void
  onChange: (next: AnalysisParams) => void
}

const FIELDS: { name: ParamName; label: string; unit: string; help: string }[] = [
  {
    name: 'vision_threshold', label: 'Vision threshold', unit: '',
    help: 'YOLO confidence a box needs before it counts as an ambulance. Paper: 0.65.',
  },
  {
    name: 'fused_threshold', label: 'Fused trigger', unit: '',
    help: 'Fused 0.7·vision + 0.3·audio confidence that arms the safety gate. Paper: 0.60.',
  },
  {
    name: 'safety_buffer_s', label: 'Safety buffer', unit: 's',
    help: 'Minimum ETA before a phase change is allowed, so the change is never abrupt. Paper: 5 s.',
  },
  {
    name: 'min_ttc_s', label: 'Min. time-to-collision', unit: 's',
    help: 'Conflicting traffic closer than this blocks the grant. Paper: 2 s. (No conflicts are injected in this demo.)',
  },
]

export function defaultsFrom(bounds: Record<ParamName, ParamBound>): AnalysisParams {
  return {
    vision_threshold: bounds.vision_threshold.default,
    fused_threshold: bounds.fused_threshold.default,
    safety_buffer_s: bounds.safety_buffer_s.default,
    min_ttc_s: bounds.min_ttc_s.default,
  }
}

/** Compact popover with the four per-run thresholds. */
export function ParametersButton({ params, bounds, disabled, recordedAvailable, open, onOpenChange, onChange }: Props) {
  const setOpen = onOpenChange
  const ref = useRef<HTMLDivElement>(null)
  const defaults = bounds ? defaultsFrom(bounds) : null
  const modified = defaults ? !paramsEqual(params, defaults) : false

  useEffect(() => {
    if (!open) return
    const onDoc = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false)
    }
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') { setOpen(false); e.stopPropagation() }
    }
    document.addEventListener('mousedown', onDoc)
    document.addEventListener('keydown', onKey, true)
    return () => {
      document.removeEventListener('mousedown', onDoc)
      document.removeEventListener('keydown', onKey, true)
    }
  }, [open, setOpen])

  return (
    <div className="popover-anchor" ref={ref}>
      <button
        className="btn btn-sm"
        data-modified={modified}
        onClick={() => setOpen(!open)}
        aria-expanded={open}
        aria-haspopup="dialog"
        title="Per-run thresholds (P)"
      >
        <SlidersHorizontal size={13} />
        <span className="hide-sm">Parameters</span>
        {modified && <span className="dot-badge" aria-label="modified" />}
      </button>

      {open && (
        <div className="popover" role="dialog" aria-label="Analysis parameters">
          <div className="popover-head">
            <span>Parameters for the next run</span>
            <button
              className="btn btn-ghost btn-sm"
              disabled={!defaults || !modified}
              onClick={() => defaults && onChange(defaults)}
              title="Reset to the paper's defaults"
            >
              <RotateCcw size={12} /> Reset
            </button>
          </div>

          {!bounds ? (
            <p className="muted" style={{ padding: '0.5rem 0' }}>Loading limits from the server{'…'}</p>
          ) : (
            <div className="param-list">
              {FIELDS.map((f) => {
                const b = bounds[f.name]
                const v = params[f.name]
                const isDefault = Math.abs(v - b.default) < 1e-9
                return (
                  <div className="param" key={f.name}>
                    <div className="param-top">
                      <label htmlFor={`param-${f.name}`}>{f.label}</label>
                      <span className="param-val mono" data-default={isDefault}>
                        {v}{f.unit ? ` ${f.unit}` : ''}
                        {!isDefault && <em> (default {b.default})</em>}
                      </span>
                    </div>
                    <div className="param-controls">
                      <input
                        id={`param-${f.name}`}
                        type="range"
                        min={b.min}
                        max={b.max}
                        step={b.step}
                        value={v}
                        disabled={disabled}
                        onChange={(e) => onChange({ ...params, [f.name]: Number(e.target.value) })}
                      />
                      <input
                        type="number"
                        className="param-num mono"
                        min={b.min}
                        max={b.max}
                        step={b.step}
                        value={v}
                        disabled={disabled}
                        aria-label={`${f.label} value`}
                        onChange={(e) => {
                          const n = Number(e.target.value)
                          if (Number.isFinite(n)) onChange({ ...params, [f.name]: Math.min(b.max, Math.max(b.min, n)) })
                        }}
                      />
                    </div>
                    <div className="param-help">{f.help}</div>
                  </div>
                )
              })}
            </div>
          )}

          <div className="popover-foot">
            {disabled ? (
              <span>Parameters lock while a run is in progress.</span>
            ) : recordedAvailable ? (
              modified
                ? <span className="warn-text">Non-default values: this demo will be analysed <strong>live on the CPU</strong> instead of replaying its recording {'—'} minutes rather than seconds on the free tier.</span>
                : <span>At the defaults, the demo replays its recorded analysis instantly.</span>
            ) : (
              <span>Applied to the next run; the server validates the same ranges.</span>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

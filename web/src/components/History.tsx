import { Film, History as HistoryIcon, Trash2, Upload, X } from 'lucide-react'
import { relativeTime, type RunRecord } from '../lib/history'

interface Props {
  runs: RunRecord[]
  activeId: string | null
  busy: boolean
  onSelect: (run: RunRecord) => void
  onRemove: (id: string) => void
  onClear: () => void
}

function verdictOf(run: RunRecord): { label: string; tone: string } {
  if (run.status === 'error') return { label: 'failed', tone: 'stop' }
  if (run.status === 'cancelled') return { label: 'stopped', tone: 'mute' }
  const granted = run.summary?.preemption_granted ?? run.frames.some((f) => f.decision === 'granted')
  return granted ? { label: 'granted', tone: 'go' } : { label: 'withheld', tone: 'warn' }
}

export function History({ runs, activeId, busy, onSelect, onRemove, onClear }: Props) {
  if (!runs.length) {
    return (
      <div className="history-empty">
        <HistoryIcon size={16} />
        <span>Runs from this browser will be listed here. The last twelve are kept, including across reloads.</span>
      </div>
    )
  }
  return (
    <div className="history">
      <ul className="history-list">
        {runs.map((r) => {
          const v = verdictOf(r)
          return (
            <li key={r.id}>
              <div className="history-row" data-active={r.id === activeId}>
                <button
                  className="history-main"
                  onClick={() => onSelect(r)}
                  disabled={busy}
                  title={r.framesDropped ? 'Frames were dropped to stay within storage budget' : 'Load this run'}
                >
                  <span className="history-icon">{r.clip.kind === 'demo' ? <Film size={13} /> : <Upload size={13} />}</span>
                  <span className="history-text">
                    <span className="history-name">{r.clip.label}</span>
                    <span className="history-meta">
                      {relativeTime(r.createdAt)} {'·'} {r.frames.length || (r.framesDropped ? 'no' : 0)} frames
                      {r.precomputed && ' · recorded'}
                    </span>
                  </span>
                  <span className="tag" data-tone={v.tone}>{v.label}</span>
                </button>
                <button
                  className="history-remove"
                  onClick={() => onRemove(r.id)}
                  aria-label={`Remove run of ${r.clip.label}`}
                  title="Remove"
                >
                  <X size={12} />
                </button>
              </div>
            </li>
          )
        })}
      </ul>
      <button className="btn btn-ghost btn-sm" onClick={onClear} style={{ marginTop: 6 }}>
        <Trash2 size={12} /> Clear history
      </button>
    </div>
  )
}

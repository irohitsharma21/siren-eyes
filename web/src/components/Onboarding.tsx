import { useEffect, useRef, useState } from 'react'
import type { DragEvent } from 'react'
import { Ear, Eye, Keyboard, Navigation, Play, ShieldCheck, Upload, X } from 'lucide-react'
import type { SystemConfig } from '../lib/api'

/* ── First-run explainer ──────────────────────────────────────────── */

const EXPLAINER_KEY = 'siren-eyes.explainer.dismissed'

export function useExplainer() {
  const [visible, setVisible] = useState(() => {
    try {
      return localStorage.getItem(EXPLAINER_KEY) !== '1'
    } catch {
      return true
    }
  })
  const dismiss = () => {
    setVisible(false)
    try {
      localStorage.setItem(EXPLAINER_KEY, '1')
    } catch {
      /* fine */
    }
  }
  const show = () => setVisible(true)
  return { visible, dismiss, show }
}

export function Explainer({ onDismiss, onRunDemo, canRun }: { onDismiss: () => void; onRunDemo: () => void; canRun: boolean }) {
  const steps = [
    { Icon: Eye, title: 'Detect', copy: 'YOLOv8 finds the ambulance in every third frame; boxes under 0.65 confidence are shown dimmed, not hidden.' },
    { Icon: Ear, title: 'Confirm', copy: 'A CNN scores the stereo audio for a siren, and ITD/ILD cues place it on a bearing.' },
    { Icon: Navigation, title: 'Locate', copy: 'Box width through a pinhole model gives distance; a least-squares fit gives closing speed and ETA.' },
    { Icon: ShieldCheck, title: 'Decide', copy: 'The gate grants green only if the vehicle is approaching, outside the 5 s buffer, on-axis, with no conflict inside 2 s TTC.' },
  ]
  return (
    <div className="explainer" role="region" aria-label="How it works">
      <div className="explainer-head">
        <span className="explainer-title">How it works</span>
        <div className="explainer-actions">
          <button className="btn btn-primary btn-sm" onClick={onRunDemo} disabled={!canRun}>
            <Play size={12} /> Run the demo
          </button>
          <button className="btn btn-ghost btn-sm" onClick={onDismiss} aria-label="Dismiss explainer">
            <X size={13} />
          </button>
        </div>
      </div>
      <ol className="explainer-steps">
        {steps.map((s, i) => (
          <li key={s.title}>
            <span className="explainer-num">{i + 1}</span>
            <s.Icon size={15} className="explainer-icon" />
            <div>
              <div className="explainer-step-title">{s.title}</div>
              <div className="explainer-step-copy">{s.copy}</div>
            </div>
          </li>
        ))}
      </ol>
      <div className="explainer-foot">
        Seeing an ambulance is not sufficient reason to change a traffic light: the interesting part is the refusal to act.
      </div>
    </div>
  )
}

/* ── Keyboard shortcut sheet ──────────────────────────────────────── */

export const SHORTCUTS: [string, string][] = [
  ['Space', 'Play / pause'],
  ['← / →', 'Step one analysed frame'],
  ['Shift + ← / →', 'Step ten frames'],
  ['Home / End', 'Jump to start / end'],
  ['R', 'Run analysis'],
  ['Esc', 'Stop a running analysis / close dialogs'],
  ['S', 'Download snapshot of the current frame'],
  ['C', 'Toggle raw detector vs gated overlay'],
  ['P', 'Open parameters'],
  ['U', 'Upload a video'],
  ['?', 'This sheet'],
]

export function ShortcutSheet({ open, onClose }: { open: boolean; onClose: () => void }) {
  const ref = useRef<HTMLDivElement>(null)
  useEffect(() => {
    if (open) ref.current?.focus()
  }, [open])
  if (!open) return null
  return (
    <div className="modal-backdrop" onMouseDown={(e) => { if (e.target === e.currentTarget) onClose() }}>
      <div className="modal" role="dialog" aria-modal="true" aria-label="Keyboard shortcuts" ref={ref} tabIndex={-1}>
        <div className="modal-head">
          <span className="panel-title"><Keyboard size={12} /> Keyboard shortcuts</span>
          <button className="btn btn-ghost btn-sm" onClick={onClose} aria-label="Close"><X size={13} /></button>
        </div>
        <dl className="shortcuts">
          {SHORTCUTS.map(([keys, what]) => (
            <div key={keys} className="shortcut-row">
              <dt><kbd>{keys}</kbd></dt>
              <dd>{what}</dd>
            </div>
          ))}
        </dl>
        <p className="muted" style={{ marginTop: 10 }}>Shortcuts pause while you are typing in a field.</p>
      </div>
    </div>
  )
}

/* ── Upload zone with validation ─────────────────────────────────── */

export function validateUpload(file: File, config: SystemConfig | null): string | null {
  const maxMb = config?.upload.max_mb ?? 200
  const exts = config?.upload.extensions ?? ['.mp4', '.mov', '.m4v', '.webm', '.mkv', '.avi']
  const ext = (file.name.match(/\.[a-z0-9]+$/i)?.[0] ?? '').toLowerCase()
  const isVideoMime = file.type.startsWith('video/')
  if (!isVideoMime && !exts.includes(ext)) {
    return `"${file.name}" is not a video. Accepted: ${exts.join(', ')}.`
  }
  if (file.size === 0) return `"${file.name}" is empty.`
  if (file.size > maxMb * 1024 * 1024) {
    return `"${file.name}" is ${(file.size / 1048576).toFixed(0)} MB; the limit is ${maxMb} MB. Trim the clip and try again.`
  }
  return null
}

interface UploadZoneProps {
  config: SystemConfig | null
  pendingFile: File | null
  error: string | null
  disabled: boolean
  progress: number | null
  inputRef: React.RefObject<HTMLInputElement>
  onFile: (file: File) => void
}

export function UploadZone({ config, pendingFile, error, disabled, progress, inputRef, onFile }: UploadZoneProps) {
  const [drag, setDrag] = useState(false)
  const maxMb = config?.upload.max_mb ?? 200
  const exts = config?.upload.extensions ?? []

  const onDrop = (e: DragEvent) => {
    e.preventDefault()
    setDrag(false)
    if (disabled) return
    const f = e.dataTransfer.files?.[0]
    if (f) onFile(f)
  }

  return (
    <div>
      <label
        className="dropzone"
        data-drag={drag}
        data-disabled={disabled}
        onDragOver={(e) => { e.preventDefault(); if (!disabled) setDrag(true) }}
        onDragLeave={() => setDrag(false)}
        onDrop={onDrop}
      >
        <Upload size={16} />
        <span className="dropzone-title">
          {pendingFile ? pendingFile.name : 'Drop a video here, or click to browse'}
        </span>
        <span className="dropzone-meta">
          {pendingFile
            ? `${(pendingFile.size / 1048576).toFixed(1)} MB · analysed live on the CPU`
            : `${exts.length ? exts.map((e) => e.slice(1)).join(' · ') : 'video'} · up to ${maxMb} MB · stereo audio enables direction`}
        </span>
        <input
          ref={inputRef}
          type="file"
          accept={['video/*', ...exts].join(',')}
          style={{ display: 'none' }}
          disabled={disabled}
          onChange={(e) => {
            const f = e.target.files?.[0]
            if (f) onFile(f)
            e.target.value = ''
          }}
        />
      </label>
      {progress != null && (
        <div className="upload-progress" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(progress * 100)}>
          <div className="progress"><div className="progress-fill" style={{ width: `${progress * 100}%` }} /></div>
          <span className="mono">{progress < 1 ? `Uploading ${(progress * 100).toFixed(0)}%` : 'Upload complete — queuing analysis'}</span>
        </div>
      )}
      {error && <div className="field-error" role="alert">{error}</div>}
    </div>
  )
}

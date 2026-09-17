/**
 * Run history for this browser.
 *
 * Every run is kept in memory for the session. Metadata and frames are also
 * mirrored to localStorage under a size budget so the list survives a reload:
 * a 165-frame run is roughly 60 KB, which is small enough that keeping the
 * frames themselves is cheaper than a second storage mechanism. When the
 * budget is exceeded the oldest runs lose their frames first, then drop off.
 *
 * Video is not stored. Demo clips reload from the server; an uploaded file's
 * object URL only lives as long as the page, so after a reload an upload run
 * still shows its timeline and results but not the footage behind them.
 */

import type { AnalysisParams, AnalysisSummary, FrameResult } from './api'

export type RunStatus = 'done' | 'cancelled' | 'error'

export interface RunRecord {
  id: string
  createdAt: string
  clip: {
    label: string
    kind: 'demo' | 'upload'
    /** Demo file name, so the footage can be reloaded from the server. */
    demoFile?: string
  }
  params: AnalysisParams
  status: RunStatus
  precomputed: boolean
  frames: FrameResult[]
  summary: AnalysisSummary | null
  sourceSize: { w: number; h: number } | null
  error?: string
  /** Set when the frames were shed to fit the storage budget. */
  framesDropped?: boolean
}

const KEY = 'siren-eyes.history.v1'
const MAX_RUNS = 12
const BUDGET_BYTES = 3_000_000

/** Object URLs for uploads, valid for this page's lifetime only. */
const sourceUrls = new Map<string, string>()

export function rememberSourceUrl(id: string, url: string) {
  sourceUrls.set(id, url)
}

export function sourceUrlFor(id: string): string | undefined {
  return sourceUrls.get(id)
}

export function loadHistory(): RunRecord[] {
  try {
    const raw = localStorage.getItem(KEY)
    if (!raw) return []
    const parsed = JSON.parse(raw)
    if (!Array.isArray(parsed)) return []
    return parsed.filter((r) => r && typeof r.id === 'string' && Array.isArray(r.frames))
  } catch {
    return []
  }
}

function persist(runs: RunRecord[]) {
  // Newest first; trim to the run cap, then to the byte budget by shedding
  // frames from the oldest entries before dropping entries outright.
  const list = runs.slice(0, MAX_RUNS).map((r) => ({ ...r }))
  const size = () => JSON.stringify(list).length
  // Shed frames from the oldest entries first, keeping their metadata.
  for (let i = list.length - 1; i >= 0 && size() > BUDGET_BYTES; i--) {
    if (list[i].frames.length) list[i] = { ...list[i], frames: [], framesDropped: true }
  }
  // Still over budget: drop whole entries from the old end.
  while (list.length > 1 && size() > BUDGET_BYTES) list.pop()
  try {
    localStorage.setItem(KEY, JSON.stringify(list))
  } catch {
    // Quota or blocked storage: keep halving until it fits or give up.
    try {
      localStorage.setItem(KEY, JSON.stringify(list.slice(0, Math.floor(list.length / 2))))
    } catch {
      /* the in-memory list still works for this session */
    }
  }
}

export function saveRun(runs: RunRecord[], record: RunRecord): RunRecord[] {
  const next = [record, ...runs.filter((r) => r.id !== record.id)].slice(0, MAX_RUNS)
  persist(next)
  return next
}

export function removeRun(runs: RunRecord[], id: string): RunRecord[] {
  const next = runs.filter((r) => r.id !== id)
  persist(next)
  sourceUrls.delete(id)
  return next
}

export function clearHistory(): RunRecord[] {
  try {
    localStorage.removeItem(KEY)
  } catch {
    /* nothing to clear */
  }
  sourceUrls.clear()
  return []
}

export function relativeTime(iso: string, now = Date.now()): string {
  const then = new Date(iso).getTime()
  if (Number.isNaN(then)) return ''
  const s = Math.max(0, Math.round((now - then) / 1000))
  if (s < 45) return 'just now'
  const m = Math.round(s / 60)
  if (m < 60) return `${m} min ago`
  const h = Math.round(m / 60)
  if (h < 24) return `${h} h ago`
  const d = Math.round(h / 24)
  return d === 1 ? 'yesterday' : `${d} d ago`
}

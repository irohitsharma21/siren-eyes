import { useEffect, useRef, useState } from 'react'
import type { CSSProperties, KeyboardEvent } from 'react'
import { Monitor, Moon, Sun } from 'lucide-react'

/* ═══════════════════════════════════════════════════════════════════════
   Appearance
   ───────────────────────────────────────────────────────────────────────
   Three states — light, dark and system — rather than a binary flip, because
   "follow the OS" is a real preference that a two-way toggle silently drops.

   How the choice lands on the page:
     · light / dark → `data-theme` is stamped on <html> and the matching token
       block in styles/app.css wins.
     · system       → the attribute is *removed*, handing the decision to
       `@media (prefers-color-scheme: dark)`. CSS then tracks the OS live.

   The same three lines are duplicated as an inline script in index.html so the
   attribute exists before first paint; keep the storage key and the "dark"
   default in step with it.
   ═══════════════════════════════════════════════════════════════════════ */

export type ThemeChoice = 'light' | 'dark' | 'system'

const THEME_KEY = 'siren-eyes.theme'
const DARK_QUERY = '(prefers-color-scheme: dark)'
const THEME_META: Record<'light' | 'dark', string> = { light: '#eef1f5', dark: '#070809' }

function readStoredTheme(): ThemeChoice {
  try {
    const raw = localStorage.getItem(THEME_KEY)
    if (raw === 'light' || raw === 'dark' || raw === 'system') return raw
  } catch {
    /* blocked storage — fall through to the default */
  }
  return 'dark'
}

function resolveTheme(choice: ThemeChoice): 'light' | 'dark' {
  if (choice !== 'system') return choice
  return window.matchMedia?.(DARK_QUERY).matches ? 'dark' : 'light'
}

function applyTheme(choice: ThemeChoice) {
  const root = document.documentElement
  if (choice === 'system') root.removeAttribute('data-theme')
  else root.setAttribute('data-theme', choice)

  const meta = document.querySelector<HTMLMetaElement>('meta[name="theme-color"]')
  if (meta) meta.content = THEME_META[resolveTheme(choice)]
}

/**
 * Owns the persisted choice and keeps the document in step with it. The
 * matchMedia listener is there for the `system` case: CSS re-resolves itself,
 * but the browser-chrome colour and the control's own label would go stale if
 * the OS flipped theme mid-session.
 */
export function useTheme() {
  const [choice, setChoice] = useState<ThemeChoice>(readStoredTheme)
  const [resolved, setResolved] = useState<'light' | 'dark'>(() => resolveTheme(readStoredTheme()))
  const mounted = useRef(false)

  useEffect(() => {
    // Cross-fade the switch, but never the first paint, and never under
    // reduced motion — the blanket transition rule carries !important.
    const animate = mounted.current
      && !window.matchMedia('(prefers-reduced-motion: reduce)').matches
    mounted.current = true

    let timer: number | undefined
    if (animate) {
      document.documentElement.classList.add('theme-transition')
      timer = window.setTimeout(
        () => document.documentElement.classList.remove('theme-transition'),
        240,
      )
    }

    applyTheme(choice)
    setResolved(resolveTheme(choice))
    try {
      localStorage.setItem(THEME_KEY, choice)
    } catch {
      /* the session still themes correctly, it just will not be remembered */
    }

    return () => {
      if (timer === undefined) return
      window.clearTimeout(timer)
      document.documentElement.classList.remove('theme-transition')
    }
  }, [choice])

  useEffect(() => {
    if (choice !== 'system') return
    const mql = window.matchMedia(DARK_QUERY)
    const onChange = () => {
      applyTheme('system')
      setResolved(mql.matches ? 'dark' : 'light')
    }
    mql.addEventListener('change', onChange)
    return () => mql.removeEventListener('change', onChange)
  }, [choice])

  return { choice, resolved, setChoice }
}

const THEME_OPTIONS: { value: ThemeChoice; label: string; Icon: typeof Sun }[] = [
  { value: 'light', label: 'Light', Icon: Sun },
  { value: 'dark', label: 'Dark', Icon: Moon },
  { value: 'system', label: 'Match system', Icon: Monitor },
]

/**
 * Compact segmented control. One tab stop with roving arrow keys, because
 * three tab stops for a single setting is noise in the tab order.
 */
export function ThemeToggle() {
  const { choice, resolved, setChoice } = useTheme()
  const index = THEME_OPTIONS.findIndex((o) => o.value === choice)

  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    const delta = event.key === 'ArrowRight' || event.key === 'ArrowDown' ? 1
      : event.key === 'ArrowLeft' || event.key === 'ArrowUp' ? -1
        : 0
    if (!delta) return
    event.preventDefault()
    event.stopPropagation()
    const next = THEME_OPTIONS[(index + delta + THEME_OPTIONS.length) % THEME_OPTIONS.length]
    setChoice(next.value)
    const group = event.currentTarget
    requestAnimationFrame(() => {
      group.querySelector<HTMLButtonElement>('[aria-checked="true"]')?.focus()
    })
  }

  return (
    <div
      className="theme-switch"
      role="radiogroup"
      aria-label="Appearance"
      onKeyDown={onKeyDown}
      style={{ '--theme-index': Math.max(index, 0) } as CSSProperties}
    >
      {THEME_OPTIONS.map(({ value, label, Icon }) => {
        const active = value === choice
        return (
          <button
            key={value}
            type="button"
            role="radio"
            aria-checked={active}
            aria-label={value === 'system' ? `Match system appearance (currently ${resolved})` : label}
            title={value === 'system' ? `Match system (${resolved})` : label}
            tabIndex={active || (index === -1 && value === 'dark') ? 0 : -1}
            className="theme-switch-btn"
            onClick={() => setChoice(value)}
          >
            <Icon size={13} strokeWidth={2.1} aria-hidden="true" />
          </button>
        )
      })}
    </div>
  )
}

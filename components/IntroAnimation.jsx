import { useCallback, useEffect, useRef, useState } from 'react'
import './IntroAnimation.css'

/* ---------- tiny math helpers (no external deps) ---------- */
const clamp = (v, a = 0, b = 1) => Math.min(b, Math.max(a, v))
const map = (v, inMin, inMax, outMin = 0, outMax = 1) =>
  clamp((v - inMin) / (inMax - inMin)) * (outMax - outMin) + outMin
/* smooth in/out easing so scrubbing feels cinematic, not linear */
const ease = (t) => t * t * (3 - 2 * t)
const seg = (p, a, b) => ease(map(p, a, b))

/* ---------- skip memory ----------
   When someone skips, we remember it for the current browser session,
   so the intro doesn't replay on every reload / revisit in the same tab.
   Set REMEMBER_SKIP to false if you want the intro to show on every load. */
const REMEMBER_SKIP = true
const SKIP_KEY = 'interntrust-intro-skipped'

const wasSkippedBefore = () => {
  if (!REMEMBER_SKIP) return false
  try {
    return sessionStorage.getItem(SKIP_KEY) === '1'
  } catch {
    return false
  }
}

const rememberSkip = () => {
  if (!REMEMBER_SKIP) return
  try {
    sessionStorage.setItem(SKIP_KEY, '1')
  } catch {
    /* storage unavailable (private mode etc.) — ignore */
  }
}

const MESSAGES = [
  { id: 'm1', text: '₹50,000 / MONTH', tone: 'money', side: 'left' },
  { id: 'm2', text: 'NO EXPERIENCE REQUIRED', tone: 'neutral', side: 'right' },
  { id: 'm3', text: 'URGENT HIRING', tone: 'amber', side: 'left' },
  { id: 'm4', text: 'PAY ₹999 TO APPLY', tone: 'danger', side: 'right' },
]

function IntroAnimation({ onComplete }) {
  const scrollerRef = useRef(null)
  const rafRef = useRef(0)
  const doneRef = useRef(false)
  const [p, setP] = useState(0)
  const [skipped] = useState(wasSkippedBefore)

  const finish = useCallback(() => {
    if (doneRef.current) return
    doneRef.current = true
    onComplete?.()
  }, [onComplete])

  /* ---------- skip: jump straight to the home page ---------- */
  const skip = useCallback(() => {
    rememberSkip()
    finish()
  }, [finish])

  /* if the visitor already skipped earlier this session, leave immediately */
  useEffect(() => {
    if (skipped) finish()
  }, [skipped, finish])

  /* ---------- scroll driver (scoped to this component only) ---------- */
  useEffect(() => {
    if (skipped) return
    const el = scrollerRef.current
    if (!el) return

    // lock the page behind the intro; restore on unmount (StrictMode safe)
    const prevOverflow = document.body.style.overflow
    const prevOverflowX = document.documentElement.style.overflowX
    document.body.style.overflow = 'hidden'
    document.documentElement.style.overflowX = 'hidden'

    const read = () => {
      rafRef.current = 0
      const max = el.scrollHeight - el.clientHeight
      const next = max > 0 ? clamp(el.scrollTop / max) : 0
      setP(next)
      if (next > 0.995) finish()
    }

    const onScroll = () => {
      if (rafRef.current) return
      rafRef.current = requestAnimationFrame(read)
    }

    // keyboard shortcut: Esc also skips
    const onKey = (e) => {
      if (e.key === 'Escape') skip()
    }

    el.addEventListener('scroll', onScroll, { passive: true })
    window.addEventListener('resize', onScroll)
    window.addEventListener('keydown', onKey)
    read()

    return () => {
      el.removeEventListener('scroll', onScroll)
      window.removeEventListener('resize', onScroll)
      window.removeEventListener('keydown', onKey)
      if (rafRef.current) cancelAnimationFrame(rafRef.current)
      rafRef.current = 0
      document.body.style.overflow = prevOverflow
      document.documentElement.style.overflowX = prevOverflowX
    }
  }, [finish, skip, skipped])

  // already skipped earlier this session: render nothing
  if (skipped) return null

  /* ---------- derived scroll progress per phase ---------- */
  const headline = seg(p, 0.0, 0.22)        // sharp -> blurred
  const card = seg(p, 0.10, 0.34)           // blue card rises
  const enter = seg(p, 0.30, 0.44)          // "inside the card"
  const phone = seg(p, 0.36, 0.60)          // phone rises
  const detect = seg(p, 0.70, 0.86)         // scam reveal
  const verify = seg(p, 0.80, 0.92)         // "verify before you trust"
  const exit = seg(p, 0.90, 1.0)            // scene leaves upward

  const glitch = Math.sin(clamp(map(p, 0.70, 0.88)) * Math.PI) // peak at reveal

  const sceneStyle = {
    '--p': p,
    '--headline': headline,
    '--card': card,
    '--enter': enter,
    '--phone': phone,
    '--detect': detect,
    '--verify': verify,
    '--exit': exit,
    '--glitch': glitch,
  }

  return (
    <div className="it-intro" ref={scrollerRef} role="presentation">
      {/* pinned cinematic stage */}
      <div className="it-stage" style={sceneStyle}>
        <div className="it-vignette" />

        {/* SKIP — always visible at the top, goes straight to the home page */}
        <button
          type="button"
          className="it-skip"
          onClick={skip}
          aria-label="Skip intro animation and go to the home page"
        >
          <span>Skip intro</span>
          <svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true" focusable="false">
            <path
              d="M3 3l6 5-6 5V3zm7 0l3 5-3 5"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.6"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </button>

        {/* PHASE 1 — black screen + big headline */}
        <h1 className="it-headline">
          <span>NOT EVERY INTERNSHIP</span>
          <span>IS WHAT IT SEEMS.</span>
        </h1>

        {/* PHASE 2/3 — blue card becomes the scene */}
        <div className="it-card">
          <div className="it-card__glow" />
          <div className="it-card__label">INTERNTRUST</div>

          {/* PHASE 4 — phone rises from bottom of the card */}
          <div className="it-phone">
            <div className="it-phone__frame">
              <div className="it-phone__notch" />
              <div className="it-phone__screen">
                <div className="it-phone__bar">
                  <span>InternTrust Inbox</span>
                  <span className="it-phone__dot" />
                </div>

                {/* PHASE 5 — messages appear progressively */}
                <div className="it-msgs">
                  {MESSAGES.map((m, i) => {
                    const start = 0.46 + i * 0.055
                    const a = seg(p, start, start + 0.08)
                    return (
                      <div
                        key={m.id}
                        className={`it-msg it-msg--${m.tone} it-msg--${m.side}`}
                        style={{ '--a': a }}
                      >
                        <span className="it-msg__text">{m.text}</span>
                        <span className="it-msg__tick">✓</span>
                      </div>
                    )
                  })}
                </div>

                {/* emphasized suspicious offer */}
                <div className="it-offer" style={{ '--a': seg(p, 0.62, 0.72) }}>
                  <b>₹50,000 / MONTH</b>
                  <b>URGENT HIRING</b>
                  <b className="it-offer__danger">PAY ₹999 TO APPLY</b>
                </div>

                {/* scanning sweep just before the reveal */}
                <div className="it-scan" style={{ '--a': seg(p, 0.60, 0.74) }}>
                  <i />
                </div>
              </div>
            </div>
          </div>

          {/* PHASE 6 — reveal */}
          <div className="it-verdict" data-text="FAKE INTERNSHIP DETECTED">
            FAKE INTERNSHIP DETECTED
          </div>
          <div className="it-verify">VERIFY BEFORE YOU TRUST.</div>
        </div>

        {/* scroll hint */}
        <div className="it-hint">
          <span>scroll</span>
          <i />
        </div>
      </div>

      {/* scroll runway — gives the story room to breathe */}
      <div className="it-runway" aria-hidden="true" />
    </div>
  )
}

export default IntroAnimation
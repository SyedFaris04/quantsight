/**
 * frontend/src/lib/gameFx.js
 * ─────────────────────────────────────────────────────────────────
 * Presentation-only "juice" for the Game page — sound, confetti, and
 * count-up numbers. None of this touches scoring or game state; it's
 * purely reacting to results the backend already computed.
 * ─────────────────────────────────────────────────────────────────
 */

import { useEffect, useRef, useState } from "react";

export function prefersReducedMotion() {
  return typeof window !== "undefined" &&
    window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
}

// ── Sound — synthesized tones, no audio files. Muted state persists. ──
const SOUND_KEY = "quantsight_game_sound";

export function isSoundOn() {
  const raw = localStorage.getItem(SOUND_KEY);
  return raw === null ? true : raw === "1";
}

export function setSoundOn(on) {
  localStorage.setItem(SOUND_KEY, on ? "1" : "0");
}

let audioCtx = null;
function getAudioCtx() {
  if (!audioCtx) {
    const Ctx = window.AudioContext || window.webkitAudioContext;
    if (!Ctx) return null;
    audioCtx = new Ctx();
  }
  if (audioCtx.state === "suspended") audioCtx.resume();
  return audioCtx;
}

function tone(ctx, { freq, start, dur, type = "sine", gain = 0.15 }) {
  const osc = ctx.createOscillator();
  const g = ctx.createGain();
  osc.type = type;
  osc.frequency.value = freq;
  g.gain.setValueAtTime(0, ctx.currentTime + start);
  g.gain.linearRampToValueAtTime(gain, ctx.currentTime + start + 0.015);
  g.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + start + dur);
  osc.connect(g);
  g.connect(ctx.destination);
  osc.start(ctx.currentTime + start);
  osc.stop(ctx.currentTime + start + dur + 0.05);
}

const SEQUENCES = {
  correct:   [{ freq: 587.33, start: 0,    dur: 0.14 }, { freq: 880,    start: 0.09, dur: 0.22 }],
  wrong:     [{ freq: 196,    start: 0,    dur: 0.28, type: "sawtooth", gain: 0.1 }],
  milestone: [{ freq: 523.25, start: 0,    dur: 0.12 }, { freq: 659.25, start: 0.09, dur: 0.12 }, { freq: 987.77, start: 0.18, dur: 0.3 }],
  levelup:   [{ freq: 523.25, start: 0,    dur: 0.12 }, { freq: 659.25, start: 0.1,  dur: 0.12 }, { freq: 783.99, start: 0.2,  dur: 0.12 }, { freq: 1046.5, start: 0.3, dur: 0.4 }],
};

export function playTone(name) {
  if (!isSoundOn()) return;
  const ctx = getAudioCtx();
  if (!ctx) return;
  (SEQUENCES[name] || []).forEach(spec => tone(ctx, spec));
}

// ── Count-up — animates a displayed number toward its real value. ──
export function useCountUp(value, duration = 600) {
  const [display, setDisplay] = useState(value);
  const prevRef = useRef(value);

  useEffect(() => {
    const from = prevRef.current;
    const to = value;
    if (from === to) return;
    if (prefersReducedMotion()) {
      setDisplay(to);
      prevRef.current = to;
      return;
    }
    const start = performance.now();
    const d = duration;
    let raf;
    const tick = (now) => {
      const t = Math.min(1, (now - start) / d);
      const eased = 1 - Math.pow(1 - t, 3);
      setDisplay(Math.round(from + (to - from) * eased));
      if (t < 1) {
        raf = requestAnimationFrame(tick);
      } else {
        prevRef.current = to;
      }
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [value, duration]);

  return display;
}

// ── Confetti — lightweight canvas burst, no dependency. ──
const CONFETTI_COLORS = ["#6366f1", "#818cf8", "#16a34a", "#22c55e", "#d97706", "#fbbf24"];

export function burstConfetti({ x, y, count = 60, colors = CONFETTI_COLORS } = {}) {
  if (prefersReducedMotion()) return;

  const canvas = document.createElement("canvas");
  canvas.style.cssText = "position:fixed;inset:0;width:100vw;height:100vh;pointer-events:none;z-index:9999;";
  const dpr = window.devicePixelRatio || 1;
  canvas.width = window.innerWidth * dpr;
  canvas.height = window.innerHeight * dpr;
  document.body.appendChild(canvas);
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);

  const originX = x ?? window.innerWidth / 2;
  const originY = y ?? window.innerHeight / 3;

  const particles = Array.from({ length: count }, () => {
    const angle = Math.random() * Math.PI * 2;
    const speed = 4 + Math.random() * 7;
    return {
      x: originX,
      y: originY,
      vx: Math.cos(angle) * speed,
      vy: Math.sin(angle) * speed - 3,
      size: 5 + Math.random() * 5,
      color: colors[Math.floor(Math.random() * colors.length)],
      rotation: Math.random() * Math.PI,
      spin: (Math.random() - 0.5) * 0.4,
      life: 1,
    };
  });

  const startTime = performance.now();
  const maxDuration = 1800;

  function frame(now) {
    const elapsed = now - startTime;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    let alive = false;
    for (const p of particles) {
      p.vy += 0.18;
      p.vx *= 0.985;
      p.x += p.vx;
      p.y += p.vy;
      p.rotation += p.spin;
      p.life = Math.max(0, 1 - elapsed / maxDuration);
      if (p.life <= 0) continue;
      alive = true;

      ctx.save();
      ctx.globalAlpha = p.life;
      ctx.translate(p.x, p.y);
      ctx.rotate(p.rotation);
      ctx.fillStyle = p.color;
      ctx.fillRect(-p.size / 2, -p.size / 4, p.size, p.size / 2);
      ctx.restore();
    }

    if (alive && elapsed < maxDuration) {
      requestAnimationFrame(frame);
    } else {
      canvas.remove();
    }
  }
  requestAnimationFrame(frame);
}

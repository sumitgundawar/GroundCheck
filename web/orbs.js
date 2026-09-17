// Loading orbs, drawn with the thinking-orbs engine (MIT, Jakub Antalik),
// vendored unchanged in /vendor/thinking-orbs, as on groundcheckhealth.com.
//
// Markup: <canvas data-orb="searching" data-orb-size="20"></canvas>
//   data-orb        one of the engine's states (working, searching, solving, ...)
//   data-orb-size   20 or 64
//   data-orb-theme  "inverse" for an orb on an ink-coloured button; otherwise
//                   it follows the page's theme
//
// Reduced motion shows one still frame. An orb pauses when hidden or off
// screen. window.gcOrbs.mount(canvas) sets up orbs added later.

import { MODE_DRAWS, resolvePreset } from "/vendor/thinking-orbs/engine.es.js";

const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
const STILL = 0.6;
const orbs = new Map();

const observer = new IntersectionObserver((entries) => {
  for (const entry of entries) {
    const orb = orbs.get(entry.target);
    if (orb) { orb.visible = entry.isIntersecting; orb.sync(); }
  }
});

function mount(canvas) {
  if (orbs.has(canvas)) return;
  const size = canvas.dataset.orbSize === "64" ? 64 : 20;
  const ratio = Math.min(2, window.devicePixelRatio || 1);
  canvas.width = Math.round(size * ratio);
  canvas.height = Math.round(size * ratio);
  canvas.style.width = `${size}px`;
  canvas.style.height = `${size}px`;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const { mode, speed, opts } = resolvePreset(canvas.dataset.orb || "working", size);
  const draw = MODE_DRAWS[mode];
  const dark = () => {
    const pageDark = window.gcTheme ? window.gcTheme.isDark() : false;
    return canvas.dataset.orbTheme === "inverse" ? !pageDark : pageDark;
  };
  const orb = {
    visible: false,
    frame: 0,
    paint(t) {
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      ctx.clearRect(0, 0, size, size);
      draw(ctx, size, t, dark(), opts);
    },
    running() { return this.visible && !document.hidden && !reducedMotion.matches; },
    tick() {
      this.paint((performance.now() / 1000) * speed);
      this.frame = this.running() ? requestAnimationFrame(() => this.tick()) : 0;
    },
    sync() {
      if (this.running()) { if (!this.frame) this.tick(); }
      else { cancelAnimationFrame(this.frame); this.frame = 0; this.paint(STILL); }
    },
  };
  orb.paint(STILL);
  orbs.set(canvas, orb);
  observer.observe(canvas);
}

const syncAll = () => orbs.forEach((orb) => orb.sync());
document.addEventListener("visibilitychange", syncAll);
reducedMotion.addEventListener("change", syncAll);
document.addEventListener("themechange", () => orbs.forEach((orb) => orb.paint(STILL)));

document.querySelectorAll("canvas[data-orb]").forEach(mount);
window.gcOrbs = { mount };

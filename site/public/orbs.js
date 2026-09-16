// Loading orbs, drawn with the thinking-orbs engine (MIT, Jakub Antalik),
// vendored unchanged in /vendor/thinking-orbs. This is a small framework-free
// equivalent of the package's React component.
//
// Markup: <canvas data-orb="working" data-orb-size="20" aria-label="…"></canvas>
//   data-orb       one of the nine states (working, searching, solving, …)
//   data-orb-size  20 or 64, the two tuned presets
//   data-orb-theme "dark" for light dots on a dark background, "light" for
//                  dark dots; omitted, it follows the colour scheme
//
// Like the original: reduced motion renders one static frame, and each orb
// pauses when it is off screen, hidden, or the tab is in the background.

import { MODE_DRAWS, resolvePreset } from "/vendor/thinking-orbs/engine.es.js";

const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
const darkScheme = window.matchMedia("(prefers-color-scheme: dark)");
const STATIC_FRAME_T = 0.6;

const orbs = new Set();

const setUp = (canvas) => {
  const state = canvas.dataset.orb || "working";
  const size = canvas.dataset.orbSize === "64" ? 64 : 20;
  const ratio = Math.min(2, window.devicePixelRatio || 1);

  canvas.width = Math.round(size * ratio);
  canvas.height = Math.round(size * ratio);
  canvas.setAttribute("role", "img");

  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  const { mode, speed, opts } = resolvePreset(state, size);
  const theme = canvas.dataset.orbTheme;
  const isDark = () => (theme === "dark" ? true : theme === "light" ? false : darkScheme.matches);
  const draw = MODE_DRAWS[mode];

  const orb = {
    canvas,
    visible: false,
    frame: 0,
    paint(t) {
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      ctx.clearRect(0, 0, size, size);
      draw(ctx, size, t, isDark(), opts);
    },
    running() {
      return this.visible && !document.hidden && !reducedMotion.matches;
    },
    tick() {
      this.paint((performance.now() / 1000) * speed);
      this.frame = this.running() ? requestAnimationFrame(() => this.tick()) : 0;
    },
    sync() {
      if (this.running()) {
        if (!this.frame) this.tick();
      } else {
        cancelAnimationFrame(this.frame);
        this.frame = 0;
        if (reducedMotion.matches) this.paint(STATIC_FRAME_T);
      }
    },
  };

  orb.paint(reducedMotion.matches ? STATIC_FRAME_T : 0);
  orbs.add(orb);
  observer.observe(canvas);
};

// A hidden element (display: none) never intersects, so toggling an orb's
// container with the hidden attribute also starts and stops its animation.
const observer = new IntersectionObserver((entries) => {
  for (const entry of entries) {
    for (const orb of orbs) {
      if (orb.canvas === entry.target) {
        orb.visible = entry.isIntersecting;
        orb.sync();
      }
    }
  }
});

const syncAll = () => orbs.forEach((orb) => orb.sync());
document.addEventListener("visibilitychange", syncAll);
reducedMotion.addEventListener("change", syncAll);
darkScheme.addEventListener("change", () =>
  orbs.forEach((orb) => (orb.running() ? null : orb.paint(STATIC_FRAME_T))),
);

document.querySelectorAll("canvas[data-orb]").forEach(setUp);

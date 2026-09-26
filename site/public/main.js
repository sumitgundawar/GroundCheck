// GroundCheckHealth site. Progressive enhancement only: every section is complete
// without this script.

(() => {
  "use strict";

  const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  // Two real runs of the pipeline, recorded from its trace output. The first
  // matches the static HTML, so the page is complete without this script.
  const RUNS = [
    {
      decision: "refuse",
      id: "82f38f41",
      totalMs: 6,
      question: "What is the recommended dose of Zalortin for a patient with Veltris syndrome?",
      verdict: "Refused",
      reason: "Routed for review. No model was called.",
      stages: [
        ["pass", "PII redaction", "no personal data found", 0],
        ["pass", "Scope check", "no injection patterns", 0],
        ["pass", "Rate limit", "within limit", 0],
        ["pass", "Retrieve", "top 4 passages, best score 0.71", 5],
        ["pass", "Retrieval gate", "0.71 clears the 0.30 threshold", 0],
        ["fail", "Source coverage", "“zalortin” is in no trusted source", 1],
        ["skip", "Generate", "not reached", null],
        ["skip", "Schema validation", "not reached", null],
        ["skip", "Grounding check", "not reached", null],
        ["skip", "Dosage guard", "not reached", null],
      ],
    },
    {
      decision: "answer",
      id: "72adc72d",
      totalMs: 57,
      question: "What is the dose of Caloradine, and can it be combined with Mendel solution?",
      verdict: "Answered",
      reason: "4 claims, each cited to its source.",
      stages: [
        ["pass", "PII redaction", "no personal data found", 0],
        ["pass", "Scope check", "no injection patterns", 0],
        ["pass", "Rate limit", "within limit", 0],
        ["pass", "Retrieve", "top 4 passages, best score 0.72", 5],
        ["pass", "Retrieval gate", "0.72 clears the 0.30 threshold", 0],
        ["pass", "Source coverage", "every key term is in a source", 1],
        ["info", "Generate", "4 claims drafted from sources", 0],
        ["pass", "Schema validation", "4 claims match the schema", 0],
        ["pass", "Grounding check", "all 4 claims grounded", 51],
        ["pass", "Dosage guard", "15 mg verified against sources", 0],
      ],
    },
  ];

  const STATUS_LABEL = { pass: "Passed", info: "Passed", fail: "Refused", skip: "Not reached" };

  const heroRecord = () => {
    const record = document.querySelector("[data-record]");
    if (!record) return;

    const el = {
      id: record.querySelector("[data-record-id]"),
      total: record.querySelector("[data-record-total]"),
      question: record.querySelector("[data-record-question]"),
      trace: record.querySelector("[data-record-trace]"),
      verdict: record.querySelector("[data-record-verdict]"),
      reason: record.querySelector("[data-record-reason]"),
      verdictRow: record.querySelector("[data-verdict]"),
      controls: record.querySelector("[data-record-controls]"),
      status: record.querySelector("[data-record-status]"),
      pause: record.querySelector("[data-record-pause]"),
      runButtons: Array.from(record.querySelectorAll("[data-run]")),
    };
    if (Object.values(el).some((v) => !v)) return;

    const reducedMotion = prefersReducedMotion;
    const STEP_MS = 110;
    const HOLD_AFTER_FAIL_MS = 450;
    const HOLD_RESULT_MS = 4200;
    const SWITCH_MS = 240;

    let current = 0;
    let timers = [];
    let userPaused = false;
    let onScreen = true;

    const clearTimers = () => {
      timers.forEach((t) => window.clearTimeout(t));
      timers = [];
      el.status.hidden = true;
      el.total.hidden = false;
    };
    const later = (fn, ms) => timers.push(window.setTimeout(fn, ms));

    const render = (run) => {
      record.dataset.decision = run.decision;
      el.id.textContent = run.id;
      el.total.textContent = `${run.totalMs} ms`;
      el.question.textContent = `“${run.question}”`;
      el.verdict.textContent = run.verdict;
      el.reason.textContent = run.reason;

      const items = run.stages.map(([status, name, detail, ms]) => {
        const li = document.createElement("li");
        li.className = "stage";
        li.dataset.status = status;

        const flag = document.createElement("span");
        flag.className = "flag";
        flag.setAttribute("aria-hidden", "true");
        const label = document.createElement("span");
        label.className = "visually-hidden";
        label.textContent = `${STATUS_LABEL[status]}: `;
        const nameEl = document.createElement("span");
        nameEl.className = "stage-name";
        nameEl.textContent = name;
        const detailEl = document.createElement("span");
        detailEl.className = "stage-detail";
        detailEl.textContent = detail;
        const msEl = document.createElement("span");
        msEl.className = "stage-ms mono";
        msEl.textContent = ms === null ? "–" : ms < 1 ? "<1" : String(ms);

        li.append(flag, label, nameEl, detailEl, msEl);
        return li;
      });
      el.trace.replaceChildren(...items);

      el.runButtons.forEach((button, i) => button.setAttribute("aria-pressed", String(i === current)));
    };

    // Reveal the trace stage by stage, then the verdict. Calls done() after.
    const replay = (done) => {
      const stages = Array.from(el.trace.children);
      const steps = [...stages, el.verdictRow];
      steps.forEach((s) => s.classList.remove("is-shown"));
      record.classList.add("is-replaying");
      el.status.hidden = false;
      el.total.hidden = true;

      let delay = 150;
      steps.forEach((step) => {
        later(() => step.classList.add("is-shown"), delay);
        delay += step.dataset.status === "fail" ? HOLD_AFTER_FAIL_MS : STEP_MS;
      });
      later(() => {
        record.classList.remove("is-replaying");
        steps.forEach((s) => s.classList.remove("is-shown"));
        el.status.hidden = true;
        el.total.hidden = false;
        done?.();
      }, delay + 200);
    };

    const running = () => !userPaused && onScreen && !document.hidden && !reducedMotion;

    // One cycle: show the current run, hold, cross-fade to the next.
    const cycle = () => {
      clearTimers();
      // A cycle interrupted mid cross-fade (tab hidden, scrolled away) would
      // otherwise leave the record faded out.
      record.classList.remove("is-switching");
      replay(() => {
        later(() => {
          if (!running()) return;
          record.classList.add("is-switching");
          later(() => {
            current = (current + 1) % RUNS.length;
            render(RUNS[current]);
            record.classList.remove("is-switching");
            cycle();
          }, SWITCH_MS);
        }, HOLD_RESULT_MS);
      });
    };

    const resume = () => {
      if (running()) cycle();
    };

    const setUserPaused = (paused) => {
      userPaused = paused;
      el.pause.setAttribute("aria-pressed", String(paused));
      el.pause.textContent = paused ? "Play" : "Pause";
      if (paused) {
        clearTimers();
        record.classList.remove("is-replaying", "is-switching");
      } else {
        resume();
      }
    };

    // Choosing a run shows it and stops the loop, so it stays put to read.
    el.runButtons.forEach((button) => {
      button.addEventListener("click", () => {
        const i = Number(button.dataset.run);
        clearTimers();
        record.classList.remove("is-replaying", "is-switching");
        current = i;
        render(RUNS[current]);
        if (!reducedMotion) {
          setUserPaused(true);
          replay();
        }
      });
    });

    el.controls.hidden = false;
    if (reducedMotion) {
      el.pause.hidden = true;
      return;
    }

    el.pause.addEventListener("click", () => setUserPaused(!userPaused));

    document.addEventListener("visibilitychange", () => {
      if (document.hidden) clearTimers();
      else resume();
    });

    if ("IntersectionObserver" in window) {
      new IntersectionObserver(([entry]) => {
        const wasOnScreen = onScreen;
        onScreen = entry.isIntersecting;
        if (!onScreen) clearTimers();
        else if (!wasOnScreen) resume();
      }).observe(record);
    }

    render(RUNS[current]);
    cycle();
  };

  // Copy buttons on code blocks. They ship hidden and are revealed only where
  // the Clipboard API exists, so no button is ever shown that cannot work.
  const wireCopyButtons = () => {
    if (!navigator.clipboard) return;

    document.querySelectorAll("[data-copy]").forEach((button) => {
      const code = button.closest(".code-block")?.querySelector("code");
      if (!code) return;

      button.hidden = false;
      const label = button.textContent;
      let resetTimer;

      button.addEventListener("click", async () => {
        try {
          await navigator.clipboard.writeText(code.textContent ?? "");
          button.textContent = "Copied";
        } catch {
          button.textContent = "Copy failed";
        }
        window.clearTimeout(resetTimer);
        resetTimer = window.setTimeout(() => { button.textContent = label; }, 1800);
      });
    });
  };

  // Dashboard tour: an ARIA tabs pattern with automatic activation.
  // https://www.w3.org/WAI/ARIA/apg/patterns/tabs/
  const wireTour = () => {
    const tour = document.querySelector("[data-tour]");
    const tablist = tour?.querySelector('[role="tablist"]');
    if (!tour || !tablist) return;

    const tabs = Array.from(tablist.querySelectorAll('[role="tab"]'));
    const panelFor = (tab) => document.getElementById(tab.getAttribute("aria-controls"));

    const select = (next, { focus = false } = {}) => {
      tabs.forEach((tab) => {
        const selected = tab === next;
        tab.setAttribute("aria-selected", String(selected));
        tab.tabIndex = selected ? 0 : -1;
        const panel = panelFor(tab);
        if (panel) panel.hidden = !selected;
      });
      if (focus) {
        next.focus();
        next.scrollIntoView({ block: "nearest", inline: "nearest" });
      }
    };

    tabs.forEach((tab) => {
      tab.addEventListener("click", () => select(tab));
      tab.addEventListener("keydown", (event) => {
        const i = tabs.indexOf(tab);
        const last = tabs.length - 1;
        const target = {
          ArrowDown: tabs[i === last ? 0 : i + 1],
          ArrowRight: tabs[i === last ? 0 : i + 1],
          ArrowUp: tabs[i === 0 ? last : i - 1],
          ArrowLeft: tabs[i === 0 ? last : i - 1],
          Home: tabs[0],
          End: tabs[last],
        }[event.key];
        if (!target) return;
        event.preventDefault();
        select(target, { focus: true });
      });
    });

    // The panels are visible panels in the static page; only switch to tabs
    // once the script can drive them.
    tablist.hidden = false;
    tour.classList.add("is-tabbed");
    select(tabs.find((t) => t.getAttribute("aria-selected") === "true") ?? tabs[0]);

    // Tablets and phones lay the tabs out horizontally.
    const orientation = window.matchMedia("(max-width: 52rem)");
    const syncOrientation = () =>
      tablist.setAttribute("aria-orientation", orientation.matches ? "horizontal" : "vertical");
    syncOrientation();
    orientation.addEventListener("change", syncOrientation);
  };


  // Runs fn once, the first time el scrolls into view.
  const onFirstView = (el, fn, threshold = 0.25) => {
    if (!("IntersectionObserver" in window)) return fn();
    const io = new IntersectionObserver((entries) => {
      if (entries.some((e) => e.isIntersecting)) {
        io.disconnect();
        fn();
      }
    }, { threshold });
    io.observe(el);
  };

  // Demo video: plays muted when in view (unless reduced motion), pauses
  // when out of view, with a play/pause button, chapters and a loading orb.
  const wireDemo = () => {
    const figure = document.querySelector("[data-demo]");
    const video = figure?.querySelector("[data-demo-video]");
    const toggle = figure?.querySelector("[data-demo-toggle]");
    const loading = figure?.querySelector("[data-demo-loading]");
    if (!figure || !video || !toggle || !loading) return;

    const chapters = Array.from(figure.querySelectorAll("[data-chapter]"));
    const starts = chapters.map((c) => Number(c.dataset.chapter));
    let userPaused = false;
    let inView = false;

    const setLoading = (on) => { loading.hidden = !on; };
    const play = () => {
      if (video.preload !== "auto") video.preload = "auto";
      const attempt = video.play();
      // Autoplay can be refused (browser policy, data saver). Leave the play
      // button showing so one click starts it.
      if (attempt) attempt.catch(() => { setLoading(false); figure.classList.remove("is-playing"); });
    };

    video.addEventListener("waiting", () => setLoading(true));
    video.addEventListener("playing", () => {
      setLoading(false);
      figure.classList.add("is-playing", "has-started");
      toggle.setAttribute("aria-label", "Pause demo video");
    });
    video.addEventListener("pause", () => {
      figure.classList.remove("is-playing");
      toggle.setAttribute("aria-label", "Play demo video");
    });
    video.addEventListener("canplay", () => setLoading(false));

    const togglePlayback = () => {
      if (video.paused) {
        userPaused = false;
        setLoading(video.readyState < 3);
        play();
      } else {
        userPaused = true;
        video.pause();
      }
    };
    toggle.addEventListener("click", togglePlayback);
    video.addEventListener("click", togglePlayback);
    video.addEventListener("error", () => setLoading(false), true);

    // Static hosting may not answer byte-range requests, and without them
    // Chrome can only seek within what it has downloaded. So wait until the
    // target time is seekable before jumping to it.
    const isSeekable = (t) => {
      for (let i = 0; i < video.seekable.length; i++) {
        if (t >= video.seekable.start(i) && t <= video.seekable.end(i)) return true;
      }
      return false;
    };
    let pendingSeek = null;
    const trySeek = () => {
      if (pendingSeek === null || !isSeekable(pendingSeek)) return;
      video.currentTime = pendingSeek;
      pendingSeek = null;
      setLoading(false);
      play();
    };
    ["loadedmetadata", "progress", "canplaythrough"].forEach((type) => video.addEventListener(type, trySeek));

    chapters.forEach((chapter, i) => {
      chapter.addEventListener("click", () => {
        userPaused = false;
        pendingSeek = starts[i];
        if (video.preload !== "auto") video.preload = "auto";
        if (video.readyState === 0) video.load();
        if (!isSeekable(pendingSeek)) setLoading(true);
        trySeek();
      });
    });

    video.addEventListener("timeupdate", () => {
      const t = video.currentTime;
      const end = video.duration || starts[starts.length - 1] + 12;
      chapters.forEach((chapter, i) => {
        const from = starts[i];
        const to = i + 1 < starts.length ? starts[i + 1] : end;
        const current = t >= from && t < to;
        chapter.classList.toggle("is-current", current);
        const progress = current ? (t - from) / (to - from) : t >= to ? 1 : 0;
        chapter.style.setProperty("--progress", progress.toFixed(3));
      });
    });

    if (prefersReducedMotion || !("IntersectionObserver" in window)) return;

    new IntersectionObserver(([entry]) => {
      inView = entry.isIntersecting;
      if (inView && !userPaused) {
        setLoading(video.readyState < 3);
        play();
      } else if (!inView && !video.paused) {
        video.pause();
      }
    }, { threshold: 0.4 }).observe(figure);

    document.addEventListener("visibilitychange", () => {
      if (document.hidden) video.pause();
      else if (inView && !userPaused) play();
    });
  };

  // The pipeline lights up stage by stage the first time it is seen.
  const wireFlowchart = () => {
    const chart = document.querySelector("[data-flowchart]");
    if (!chart || prefersReducedMotion) return;
    const steps = Array.from(chart.querySelectorAll(".flow-step"));
    chart.classList.add("is-animated");
    onFirstView(chart, () => {
      steps.forEach((step, i) => window.setTimeout(() => step.classList.add("is-lit"), 150 + i * 140));
    });
  };

  // Bars grow and headline numbers count up the first time they are seen.
  const wireCharts = () => {
    if (prefersReducedMotion) return;
    const groups = document.querySelectorAll(".meters, .bars");
    const format = new Intl.NumberFormat("en-US");

    groups.forEach((group) => {
      const fills = Array.from(group.querySelectorAll("[data-fill]"));
      const counters = Array.from(group.querySelectorAll("[data-count-to]"));
      fills.forEach((f) => f.style.setProperty("--fill", "0"));
      counters.forEach((c) => { c.textContent = "0"; });

      onFirstView(group, () => {
        requestAnimationFrame(() => {
          fills.forEach((f) => f.style.setProperty("--fill", f.dataset.fill));
        });
        const DURATION = 1100;
        const start = performance.now();
        const tick = (now) => {
          const p = Math.min(1, (now - start) / DURATION);
          const eased = 1 - Math.pow(1 - p, 3);
          counters.forEach((c) => {
            c.textContent = format.format(Math.round(Number(c.dataset.countTo) * eased));
          });
          if (p < 1) requestAnimationFrame(tick);
        };
        requestAnimationFrame(tick);
      });
    });
  };

  // A soft light follows the pointer across cards marked .spotlight.
  const wireSpotlights = () => {
    if (!window.matchMedia("(hover: hover)").matches) return;
    document.querySelectorAll(".spotlight").forEach((card) => {
      card.addEventListener("pointermove", (event) => {
        const rect = card.getBoundingClientRect();
        card.style.setProperty("--spot-x", `${event.clientX - rect.left}px`);
        card.style.setProperty("--spot-y", `${event.clientY - rect.top}px`);
      });
    });
  };

  // Dot fields. The hero's is a still lattice whose dots grow near the
  // pointer. The call-to-action band's (data-dotgrid="wave") also drifts in a
  // slow wave. Both are still under reduced motion, and the wave pauses when
  // off screen or in a background tab.
  const wireDotGrids = () => document.querySelectorAll("[data-dotgrid]").forEach(wireDotGrid);

  const wireDotGrid = (canvas) => {
    const host = canvas.parentElement;
    const ctx = canvas.getContext("2d");
    if (!host || !ctx) return;

    const wave = canvas.dataset.dotgrid === "wave";
    const SPACING = wave ? 22 : 26;
    const RADIUS = 1.1;
    const REACH = 150;
    const dark = window.matchMedia("(prefers-color-scheme: dark)");
    const pointerOk = !prefersReducedMotion && window.matchMedia("(pointer: fine)").matches;
    const animate = wave && !prefersReducedMotion;

    let width = 0;
    let height = 0;
    let pointer = null;
    let frame = 0;
    let visible = false;

    const resize = () => {
      const ratio = Math.min(2, window.devicePixelRatio || 1);
      width = host.clientWidth;
      height = host.clientHeight;
      canvas.width = Math.round(width * ratio);
      canvas.height = Math.round(height * ratio);
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      draw(performance.now());
    };

    const draw = (now) => {
      frame = 0;
      const t = now / 1000;
      ctx.clearRect(0, 0, width, height);
      const light = wave || dark.matches;
      const ink = light ? "232, 240, 237" : "15, 38, 34";

      for (let y = SPACING / 2; y < height; y += SPACING) {
        for (let x = SPACING / 2; x < width; x += SPACING) {
          let dy = 0;
          let lift = 0;
          if (wave) {
            // Two slow, crossing sine waves give a gentle, non-repeating swell.
            const a = Math.sin(x * 0.012 + t * 0.6) + 0.6 * Math.sin(x * 0.021 - y * 0.01 - t * 0.45);
            dy = a * 7;
            lift = (a + 1.6) / 3.2;
          }
          let boost = 0;
          if (pointer) {
            const d = Math.hypot(x - pointer.x, y + dy - pointer.y);
            if (d < REACH) boost = 1 - d / REACH;
          }
          const alpha = wave ? 0.14 + lift * 0.36 + boost * 0.35 : 0.16 + boost * 0.4;
          ctx.beginPath();
          ctx.arc(x, y + dy, RADIUS + (wave ? lift * 0.7 : 0) + boost * 1.6, 0, Math.PI * 2);
          ctx.fillStyle = `rgba(${ink}, ${alpha.toFixed(3)})`;
          ctx.fill();
        }
      }

      if (animate && visible && !document.hidden) frame = requestAnimationFrame(draw);
    };

    const schedule = () => { if (!frame) frame = requestAnimationFrame(draw); };

    if ("ResizeObserver" in window) new ResizeObserver(resize).observe(host);
    else window.addEventListener("resize", resize);
    dark.addEventListener("change", schedule);
    resize();

    if (animate && "IntersectionObserver" in window) {
      new IntersectionObserver(([entry]) => {
        visible = entry.isIntersecting;
        if (visible) schedule();
      }).observe(host);
      document.addEventListener("visibilitychange", () => { if (!document.hidden && visible) schedule(); });
    }

    if (!pointerOk) return;
    host.addEventListener("pointermove", (event) => {
      const rect = host.getBoundingClientRect();
      pointer = { x: event.clientX - rect.left, y: event.clientY - rect.top };
      schedule();
    });
    host.addEventListener("pointerleave", () => { pointer = null; schedule(); });
  };

  // Phones and tablets: a menu button opens the section list.
  const wireMenu = () => {
    const button = document.querySelector("[data-menu-button]");
    const menu = document.querySelector("[data-menu]");
    if (!button || !menu) return;

    const setOpen = (open) => {
      button.setAttribute("aria-expanded", String(open));
      menu.hidden = !open;
    };

    button.hidden = false;
    button.addEventListener("click", () => setOpen(menu.hidden));
    menu.addEventListener("click", (event) => {
      if (event.target.closest("a")) setOpen(false);
    });
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && !menu.hidden) {
        setOpen(false);
        button.focus();
      }
    });
    window.matchMedia("(min-width: 64.0625rem)").addEventListener("change", (e) => {
      if (e.matches) setOpen(false);
    });
  };

  // Marks the nav link for the section currently in view.
  const wireScrollSpy = () => {
    const links = Array.from(document.querySelectorAll("[data-nav] a[href^='#']"));
    const sections = links
      .map((link) => document.getElementById(link.getAttribute("href").slice(1)))
      .filter(Boolean);
    if (!sections.length || !("IntersectionObserver" in window)) return;

    const visible = new Map();
    const update = () => {
      let current = null;
      for (const section of sections) {
        if (visible.get(section)) { current = section; break; }
      }
      links.forEach((link) => {
        const active = current && link.getAttribute("href") === `#${current.id}`;
        if (active) link.setAttribute("aria-current", "true");
        else link.removeAttribute("aria-current");
      });
    };
    const io = new IntersectionObserver((entries) => {
      entries.forEach((e) => visible.set(e.target, e.isIntersecting));
      update();
    }, { rootMargin: "-40% 0px -55% 0px" });
    sections.forEach((section) => io.observe(section));
  };

  const init = () => {
    heroRecord();
    wireCopyButtons();
    wireTour();
    wireDemo();
    wireFlowchart();
    wireCharts();
    wireSpotlights();
    wireDotGrids();
    wireMenu();
    wireScrollSpy();
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init, { once: true });
  } else {
    init();
  }
})();

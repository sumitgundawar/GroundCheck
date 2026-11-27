// Layout checks across widths, browsers and themes, with screenshots.
//
//   uvicorn app.main:app --port 8000
//   cd tests/ui && node responsive.mjs            (SHOTS=dir keeps screenshots, BROWSERS=chromium,webkit)
//
// At each width, in light and dark: no page scrolls sideways, the feedback
// controls are at least 44px to tap on a phone, the data-protection page fits,
// and a pseudo-locale (longer text) and Spanish still fit a phone.

import fs from "node:fs";
import path from "node:path";
import { chromium, webkit } from "playwright";

const BASE = process.env.GROUNDCHECK_URL || "http://127.0.0.1:8000";
const SHOTS = process.env.SHOTS || "";
const WIDTHS = (process.env.WIDTHS || "375,768,1280,1440").split(",").map(Number);
const THEMES = (process.env.THEMES || "light,dark").split(",");
const LOCALES = (process.env.LOCALES || "pseudo,es").split(",").filter(Boolean);
const engines = { chromium, webkit };
const wanted = (process.env.BROWSERS || "chromium,webkit").split(",");
const failures = [];
let checks = 0;

function note(ok, message) {
  checks += 1;
  if (!ok) { failures.push(message); console.log("  FAIL ", message); }
}

// Extra spaces between words give each ask its own remembered-answer key.
let asked = 0;
const unique = (question) => { asked += 1; let n = Date.now() * 1000 + asked; return question.split(" ").map((w, i, all) => { const gap = i < all.length - 1 ? " ".repeat(1 + (n % 3)) : ""; n = Math.floor(n / 3); return w + gap; }).join(""); };

async function ask(page, question) {
  await page.fill("#query", unique(question));
  await page.click("#ask-btn");
  await page.waitForFunction(() => document.querySelector("#ask-btn").disabled, undefined, { timeout: 15000 }).catch(() => {});
  await page.waitForFunction(() => !document.querySelector("#ask-btn").disabled, undefined, { timeout: 60000 });
  await page.waitForSelector("#decision-panel:not([hidden])");
}

const overflow = (page) => page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);

async function visibleOverflowers(page) {
  return page.evaluate(() => {
    const vw = document.documentElement.clientWidth;
    const out = [];
    for (const el of document.querySelectorAll("main *, .main *")) {
      const r = el.getBoundingClientRect();
      if (r.width === 0 || r.height === 0 || el.closest("[hidden]")) continue;
      if (el.closest(".sprite, svg, canvas, pre, .table-scroll, [data-scroll]")) continue;
      if (r.right > vw + 1 && getComputedStyle(el).position !== "fixed") {
        out.push(`${el.tagName.toLowerCase()}#${el.id}.${String(el.className).split(" ")[0]} right=${Math.round(r.right)}`);
        if (out.length > 4) break;
      }
    }
    return out;
  });
}

async function shot(page, name) {
  if (!SHOTS) return;
  fs.mkdirSync(SHOTS, { recursive: true });
  await page.screenshot({ path: path.join(SHOTS, `${name}.png`), fullPage: false });
}

async function views(page) {
  return page.$$eval(".nav-link[data-view]:not([hidden])", (links) => links.map((l) => l.dataset.view));
}

async function go(page, view) {
  await page.evaluate((v) => { location.hash = `#/${v}`; }, view);
  await page.waitForFunction((v) => !document.querySelector(`#view-${v}`)?.hidden, view, { timeout: 15000 });
  await page.waitForTimeout(350);
}

for (const name of wanted) {
  const engine = engines[name];
  const browser = await engine.launch();
  for (const width of WIDTHS) {
    for (const theme of THEMES) {
      const tag = `${name}-${width}-${theme}`;
      const context = await browser.newContext({ viewport: { width, height: width < 600 ? 800 : 900 }, colorScheme: theme, reducedMotion: "reduce" });
      const page = await context.newPage();
      const errors = [];
      page.on("pageerror", (e) => errors.push(e.message));
      await page.goto(BASE + "/", { waitUntil: "networkidle" });
      await page.evaluate((t) => window.gcTheme && window.gcTheme.set(t), theme);
      console.log(tag);

      await ask(page, "What is the standard dose of Caloradine?");
      note((await overflow(page)) <= 1, `${tag}: the answer view scrolls sideways (${await overflow(page)}px)`);
      await shot(page, `${tag}-answer`);

      const buttons = await page.evaluate(() => ["feedback-up", "feedback-down"].map((id) => {
        const r = document.getElementById(id).getBoundingClientRect();
        return { id, w: Math.round(r.width), h: Math.round(r.height) };
      }));
      const floor = width <= 900 ? 44 : 24; // touch widths: 44px; a mouse: WCAG 2.2 minimum of 24px
      for (const b of buttons) note(b.h >= floor && b.w >= floor, `${tag}: #${b.id} is ${b.w}x${b.h}px, under the tap size`);
      await page.click("#feedback-down");
      const send = await page.locator("#feedback-send").boundingBox();
      note(send && send.height >= floor && send.width >= floor, `${tag}: #feedback-send is ${send && Math.round(send.width)}x${send && Math.round(send.height)}px`);
      note((await overflow(page)) <= 1, `${tag}: the feedback form scrolls sideways`);
      await shot(page, `${tag}-feedback`);

      await ask(page, "What is the dose of Zalortin?");
      note((await overflow(page)) <= 1, `${tag}: a refusal scrolls sideways`);
      await shot(page, `${tag}-refusal`);

      for (const view of await views(page)) {
        await go(page, view);
        const over = await overflow(page);
        const wide = over > 1 ? await visibleOverflowers(page) : [];
        note(over <= 1, `${tag}: ${view} scrolls sideways by ${over}px ${wide.join("; ")}`);
        if (view === "data-protection" || view === "review" || view === "usage" || view === "monitoring") await shot(page, `${tag}-${view}`);
      }
      note(errors.length === 0, `${tag}: script errors: ${errors.slice(0, 2).join(" | ")}`);
      await context.close();
    }
  }

  // Longer text and Spanish at every width.
  for (const locale of LOCALES) {
    for (const width of WIDTHS) {
      for (const theme of THEMES) {
        const tag = `${name}-${width}-${theme}-${locale}`;
        const context = await browser.newContext({ viewport: { width, height: width < 600 ? 800 : 900 }, colorScheme: theme, reducedMotion: "reduce" });
        const page = await context.newPage();
        const errors = [];
        page.on("pageerror", (e) => errors.push(e.message));
        await page.goto(`${BASE}/?locale=${locale}`, { waitUntil: "networkidle" });
        await page.evaluate(() => window.gcI18n.ready);
        console.log(tag);
        const label = ((await page.textContent(".nav-link[data-view=\"ask\"]")) || "").trim();
        note(locale === "es" ? /Preguntar/.test(label) : /^\[/.test(label), `${tag}: the locale was not applied (topbar says ${label})`);
        await ask(page, "What is the dose of Zalortin?");
        note((await overflow(page)) <= 1, `${tag}: the Ask view scrolls sideways (${await overflow(page)}px)`);
        if (locale === "es") {
          const said = ((await page.textContent("#refuse-callout")) || "") + ((await page.textContent("#trace")) || "");
          note(/no aparece en ninguna fuente/.test(said), `${tag}: the refusal reason is not in Spanish`);
          note(!/does not appear in any|not reached|within rate limit/.test(said), `${tag}: server English is showing in the refusal or trace`);
        }
        await shot(page, `${tag}-refusal`);
        for (const view of await views(page)) {
          await go(page, view);
          const over = await overflow(page);
          note(over <= 1, `${tag}: ${view} scrolls sideways by ${over}px ${over > 1 ? (await visibleOverflowers(page)).join("; ") : ""}`);
          if (view === "review") await shot(page, `${tag}-review`);
        }
        note(errors.length === 0, `${tag}: script errors: ${errors.slice(0, 2).join(" | ")}`);
        await context.close();
      }
    }
  }
  await browser.close();
}

console.log(`\n${checks - failures.length}/${checks} layout checks passed`);
if (failures.length) { console.log("\n" + failures.join("\n")); process.exit(1); }

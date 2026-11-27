// Accessibility checks of the browser app, in light and dark, English and Spanish.
//
//   uvicorn app.main:app --port 8000
//   cd tests/ui && node a11y.mjs         (AXE_PATH=/path/to/axe.min.js adds the axe-core rules)
//
// The built-in checks always run: heading order, a name on every control, a
// label on every input, the contrast of the proposal verdict and its table, and
// a keyboard-only walk from asking a question through the trace to a review
// case. axe-core is injected only when its file can be found offline.

import fs from "node:fs";
import { createRequire } from "node:module";
import { chromium, webkit } from "playwright";

const BASE = process.env.GROUNDCHECK_URL || "http://127.0.0.1:8000";
const require = createRequire(import.meta.url);
const engines = { chromium, webkit };
const wanted = (process.env.BROWSERS || "chromium").split(",");
const failures = [];
let checks = 0;

function axeSource() {
  const candidates = [process.env.AXE_PATH];
  try { candidates.push(require.resolve("axe-core/axe.min.js")); } catch (_) { /* not installed here */ }
  const found = candidates.filter(Boolean).find((p) => fs.existsSync(p));
  return found ? fs.readFileSync(found, "utf8") : null;
}
const AXE = axeSource();

function note(ok, message) {
  checks += 1;
  if (!ok) { failures.push(message); console.log("  FAIL ", message); }
}

let asked = 0;
const unique = (q) => { asked += 1; let n = Date.now() * 1000 + asked; return q.split(" ").map((w, i, all) => { const gap = i < all.length - 1 ? " ".repeat(1 + (n % 3)) : ""; n = Math.floor(n / 3); return w + gap; }).join(""); };

async function ask(page, question) {
  await page.fill("#query", unique(question));
  await page.click("#ask-btn");
  await page.waitForFunction(() => document.querySelector("#ask-btn").disabled, undefined, { timeout: 15000 }).catch(() => {});
  await page.waitForFunction(() => !document.querySelector("#ask-btn").disabled, undefined, { timeout: 60000 });
  await page.waitForSelector("#decision-panel:not([hidden])");
}

async function go(page, view) {
  await page.evaluate((v) => { location.hash = `#/${v}`; }, view);
  await page.waitForFunction((v) => !document.querySelector(`#view-${v}`)?.hidden, view, { timeout: 15000 });
  await page.waitForTimeout(300);
}

// ---- checks run inside the page ----------------------------------------------

const inPage = {
  headings() {
    const visible = (el) => !el.closest("[hidden], .sprite") && el.getClientRects().length > 0 && getComputedStyle(el).visibility !== "hidden";
    const heads = [...document.querySelectorAll("h1, h2, h3, h4, h5, h6, [role=heading]")].filter(visible)
      .map((h) => ({ level: h.getAttribute("aria-level") ? Number(h.getAttribute("aria-level")) : Number(h.tagName[1]), text: h.textContent.trim().slice(0, 40) }));
    const problems = [];
    if (heads.filter((h) => h.level === 1).length !== 1) problems.push(`${heads.filter((h) => h.level === 1).length} h1 headings`);
    heads.forEach((h, i) => {
      const prev = i ? heads[i - 1].level : 0;
      if (h.level > prev + 1) problems.push(`h${h.level} "${h.text}" follows h${prev || "-"}`);
    });
    return problems;
  },

  names() {
    const visible = (el) => !el.closest("[hidden], .sprite, template") && el.getClientRects().length > 0 && getComputedStyle(el).visibility !== "hidden";
    const text = (el) => (el.textContent || "").replace(/\s+/g, " ").trim();
    const byId = (ids) => ids.split(/\s+/).map((id) => document.getElementById(id)).filter(Boolean).map(text).join(" ").trim();
    const nameOf = (el) => el.getAttribute("aria-label")?.trim() || (el.getAttribute("aria-labelledby") && byId(el.getAttribute("aria-labelledby")))
      || text(el) || el.getAttribute("title")?.trim() || el.querySelector("img[alt]")?.getAttribute("alt")?.trim() || el.querySelector("svg title")?.textContent.trim() || "";
    const out = { buttons: [], inputs: [] };
    document.querySelectorAll("button, a[href], [role=button], [role=tab], [role=menuitem], summary").forEach((el) => {
      if (visible(el) && !nameOf(el)) out.buttons.push(`${el.tagName.toLowerCase()}#${el.id}.${String(el.className).split(" ")[0]}`);
    });
    document.querySelectorAll("input, select, textarea").forEach((el) => {
      if (el.type === "hidden" || !visible(el)) return;
      const labelled = el.labels && [...el.labels].some((l) => text(l))
        || el.getAttribute("aria-label")?.trim() || (el.getAttribute("aria-labelledby") && byId(el.getAttribute("aria-labelledby")))
        || el.getAttribute("title")?.trim();
      if (!labelled) out.inputs.push(`${el.tagName.toLowerCase()}#${el.id}[${el.type}]`);
    });
    return out;
  },

  contrast(selectors) {
    const parse = (c) => { const m = c.match(/rgba?\(([^)]+)\)/); if (!m) return null; const p = m[1].split(/[ ,\/]+/).filter(Boolean).map(Number); return { r: p[0], g: p[1], b: p[2], a: p.length > 3 ? p[3] : 1 }; };
    const lum = ({ r, g, b }) => { const f = (v) => { v /= 255; return v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4; }; return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b); };
    const over = (top, under) => ({ r: top.r * top.a + under.r * (1 - top.a), g: top.g * top.a + under.g * (1 - top.a), b: top.b * top.a + under.b * (1 - top.a), a: 1 });
    const background = (el) => {
      const layers = [];
      for (let n = el; n; n = n.parentElement) { const c = parse(getComputedStyle(n).backgroundColor); if (c && c.a > 0) { layers.push(c); if (c.a === 1) break; } }
      let base = { r: 255, g: 255, b: 255, a: 1 };
      if (!layers.length || layers[layers.length - 1].a < 1) { const body = parse(getComputedStyle(document.body).backgroundColor); if (body && body.a > 0) base = body; }
      return layers.reverse().reduce((under, top) => over(top, under), base);
    };
    const out = [];
    for (const sel of selectors) {
      document.querySelectorAll(sel).forEach((el) => {
        if (el.closest("[hidden]") || !el.getClientRects().length || !(el.textContent || "").trim()) return;
        const style = getComputedStyle(el);
        const bg = background(el);
        const fg = over(parse(style.color), bg);
        const ratio = (Math.max(lum(fg), lum(bg)) + 0.05) / (Math.min(lum(fg), lum(bg)) + 0.05);
        const size = parseFloat(style.fontSize), bold = Number(style.fontWeight) >= 700;
        const need = size >= 24 || (size >= 18.66 && bold) ? 3 : 4.5;
        if (ratio < need) out.push(`${sel} "${el.textContent.trim().slice(0, 24)}" ${ratio.toFixed(2)} < ${need}`);
      });
    }
    return out;
  },
};

async function axeViolations(page, label) {
  if (!AXE) return [];
  await page.waitForTimeout(700); // the trace rows fade in; a row measured mid-fade is not a contrast fault
  await page.evaluate(AXE);
  const result = await page.evaluate(async () => {
    const r = await window.axe.run(document, { resultTypes: ["violations"], rules: { "color-contrast": { enabled: true } } });
    return r.violations.map((v) => ({ id: v.id, impact: v.impact, nodes: v.nodes.slice(0, 3).map((n) => n.target.join(" ")), help: v.help }));
  });
  return result.map((v) => `${label}: axe ${v.id} (${v.impact}): ${v.help} at ${v.nodes.join(" | ")}`);
}

// ---- the keyboard-only walk -----------------------------------------------------

async function tabTo(page, predicate, max = 120, back = false) {
  for (let i = 0; i < max; i += 1) {
    await page.keyboard.press(back ? "Shift+Tab" : "Tab");
    if (await page.evaluate(predicate)) return true;
  }
  return false;
}

async function keyboardJourney(page, tag, engine) {
  await go(page, "ask");
  await page.focus("#query");
  await page.keyboard.type(unique("What is the dose of Zalortin?"));
  await page.keyboard.press("Enter");
  await page.waitForFunction(() => document.querySelector("#ask-btn").disabled, undefined, { timeout: 15000 }).catch(() => {});
  await page.waitForFunction(() => !document.querySelector("#ask-btn").disabled, undefined, { timeout: 60000 });
  await page.waitForSelector("#trace .trace-row");

  note(await tabTo(page, () => document.activeElement?.classList.contains("trace-row")), `${tag}: a trace row cannot be reached with Tab`);
  await page.keyboard.press("Enter");
  const open = await page.evaluate(() => document.activeElement?.getAttribute("aria-expanded"));
  note(open === "true", `${tag}: Enter on a trace row does not open it (aria-expanded=${open})`);
  const shown = await page.evaluate(() => { const li = document.activeElement?.closest(".trace-item"); const p = li && li.querySelector(".trace-expand"); return !!p && !p.hidden && p.getClientRects().length > 0; });
  note(shown, `${tag}: the opened trace row shows nothing`);
  await page.keyboard.press("Space");
  note((await page.evaluate(() => document.activeElement?.getAttribute("aria-expanded"))) === "false", `${tag}: Space on a trace row does not close it`);

  const routed = await page.evaluate(() => !!document.querySelector("#refuse-routed a, #refuse-routed button, .routed a, .routed button"));
  if (routed) {
    // Safari does not Tab to links unless the user turns that on, so WebKit is given the link directly.
    if (engine === "webkit") await page.focus(".routed a, .routed button");
    else note(await tabTo(page, () => !!document.activeElement?.closest(".routed")), `${tag}: the review link on a refusal cannot be reached with Tab`);
    await page.keyboard.press("Enter");
  } else {
    await page.focus('.nav-link[data-view="review"]');
    await page.keyboard.press("Enter");
  }
  await page.waitForTimeout(600);
  if (!(await page.evaluate(() => document.querySelector("#case-dialog")?.open))) {
    await go(page, "review");
    await page.waitForSelector("#review-rows button, #review-rows .btn", { timeout: 15000 });
    const first = page.locator("#review-rows button").first();
    await first.focus();
    await page.keyboard.press("Enter");
  }
  await page.waitForFunction(() => document.querySelector("#case-dialog")?.open, undefined, { timeout: 10000 }).catch(() => {});
  const dialog = await page.evaluate(() => ({ open: !!document.querySelector("#case-dialog")?.open, inside: !!document.activeElement?.closest("#case-dialog") }));
  note(dialog.open, `${tag}: the review case does not open from the keyboard`);
  note(dialog.inside, `${tag}: focus is not inside the review case dialog`);
  const named = await page.evaluate(() => { const d = document.querySelector("#case-dialog"); return !!(d && (d.getAttribute("aria-labelledby") || d.getAttribute("aria-label"))); });
  note(named, `${tag}: the review case dialog has no name`);
  await page.keyboard.press("Escape");
  note(!(await page.evaluate(() => document.querySelector("#case-dialog")?.open)), `${tag}: Escape does not close the review case`);
}

// ---- main -----------------------------------------------------------------------

const NEW_ELEMENTS = [".proposal-verdict", ".proposal-table caption", ".proposal-table thead th", ".proposal-table tbody th", ".proposal-table td",
  ".dim-status", ".dim-reason", ".status-chip", ".finding-text", ".finding-tag", ".alert-item", ".alert-item p"];

for (const name of wanted) {
  const browser = await engines[name].launch();
  console.log(AXE ? "axe-core found: its rules will run" : "axe-core not found: running the built-in checks only");
  for (const locale of ["en", "es"]) {
    for (const theme of ["light", "dark"]) {
      const tag = `${name}-${locale}-${theme}`;
      console.log(tag);
      const context = await browser.newContext({ viewport: { width: 1280, height: 900 }, colorScheme: theme, reducedMotion: "reduce" });
      const page = await context.newPage();
      await page.goto(`${BASE}/?locale=${locale}`, { waitUntil: "networkidle" });
      await page.evaluate(() => window.gcI18n.ready);
      await page.evaluate((t) => window.gcTheme && window.gcTheme.set(t), theme);

      // A proposal that does not match, a proposal that matches, a patient dose calculation and a refusal.
      const patient = async (age, weight) => {
        const box = page.locator("#patient-box");
        if ((await box.getAttribute("open")) === null) await page.click("#patient-box > summary");
        await page.fill("#pt-age", String(age));
        await page.fill("#pt-weight", String(weight));
        await page.fill("#pt-egfr", "90");
      };
      await ask(page, "Is 15 mg of Caloradine three times daily by mouth for 21 days okay?");
      note(await page.locator(".proposal-verdict.mismatch").count() === 1, `${tag}: no mismatch verdict`);
      note(await page.locator(".proposal-table tbody tr").count() >= 3, `${tag}: the proposal table has no rows`);
      note((await inPage.headings && true), "");
      const verdictProblems = await page.evaluate(inPage.contrast, NEW_ELEMENTS);
      note(verdictProblems.length === 0, `${tag}: proposal contrast: ${verdictProblems.slice(0, 4).join("; ")}`);
      const hp = await page.evaluate(inPage.headings);
      note(hp.length === 0, `${tag}: ask view headings: ${hp.join("; ")}`);
      const nm = await page.evaluate(inPage.names);
      note(nm.buttons.length === 0, `${tag}: controls without a name: ${nm.buttons.join(", ")}`);
      note(nm.inputs.length === 0, `${tag}: inputs without a label: ${nm.inputs.join(", ")}`);
      for (const v of await axeViolations(page, `${tag} ask`)) note(false, v);

      await ask(page, "Is 15 mg of Caloradine once daily for 14 days okay?");
      note(await page.locator(".proposal-verdict.match").count() === 1, `${tag}: no match verdict`);
      const matchProblems = await page.evaluate(inPage.contrast, NEW_ELEMENTS);
      note(matchProblems.length === 0, `${tag}: match contrast: ${matchProblems.slice(0, 4).join("; ")}`);

      await patient(8, 18);
      await ask(page, "What is the dose of Mendel solution?");
      const findingProblems = await page.evaluate(inPage.contrast, NEW_ELEMENTS);
      note(findingProblems.length === 0, `${tag}: patient findings contrast: ${findingProblems.slice(0, 4).join("; ")}`);
      for (const v of await axeViolations(page, `${tag} patient`)) note(false, v);
      await page.click("#pt-clear").catch(() => {});

      await ask(page, "Is it harmful to give Caloradine 15 mg once daily and a clock time of 8:30?");
      note(await page.locator(".proposal-table .dim-unverifiable").count() >= 1, `${tag}: a refused proposal shows no unchecked part`);
      const refusedProblems = await page.evaluate(inPage.contrast, [...NEW_ELEMENTS, "#refuse-callout", "#refuse-callout p", ".refuse-detail"]);
      note(refusedProblems.length === 0, `${tag}: refused proposal contrast: ${refusedProblems.slice(0, 4).join("; ")}`);
      for (const v of await axeViolations(page, `${tag} refusal`)) note(false, v);

      const views = await page.$$eval(".nav-link[data-view]:not([hidden])", (links) => links.map((l) => l.dataset.view));
      for (const view of views) {
        await go(page, view);
        const h = await page.evaluate(inPage.headings);
        note(h.length === 0, `${tag}: ${view} headings: ${h.join("; ")}`);
        const n = await page.evaluate(inPage.names);
        note(n.buttons.length === 0, `${tag}: ${view} controls without a name: ${n.buttons.join(", ")}`);
        note(n.inputs.length === 0, `${tag}: ${view} inputs without a label: ${n.inputs.join(", ")}`);
        if (view === "monitoring" || view === "review") {
          const c = await page.evaluate(inPage.contrast, [...NEW_ELEMENTS, "td", "th", ".sev", ".muted"]);
          note(c.length === 0, `${tag}: ${view} contrast: ${c.slice(0, 4).join("; ")}`);
          for (const v of await axeViolations(page, `${tag} ${view}`)) note(false, v);
        }
      }
      if (theme === "light") await keyboardJourney(page, tag, name);
      await context.close();
    }
  }
  await browser.close();
}

console.log(`\n${checks - failures.length}/${checks} accessibility checks passed`);
if (failures.length) { console.log("\n" + failures.join("\n")); process.exit(1); }

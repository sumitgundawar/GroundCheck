// Runs web/i18n.js against a minimal DOM stub and checks locale handling.
import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const read = (p) => fs.readFileSync(path.join(root, p), "utf8");
const files = { "/i18n/en.json": read("web/i18n/en.json"), "/i18n/es.json": read("web/i18n/es.json") };
const results = [];
const check = (name, ok, detail = "") => results.push({ name, ok: !!ok, detail });

function makeEl(attrs, text) {
  const el = { attrs: { ...attrs }, textContent: text,
    getAttribute: (n) => (n in el.attrs ? el.attrs[n] : null), setAttribute: (n, v) => { el.attrs[n] = v; }, value: "en",
    listeners: {}, addEventListener(n, f) { el.listeners[n] = f; } };
  return el;
}

async function boot({ stored = null, search = "", failEs = false, els }) {
  const listeners = {};
  const doc = {
    documentElement: { lang: "en" }, title: "x", listeners,
    addEventListener: (n, f) => { listeners[n] = f; }, dispatchEvent: (e) => { (listeners[e.type] || (() => {}))(e); },
    querySelectorAll: (sel) => (sel === "[data-i18n]" ? els.filter((e) => e.attrs["data-i18n"] !== undefined)
      : sel === "[data-i18n-attr]" ? els.filter((e) => e.attrs["data-i18n-attr"] !== undefined)
      : sel === ".locale-select" ? els.filter((e) => e.cls === "locale-select") : []),
  };
  const store = {};
  if (stored) store["gc-locale"] = stored;
  const win = {
    document: doc, location: { search },
    localStorage: { getItem: (k) => store[k] ?? null, setItem: (k, v) => { store[k] = v; } },
    URLSearchParams, CustomEvent: class { constructor(type, init) { this.type = type; this.detail = init && init.detail; } },
    Promise, Object, Array, String, JSON, Math, Error,
    fetch: async (url) => {
      const u = new URL(url, "http://x").pathname;
      if (failEs && u.endsWith("es.json")) return { ok: false, json: async () => ({}) };
      return files[u] ? { ok: true, json: async () => JSON.parse(files[u]) } : { ok: false, json: async () => ({}) };
    },
  };
  win.window = win;
  const ctx = vm.createContext(win);
  vm.runInContext(read("web/i18n.js"), ctx);
  doc.listeners.DOMContentLoaded();
  await win.gcI18n.ready;
  return { win, doc, store };
}

const en = JSON.parse(files["/i18n/en.json"]), es = JSON.parse(files["/i18n/es.json"]);

{
  const els = [makeEl({ "data-i18n": "sidebar.ask" }, "Ask"), makeEl({ "data-i18n": "sidebar.documents" }, "Documents"),
    makeEl({ "data-i18n-attr": "aria-label:locale.language", "aria-label": "Language" }, ""),
    makeEl({ "data-i18n": "no.such.key" }, "Hand written")];
  const { win } = await boot({ els });
  check("english is the default", win.gcI18n.locale() === "en" && els[0].textContent === "Ask");
  check("an unknown key keeps the page text", els[3].textContent === "Hand written");
  check("t falls back to the default argument", win.t("no.such.key", "fallback") === "fallback");
  check("t falls back to the key itself", win.t("no.such.key") === "no.such.key");
  await win.gcI18n.setLocale("es");
  check("spanish is applied", els[0].textContent === es["sidebar.ask"] && els[2].attrs["aria-label"] === es["locale.language"]);
  check("document language follows", win.document.documentElement.lang === "es");
  win.gcI18n.apply(win.document);
  await win.gcI18n.setLocale("en");
  check("back to english", els[0].textContent === "Ask" && win.document.documentElement.lang === "en");
}
{
  const els = [makeEl({ "data-i18n": "sidebar.ask" }, "Ask")];
  const { win } = await boot({ stored: "es", els });
  check("the stored locale is used", win.gcI18n.locale() === "es" && els[0].textContent === es["sidebar.ask"]);
}
{
  const els = [makeEl({ "data-i18n": "sidebar.ask" }, "Ask")];
  const { win } = await boot({ stored: "es", failEs: true, els });
  check("a locale file that fails to load leaves english", win.gcI18n.locale() === "en" && els[0].textContent === "Ask");
}
{
  const els = [makeEl({ "data-i18n": "sidebar.ask" }, "Ask")];
  const { win } = await boot({ stored: "xx", els });
  check("an unknown locale is english", win.gcI18n.locale() === "en");
}
{
  const els = [makeEl({ "data-i18n": "sidebar.ask" }, "Ask")];
  const { win } = await boot({ search: "?locale=pseudo", els });
  check("the pseudo locale is longer and marked", win.gcI18n.locale() === "pseudo" && els[0].textContent.length > 6
    && els[0].textContent.startsWith("[") && els[0].textContent !== "Ask", els[0].textContent);
}
{
  // Text a script changed since the page loaded is not overwritten by a locale change.
  const els = [makeEl({ "data-i18n": "sidebar.ask" }, "Ask")];
  const { win } = await boot({ els });
  els[0].textContent = "Dynamic 42";
  await win.gcI18n.setLocale("es");
  check("script-written text survives a locale change", els[0].textContent === "Dynamic 42");
}
{
  const els = [makeEl({ "data-i18n": "reason.term_not_covered" }, "x")];
  const { win } = await boot({ els });
  await win.gcI18n.setLocale("es");
  check("t returns the spanish reason", win.t("reason.term_not_covered", "d") === es["reason.term_not_covered"]);
}
{
  // A key the Spanish file lacks shows English.
  const saved = files["/i18n/es.json"];
  const trimmed = { ...es };
  delete trimmed["sidebar.documents"];
  files["/i18n/es.json"] = JSON.stringify(trimmed);
  const els = [makeEl({ "data-i18n": "sidebar.documents" }, "Documents")];
  const { win } = await boot({ stored: "es", els });
  check("a missing spanish key falls back to english", els[0].textContent === en["sidebar.documents"]);
  files["/i18n/es.json"] = saved;
}
{
  const els = [makeEl({ "data-i18n": "sidebar.ask" }, "Ask")];
  const { win } = await boot({ els });
  check("t fills {name} from vars", win.t("js.chosen_mb", "x", { files: "3 files", bytes: "1.5" }) === "3 files chosen, 1.5 MB");
  check("t leaves an unknown placeholder alone", win.t("k", "a {b} c", { x: 1 }) === "a {b} c");
  check("t inserts text, not markup", win.t("k", "{v}", { v: "<b>x</b>" }) === "<b>x</b>");
  await win.gcI18n.setLocale("es");
  check("spanish fills vars", win.t("js.chosen_mb", "x", { files: "3 archivos", bytes: "1.5" }) === es["js.chosen_mb"].replace("{files}", "3 archivos").replace("{bytes}", "1.5"));
}
{
  const els = [makeEl({ "data-i18n": "sidebar.ask" }, "Ask")];
  const { win } = await boot({ search: "?locale=pseudo", els });
  check("pseudo keeps placeholders usable", /3 files/.test(win.t("js.chosen_mb", "x", { files: "3 files", bytes: "1" })));
}
console.log(JSON.stringify(results));
process.exit(results.every((r) => r.ok) ? 0 : 1);

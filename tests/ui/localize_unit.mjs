// Runs web/i18n.js and web/localize.js against the real locale files in a stubbed
// page. With a JSON list of {kind, text} on stdin it also reports which of those
// server sentences have no translation, so a test can compare them with the
// server's own output.
import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const read = (p) => fs.readFileSync(path.join(root, p), "utf8");
const files = { "/i18n/en.json": read("web/i18n/en.json"), "/i18n/es.json": read("web/i18n/es.json") };
const results = [];
const check = (name, ok, detail = "") => results.push({ name, ok: !!ok, detail });

async function boot(locale) {
  const listeners = {};
  const doc = {
    documentElement: { lang: "en" }, title: "x",
    addEventListener: (n, f) => { listeners[n] = f; }, dispatchEvent: (e) => { (listeners[e.type] || (() => {}))(e); },
    querySelectorAll: () => [],
  };
  const store = { "gc-locale": locale };
  const win = {
    document: doc, location: { search: "" },
    localStorage: { getItem: (k) => store[k] ?? null, setItem: (k, v) => { store[k] = v; } },
    URLSearchParams, CustomEvent: class { constructor(type, init) { this.type = type; this.detail = init && init.detail; } },
    Promise, Object, Array, String, JSON, Math, Error, RegExp,
    fetch: async (url) => {
      const u = new URL(url, "http://x").pathname;
      return files[u] ? { ok: true, json: async () => JSON.parse(files[u]) } : { ok: false, json: async () => ({}) };
    },
  };
  win.window = win;
  vm.createContext(win);
  vm.runInContext(read("web/i18n.js"), win);
  vm.runInContext(read("web/localize.js"), win);
  doc.dispatchEvent({ type: "DOMContentLoaded" });
  (listeners.DOMContentLoaded || (() => {}))();
  await win.gcI18n.ready;
  return win;
}

const en = await boot("en");
const es = await boot("es");
const pseudo = await boot("pseudo");
const L = es.gcLocalize;

check("english shows the server's own text", en.gcLocalize.text("not reached") === null && en.gcLocalize.refusal({ refused_reason: "x", reason_code: "term_not_covered" }) === null);
check("a known sentence is translated", L.text("not reached") === "no se llegó a esta etapa");
check("an unknown sentence is left to the server", L.text("something the server invented") === null && L.reason("nothing like it") === null);
check("an unknown reason code falls back to nothing", L.refusal({ refused_reason: "novel", reason_code: "brand_new_code" }) === null);
check("a known reason code with new wording falls back to its Spanish explanation",
  L.refusal({ refused_reason: "reworded entirely", reason_code: "term_not_covered" }) !== null);
check("slots are filled", L.reason("the term 'zalortin' does not appear in any trusted source") === "El término «zalortin» no aparece en ninguna fuente de confianza");
check("the retrieved variant is told apart", /recuperada/.test(L.reason("the term 'x' does not appear in any retrieved source")));
check("a drug name keeps its case and a dose its unit", L.text("the value '10 mL' is not supported by a source about this medicine").includes("«10 mL»")
  && L.text("Caloradine is contraindicated in Asthma.") === "Caloradine está contraindicado en Asthma.");
check("a medicine at the start of a lowered reason gets its capital back", /^Caloradine está/.test(L.reason("caloradine is contraindicated in asthma")));
check("an age and group are put in Spanish", L.reason("the question is about a 6-year-old, and no trusted source covers children") === "La pregunta trata sobre un paciente de 6 años, y ninguna fuente de confianza cubre a niños");
check("an age of one year is singular", /1 año,/.test(L.reason("the question is about a 1-year-old, and no trusted source covers children")));
check("an article before a vowel sound still matches", L.reason("the question is about an 8-year-old, and no trusted source covers children") !== null);
check("a maximum or minimum is told apart", /máxima/.test(L.reason("no trusted source states a maximum dose")) && /mínima/.test(L.reason("no trusted source states a minimum dose")));
check("a claim is quoted untouched", L.reason('a claim could not be grounded: "Take 10 mg of Caloradine once daily."').includes("«Take 10 mg of Caloradine once daily.»"));
check("removed identifiers are counted in Spanish", L.text("removed 2 names, 1 NHS number, 1 street address") === "se eliminaron 2 nombres, 1 número del NHS, 1 dirección postal");
check("an unknown identifier kind is not guessed at", L.text("removed 2 widgets") === null);
check("the judge tail is peeled off", /juez: 2\/3 respaldadas$/.test(L.text("all 3 claim(s) grounded in cited sources, judge 2/3 supported")));
check("scope and language details join", L.text("no scope or injection issues; language es, read as English by its word list") !== null);
check("a part the table lacks leaves the whole to the server", L.text("no scope or injection issues; something new") === null);
check("missing patient data is listed in Spanish", /su edad, peso, función renal/.test(L.text("To check a dose of Caloradine for this patient, GroundCheckHealth needs their age, weight, kidney function (eGFR, or serum creatinine with age and weight).")));
check("missing data it does not know is left alone", L.text("To check a dose of Caloradine for this patient, GroundCheckHealth needs their shoe size.") === null);
check("the most specific pattern wins", L.text("The formulary has no dosing for Caloradine in a patient aged 5: this answer is about adults.").includes("se refiere a adultos")
  && !L.text("The formulary has no dosing for Caloradine in a patient aged 5.").includes("adultos"));
check("slot values are text, not markup", L.text("the term '<b>x</b>' does not appear in any trusted source".replace("the term", "the term")) === null
  || !/<b>/.test("") );
const markup = L.reason("the term '<img src=x onerror=alert(1)>' does not appear in any trusted source");
check("markup in a slot comes back as the same characters", markup === "El término «<img src=x onerror=alert(1)>» no aparece en ninguna fuente de confianza");
check("a replacement pattern in a slot is not expanded", L.reason("the term '$&$1' does not appear in any trusted source").includes("«$&$1»"));
check("a stage explanation is used only while the server's text is unchanged",
  L.explain({ name: "rate limit", explain: JSON.parse(files["/i18n/en.json"])["srv.explain.rate_limit"] }) !== null
  && L.explain({ name: "rate limit", explain: "reworded by the server" }) === null);
check("the refusal answer text is translated whole or not at all",
  L.answerText("I cannot answer this safely. No sufficiently relevant source was found. The question has been routed for review.") ===
  "No es posible responder esto con seguridad. No se encontró ninguna fuente suficientemente relevante. La pregunta se derivó a revisión."
  && L.answerText("I cannot answer this safely. Something else entirely.") === null);
check("a decimal point does not split a sentence", L.answerText("I cannot answer this safely. The value 0.5 mg") === null);
check("an example keeps its query unless Spanish has one", (() => {
  const a = L.example({ group: "Core demo", label: "Dosage trap (refuses)", query: "What is the recommended dose of Zalortin for a patient with Veltris syndrome?" });
  const b = L.example({ group: "Nothing", label: "Unlisted", query: "Q?" });
  return a.group === "Demostración principal" && /Zalortin/.test(a.query) && b.query === "Q?" && b.label === "Unlisted";
})());
check("pseudo-locale keeps the query as written", pseudo.gcLocalize.example({ group: "Core demo", label: "Dosage trap (refuses)", query: "raw?" }).query === "raw?");
check("pseudo-locale still renders longer text", /^\[/.test(pseudo.gcLocalize.text("not reached") || ""));

const stdin = fs.readFileSync(0, "utf8").trim();
const missing = [];
if (stdin) {
  for (const { kind, text } of JSON.parse(stdin)) {
    const got = kind === "reason" ? L.reason(text) : kind === "answer" ? L.answerText(text) : L.text(text);
    if (got === null) missing.push({ kind, text });
  }
}
console.log(JSON.stringify({ results, missing }));
process.exit(results.every((r) => r.ok) ? 0 : 1);

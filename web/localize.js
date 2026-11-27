// Text the server writes in English (refusal reasons, trace details, stage
// explanations, patient findings, review notes), shown in the chosen language.
// The English wording is the key: web/i18n/en.json holds each server sentence
// under "srv.exact.*" (whole sentences) or "srv.pat.*" / "srv.tail.*" (with
// {name} slots), and the locale file holds its translation. A sentence the
// table does not know comes back as null and the caller shows the server's
// English. Slots are filled as text and never as markup; anything that is a
// name, a dose or a value is copied exactly as the server sent it.
(function (root) {
  "use strict";

  var WORD_VARS = { group: 1, which: 1, where: 1, mode: 1, lang_name: 1, name: 1, allowed: 1 };
  var NESTED_VARS = { reason: 1, age: 1, note: 1 };
  var NUMBER_VARS = { n: 1, a: 1, b: 1, id: 1, years: 1, score: 1, threshold: 1 };
  var CODES = [
    "input_blocked", "rate_limited", "no_relevant_source", "term_not_covered", "population_not_covered",
    "stated_value_unsupported", "form_unsupported", "limit_not_stated", "combination_forbidden",
    "insufficient_context", "claim_ungrounded", "dose_unsupported", "patient_unsafe", "unsupported_language",
  ];

  var built = null;
  var index = null;

  function i18n() { return root.gcI18n; }
  function active() { var i = i18n(); return !!i && typeof i.locale === "function" && i.locale() !== "en"; }
  function en() { return i18n().en(); }
  function tr(key, vars) { return root.t(key, undefined, vars); }
  function has(key) { return Object.prototype.hasOwnProperty.call(en(), key); }
  function slug(s) { return String(s).toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, ""); }
  function escapeRe(s) { return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"); }
  function capFirst(s) { return s ? s.charAt(0).toUpperCase() + s.slice(1) : s; }

  function build() {
    var table = en();
    if (built === table) return index;
    var exact = {}, pats = [], tails = [];
    Object.keys(table).forEach(function (key) {
      var kind = /^srv\.(exact|pat|tail)\./.exec(key);
      if (!kind) return;
      var value = table[key];
      if (!/\{\w+\}/.test(value)) { exact[value] = key; return; }
      var names = [];
      var body = value.split(/(\{\w+\})/).map(function (part) {
        var m = /^\{(\w+)\}$/.exec(part);
        if (m) { names.push(m[1]); return NUMBER_VARS[m[1]] ? "(\\d+(?:\\.\\d+)?)" : "(.+?)"; }
        return escapeRe(part);
      }).join("");
      var entry = { key: key, names: names, weight: value.replace(/\{\w+\}/g, "").length };
      if (kind[1] === "tail") { entry.re = new RegExp(body + "$"); tails.push(entry); }
      else { entry.re = new RegExp("^" + body + "$"); pats.push(entry); }
    });
    pats.sort(function (a, b) { return b.weight - a.weight; });
    built = table;
    index = { exact: exact, pats: pats, tails: tails };
    return index;
  }

  function word(value) {
    var one = function (v) { var k = "srv.word." + slug(v); return has(k) ? tr(k) : null; };
    var whole = one(value);
    if (whole !== null) return whole;
    var parts = String(value).split(", ");
    if (parts.length > 1) {
      var out = parts.map(one);
      if (out.every(function (x) { return x !== null; })) return out.join(", ");
    }
    return value;
  }

  // "2 names, 1 NHS number": the server's own kind names, with an "s" when
  // there is more than one.
  function kindLabel(kind, n) {
    var k = "srv.kind." + slug(kind);
    if (!has(k)) return null;
    return n + " " + (String(n) === "1" ? tr(k) : tr("srv.kindpl." + slug(kind)));
  }

  function redactionList(value) {
    var table = en();
    var kinds = Object.keys(table).filter(function (k) { return k.indexOf("srv.kind.") === 0; }).map(function (k) { return table[k]; })
      .sort(function (a, b) { return b.length - a.length; });
    var out = [];
    var parts = value.split(", ");
    for (var i = 0; i < parts.length; i += 1) {
      var m = /^(\d+) (.+)$/.exec(parts[i]);
      var kind = m && kinds.filter(function (k) { return m[2] === k || m[2] === k + "s"; })[0];
      var label = kind && kindLabel(kind, m[1]);
      if (!label) return null;
      out.push(label);
    }
    return out.join(", ");
  }

  function missingItems(value) {
    var table = en();
    var known = Object.keys(table).filter(function (k) { return k.indexOf("srv.item.") === 0; })
      .map(function (k) { return [table[k], k]; }).sort(function (a, b) { return b[0].length - a[0].length; });
    var rest = value, out = [];
    while (rest) {
      var hit = null;
      for (var i = 0; i < known.length; i += 1) {
        if (rest === known[i][0] || rest.indexOf(known[i][0] + ", ") === 0) { hit = known[i]; break; }
      }
      if (!hit) return null;
      out.push(tr(hit[1]));
      rest = rest.slice(hit[0].length + 2);
    }
    return out.join(", ");
  }

  function agePhrase(value) {
    var m = /^(\d+(?:\.\d+)?)-(year|month|week|day)-old$/.exec(value);
    if (!m) return null;
    var unit = tr("srv.unit." + m[2] + (m[1] === "1" ? ".one" : ".other"));
    return tr("srv.age.phrase", { n: m[1], unit: unit });
  }

  function slot(name, value, depth) {
    if (NESTED_VARS[name]) { var nested = name === "reason" ? reason(value, depth + 1) : text(value, depth + 1); return nested === null ? value : nested; }
    if (WORD_VARS[name]) return word(value);
    if (name === "list") return redactionList(value);
    if (name === "items") return missingItems(value);
    if (name === "phrase") return agePhrase(value);
    return value;
  }

  function fill(entry, m, depth) {
    var vars = {};
    for (var i = 0; i < entry.names.length; i += 1) {
      var v = slot(entry.names[i], m[i + 1], depth);
      if (v === null) return null;
      vars[entry.names[i]] = v;
    }
    return tr(entry.key, vars);
  }

  // The translation of one server sentence, or null.
  function text(s, depth) {
    depth = depth || 0;
    if (!active() || typeof s !== "string" || !s || depth > 4) return null;
    var idx = build();
    if (Object.prototype.hasOwnProperty.call(idx.exact, s)) return tr(idx.exact[s]);
    for (var i = 0; i < idx.pats.length; i += 1) {
      var m = idx.pats[i].re.exec(s);
      if (m) { var out = fill(idx.pats[i], m, depth); if (out !== null) return out; }
    }
    for (var j = 0; j < idx.tails.length; j += 1) {
      var t = idx.tails[j].re.exec(s);
      if (t && t.index > 0) {
        var head = text(s.slice(0, t.index), depth + 1);
        var tail = head === null ? null : fill(idx.tails[j], t, depth);
        if (tail !== null) return head + tail;
      }
    }
    if (s.indexOf("; ") > 0) {
      var parts = s.split("; ").map(function (p) { return text(p, depth + 1); });
      if (parts.every(function (p) { return p !== null; })) return parts.join("; ");
    }
    return null;
  }

  // A refusal reason as the server writes it: lower case, no full stop; and a
  // patient finding lowered to read as a reason.
  function reason(s, depth) {
    if (!active() || typeof s !== "string" || !s) return null;
    return text(s, depth) || text(capFirst(s), depth) || text(capFirst(s) + ".", depth);
  }

  function coded(code) { return CODES.indexOf(code) >= 0 && has("reason." + code) ? tr("reason." + code) : null; }

  // The reason line for a refusal.
  function refusal(data) {
    if (!active() || !data) return null;
    if (data.reason_code === "patient_unsafe") {
      var block = (data.patient_findings || []).filter(function (f) { return f.severity === "block"; })[0];
      var said = block && text(block.message);
      if (said) return said;
    }
    return reason(data.refused_reason) || coded(data.reason_code);
  }

  function traceDetail(step) { return step ? text(step.detail) : null; }

  function explain(step) {
    if (!active() || !step || !step.explain) return null;
    var key = "srv.explain." + slug(step.name);
    return has(key) && en()[key] === step.explain ? tr(key) : null;
  }

  function finding(f) { return f ? text(f.message) : null; }

  // A refusal's answer text: every sentence known, or none of it. A known
  // message can run over several sentences, so the longest run is tried first.
  function answerText(s) {
    if (!active() || typeof s !== "string" || !s) return null;
    var sentences = s.replace(/([.!?])\s+/g, "$1\u0001").split("\u0001");
    var out = [];
    var i = 0;
    while (i < sentences.length) {
      var hit = null;
      for (var j = sentences.length; j > i && !hit; j -= 1) {
        var one = text(sentences.slice(i, j).join(" "));
        if (one !== null) hit = [one, j];
      }
      if (!hit) return null;
      out.push(hit[0]);
      i = hit[1];
    }
    return out.join(" ");
  }

  function example(ex) {
    if (!active() || !ex) return ex;
    var pick = function (kind, value) { var k = "example." + kind + "." + slug(value); return has(k) ? tr(k) : value; };
    var q = "example.query." + slug(ex.label);
    return {
      group: ex.group ? pick("group", ex.group) : ex.group,
      label: pick("label", ex.label),
      query: i18n().locale() !== "pseudo" && has(q) ? tr(q) : ex.query,
    };
  }

  root.gcLocalize = {
    active: active, text: text, reason: reason, refusal: refusal, traceDetail: traceDetail, explain: explain,
    finding: finding, kind: kindLabel, answerText: answerText, example: example, slug: slug,
  };
  if (typeof module !== "undefined" && module.exports) module.exports = root.gcLocalize;
})(typeof window !== "undefined" ? window : globalThis);

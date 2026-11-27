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
      var kind = /^(?:srv\.(exact|pat|tail)|msg\.(note))\./.exec(key);
      if (kind) kind[1] = kind[1] || "pat";
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

  // "a; b; c", where one reason may itself contain "; ": the longest run of
  // pieces that is a known sentence is taken first.
  function reasonList(value, depth) {
    var pieces = value.split("; "), out = [], i = 0;
    while (i < pieces.length) {
      var hit = null;
      for (var j = pieces.length; j > i && !hit; j -= 1) {
        var one = text(pieces.slice(i, j).join("; "), depth + 1);
        if (one !== null) hit = [one, j];
      }
      if (!hit) return null;
      out.push(hit[0]);
      i = hit[1];
    }
    return out.join("; ");
  }

  function slot(name, value, depth) {
    if (NESTED_VARS[name]) { var nested = name === "reason" ? reason(value, depth + 1) : text(value, depth + 1); return nested === null ? value : nested; }
    if (WORD_VARS[name]) return word(value);
    if (name === "list") return redactionList(value);
    if (name === "items") return missingItems(value);
    if (name === "phrase") return agePhrase(value);
    if (name === "reasons") return reasonList(value, depth);
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
      var said = block && (structured(block) || text(block.message));
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

  // Structured findings: written from the code and params, every number, unit,
  // medicine and condition copied from the params. A part the table has no
  // wording for gives null, and the caller falls back to the sentence table.
  function fmtG(v) {
    if (v === null || v === undefined) return "";
    if (typeof v !== "number") return String(v);
    return String(Number(v.toPrecision(6)));
  }
  var PHRASES = [
    ["range", /^(\d+(?:\.\d+)?) to (\d+(?:\.\d+)?) (day|week|month|year)s?/],
    ["unit", /^(\d+(?:\.\d+)?) (day|week|month|year)s?/],
    ["msg.pvp.times_daily", /^(\d+(?:\.\d+)?) times daily/],
    ["msg.pvp.every_hours", /^every (\d+(?:\.\d+)?) hours?/],
    ["msg.pvp.once_every_weeks", /^once every (\d+(?:\.\d+)?) weeks/],
    ["msg.pvp.once_every_days", /^once every (\d+(?:\.\d+)?) days/],
    ["quantity", /^\d[\d,]*(?:\.\d+)? (?:mg|mcg|g|mL|L|units)\b/],
  ];
  function unitWord(unit, n) { var k = "srv.unit." + unit + (n === "1" ? ".one" : ".other"); return has(k) ? tr(k) : null; }
  function phrase(value) {
    if (typeof value !== "string") return null;
    var table = en();
    var fixed = Object.keys(table).filter(function (k) { return k.indexOf("msg.pv.") === 0; })
      .map(function (k) { return [table[k], k]; }).sort(function (a, b) { return b[0].length - a[0].length; });
    var text = value.replace(/^\s+|\s+$/g, ""), out = "", pos = 0;
    while (pos < text.length) {
      if (text.charAt(pos) === " ") { out += " "; pos += 1; continue; }
      var rest = text.slice(pos), hit = false;
      for (var i = 0; i < PHRASES.length && !hit; i += 1) {
        var m = PHRASES[i][1].exec(rest);
        if (!m) continue;
        var kind = PHRASES[i][0];
        if (kind === "quantity") out += m[0];
        else if (kind === "unit") { var w = unitWord(m[2], m[1]); if (w === null) return null; out += m[1] + " " + w; }
        else if (kind === "range") { var w2 = unitWord(m[3], m[2]); if (w2 === null) return null; out += tr("msg.pvp.range", { a: m[1], b: m[2], unit: w2 }); }
        else out += tr(kind, { n: m[1] });
        pos += m[0].length;
        hit = true;
      }
      if (hit) continue;
      for (var j = 0; j < fixed.length && !hit; j += 1) {
        var word = fixed[j][0], nextc = rest.charAt(word.length);
        if (rest.indexOf(word) === 0 && (!nextc || !/[A-Za-z0-9]/.test(nextc))) { out += tr(fixed[j][1]); pos += word.length; hit = true; }
      }
      if (!hit) return null;
    }
    return out;
  }

  function wordOf(prefix, value) { var k = prefix + "." + slug(value).replace(/-/g, "_"); return has(k) ? tr(k) : null; }
  function whereOf(code, p) {
    if (code.indexOf("hepatic") === 0) return tr("msg.where.liver", { "class": p.child_pugh });
    return tr(p.measure === "eGFR" ? "msg.where.egfr" : "msg.where.crcl", { value: fmtG(p.value) });
  }
  function noteOf(p) { return p.note ? text(p.note) : null; }

  function structured(f) {
    var p = f.params || {}, code = f.code, sev = f.severity;
    var sub = has("msg.subject." + (p.subject || "answer")) ? tr("msg.subject." + (p.subject || "answer")) : null;
    var base = { medicine: p.medicine || "" };
    if (sub) { base.subject = sub; base.Subject = capFirst(sub); }
    var n, w;
    if (code.indexOf("interaction_") === 0) {
      if (p.note) { n = noteOf(p); return n === null ? null : tr("msg.find.interaction", { medicine: base.medicine, other: p.other, note: n }); }
      w = wordOf("msg.sev", p.severity);
      return w === null ? null : tr("msg.find.interaction_sev", { medicine: base.medicine, other: p.other, severity: w });
    }
    if (code === "duplicate_class") {
      w = wordOf("msg.class", p["class"]);
      return w === null ? null : tr("msg.find.duplicate_class", { medicine: base.medicine, other: p.other, "class": w });
    }
    if (code.indexOf("lab_") === 0) {
      n = noteOf(p);
      var lk = "msg.lab." + p.lab;
      if (n === null || !has(lk)) return null;
      return tr("msg.find.lab", { note: n, sentence: tr(lk, { value: fmtG(p.value) }) });
    }
    if (code === "renal" || code === "renal_caution" || code === "hepatic" || code === "hepatic_caution") {
      n = noteOf(p);
      if (n === null) return null;
      w = whereOf(code, p);
      return tr("msg.find.rule", { note: n, sentence: capFirst(w) + "." });
    }
    if (code === "renal_dose" || code === "hepatic_dose") {
      n = noteOf(p);
      var dose = phrase(p.dose || "");
      if (n === null || dose === null || !sub) return null;
      w = whereOf(code, p);
      var high = p.stated !== null && p.stated !== undefined
        ? tr("msg.find.high", { subject: base.subject, Subject: base.Subject, stated: fmtG(p.stated), unit: p.unit, where: w }) : "";
      return tr("msg.find.rule_dose", { high: high, note: n, dose: dose });
    }
    if (code === "paediatric_dose") {
      var fq = phrase(p.frequency || "");
      if (fq === null || !sub) return null;
      var cap = p.max_single !== null && p.max_single !== undefined ? tr("msg.calc.child_cap", { max_single: fmtG(p.max_single), unit: p.unit }) : "";
      var calc = tr("msg.calc.child", { per_kg: fmtG(p.per_kg), unit: p.unit, weight: fmtG(p.weight), uncapped: fmtG(p.uncapped), cap: cap });
      return tr(sev === "block" ? "msg.find.paed_block" : "msg.find.paed_info",
        { subject: base.subject, Subject: base.Subject, amount: fmtG(p.amount), unit: p.unit, frequency: fq, calc: calc });
    }
    if (code === "weight_dose" || code === "dose_above_maximum") {
      if (code === "dose_above_maximum" && Object.prototype.hasOwnProperty.call(p, "dose")) {
        var d = phrase(p.dose);
        if (d === null || !sub) return null;
        return tr("msg.find.above_adult", { subject: base.subject, Subject: base.Subject, stated: fmtG(p.stated), unit: p.unit, medicine: base.medicine, dose: d });
      }
      var f2 = phrase(p.frequency || "");
      if (f2 === null) return null;
      var cap2 = p.max_single !== null && p.max_single !== undefined ? tr("msg.calc.weight_cap", { max_single: fmtG(p.max_single), unit: p.unit }) : "";
      var calculated = tr("msg.find.weight_dose", { per_kg: fmtG(p.per_kg), unit: p.unit, weight: fmtG(p.weight), amount: fmtG(p.amount), frequency: f2, cap: cap2 });
      if (code === "weight_dose") return calculated;
      if (!sub) return null;
      return tr("msg.find.above_weight", { subject: base.subject, Subject: base.Subject, stated: fmtG(p.stated), unit: p.unit, medicine: base.medicine, calculated: calculated });
    }
    return null;
  }

  // The proposal verdict, written from the dimensions of the check.
  function partPhrase(dims, field) {
    var parts = [];
    for (var i = 0; i < dims.length; i += 1) {
      var d = dims[i], v = field === "proposed" ? d.proposed : d.documented;
      if (v === null || v === undefined) continue;
      var out;
      if (d.name === "max_daily") {
        var shown = phrase(field === "documented" ? v.replace(" a day", "") : v);
        if (shown === null) return null;
        out = tr(field === "documented" ? "msg.part.max_documented" : "msg.part.max_proposed", { value: shown });
      } else {
        var pv = phrase(v);
        if (pv === null) return null;
        out = pv;
        if (d.name === "duration") out = tr("msg.part.duration", { value: pv });
        else if (d.name === "route") out = tr(dims.length > 1 ? "msg.part.route_many" : "msg.part.route_one", { value: pv });
        else if (d.name === "timing" && dims.length > 1) out = tr("msg.part.timing_many", { value: pv });
      }
      parts.push(out);
    }
    return parts.join(" ");
  }

  function verdict(data) {
    var check = data && data.proposal_check;
    if (!active() || !check || check.status === "unverifiable" || data.decision !== "answer") return null;
    var hedged = /^(?:Yes|No) — the proposed regimen/.test(data.answer_text || "");
    var checked = check.dimensions.filter(function (d) { return d.status !== "unverifiable" && d.documented !== null && d.documented !== undefined; });
    if (check.status === "match") {
      var a = partPhrase(checked.filter(function (d) { return d.name !== "max_daily"; }), "proposed");
      var b = partPhrase(checked.filter(function (d) { return d.name === "max_daily"; }), "documented");
      if (a === null || b === null) return null;
      return tr(hedged ? "msg.verdict.yes_hedged" : "msg.verdict.yes", { what: [a, b].filter(Boolean).join(" ") });
    }
    var documented = partPhrase(checked, "documented"), proposed = partPhrase(checked, "proposed");
    if (documented === null || proposed === null) return null;
    return tr(hedged ? "msg.verdict.no_hedged" : "msg.verdict.no", { documented: documented, proposed: proposed });
  }

  // One cell of the dimension table: a dose, frequency, duration, route or timing in the language; a name as sent.
  function value(name, v) {
    if (v === null || v === undefined || !active() || name === "drug") return v;
    var out = phrase(v);
    return out === null ? v : out;
  }

  // Monitoring: an alert's title and detail from its rule and params.
  function pluralKey(base, n) { return base + (String(n) === "1" ? ".one" : ".other"); }
  function alertText(a) {
    if (!a || !active() || !a.params) return null;
    var p = a.params, rule = a.rule, k = "msg.alert." + rule + ".";
    if (!has(k + "title") && !has(k + "title.expired")) return null;
    var title = null, detail = null;
    if (rule === "refusal_rate") {
      detail = tr(k + "detail", { rate: p.rate, n: p.n, minutes: p.minutes, base: p.base });
    } else if (rule === "latency") {
      detail = tr(k + "detail", { minutes: p.minutes, seconds: p.seconds, limit: p.limit });
    } else if (rule === "retrieval_drift") {
      detail = tr(k + "detail", { psi: p.psi, threshold: p.threshold, lower: p.lower ? tr(k + "lower") : "" });
    } else if (rule === "review_overdue") {
      detail = tr(pluralKey(k + "detail", p.n), { n: p.n });
    } else if (rule === "documents_expiring") {
      var more = p.more > 0 ? tr(k + "more", { n: p.more }) : "";
      var state = p.expired > 0 ? "expired" : "soon";
      title = tr(k + "title." + state);
      detail = tr(pluralKey(k + "detail." + state, state === "expired" ? p.expired : p.total),
        { n: state === "expired" ? p.expired : p.total, days: p.days, names: p.names, more: more });
    } else if (rule === "audit_chain") {
      if (p.error) return null;
      var pk = k + "problem." + slug(String(p.problem || "").replace(/\.$/, ""));
      if (!has(pk)) return null;
      detail = tr(k + "detail", { at: p.seq !== null && p.seq !== undefined ? tr(k + "at", { seq: p.seq }) : "", problem: tr(pk) });
    } else if (rule === "server_errors") {
      detail = tr(k + "detail", { share: p.share, n: p.n });
    } else if (rule === "imaging_failures") {
      detail = tr(pluralKey(k + "detail", p.n), { n: p.n });
    } else if (rule === "release_blocked") {
      var rk = "msg.release.reason." + slug(p.reason || "");
      if (!has(rk)) return null;
      var dk = p.unsafe ? pluralKey(k + "detail", p.unsafe) : k + "detail.unchecked";
      detail = tr(dk, { number: p.number, reason: tr(rk), unsafe: p.unsafe });
    } else return null;
    return { title: title || tr(k + "title"), detail: detail };
  }

  function ruleText(r) {
    if (!r || !active() || !r.params || !has("msg.rule." + r.rule + ".name")) return null;
    return { name: tr("msg.rule." + r.rule + ".name"), limit: tr("msg.rule." + r.rule + ".limit", r.params) };
  }

  // A rule's source label ("Formulary"), when the table has wording for it.
  function sourceLabel(label) {
    if (!active() || typeof label !== "string") return null;
    var k = "msg.source." + slug(label);
    return has(k) ? tr(k) : null;
  }

  function finding(f) {
    if (!f || !active()) return null;
    return structured(f) || text(f.message);
  }

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
    finding: finding, structured: structured, phrase: phrase, verdict: verdict, value: value, alert: alertText, rule: ruleText, source: sourceLabel, kind: kindLabel, answerText: answerText, example: example, slug: slug,
  };
  if (typeof module !== "undefined" && module.exports) module.exports = root.gcLocalize;
})(typeof window !== "undefined" ? window : globalThis);

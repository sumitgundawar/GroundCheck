// Interface strings. The English text is in the page itself; this loads
// /i18n/en.json and the chosen locale's file, and swaps in the locale's string
// for every element that carries a key. A key the locale lacks shows English.
(function () {
  var KEY = "gc-locale";
  var LOCALES = { en: "English", es: "Español" };
  var strings = { en: {}, current: {} };
  var locale = "en";
  var pseudo = false;

  function stored() {
    try {
      var fromUrl = new URLSearchParams(location.search).get("locale");
      if (fromUrl) return fromUrl;
      return localStorage.getItem(KEY) || "en";
    } catch (e) { return "en"; }
  }
  function squash(s) { return String(s).replace(/\s+/g, " ").trim(); }

  var ACCENT = { a: "á", e: "é", i: "í", o: "ó", u: "ú", c: "ç", n: "ñ", y: "ý", A: "Á", E: "É", I: "Í", O: "Ó", U: "Ú" };
  function pseudoText(s) {
    var out = String(s).split(/(\{\w+\})/).map(function (part) {
      return /^\{\w+\}$/.test(part) ? part : part.replace(/[A-Za-z]/g, function (c) { return ACCENT[c] || c; });
    }).join("");
    var pad = Math.ceil(out.length * 0.4);
    return "[" + out + " " + new Array(Math.ceil(pad / 3) + 1).join("~· ").trim() + "]";
  }

  function lookup(table, key) { return Object.prototype.hasOwnProperty.call(table, key) ? table[key] : undefined; }

  function fill(value, vars) {
    if (!vars) return value;
    return String(value).replace(/\{(\w+)\}/g, function (whole, name) {
      return Object.prototype.hasOwnProperty.call(vars, name) ? String(vars[name]) : whole;
    });
  }

  // t(key, fallback, vars): the locale's string, else English, else the fallback,
  // with {name} filled from vars. Values are inserted as text, never as markup.
  function t(key, fallback, vars) {
    var value = pseudo ? lookup(strings.en, key) : lookup(strings.current, key);
    if (value === undefined) value = lookup(strings.en, key);
    if (value === undefined) value = fallback !== undefined ? fallback : key;
    return fill(pseudo ? pseudoText(value) : value, vars);
  }

  function apply(root) {
    root = root || document;
    var previous = window.gcI18n._previous || strings.en;
    root.querySelectorAll("[data-i18n]").forEach(function (el) {
      var key = el.getAttribute("data-i18n");
      var was = lookup(previous, key);
      if (was !== undefined && squash(el.textContent) !== squash(was) && squash(el.textContent) !== squash(lookup(strings.en, key) || "")) return;
      var value = t(key, el.textContent);
      if (el.textContent !== value) el.textContent = value;
    });
    root.querySelectorAll("[data-i18n-attr]").forEach(function (el) {
      el.getAttribute("data-i18n-attr").split(";").forEach(function (pair) {
        var i = pair.indexOf(":");
        if (i < 0) return;
        var name = pair.slice(0, i), key = pair.slice(i + 1);
        el.setAttribute(name, t(key, el.getAttribute(name)));
      });
    });
    window.gcI18n._previous = pseudo ? strings.en : Object.assign({}, strings.en, strings.current);
    document.documentElement.lang = pseudo ? "en" : locale;
    var title = lookup(strings.en, "app.title");
    if (title !== undefined) document.title = t("app.title", document.title);
  }

  function fetchJson(code) {
    return fetch("/i18n/" + code + ".json", { cache: "no-cache" }).then(function (r) {
      if (!r.ok) throw new Error(code);
      return r.json();
    });
  }

  function load(code) {
    pseudo = code === "pseudo";
    locale = pseudo || LOCALES[code] ? code : "en";
    var wanted = pseudo || locale === "en" ? Promise.resolve({}) : fetchJson(locale).catch(function () { locale = "en"; return {}; });
    return wanted.then(function (table) {
      strings.current = table;
      syncSelects();
      apply(document);
      document.dispatchEvent(new CustomEvent("localechange", { detail: { locale: locale } }));
    });
  }

  function syncSelects() {
    document.querySelectorAll(".locale-select").forEach(function (sel) { sel.value = pseudo ? "en" : locale; });
  }

  function setLocale(code) {
    try { localStorage.setItem(KEY, code); } catch (e) { /* this page only */ }
    return load(code);
  }

  window.gcI18n = {
    t: t,
    en: function () { return strings.en; },
    apply: apply,
    locale: function () { return pseudo ? "pseudo" : locale; },
    locales: LOCALES,
    setLocale: setLocale,
    _previous: null,
    ready: null,
  };
  window.t = t;

  document.addEventListener("DOMContentLoaded", function () {
    document.querySelectorAll(".locale-select").forEach(function (sel) {
      sel.addEventListener("change", function () { setLocale(sel.value); });
    });
    window.gcI18n.ready = fetchJson("en").then(function (en) {
      strings.en = en;
      window.gcI18n._previous = en;
      return load(stored());
    }).catch(function () { /* the page is already in English */ });
  });
})();

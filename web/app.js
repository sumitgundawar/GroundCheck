"use strict";

const $ = (id) => document.getElementById(id);
const REDUCED_MOTION = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

let DEFAULTS = null;
let llmAvailable = false;
let lastRetrievedIds = [];

// Static stage list for the "How it works" panel. The authoritative,
// per-run descriptions come from the trace itself; this mirrors them for the
// explainer shown before any question is asked.
const PIPELINE_STAGES = [
  ["input guards", "Redact PII, reject injection and out-of-scope input, rate limit."],
  ["embed + retrieve", "Embed the query and rank passages by a blend of embedding similarity and keyword match, with cosine scores."],
  ["retrieval gate", "If the best passage is below threshold, refuse now, before any generation."],
  ["source coverage", "Check the question's entities actually appear in the retrieved sources."],
  ["generate", "LLM returns structured, cited claims. Extractive fallback if no LLM is available."],
  ["schema validate", "Validate the output against a strict schema. Retry once, then fall back."],
  ["grounding check", "Deterministic claim-to-source similarity, with an LLM judge as corroboration."],
  ["dosage guard", "Every value with a clinical unit must appear verbatim in a source, or refuse."],
  ["decision", "Answer with citations, or refuse with a specific reason and route for review."],
];

// Plain-language explanation shown in the refusal callout, keyed by a stable
// phrase found in the refusal reason.
function refusalExplanation(reason) {
  const r = (reason || "").toLowerCase();
  if (r.includes("not appear in any trusted source") || r.includes("do not cover"))
    return "The question names something the trusted sources never mention, so there is nothing to ground an answer on.";
  if (r.includes("no sufficiently relevant source"))
    return "No document was similar enough to the question to be trustworthy, so the system refused before generating anything.";
  if (r.includes("not supported by any source"))
    return "The answer contained a value with a clinical unit that does not appear, character for character, in any retrieved source.";
  if (r.includes("could not be grounded"))
    return "A statement in the draft answer was not sufficiently supported by its cited source.";
  if (r.includes("enough information"))
    return "The retrieved sources did not contain enough to answer safely.";
  if (r.includes("input guard") || r.includes("blocked"))
    return "The query was stopped by an input guard before reaching the pipeline.";
  if (r.includes("rate limit"))
    return "Too many requests in a short window. This protects the live service.";
  return "The pipeline could not produce a fully grounded answer, so it declined.";
}

// ---------- Boot ----------
document.addEventListener("DOMContentLoaded", async () => {
  loadLogo();
  loadHealth(); // public, so the header is complete even on the sign-in screen
  wireAccounts();
  const ready = await checkSession();
  if (ready) startDashboard();
});

let dashboardStarted = false;

function startDashboard() {
  if (dashboardStarted) return;
  dashboardStarted = true;
  loadLocalAI();
  loadExamples();
  loadSettings();
  loadEvalSummary();
  loadCorpus();
  renderHowStages();
  wireForm();
  wireCollapsibles();
  wireNavigation();
}

// ---------- Accounts ----------
const account = { user: null, authRequired: false };

async function api(path, options = {}) {
  const res = await fetch(path, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    body: options.body === undefined ? undefined : JSON.stringify(options.body),
  });
  let data = null;
  try { data = await res.json(); } catch (_) { /* empty body */ }
  if (res.status === 401 && account.authRequired && !path.startsWith("/api/auth/")) {
    showAuthScreen("login", "Your session ended. Sign in again.");
  }
  if (!res.ok) {
    const detail = data && typeof data.detail === "string" ? data.detail : `Request failed (${res.status}).`;
    throw new Error(detail);
  }
  return data;
}

// Returns true when the dashboard can be shown.
async function checkSession() {
  let me;
  try { me = await api("/api/auth/me"); }
  catch (_) { return true; }
  account.authRequired = me.auth_required;
  account.user = me.user;
  renderAccountMenu();
  if (!me.auth_required || me.user) {
    hideAuthScreen();
    return true;
  }
  if (me.needs_first_admin) showAuthScreen("first-admin");
  else if (me.mfa_pending) showAuthScreen("mfa");
  else showAuthScreen("login");
  return false;
}

function showAuthScreen(mode, message) {
  const titles = {
    login: ["Sign in", "Sign in to ask questions and review answers."],
    mfa: ["Two-factor authentication", "Enter the code from your authenticator app to finish signing in."],
    "first-admin": ["Create the first admin", "No accounts exist yet. The first account is an admin, who can add everyone else."],
  };
  $("dashboard").hidden = true;
  $("auth-screen").hidden = false;
  $("auth-title").textContent = titles[mode][0];
  $("auth-lede").textContent = titles[mode][1];
  $("login-form").hidden = mode !== "login";
  $("mfa-form").hidden = mode !== "mfa";
  $("first-admin-form").hidden = mode !== "first-admin";
  setAuthError(message || "");
  const first = { login: "login-email", mfa: "mfa-code", "first-admin": "admin-name" }[mode];
  requestAnimationFrame(() => $(first).focus());
}

function hideAuthScreen() {
  $("auth-screen").hidden = true;
  $("dashboard").hidden = false;
}

function setAuthError(text) {
  $("auth-error").textContent = text;
  $("auth-error").hidden = !text;
}

async function signedIn(user) {
  account.user = user;
  renderAccountMenu();
  hideAuthScreen();
  startDashboard();
}

function renderAccountMenu() {
  const user = account.user;
  $("account-menu").hidden = !user;
  if (!user) return;
  $("account-label").textContent = user.name || user.email;
  $("account-email").textContent = user.email;
  $("account-role").textContent = user.role;
  const initials = (user.name || user.email).split(/[\s@.]+/).filter(Boolean).slice(0, 2).map((w) => w[0].toUpperCase()).join("");
  $("account-avatar").textContent = initials;
  applyRoleNavigation();
}

// Which pages a person sees depends on their role when sign-in is required.
// Without sign-in (the local demo), every page is shown; the server still
// decides what each request may do.
function canSee(view) {
  const role = account.user?.role;
  if (!account.authRequired) return view !== "users";
  const rank = { clinician: 0, reviewer: 1, admin: 2 }[role] ?? -1;
  const needs = { usage: 1, review: 1, documents: 1, "data-protection": 1, users: 2, "local-ai": 0 }[view] ?? 0;
  return rank >= needs;
}

function applyRoleNavigation() {
  document.querySelectorAll(".nav-link").forEach((link) => { link.hidden = !canSee(link.dataset.view); });
  document.querySelectorAll(".nav-group").forEach((group) => {
    group.hidden = ![...group.querySelectorAll(".nav-link")].some((l) => !l.hidden);
  });
  $("hazard-add").hidden = account.authRequired && account.user?.role !== "admin";
  if (dashboardStarted) route();
}

function canEditHazards() {
  return !account.authRequired || account.user?.role === "admin";
}

function formBusy(form, busy) {
  form.querySelectorAll("button, input, select").forEach((el) => { el.disabled = busy; });
}

function wireAccounts() {
  $("login-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const form = e.currentTarget;
    formBusy(form, true);
    try {
      const data = await api("/api/auth/login", { method: "POST", body: { email: $("login-email").value, password: $("login-password").value } });
      $("login-password").value = "";
      if (data.mfa_required) showAuthScreen("mfa");
      else await signedIn(data.user);
    } catch (err) { setAuthError(err.message); }
    formBusy(form, false);
  });

  $("mfa-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const form = e.currentTarget;
    formBusy(form, true);
    try {
      const data = await api("/api/auth/mfa", { method: "POST", body: { code: $("mfa-code").value } });
      $("mfa-code").value = "";
      await signedIn(data.user);
    } catch (err) { setAuthError(err.message); }
    formBusy(form, false);
  });

  $("mfa-cancel").addEventListener("click", async () => {
    await api("/api/auth/logout", { method: "POST" }).catch(() => {});
    showAuthScreen("login");
  });

  $("first-admin-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const form = e.currentTarget;
    formBusy(form, true);
    try {
      const data = await api("/api/auth/first-admin", { method: "POST", body: {
        name: $("admin-name").value, email: $("admin-email").value, password: $("admin-password").value,
      } });
      $("admin-password").value = "";
      await signedIn(data.user);
    } catch (err) { setAuthError(err.message); }
    formBusy(form, false);
  });

  // Account menu
  const button = $("account-button"), dropdown = $("account-dropdown");
  const setMenu = (open) => { dropdown.hidden = !open; button.setAttribute("aria-expanded", String(open)); };
  button.addEventListener("click", () => setMenu(dropdown.hidden));
  document.addEventListener("click", (e) => { if (!$("account-menu").contains(e.target)) setMenu(false); });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") setMenu(false); });

  $("sign-out").addEventListener("click", async () => {
    await api("/api/auth/logout", { method: "POST" }).catch(() => {});
    account.user = null;
    if (account.authRequired) window.location.reload();
    else { renderAccountMenu(); setMenu(false); }
  });

  $("open-account").addEventListener("click", () => { setMenu(false); openAccountDialog(); });
  document.querySelectorAll("dialog [data-close]").forEach((b) => b.addEventListener("click", () => b.closest("dialog").close()));

  $("password-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const form = e.currentTarget;
    formBusy(form, true);
    try {
      await api("/api/auth/password", { method: "POST", body: { current_password: $("current-password").value, new_password: $("new-password").value } });
      form.reset();
      showMessage("account-message", "Password changed. Other sessions were signed out.");
    } catch (err) { showMessage("account-message", err.message, true); }
    formBusy(form, false);
  });

  $("twofa-start").addEventListener("click", async () => {
    try {
      const data = await api("/api/auth/mfa/setup", { method: "POST" });
      $("twofa-qr").src = data.qr_svg_data_uri;
      $("twofa-secret").textContent = data.secret;
      $("twofa-setup").hidden = false;
      $("twofa-start").hidden = true;
      $("twofa-code").focus();
    } catch (err) { showMessage("account-message", err.message, true); }
  });

  $("twofa-confirm-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const form = e.currentTarget; // currentTarget is null once an await has run
    try {
      await api("/api/auth/mfa/confirm", { method: "POST", body: { code: $("twofa-code").value } });
      account.user = { ...account.user, mfa_enabled: true };
      form.reset();
      renderTwoFactor();
      showMessage("account-message", "Two-factor authentication is on.");
    } catch (err) { showMessage("account-message", err.message, true); }
  });

  $("twofa-disable-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const form = e.currentTarget;
    try {
      await api("/api/auth/mfa/disable", { method: "POST", body: { password: $("twofa-disable-password").value } });
      account.user = { ...account.user, mfa_enabled: false };
      form.reset();
      renderTwoFactor();
      showMessage("account-message", "Two-factor authentication is off.");
    } catch (err) { showMessage("account-message", err.message, true); }
  });

  $("add-user-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const form = e.currentTarget;
    formBusy(form, true);
    try {
      await api("/api/users", { method: "POST", body: {
        name: $("new-user-name").value, email: $("new-user-email").value,
        password: $("new-user-password").value, role: $("new-user-role").value,
      } });
      form.reset();
      await loadUsers();
      showMessage("users-message", "User added. Share the temporary password with them securely.");
    } catch (err) { showMessage("users-message", err.message, true); }
    formBusy(form, false);
  });
}

function showMessage(id, text, isError = false) {
  const el = $(id);
  el.textContent = text;
  el.hidden = false;
  el.classList.toggle("error", isError);
}

function renderTwoFactor() {
  const on = !!account.user?.mfa_enabled;
  $("twofa-status").textContent = on ? "On. You’ll be asked for a code when you sign in." : "Off.";
  $("twofa-start").hidden = on;
  $("twofa-setup").hidden = true;
  $("twofa-disable-form").hidden = !on;
}

function openAccountDialog() {
  $("account-message").hidden = true;
  renderTwoFactor();
  $("account-dialog").showModal();
}

async function loadUsers() {
  const body = $("users-body");
  let data;
  try { data = await api("/api/users"); }
  catch (err) { showMessage("users-message", err.message, true); return; }
  body.textContent = "";
  data.users.forEach((u) => {
    const tr = document.createElement("tr");

    const who = document.createElement("td");
    const name = document.createElement("div");
    name.className = "user-name";
    name.textContent = u.name || u.email;
    const email = document.createElement("div");
    email.className = "user-email mono";
    email.textContent = u.email;
    who.append(name, email);

    const roleCell = document.createElement("td");
    const select = document.createElement("select");
    select.setAttribute("aria-label", `Role for ${u.email}`);
    ["clinician", "reviewer", "admin"].forEach((r) => {
      const opt = document.createElement("option");
      opt.value = r; opt.textContent = r; opt.selected = r === u.role;
      select.appendChild(opt);
    });
    select.addEventListener("change", () => updateUser(u.id, { role: select.value }, () => { select.value = u.role; }));
    roleCell.appendChild(select);

    const mfa = document.createElement("td");
    mfa.textContent = u.mfa_enabled ? "On" : "Off";

    const last = document.createElement("td");
    last.className = "mono";
    last.textContent = u.last_login_at ? new Date(u.last_login_at).toLocaleString() : "Never";

    const status = document.createElement("td");
    const toggle = document.createElement("button");
    toggle.type = "button";
    toggle.className = "btn btn-sm btn-ghost";
    toggle.textContent = u.is_active ? "Deactivate" : "Reactivate";
    toggle.disabled = account.user && u.id === account.user.id;
    toggle.addEventListener("click", () => updateUser(u.id, { is_active: !u.is_active }));
    const deleted = u.email.endsWith("@deleted.invalid");
    status.append(document.createTextNode(deleted ? "Deleted " : u.is_active ? "Active " : "Inactive "));
    if (!deleted) {
      status.appendChild(toggle);
      const remove = document.createElement("button");
      remove.type = "button";
      remove.className = "btn btn-sm btn-ghost";
      remove.textContent = "Delete";
      remove.disabled = account.user && u.id === account.user.id;
      remove.addEventListener("click", () => deleteUser(u));
      status.append(" ", remove);
    } else {
      select.disabled = true;
    }

    tr.append(who, roleCell, mfa, last, status);
    body.appendChild(tr);
  });
}

async function deleteUser(u) {
  if (!window.confirm(`Delete ${u.email}? Their name, email and sign-in details are removed. Their audit records stay, under an anonymous account number.`)) return;
  try {
    await api(`/api/users/${u.id}`, { method: "DELETE" });
    showMessage("users-message", "Deleted.");
  } catch (err) { showMessage("users-message", err.message, true); }
  await loadUsers();
}

async function updateUser(id, changes, revert) {
  try {
    await api(`/api/users/${id}`, { method: "PATCH", body: changes });
    showMessage("users-message", "Saved.");
  } catch (err) {
    if (revert) revert();
    showMessage("users-message", err.message, true);
  }
  await loadUsers();
}

async function loadLogo() {
  try { $("brand-mark").innerHTML = await (await fetch("/assets/logo.svg")).text(); }
  catch (_) {}
}

async function loadHealth() {
  const dot = $("provider-dot"), label = $("provider-label");
  try {
    const data = await (await fetch("/api/health")).json();
    llmAvailable = !!data.llm;
    const provider = data.provider;
    if (provider && provider.kind === "local") { dot.className = "dot ok"; label.textContent = `llm: local ${provider.model}`; }
    else if (provider) { dot.className = "dot ok"; label.textContent = `llm: ${provider.model}`; }
    else { dot.className = "dot warn"; label.textContent = "llm: extractive"; }
    if (typeof data.corpus === "number") $("corpus-count").textContent = data.corpus.toLocaleString();
  } catch (_) { dot.className = "dot warn"; label.textContent = "llm: extractive"; }
}

// ---------- Examples (grouped) ----------
async function loadExamples() {
  try {
    const examples = await (await fetch("/api/examples")).json();
    const wrap = $("chips");
    wrap.innerHTML = "";
    const groups = [];
    const byGroup = {};
    examples.forEach((ex) => {
      const g = ex.group || "Examples";
      if (!byGroup[g]) { byGroup[g] = []; groups.push(g); }
      byGroup[g].push(ex);
    });
    groups.forEach((g) => {
      const row = document.createElement("div");
      row.className = "chip-group";
      const label = document.createElement("span");
      label.className = "chip-group-label eyebrow";
      label.textContent = g;
      row.appendChild(label);
      const chips = document.createElement("div");
      chips.className = "chips-row";
      byGroup[g].forEach((ex) => {
        const chip = document.createElement("button");
        chip.type = "button";
        chip.className = "chip";
        chip.textContent = ex.label;
        chip.title = ex.query;
        chip.addEventListener("click", () => { $("query").value = ex.query; submitQuery(ex.query); });
        chips.appendChild(chip);
      });
      row.appendChild(chips);
      wrap.appendChild(row);
    });
  } catch (_) {}
}

// ---------- Tuning ----------
const RANGE_CONTROLS = [
  { id: "set-retrieval", key: "retrieval_min_score", out: "set-retrieval-val", fixed: 2 },
  { id: "set-grounding", key: "grounding_min", out: "set-grounding-val", fixed: 2 },
  { id: "set-topk", key: "top_k", out: "set-topk-val", fixed: 0 },
  { id: "set-temperature", key: "temperature", out: "set-temperature-val", fixed: 1 },
];
const TOGGLE_CONTROLS = [
  { id: "set-extractive", key: "force_extractive" },
  { id: "set-pii", key: "enable_pii_redaction" },
  { id: "set-injection", key: "enable_injection_guard" },
  { id: "set-coverage", key: "enable_coverage_guard" },
  { id: "set-grounding-guard", key: "enable_grounding_guard" },
  { id: "set-dosage", key: "enable_dosage_guard" },
  { id: "set-judge", key: "use_llm_judge" },
];

async function loadSettings() {
  try {
    const data = await (await fetch("/api/settings")).json();
    DEFAULTS = data.defaults;
    llmAvailable = !!data.llm_available;
    RANGE_CONTROLS.forEach((c) => {
      const el = $(c.id);
      const b = (data.bounds || {})[c.key];
      if (b) { el.min = b.min; el.max = b.max; el.step = b.step; }
      el.value = DEFAULTS[c.key];
      syncRangeOutput(c);
      el.addEventListener("input", () => { syncRangeOutput(c); updateTuningStatus(); });
    });
    TOGGLE_CONTROLS.forEach((c) => {
      const el = $(c.id);
      el.checked = !!DEFAULTS[c.key];
      el.addEventListener("change", updateTuningStatus);
    });
    if (!llmAvailable) {
      const sw = $("judge-switch");
      sw.classList.add("disabled");
      sw.title = "Available only when a live LLM provider is configured.";
      $("set-judge").checked = false;
      $("set-judge").disabled = true;
    }
    $("tuning-reset").addEventListener("click", resetSettings);
    updateTuningStatus();
  } catch (_) {}
}

function syncRangeOutput(c) { $(c.out).textContent = Number($(c.id).value).toFixed(c.fixed); }

function readSettings() {
  const s = {};
  RANGE_CONTROLS.forEach((c) => { s[c.key] = c.fixed === 0 ? parseInt($(c.id).value, 10) : parseFloat($(c.id).value); });
  TOGGLE_CONTROLS.forEach((c) => { s[c.key] = $(c.id).checked; });
  return s;
}

function isDefault() {
  if (!DEFAULTS) return true;
  const s = readSettings();
  return Object.keys(s).every((k) => {
    // Without a live model the judge toggle is forced off and can't be changed,
    // so it is not a user modification.
    if (k === "use_llm_judge" && !llmAvailable) return true;
    return typeof s[k] === "number"
      ? Math.abs(s[k] - DEFAULTS[k]) < 1e-9 : s[k] === DEFAULTS[k];
  });
}

function updateTuningStatus() {
  const el = $("tuning-status");
  if (isDefault()) { el.textContent = "defaults"; el.classList.remove("modified"); }
  else { el.textContent = "modified"; el.classList.add("modified"); }
}

function resetSettings() {
  if (!DEFAULTS) return;
  RANGE_CONTROLS.forEach((c) => { $(c.id).value = DEFAULTS[c.key]; syncRangeOutput(c); });
  TOGGLE_CONTROLS.forEach((c) => {
    if (c.key === "use_llm_judge" && !llmAvailable) return;
    $(c.id).checked = !!DEFAULTS[c.key];
  });
  updateTuningStatus();
}

// ---------- Ask ----------
function wireForm() {
  $("ask-form").addEventListener("submit", (e) => {
    e.preventDefault();
    const q = $("query").value.trim();
    if (q) submitQuery(q);
  });
}

let inflight = false;
async function submitQuery(query) {
  if (inflight) return;
  inflight = true;
  setLoading(true);
  try {
    const res = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, settings: readSettings() }),
    });
    render(await res.json());
    refreshReviewCount();
  } catch (_) {
    render({ decision: "refuse", answer_text: "The service is unreachable. Please try again.",
      refused_reason: "network error", claims: [], sources: [], trace: [], audit_id: "", total_ms: 0, llm_used: false });
  } finally { setLoading(false); inflight = false; }
}

function setLoading(on) {
  $("ask-btn").disabled = on;
  $("spinner").hidden = !on;
  $("ask-btn").querySelector(".btn-label").textContent = on ? "Asking" : "Ask";
}

// ---------- Render ----------
function render(data) {
  renderDecision(data);
  renderSources(data);
  renderTrace(data.trace || []);
  renderAudit(data.audit_id);
  lastRetrievedIds = (data.sources || []).map((s) => s.id);
  highlightCorpus(lastRetrievedIds);
  $("results-grid").hidden = false;
  $("decision-panel").scrollIntoView({ behavior: REDUCED_MOTION ? "auto" : "smooth", block: "nearest" });
}

function renderDecision(data) {
  $("decision-panel").hidden = false;
  const chip = $("decision-chip"), reason = $("decision-reason");
  const body = $("answer-body"), callout = $("refuse-callout");
  body.innerHTML = "";
  if (data.decision === "answer") {
    chip.className = "status-chip answer";
    chip.textContent = "ANSWER";
    const ids = citedIds(data);
    reason.textContent = ids.length ? `Grounded in ${ids.join(", ")}.` : "Grounded in cited sources.";
    body.appendChild(renderAnswerWithCitations(data.answer_text));
    callout.hidden = true;
    resetFlag(data.audit_id);
  } else {
    chip.className = "status-chip refuse";
    chip.textContent = "REFUSED";
    resetFlag(null);
    reason.textContent = data.refused_reason ? capitalize(data.refused_reason) : "";
    callout.hidden = false;
    $("refuse-detail").textContent = data.refused_reason || "No grounded answer available.";
    $("refuse-explain").textContent = refusalExplanation(data.refused_reason);
  }
}

function citedIds(data) {
  const ids = new Set();
  (data.claims || []).forEach((c) => (c.source_ids || []).forEach((id) => ids.add(id)));
  return [...ids];
}

function renderAnswerWithCitations(text) {
  const frag = document.createDocumentFragment();
  const re = /\[([A-Z]+-[A-Za-z0-9]+)\]/g;
  let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) frag.appendChild(document.createTextNode(text.slice(last, m.index)));
    const id = m[1];
    const a = document.createElement("a");
    a.className = "citation";
    a.textContent = `[${id}]`;
    a.href = `#src-${id}`;
    a.addEventListener("click", (e) => {
      e.preventDefault();
      const card = $(`src-${id}`);
      if (card) { card.scrollIntoView({ behavior: REDUCED_MOTION ? "auto" : "smooth", block: "center" }); flash(card); }
    });
    frag.appendChild(a);
    last = re.lastIndex;
  }
  if (last < text.length) frag.appendChild(document.createTextNode(text.slice(last)));
  return frag;
}

function flash(el) { el.classList.add("flash"); setTimeout(() => el.classList.remove("flash"), 800); }

const KIND_LABEL = { condition: "condition", drug: "drug", reference: "reference" };

function renderSources(data) {
  const wrap = $("sources");
  wrap.innerHTML = "";
  const cited = new Set(citedIds(data));
  const sources = data.sources || [];
  if (!sources.length) {
    const p = document.createElement("p");
    p.className = "source-snippet";
    p.textContent = "No sources retrieved for this question.";
    wrap.appendChild(p);
    return;
  }
  sources.forEach((s) => {
    const card = document.createElement("div");
    card.className = "source-card" + (cited.has(s.id) ? " cited" : "");
    card.id = `src-${s.id}`;

    const head = document.createElement("div");
    head.className = "source-card-head";
    const left = document.createElement("div");
    left.className = "source-id-row";
    const rank = document.createElement("span");
    rank.className = "source-rank";
    rank.textContent = `#${s.rank || "?"}`;
    const id = document.createElement("span");
    id.className = "source-id";
    id.textContent = s.id;
    left.append(rank, id);
    if (cited.has(s.id)) {
      const badge = document.createElement("span");
      badge.className = "src-badge cited-badge";
      badge.textContent = "cited";
      left.appendChild(badge);
    }
    if (s.above_gate === false) {
      const badge = document.createElement("span");
      badge.className = "src-badge below-gate";
      badge.textContent = "below gate";
      left.appendChild(badge);
    }
    const score = document.createElement("span");
    score.className = "source-score-tag";
    score.textContent = (s.score ?? 0).toFixed(2);
    head.append(left, score);

    const title = document.createElement("div");
    title.className = "source-title-row";
    title.textContent = s.title;

    const meta = document.createElement("div");
    meta.className = "source-meta";
    if (s.kind) meta.appendChild(tag(KIND_LABEL[s.kind] || s.kind, "tag-kind"));
    if (s.section) meta.appendChild(tag(s.section, "tag-section"));
    if (s.topic) meta.appendChild(tag(s.topic, "tag-topic"));

    const snippet = document.createElement("p");
    snippet.className = "source-snippet";
    snippet.textContent = s.snippet;

    const scoreRow = document.createElement("div");
    scoreRow.className = "score-row";
    const bar = document.createElement("div");
    bar.className = "score-bar";
    const fill = document.createElement("div");
    fill.className = "score-bar-fill";
    fill.style.width = `${Math.round(Math.max(0, Math.min(1, s.score)) * 100)}%`;
    bar.appendChild(fill);
    const num = document.createElement("span");
    num.className = "score-num";
    num.textContent = "cosine";
    scoreRow.append(bar, num);

    card.append(head, title, meta, snippet, scoreRow);
    wrap.appendChild(card);
  });
}

function tag(text, cls) {
  const el = document.createElement("span");
  el.className = `meta-tag ${cls || ""}`;
  el.textContent = text;
  return el;
}

// ---------- Trace (clickable, explained) ----------
function renderTrace(steps) {
  const list = $("trace");
  list.innerHTML = "";
  steps.forEach((step, i) => {
    const li = document.createElement("li");
    li.className = `trace-item ${step.status}`;
    if (!REDUCED_MOTION) li.style.animationDelay = `${i * 60}ms`;

    const row = document.createElement("button");
    row.type = "button";
    row.className = "trace-row";
    row.setAttribute("aria-expanded", "false");

    const glyph = document.createElement("span");
    glyph.className = `trace-glyph ${step.status}`;
    const main = document.createElement("div");
    main.className = "trace-main";
    const name = document.createElement("div");
    name.className = "trace-name";
    name.textContent = step.name;
    const detail = document.createElement("div");
    detail.className = "trace-detail";
    detail.textContent = step.detail;
    main.append(name, detail);
    const right = document.createElement("div");
    right.className = "trace-right";
    const ms = document.createElement("span");
    ms.className = "trace-ms";
    ms.textContent = `${step.ms} ms`;
    const caret = document.createElement("span");
    caret.className = "trace-caret";
    caret.textContent = "▸";
    right.append(ms, caret);
    row.append(glyph, main, right);

    const panel = document.createElement("div");
    panel.className = "trace-expand";
    panel.hidden = true;
    panel.appendChild(buildStageDetail(step));

    row.addEventListener("click", () => {
      const open = row.getAttribute("aria-expanded") === "true";
      row.setAttribute("aria-expanded", String(!open));
      panel.hidden = open;
      li.classList.toggle("open", !open);
    });

    li.append(row, panel);
    list.appendChild(li);
  });
}

function buildStageDetail(step) {
  const frag = document.createDocumentFragment();
  if (step.explain) {
    const p = document.createElement("p");
    p.className = "stage-explain";
    p.textContent = step.explain;
    frag.appendChild(p);
  }
  const d = step.data;
  if (!d) return frag;

  if (Array.isArray(d.results)) { frag.appendChild(retrieveDetail(d.results)); return frag; }
  if (Array.isArray(d.claims)) { frag.appendChild(groundingDetail(d)); return frag; }
  if (d.detected_values) { frag.appendChild(dosageDetail(d)); return frag; }
  // Generic key/value detail for any other stage data (gate, coverage, etc.).
  frag.appendChild(kvDetail(d));
  return frag;
}

function kvDetail(obj) {
  const wrap = document.createElement("div");
  wrap.className = "stage-data";
  for (const [k, v] of Object.entries(obj)) {
    const row = document.createElement("div");
    row.className = "data-line";
    const key = document.createElement("span");
    key.className = "mono strong"; key.textContent = k.replace(/_/g, " ");
    const val = document.createElement("span");
    val.className = "data-muted";
    val.textContent = Array.isArray(v) ? (v.length ? v.join(", ") : "(none)") : String(v);
    row.append(key, val);
    wrap.appendChild(row);
  }
  return wrap;
}

function retrieveDetail(results) {
  const wrap = document.createElement("div");
  wrap.className = "stage-data";
  results.forEach((r) => {
    const row = document.createElement("div");
    row.className = "data-line";
    const id = document.createElement("span");
    id.className = "mono strong";
    id.textContent = r.id;
    const t = document.createElement("span");
    t.className = "data-muted";
    t.textContent = r.title;
    const sc = document.createElement("span");
    sc.className = "mono data-score";
    sc.textContent = (r.score ?? 0).toFixed(3);
    row.append(id, t, sc);
    wrap.appendChild(row);
  });
  return wrap;
}

function groundingDetail(d) {
  const wrap = document.createElement("div");
  wrap.className = "stage-data";

  const method = document.createElement("p");
  method.className = "data-note";
  method.textContent = `Method: ${d.method}. Judge: ${d.judge_summary}` +
    (d.judge_model ? ` (${d.judge_model}).` : ".");
  wrap.appendChild(method);

  d.claims.forEach((c) => {
    const box = document.createElement("div");
    box.className = "claim-box";

    const claim = document.createElement("p");
    claim.className = "claim-text";
    claim.textContent = `"${c.claim}"`;
    box.appendChild(claim);

    const meta = document.createElement("div");
    meta.className = "claim-meta";
    meta.appendChild(tag(`cites ${c.source_ids.join(", ") || "nothing"}`, "tag-topic"));
    const det = document.createElement("span");
    det.className = "claim-stat " + (c.grounded ? "ok" : "bad");
    det.textContent = `similarity ${c.deterministic_score.toFixed(2)} vs ${c.threshold.toFixed(2)} ${c.grounded ? "PASS" : "FAIL"}`;
    meta.appendChild(det);
    box.appendChild(meta);

    if (c.judge) {
      const judge = document.createElement("div");
      judge.className = "judge-row";
      const verdict = document.createElement("span");
      verdict.className = "judge-verdict " + (c.judge.supported ? "ok" : "bad");
      verdict.textContent = c.judge.supported ? "judge: supported" : "judge: not supported";
      judge.appendChild(verdict);
      if (c.judge.reason) {
        const reason = document.createElement("span");
        reason.className = "judge-reason";
        reason.textContent = c.judge.reason;
        judge.appendChild(reason);
      }
      box.appendChild(judge);
    } else {
      const none = document.createElement("p");
      none.className = "data-muted small";
      none.textContent = "LLM judge not run (no live provider, or judge disabled). Deterministic check is authoritative.";
      box.appendChild(none);
    }
    wrap.appendChild(box);
  });
  return wrap;
}

function dosageDetail(d) {
  const wrap = document.createElement("div");
  wrap.className = "stage-data";
  const rule = document.createElement("p");
  rule.className = "data-note";
  rule.textContent = d.rule ? `Rule: ${d.rule}.` : "Dosage guard disabled.";
  wrap.appendChild(rule);

  if (!d.detected_values.length) {
    const none = document.createElement("p");
    none.className = "data-muted";
    none.textContent = "No values with clinical units were present in the answer, so nothing to verify.";
    wrap.appendChild(none);
    return wrap;
  }
  d.detected_values.forEach((v) => {
    const row = document.createElement("div");
    row.className = "data-line";
    const val = document.createElement("span");
    val.className = "mono strong";
    val.textContent = v;
    const verified = (d.verified || []).includes(v);
    const status = document.createElement("span");
    status.className = "claim-stat " + (verified ? "ok" : "bad");
    status.textContent = verified ? "verbatim in a source" : "NOT found in any source";
    row.append(val, status);
    wrap.appendChild(row);
  });
  return wrap;
}

// ---------- Audit ----------
let currentAuditId = null;
function renderAudit(auditId) {
  const panel = $("audit-panel");
  if (!auditId) { panel.hidden = true; return; }
  panel.hidden = false;
  currentAuditId = auditId;
  $("audit-toggle").setAttribute("aria-expanded", "false");
  const pre = $("audit-json");
  pre.hidden = true; pre.textContent = "";
}
let currentAuditData = null;
async function fetchAudit() {
  if (!currentAuditId) return;
  const wrap = $("audit-json");
  try {
    currentAuditData = await (await fetch(`/api/audit/${currentAuditId}`)).json();
    wrap.innerHTML = "";
    wrap.appendChild(jsonNode(currentAuditData, null, true));
    $("audit-controls").hidden = false;
  } catch (_) { wrap.textContent = "Audit record unavailable."; }
}

// Render a JSON value as a collapsible tree. Objects/arrays use <details> so
// each level can be folded; the top two levels start open.
function jsonNode(value, key, open, depth = 0) {
  const isObj = value && typeof value === "object";
  if (!isObj) {
    const line = document.createElement("div");
    line.className = "j-line";
    if (key !== null) {
      const k = document.createElement("span"); k.className = "j-key"; k.textContent = key + ": ";
      line.appendChild(k);
    }
    const v = document.createElement("span");
    v.className = "j-val j-" + (value === null ? "null" : typeof value);
    v.textContent = typeof value === "string" ? `"${value}"` : String(value);
    line.appendChild(v);
    return line;
  }
  const arr = Array.isArray(value);
  const entries = arr ? value.map((v, i) => [i, v]) : Object.entries(value);
  const det = document.createElement("details");
  det.className = "j-node";
  if (open && depth < 2) det.open = true;
  const sum = document.createElement("summary");
  sum.className = "j-summary";
  const label = key !== null ? `${key}` : (arr ? "array" : "object");
  sum.innerHTML = `<span class="j-key">${label}</span> <span class="j-meta">${arr ? "[" + entries.length + "]" : "{" + entries.length + "}"}</span>`;
  det.appendChild(sum);
  const body = document.createElement("div");
  body.className = "j-body";
  for (const [k, v] of entries) body.appendChild(jsonNode(v, String(k), open, depth + 1));
  det.appendChild(body);
  return det;
}

function setAllAuditOpen(open) {
  $("audit-json").querySelectorAll("details").forEach((d) => { d.open = open; });
}

// ---------- How it works ----------
function renderHowStages() {
  const list = $("how-stages");
  if (!list) return;
  list.innerHTML = "";
  PIPELINE_STAGES.forEach(([name, desc], i) => {
    const li = document.createElement("li");
    li.className = "how-stage";
    const n = document.createElement("span");
    n.className = "how-stage-num mono";
    n.textContent = String(i + 1).padStart(2, "0");
    const body = document.createElement("div");
    const t = document.createElement("div");
    t.className = "how-stage-name mono";
    t.textContent = name;
    const d = document.createElement("div");
    d.className = "how-stage-desc";
    d.textContent = desc;
    body.append(t, d);
    li.append(n, body);
    list.appendChild(li);
  });
}

// ---------- Corpus map (3D canvas) ----------
const KIND_COLOR = {
  condition: "#8f8f8f", drug: "#0969da", reference: "#9a6700",
  marker: "#0d9488", procedure: "#8250df",
};
// Concrete colours for the canvas (CSS vars are not available to canvas calls).
const KIND_HEX = {
  condition: "#8f8f8f", drug: "#0969da", reference: "#9a6700",
  marker: "#0d9488", procedure: "#8250df",
};
const HIT_HEX = "#171717";

const map3d = {
  points: [],            // [{x,y,z,kind,id,title,section}]
  proj: [],              // last projected screen positions, for hover hit-testing
  hits: new Set(),       // retrieved ids to highlight
  hoverId: null,         // id currently under the cursor
  rotX: -0.45, rotY: 0.6,
  dragging: false, lastX: 0, lastY: 0,
  auto: !REDUCED_MOTION,
  raf: null,
};

async function loadCorpus() {
  try {
    const data = await (await fetch("/api/corpus")).json();
    renderCorpusTiles(data.stats);
    setupCorpusCanvas(data.points);
    renderCorpusLegend(data.stats);
    renderSectionBars(data.stats);
    $("corpus-summary").textContent =
      `${data.stats.total.toLocaleString()} passages, ${data.stats.topics} topics`;
  } catch (_) { $("corpus-summary").textContent = "unavailable"; }
}

let corpusTopicList = [];

function renderCorpusTiles(stats) {
  const wrap = $("corpus-tiles");
  wrap.innerHTML = "";
  corpusTopicList = stats.topic_list || [];
  const tiles = [
    ["documents", stats.total, false],
    ["conditions", stats.by_kind.condition || 0, false],
    ["drug labels", stats.by_kind.drug || 0, false],
    ["lab markers", stats.by_kind.marker || 0, false],
    ["procedures", stats.by_kind.procedure || 0, false],
    ["topics", stats.topics, true],   // clickable: reveals the topic list
  ];
  tiles.forEach(([label, value, clickable]) => {
    const t = document.createElement("div");
    t.className = "corpus-tile" + (clickable ? " clickable" : "");
    const v = document.createElement("div");
    v.className = "tile-value mono";
    v.textContent = Number(value).toLocaleString();
    const l = document.createElement("div");
    l.className = "tile-label";
    l.textContent = clickable ? `${label} (click to view)` : label;
    t.append(v, l);
    if (clickable) {
      t.setAttribute("role", "button");
      t.setAttribute("tabindex", "0");
      t.addEventListener("click", toggleTopicList);
      t.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); toggleTopicList(); } });
    }
    wrap.appendChild(t);
  });
}

function toggleTopicList() {
  let panel = $("topic-list-panel");
  if (panel) { panel.remove(); return; }
  panel = document.createElement("div");
  panel.id = "topic-list-panel";
  panel.className = "topic-list-panel";
  const head = document.createElement("div");
  head.className = "topic-list-head eyebrow";
  head.textContent = `${corpusTopicList.length} topics (each topic is one fictional entity, with several documents)`;
  panel.appendChild(head);
  const grid = document.createElement("div");
  grid.className = "topic-list-grid";
  corpusTopicList.forEach((t) => {
    const item = document.createElement("div");
    item.className = "topic-item";
    const dot = document.createElement("span");
    dot.className = "topic-dot";
    dot.style.background = KIND_COLOR[t.kind] || "#8f8f8f";
    const name = document.createElement("span");
    name.className = "topic-name";
    name.textContent = t.topic;
    const count = document.createElement("span");
    count.className = "topic-count mono";
    count.textContent = `${t.count}`;
    item.append(dot, name, count);
    grid.appendChild(item);
  });
  panel.appendChild(grid);
  $("corpus-tiles").insertAdjacentElement("afterend", panel);
  panel.scrollIntoView({ behavior: REDUCED_MOTION ? "auto" : "smooth", block: "nearest" });
}

function setupCorpusCanvas(points) {
  map3d.points = points.map((p) => ({
    x: p.x, y: p.y, z: p.z ?? 0, kind: p.kind, id: p.id,
    title: p.title || "", section: p.section || "",
  }));
  const canvas = $("corpus-canvas");
  if (!canvas) return;

  const resize = () => {
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    // If the panel is collapsed the canvas has no width; skip until it is shown.
    if (rect.width < 2) return;
    const size = Math.max(1, Math.round(rect.width));
    canvas.width = size * dpr;
    canvas.height = size * dpr;
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    map3d.css = size;
    drawCorpus();
  };
  // Exposed so the collapsible toggle can resize once the panel becomes visible.
  map3d.resize = resize;

  // Drag to rotate (mouse + touch).
  const start = (x, y) => { map3d.dragging = true; map3d.auto = false; map3d.lastX = x; map3d.lastY = y;
    const h = $("corpus-drag-hint"); if (h) h.style.opacity = "0"; };
  const move = (x, y) => {
    if (!map3d.dragging) return;
    map3d.rotY += (x - map3d.lastX) * 0.01;
    map3d.rotX += (y - map3d.lastY) * 0.01;
    map3d.rotX = Math.max(-1.4, Math.min(1.4, map3d.rotX));
    map3d.lastX = x; map3d.lastY = y;
    drawCorpus();
  };
  const end = () => { map3d.dragging = false; };

  // Hover hit-testing: find the nearest projected point under the cursor and
  // show a tooltip. Auto-rotation pauses while the cursor is over the map.
  const hover = (e) => {
    if (map3d.dragging) return;
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    let best = null, bestD = 64;  // within ~8px
    for (const q of map3d.proj) {
      const dx = q.sx - mx, dy = q.sy - my, d = dx * dx + dy * dy;
      if (d < bestD) { bestD = d; best = q; }
    }
    map3d.hoverId = best ? best.id : null;
    showCorpusTooltip(best, mx, my, rect);
  };

  canvas.addEventListener("mousedown", (e) => start(e.clientX, e.clientY));
  window.addEventListener("mousemove", (e) => move(e.clientX, e.clientY));
  window.addEventListener("mouseup", end);
  canvas.addEventListener("mousemove", hover);
  canvas.addEventListener("mouseenter", () => { map3d.auto = false; });
  canvas.addEventListener("mouseleave", () => {
    map3d.auto = !REDUCED_MOTION; map3d.hoverId = null;
    const tt = $("corpus-tooltip"); if (tt) tt.hidden = true;
    drawCorpus();
  });
  canvas.addEventListener("touchstart", (e) => { const t = e.touches[0]; start(t.clientX, t.clientY); }, { passive: true });
  canvas.addEventListener("touchmove", (e) => { const t = e.touches[0]; move(t.clientX, t.clientY); }, { passive: true });
  canvas.addEventListener("touchend", end);

  // Scroll to zoom; double-click to reset the view.
  if (map3d.zoom === undefined) map3d.zoom = 1;
  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    const factor = e.deltaY < 0 ? 1.12 : 1 / 1.12;
    map3d.zoom = Math.max(0.6, Math.min(6, map3d.zoom * factor));
    drawCorpus();
  }, { passive: false });
  canvas.addEventListener("dblclick", () => {
    map3d.zoom = 1; map3d.rotX = -0.45; map3d.rotY = 0.6; drawCorpus();
  });

  window.addEventListener("resize", resize);

  // Fullscreen: expand the canvas wrap, and resize the drawing buffer to match.
  const wrapEl = $("corpus-canvas-wrap");
  const fsBtn = $("corpus-fs-btn");
  if (fsBtn && wrapEl) {
    fsBtn.addEventListener("click", () => {
      if (document.fullscreenElement) document.exitFullscreen();
      else if (wrapEl.requestFullscreen) wrapEl.requestFullscreen();
    });
    document.addEventListener("fullscreenchange", () => {
      const on = document.fullscreenElement === wrapEl;
      wrapEl.classList.toggle("fullscreen", on);
      fsBtn.textContent = on ? "⤡" : "⤢";
      // Let layout settle, then resize the canvas buffer to the new size.
      requestAnimationFrame(() => requestAnimationFrame(resize));
    });
  }

  resize();
  startCorpusLoop();
}

function startCorpusLoop() {
  if (map3d.raf) cancelAnimationFrame(map3d.raf);
  const tick = () => {
    if (map3d.auto && !map3d.dragging) { map3d.rotY += 0.0025; drawCorpus(); }
    map3d.raf = requestAnimationFrame(tick);
  };
  map3d.raf = requestAnimationFrame(tick);
}

function drawCorpus() {
  const canvas = $("corpus-canvas");
  if (!canvas || !map3d.points.length) return;
  const ctx = canvas.getContext("2d");
  const size = map3d.css || canvas.width;
  ctx.clearRect(0, 0, size, size);

  const cx = size / 2, cy = size / 2, scale = size * 0.34 * (map3d.zoom || 1);
  const cosY = Math.cos(map3d.rotY), sinY = Math.sin(map3d.rotY);
  const cosX = Math.cos(map3d.rotX), sinX = Math.sin(map3d.rotX);

  // Project every point; collect with depth for painter's-algorithm ordering.
  const proj = [];
  for (const p of map3d.points) {
    // rotate around Y then X
    let x = p.x * cosY - p.z * sinY;
    let z = p.x * sinY + p.z * cosY;
    let y = p.y * cosX - z * sinX;
    z = p.y * sinX + z * cosX;
    const depth = (z + 1.6) / 3.2;            // 0 (far) .. 1 (near)
    const persp = 1 / (1.8 - z * 0.45);        // gentle perspective
    proj.push({
      sx: cx + x * scale * persp,
      sy: cy + y * scale * persp,
      depth, kind: p.kind, id: p.id, title: p.title, section: p.section,
      hit: map3d.hits.has(p.id),
    });
  }
  proj.sort((a, b) => a.depth - b.depth);       // far first
  map3d.proj = proj;                            // kept for hover hit-testing

  for (const q of proj) {
    if (q.hit || q.id === map3d.hoverId) continue;   // draw hits/hover on top
    const r = (0.8 + q.depth * 1.4);
    ctx.globalAlpha = 0.25 + q.depth * 0.5;
    ctx.fillStyle = KIND_HEX[q.kind] || "#8f8f8f";
    ctx.beginPath();
    ctx.arc(q.sx, q.sy, r, 0, 6.2832);
    ctx.fill();
  }
  let hitCount = 0;
  for (const q of proj) {
    if (!q.hit) continue;
    hitCount++;
    ctx.globalAlpha = 1;
    ctx.fillStyle = HIT_HEX;
    ctx.beginPath();
    ctx.arc(q.sx, q.sy, 3.6 + q.depth * 1.6, 0, 6.2832);
    ctx.fill();
    ctx.lineWidth = 1.4;
    ctx.strokeStyle = "#fff";
    ctx.stroke();
    // Label each retrieved point with its id (white halo for legibility).
    ctx.font = "600 10px ui-monospace, monospace";
    ctx.lineWidth = 3; ctx.strokeStyle = "rgba(255,255,255,0.92)";
    ctx.strokeText(q.id, q.sx + 7, q.sy - 6);
    ctx.fillStyle = HIT_HEX;
    ctx.fillText(q.id, q.sx + 7, q.sy - 6);
  }
  // Caption: how many points are highlighted for the current answer.
  if (hitCount) {
    ctx.globalAlpha = 1;
    ctx.font = "600 11px ui-monospace, monospace";
    ctx.fillStyle = HIT_HEX;
    ctx.fillText(`${hitCount} sources retrieved (highlighted)`, 8, size - 8);
  }
  // Hovered point: a ring in its category colour, drawn last so it is visible.
  const hp = map3d.hoverId && proj.find((q) => q.id === map3d.hoverId);
  if (hp) {
    ctx.globalAlpha = 1;
    ctx.fillStyle = KIND_HEX[hp.kind] || "#8f8f8f";
    ctx.beginPath();
    ctx.arc(hp.sx, hp.sy, 4.5, 0, 6.2832);
    ctx.fill();
    ctx.lineWidth = 1.6;
    ctx.strokeStyle = "#171717";
    ctx.stroke();
  }
  ctx.globalAlpha = 1;
}

function showCorpusTooltip(point, mx, my, rect) {
  const tt = $("corpus-tooltip");
  if (!tt) return;
  if (!point) { tt.hidden = true; return; }
  tt.innerHTML = "";
  const id = document.createElement("span");
  id.className = "tt-id mono";
  id.textContent = point.id;
  const kind = document.createElement("span");
  kind.className = "tt-kind";
  kind.textContent = point.kind;
  kind.style.color = KIND_HEX[point.kind] || "#8f8f8f";
  const title = document.createElement("div");
  title.className = "tt-title";
  title.textContent = point.title || "(document)";
  const head = document.createElement("div");
  head.className = "tt-head";
  head.append(id, kind);
  tt.append(head, title);
  if (point.section) {
    const sec = document.createElement("div");
    sec.className = "tt-section";
    sec.textContent = "section: " + point.section;
    tt.append(sec);
  }
  if (map3d.hits.has(point.id)) {
    const ret = document.createElement("div");
    ret.className = "tt-retrieved";
    ret.textContent = "● retrieved for this answer";
    tt.append(ret);
  }
  // Position near the cursor, clamped inside the canvas.
  const pad = 12;
  let left = mx + pad, top = my + pad;
  if (left > rect.width - 180) left = mx - 180;
  if (top > rect.height - 60) top = my - 60;
  tt.style.left = `${Math.max(0, left)}px`;
  tt.style.top = `${Math.max(0, top)}px`;
  tt.hidden = false;
}

function highlightCorpus(ids) {
  map3d.hits = new Set(ids || []);
  drawCorpus();
}

function renderCorpusLegend(stats) {
  const wrap = $("corpus-legend");
  wrap.innerHTML = "";
  const items = [
    ["condition pages (illnesses)", KIND_COLOR.condition, stats.by_kind.condition || 0],
    ["drug label pages (medications)", KIND_COLOR.drug, stats.by_kind.drug || 0],
    ["lab marker pages", KIND_COLOR.marker, stats.by_kind.marker || 0],
    ["diagnostic procedure pages", KIND_COLOR.procedure, stats.by_kind.procedure || 0],
    ["general reference notes", KIND_COLOR.reference, stats.by_kind.reference || 0],
    ["retrieved for your question", "#171717", null],
  ];
  items.forEach(([label, color, count]) => {
    const el = document.createElement("span");
    el.className = "legend-item";
    const dot = document.createElement("span");
    dot.className = "legend-dot";
    dot.style.background = color;
    const txt = document.createElement("span");
    txt.textContent = count === null ? label : `${label}: ${count.toLocaleString()}`;
    el.append(dot, txt);
    wrap.appendChild(el);
  });
}

function renderSectionBars(stats) {
  const wrap = $("corpus-section-bars");
  if (!wrap) return;
  wrap.innerHTML = "";
  const entries = Object.entries(stats.by_section || {});
  if (!entries.length) return;
  const max = Math.max(...entries.map(([, v]) => v));
  const title = document.createElement("div");
  title.className = "eyebrow section-bars-title";
  title.textContent = "Documents by section";
  wrap.appendChild(title);
  entries.forEach(([name, count]) => {
    const row = document.createElement("div");
    row.className = "section-bar-row";
    const label = document.createElement("span");
    label.className = "section-bar-label";
    label.textContent = name;
    const track = document.createElement("div");
    track.className = "section-bar-track";
    const fill = document.createElement("div");
    fill.className = "section-bar-fill";
    fill.style.width = `${Math.round((count / max) * 100)}%`;
    track.appendChild(fill);
    const num = document.createElement("span");
    num.className = "section-bar-num mono";
    num.textContent = count.toLocaleString();
    row.append(label, track, num);
    wrap.appendChild(row);
  });
}

// ---------- Local AI ----------
const localAI = { status: null, downloading: new Map(), busy: null };

function formatGb(gb) { return `${gb.toFixed(gb < 10 ? 1 : 0)} GB`; }

async function loadLocalAI() {
  try {
    const res = await fetch("/api/local-ai");
    localAI.status = await res.json();
  } catch (_) {
    localAI.status = null;
  }
  renderLocalAI();
}

function renderLocalAI() {
  const s = localAI.status;
  const summary = $("local-ai-summary");
  const hw = $("local-ai-hw");
  const msg = $("local-ai-message");
  const list = $("local-ai-models");
  hw.textContent = "";
  list.textContent = "";
  msg.hidden = true;

  if (!s) { summary.textContent = "unavailable"; return; }

  const active = s.models.find((m) => m.name === s.active);
  summary.textContent = active ? `using ${active.label}`
    : s.ollama.running ? "no model selected" : "Ollama not running";

  const h = s.hardware;
  const accel = { "apple-silicon": "Apple silicon GPU", nvidia: "NVIDIA GPU", cpu: "CPU only" }[h.accelerator] || h.accelerator;
  const facts = [
    h.cpu,
    `${formatGb(h.memory_gb)} memory`,
    ...h.gpus.map((g) => `${g.name}, ${formatGb(g.memory_gb)}`),
    accel,
    `${formatGb(h.model_memory_gb)} available for a model`,
  ];
  facts.forEach((text) => {
    const pill = document.createElement("span");
    pill.className = "pill";
    pill.textContent = text;
    hw.appendChild(pill);
  });

  if (!s.ollama.running) {
    msg.hidden = false;
    msg.textContent = "";
    msg.append("Ollama isn't running at ", Object.assign(document.createElement("code"), { textContent: s.ollama.host }),
      ". Install it from ", Object.assign(document.createElement("a"), { href: "https://ollama.com/download", textContent: "ollama.com", target: "_blank", rel: "noopener" }),
      ", start it, then reopen this panel.");
  } else if (!s.can_manage) {
    msg.hidden = false;
    msg.textContent = "Models can only be downloaded or switched from the machine running GroundCheck.";
  }

  s.models.forEach((m) => list.appendChild(renderModelRow(m, s)));
}

function renderModelRow(m, s) {
  const li = document.createElement("li");
  li.className = `local-ai-model fit-${m.fit}`;

  const main = document.createElement("div");
  main.className = "local-ai-model-main";
  const name = document.createElement("span");
  name.className = "local-ai-model-name";
  name.textContent = m.label;
  main.appendChild(name);
  if (m.recommended) main.appendChild(badge("Recommended", "recommended"));
  main.appendChild(badge({ good: "Good fit", tight: "Tight fit", "too-large": "Too large" }[m.fit], `fit ${m.fit}`));

  const meta = document.createElement("div");
  meta.className = "local-ai-model-meta mono";
  meta.textContent = `${m.name}  ${m.parameters}  download ${formatGb(m.download_gb)}  needs ${formatGb(m.memory_needed_gb)}`;
  const licence = document.createElement("div");
  licence.className = "local-ai-model-licence";
  licence.textContent = m.licence;

  const actions = document.createElement("div");
  actions.className = "local-ai-model-actions";
  const progress = localAI.downloading.get(m.name);
  const disabled = !s.ollama.running || !s.can_manage || localAI.busy !== null;

  if (progress) {
    const bar = document.createElement("div");
    bar.className = "local-ai-progress";
    bar.setAttribute("role", "progressbar");
    bar.setAttribute("aria-valuemin", "0");
    bar.setAttribute("aria-valuemax", "100");
    bar.setAttribute("aria-valuenow", String(progress.percent));
    bar.setAttribute("aria-label", `Downloading ${m.label}`);
    const fill = document.createElement("span");
    fill.style.width = `${progress.percent}%`;
    bar.appendChild(fill);
    const label = document.createElement("span");
    label.className = "local-ai-progress-label mono";
    label.textContent = progress.label;
    actions.append(bar, label);
  } else if (!m.installed) {
    actions.appendChild(button(`Download ${formatGb(m.download_gb)}`, "btn-ghost",
      disabled || m.fit === "too-large", () => downloadModel(m)));
  } else if (s.selected === m.name) {
    const inUse = badge(localAI.busy === m.name ? "Loading…" : "In use", "in-use");
    actions.append(inUse, button("Stop using", "btn-ghost", disabled, () => selectModel(null)));
  } else {
    actions.appendChild(button(localAI.busy === m.name ? "Loading…" : "Use this model", "btn-primary",
      disabled, () => selectModel(m.name)));
  }

  li.append(main, meta, licence, actions);
  return li;
}

function badge(text, kind) {
  const el = document.createElement("span");
  el.className = `local-ai-badge ${kind}`;
  el.textContent = text;
  return el;
}

function button(text, style, disabled, onClick) {
  const el = document.createElement("button");
  el.type = "button";
  el.className = `btn btn-sm ${style}`;
  el.textContent = text;
  el.disabled = disabled;
  el.addEventListener("click", onClick);
  return el;
}

async function downloadModel(m) {
  localAI.downloading.set(m.name, { percent: 0, label: "starting…" });
  renderLocalAI();
  let error = null;
  try {
    const res = await fetch("/api/local-ai/pull", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ model: m.name }),
    });
    if (!res.ok || !res.body) throw new Error((await res.json().catch(() => ({}))).detail || `HTTP ${res.status}`);
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop();
      for (const line of lines) {
        if (!line.trim()) continue;
        const event = JSON.parse(line);
        if (event.status === "error") { error = event.error; continue; }
        const percent = event.total ? Math.round((event.completed / event.total) * 100) : 0;
        const label = event.total ? `${percent}%  ${formatGb(event.completed / 1e9)} of ${formatGb(event.total / 1e9)}` : event.status;
        localAI.downloading.set(m.name, { percent, label });
        renderLocalAI();
      }
    }
  } catch (err) {
    error = err.message;
  }
  localAI.downloading.delete(m.name);
  await loadLocalAI();
  if (error) showLocalAIError(`Download failed: ${error}`);
}

async function selectModel(name) {
  localAI.busy = name || localAI.status?.selected || "";
  renderLocalAI();
  let error = null;
  try {
    const res = await fetch("/api/local-ai/select", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ model: name }),
    });
    if (!res.ok) error = (await res.json().catch(() => ({}))).detail || `HTTP ${res.status}`;
  } catch (err) {
    error = err.message;
  }
  localAI.busy = null;
  await Promise.all([loadLocalAI(), loadHealth()]);
  if (error) showLocalAIError(error);
}

function showLocalAIError(text) {
  const msg = $("local-ai-message");
  msg.hidden = false;
  msg.textContent = text;
  msg.classList.add("error");
}

// ---------- Documents ----------
const STATUS_LABELS = { pending: "Pending review", approved: "Approved", rejected: "Rejected", retired: "Retired" };
let indexPoll = null;

function isoDateToIso(value) { return value ? `${value}T00:00:00` : ""; }
function shortDate(iso) { return iso ? new Date(iso).toLocaleDateString() : ""; }

async function loadDocuments() {
  let data;
  try { data = await api("/api/sources"); }
  catch (err) {
    showMessage("documents-message", err.message, true);
    $("upload-form").hidden = true;
    return;
  }
  $("upload-form").hidden = false;
  renderIndexStatus(data.index);
  const rows = $("documents-body-rows");
  rows.textContent = "";
  $("documents-empty").hidden = data.sources.length > 0;
  const approved = data.sources.filter((d) => d.status === "approved").length;
  const pending = data.sources.filter((d) => d.status === "pending").length;
  $("documents-summary").textContent = `${approved} approved, ${pending} pending review`;

  data.sources.forEach((d) => {
    const tr = document.createElement("tr");
    const doc = document.createElement("td");
    const name = document.createElement("div");
    name.className = "user-name";
    name.textContent = d.title;
    const meta = document.createElement("div");
    meta.className = "user-email mono";
    meta.textContent = `v${d.version}  ${d.filename}`;
    doc.append(name, meta);

    const status = document.createElement("td");
    const chip = document.createElement("span");
    chip.className = `status-pill status-${d.status}`;
    chip.textContent = STATUS_LABELS[d.status] || d.status;
    status.appendChild(chip);

    const owner = document.createElement("td");
    owner.textContent = d.owner || "";

    const force = document.createElement("td");
    const range = [d.effective_from ? `from ${shortDate(d.effective_from)}` : "", d.expires_on ? `until ${shortDate(d.expires_on)}` : ""].filter(Boolean).join(" ");
    force.textContent = d.status === "approved" ? `${d.in_force ? "Yes" : "No"}${range ? `, ${range}` : ""}` : (range || "");

    const sections = document.createElement("td");
    sections.className = "mono";
    sections.textContent = d.chunks;

    const actionsCell = document.createElement("td");
    const actions = document.createElement("div");
    actions.className = "document-actions";
    actionsCell.appendChild(actions);
    actions.appendChild(button("View", "btn-ghost", false, () => openDocument(d.id)));
    if (d.status === "pending") {
      actions.appendChild(button("Approve", "btn-primary", false, () => reviewDocument(d, "approve")));
      actions.appendChild(button("Reject", "btn-ghost", false, () => reviewDocument(d, "reject")));
    }
    if (d.status === "approved") {
      if (d.in_force) actions.appendChild(button("Evaluate", "btn-ghost", false, (e) => evaluateDocument(d, e.currentTarget)));
      actions.appendChild(button("Retire", "btn-ghost", false, () => reviewDocument(d, "retire")));
    }
    tr.append(doc, status, owner, force, sections, actionsCell);
    rows.appendChild(tr);
  });
}

function renderIndexStatus(index) {
  const text = $("index-status-text");
  if (index.state === "running") text.textContent = "Rebuilding the index…";
  else if (index.state === "failed") text.textContent = `Index rebuild failed: ${index.error}`;
  else if (index.finished_at) text.textContent = `Index rebuilt ${new Date(index.finished_at).toLocaleString()}: ${index.documents.toLocaleString()} passages (${index.embedded} newly embedded).`;
  else text.textContent = "The index was built when the app started.";
  $("index-status").classList.toggle("failed", index.state === "failed");
  $("index-rebuild").disabled = index.state === "running";
  if (index.state === "running" && !indexPoll) {
    indexPoll = setInterval(async () => {
      try {
        const data = await api("/api/index");
        if (data.index.state !== "running") {
          clearInterval(indexPoll);
          indexPoll = null;
          await Promise.all([loadDocuments(), loadHealth()]);
        } else {
          renderIndexStatus(data.index);
        }
      } catch (_) { clearInterval(indexPoll); indexPoll = null; }
    }, 1500);
  }
}

async function reviewDocument(d, decision) {
  const verbs = { approve: "Approve", reject: "Reject", retire: "Retire" };
  const prompts = {
    approve: `Approve “${d.title}” version ${d.version}? It will be cited in answers.`,
    reject: `Reject “${d.title}” version ${d.version}? Add a note for the uploader (optional):`,
    retire: `Retire “${d.title}” version ${d.version}? It will stop being cited.`,
  };
  let note = "";
  if (decision === "reject") {
    const answer = window.prompt(prompts.reject, "");
    if (answer === null) return;
    note = answer;
  } else if (!window.confirm(prompts[decision])) {
    return;
  }
  try {
    const data = await api(`/api/sources/${d.id}/${decision}`, { method: "POST", body: { note } });
    showMessage("documents-message", `${verbs[decision]}d “${d.title}”.${decision !== "reject" ? " Rebuilding the index." : ""}`);
    renderIndexStatus(data.index);
  } catch (err) { showMessage("documents-message", err.message, true); }
  await loadDocuments();
}

async function evaluateDocument(d, trigger) {
  trigger.disabled = true;
  trigger.textContent = "Evaluating…";
  try {
    const data = await api(`/api/sources/${d.id}/evaluate`, { method: "POST" });
    const e = data.evaluation;
    showMessage("documents-message", `“${d.title}”: ${e.passed} of ${e.total} test questions behaved as intended, ${e.unsafe_answers} unsafe answers.`, e.unsafe_answers > 0);
    await openDocument(d.id);
  } catch (err) { showMessage("documents-message", err.message, true); }
  await loadDocuments();
}

async function openDocument(id) {
  let data;
  try { data = await api(`/api/sources/${id}`); }
  catch (err) { showMessage("documents-message", err.message, true); return; }
  const d = data.source;
  $("document-dialog-title").textContent = `${d.title}, version ${d.version}`;
  const bits = [STATUS_LABELS[d.status] || d.status, d.owner && `Owner: ${d.owner}`, d.filename,
    d.review_note && `Note: ${d.review_note}`].filter(Boolean);
  $("document-meta").textContent = bits.join(". ") + ".";

  const list = $("document-sections");
  list.textContent = "";
  d.sections.forEach((s) => {
    const li = document.createElement("li");
    const head = document.createElement("div");
    head.className = "document-section-head";
    const id = document.createElement("code");
    id.textContent = s.chunk_id;
    const title = document.createElement("strong");
    title.textContent = s.section;
    head.append(id, title);
    const body = document.createElement("p");
    body.textContent = s.text;
    li.append(head, body);
    list.appendChild(li);
  });

  const evalSection = $("document-eval-section");
  evalSection.hidden = !d.evaluation;
  if (d.evaluation) {
    const e = d.evaluation;
    $("document-eval-summary").textContent = `${e.passed} of ${e.total} behaved as intended, ${e.unsafe_answers} unsafe answers, run ${new Date(e.ran_at).toLocaleString()}.`;
    const cases = $("document-eval-cases");
    cases.textContent = "";
    e.cases.forEach((c) => {
      const li = document.createElement("li");
      li.className = "eval-case";
      const glyph = document.createElement("span");
      glyph.className = `glyph ${c.ok ? "ok" : "bad"}`;
      const q = document.createElement("span");
      q.className = "eval-q";
      q.textContent = c.query;
      const r = document.createElement("span");
      r.className = "eval-r mono";
      r.textContent = `expect ${c.expect} / got ${c.got}`;
      li.append(glyph, q, r);
      cases.appendChild(li);
    });
  }
  $("document-dialog").showModal();
}

function wireDocuments() {
  $("index-rebuild").addEventListener("click", async () => {
    try {
      const data = await api("/api/index/rebuild", { method: "POST" });
      renderIndexStatus({ ...data.index, state: "running" });
    } catch (err) { showMessage("documents-message", err.message, true); }
  });
  $("upload-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const form = e.currentTarget;
    const file = $("upload-file").files[0];
    if (!file) { showMessage("documents-message", "Choose a file to upload.", true); return; }
    const body = new FormData();
    body.append("file", file);
    body.append("title", $("upload-title").value);
    body.append("owner", $("upload-owner").value);
    body.append("effective_from", isoDateToIso($("upload-effective").value));
    body.append("expires_on", isoDateToIso($("upload-expires").value));
    formBusy(form, true);
    $("upload-button").textContent = "Uploading…";
    try {
      const res = await fetch("/api/sources", { method: "POST", body });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.detail || `Upload failed (${res.status}).`);
      form.reset();
      showMessage("documents-message", `Uploaded “${data.source.title}” as version ${data.source.version}, with ${data.source.chunks} sections. It needs approval before it’s cited.`);
      await loadDocuments();
    } catch (err) { showMessage("documents-message", err.message, true); }
    formBusy(form, false);
    $("upload-button").textContent = "Upload for review";
  });
}

// ---------- Review ----------
const KIND_LABELS = { refusal: "Refusal", flagged: "Flagged answer" };
const SEVERITY = ["", "Minor", "Significant", "Considerable", "Major", "Catastrophic"];
const LIKELIHOOD = ["", "Very low", "Low", "Medium", "High", "Very high"];
const review = { tab: "queue", data: null, caseId: null, hazards: [], hazardId: null };

function relativeTime(iso) {
  const minutes = Math.round((new Date(iso) - Date.now()) / 60000);
  const fmt = new Intl.RelativeTimeFormat(undefined, { numeric: "auto" });
  const abs = Math.abs(minutes);
  if (abs < 60) return fmt.format(minutes, "minute");
  if (abs < 48 * 60) return fmt.format(Math.round(minutes / 60), "hour");
  return fmt.format(Math.round(minutes / 1440), "day");
}

// Labels for table cells, shown when rows stack as cards on narrow screens.
function labelCells(cells, labels) {
  cells.forEach((td, i) => { if (labels[i]) td.dataset.label = labels[i]; });
}

const plural = (n, word) => `${n.toLocaleString()} ${word}${n === 1 ? "" : "s"}`;

function selectTab(name) {
  review.tab = name;
  for (const tab of ["queue", "hazards", "report"]) {
    const on = tab === name;
    $(`tab-${tab}`).setAttribute("aria-selected", String(on));
    $(`tab-${tab}`).tabIndex = on ? 0 : -1;
    $(`review-${tab}`).hidden = !on;
  }
  $("review-message").hidden = true;
  loadReviewTab();
}

function loadReviewTab() {
  if (review.tab === "queue") return loadReviews();
  if (review.tab === "hazards") return loadHazards();
  return loadReport();
}

async function loadReviews() {
  const params = new URLSearchParams({ status: $("review-status").value });
  if ($("review-mine").checked) params.set("mine", "true");
  if ($("review-overdue").checked) params.set("overdue", "true");
  let data;
  try { data = await api(`/api/reviews?${params}`); }
  catch (err) { showMessage("review-message", err.message, true); return; }
  review.data = data;
  const c = data.counts;
  $("review-summary").textContent = `${c.open} open${c.overdue ? `, ${c.overdue} overdue` : ""}`;
  const counts = $("review-counts");
  counts.textContent = "";
  [["Open", c.open, ""], ["Overdue", c.overdue, c.overdue ? "bad" : ""], ["Resolved", c.resolved, ""]].forEach(([label, n, cls]) => {
    const item = document.createElement("span");
    item.className = `review-count ${cls}`;
    const num = document.createElement("strong");
    num.textContent = n.toLocaleString();
    item.append(num, ` ${label.toLowerCase()}`);
    counts.appendChild(item);
  });

  const rows = $("review-rows");
  rows.textContent = "";
  $("review-empty").hidden = data.cases.length > 0;
  data.cases.forEach((k) => {
    const tr = document.createElement("tr");
    if (k.overdue) tr.className = "overdue";

    const q = document.createElement("td");
    const text = document.createElement("div");
    text.className = "user-name review-question";
    text.textContent = k.query;
    const why = document.createElement("div");
    why.className = "user-email";
    why.textContent = k.kind === "flagged" ? `Flagged: ${k.reason}` : k.reason_category;
    q.append(text, why);

    const type = document.createElement("td");
    const pills = document.createElement("div");
    pills.className = "pill-group";
    type.appendChild(pills);
    const pill = document.createElement("span");
    pill.className = `status-pill kind-${k.kind}${k.priority === "high" ? " priority-high" : ""}`;
    pill.textContent = KIND_LABELS[k.kind] || k.kind;
    pills.appendChild(pill);
    if (k.status !== "open") {
      const st = document.createElement("span");
      st.className = "status-pill status-retired";
      st.textContent = "Resolved";
      pills.appendChild(st);
    }

    const asked = document.createElement("td");
    asked.textContent = k.occurrences === 1 ? "once" : `${k.occurrences} times`;
    asked.title = `First ${new Date(k.created_at).toLocaleString()}, last ${new Date(k.last_seen_at).toLocaleString()}`;

    const due = document.createElement("td");
    if (k.status === "open") {
      due.textContent = k.overdue ? `Overdue, ${relativeTime(k.due_at)}` : relativeTime(k.due_at);
      due.className = k.overdue ? "due-overdue" : "";
    } else {
      due.textContent = k.outcome_label;
    }

    const who = document.createElement("td");
    who.textContent = k.assigned_to_name || "Unassigned";
    if (!k.assigned_to_name) who.className = "muted";

    const actionsCell = document.createElement("td");
    const actions = document.createElement("div");
    actions.className = "document-actions";
    actions.appendChild(button(k.status === "open" ? "Review" : "View", k.status === "open" ? "btn-primary" : "btn-ghost", false, () => openCase(k.id)));
    actionsCell.appendChild(actions);

    labelCells([q, type, asked, due, who, actionsCell], ["", "Type", "Asked", k.status === "open" ? "Due" : "Outcome", "Assigned", ""]);
    tr.append(q, type, asked, due, who, actionsCell);
    rows.appendChild(tr);
  });
}

async function openCase(id) {
  let data;
  try { data = await api(`/api/reviews/${id}`); }
  catch (err) { showMessage("review-message", err.message, true); return; }
  review.caseId = id;
  renderCase(data.case);
  $("case-message").hidden = true;
  if (!$("case-dialog").open) $("case-dialog").showModal();
}

function renderCase(k) {
  $("case-dialog-title").textContent = `${KIND_LABELS[k.kind] || "Case"} #${k.id}`;
  $("case-query").textContent = k.query;
  const meta = [
    k.kind === "flagged" ? `Flagged${k.flagged_by_name ? ` by ${k.flagged_by_name}` : ""}: ${k.reason}` : `Refused: ${k.reason}`,
    `Asked ${k.occurrences === 1 ? "once" : `${k.occurrences} times`}, first ${new Date(k.created_at).toLocaleString()}`,
    k.status === "open" ? `Due ${relativeTime(k.due_at)}${k.escalated ? ", escalated" : ""}` : "",
    `Audit record ${k.last_audit_id}`,
  ].filter(Boolean).map((part) => part.replace(/[.\s]+$/, ""));
  $("case-meta").textContent = meta.join(". ") + ".";

  $("case-answer-section").hidden = !k.answer_text;
  $("case-answer").textContent = k.answer_text || "";
  $("case-sources").textContent = (k.sources || []).length
    ? `Sources retrieved: ${k.sources.map((x) => x.id).join(", ")}.` : "";

  const open = k.status === "open";
  $("case-work").hidden = !open;
  $("case-closed").hidden = open;
  if (open) {
    const select = $("case-assignee");
    select.textContent = "";
    const none = new Option("Unassigned", "");
    select.appendChild(none);
    (review.data?.reviewers || []).forEach((r) => select.appendChild(new Option(r.name, r.id)));
    if (review.data?.me && !(review.data.reviewers || []).some((r) => r.id === review.data.me)) {
      select.appendChild(new Option("Me", review.data.me));
    }
    select.value = k.assigned_to ?? "";
    const outcome = $("case-outcome");
    if (!outcome.options.length) {
      outcome.appendChild(new Option("Choose…", ""));
      Object.entries(review.data?.outcomes || {}).forEach(([key, label]) => outcome.appendChild(new Option(label, key)));
    }
    outcome.value = "";
    $("case-expected").value = k.kind === "refusal" ? "refuse" : "";
    $("case-expected-field").hidden = true;
    $("case-note").value = "";
    $("case-comment").value = "";
  } else {
    $("case-outcome-text").textContent = `Resolved${k.resolved_by_name ? ` by ${k.resolved_by_name}` : ""} ${new Date(k.resolved_at).toLocaleString()}: ${k.outcome_label}.${k.outcome_note ? ` ${k.outcome_note}` : ""}`;
    $("case-reopen-note").value = "";
  }

  const list = $("case-timeline");
  list.textContent = "";
  k.events.slice().reverse().forEach((e) => {
    const li = document.createElement("li");
    const head = document.createElement("div");
    head.className = "timeline-head";
    const action = document.createElement("strong");
    action.textContent = capitalize(e.action);
    const when = document.createElement("span");
    when.className = "muted";
    when.textContent = `${e.user ? `${e.user}, ` : ""}${new Date(e.at).toLocaleString()}`;
    head.append(action, when);
    li.appendChild(head);
    if (e.note) {
      const note = document.createElement("p");
      note.textContent = e.note;
      li.appendChild(note);
    }
    list.appendChild(li);
  });
}

async function caseAction(path, body, form, done) {
  if (form) formBusy(form, true);
  try {
    const data = await api(`/api/reviews/${review.caseId}/${path}`, { method: "POST", body });
    renderCase(data.case);
    showMessage("case-message", done);
    loadReviews();
    refreshReviewCount();
  } catch (err) { showMessage("case-message", err.message, true); }
  if (form) formBusy(form, false);
}

async function loadHazards() {
  let data;
  try { data = await api("/api/hazards"); }
  catch (err) { showMessage("review-message", err.message, true); return; }
  review.hazards = data.hazards;
  const rows = $("hazard-rows");
  rows.textContent = "";
  $("hazard-empty").hidden = data.hazards.length > 0;
  const riskCell = (r, s, l) => {
    const td = document.createElement("td");
    const pill = document.createElement("span");
    pill.className = `risk-pill risk-${r.level.replace(" ", "-")}`;
    pill.textContent = `${r.score} ${r.level}`;
    pill.title = `Severity ${s} (${SEVERITY[s]}) × likelihood ${l} (${LIKELIHOOD[l]})`;
    td.appendChild(pill);
    return td;
  };
  data.hazards.forEach((h) => {
    const tr = document.createElement("tr");
    const id = document.createElement("td");
    id.className = "mono";
    id.textContent = `H${h.id}`;
    const title = document.createElement("td");
    const name = document.createElement("div");
    name.className = "user-name";
    name.textContent = h.title;
    const controls = document.createElement("div");
    controls.className = "user-email";
    controls.textContent = h.controls ? `Controls: ${h.controls}` : "No controls recorded";
    title.append(name, controls);
    const status = document.createElement("td");
    status.textContent = capitalize(h.status);
    const owner = document.createElement("td");
    owner.textContent = h.owner;
    const actionsCell = document.createElement("td");
    const actions = document.createElement("div");
    actions.className = "document-actions";
    actions.appendChild(button(canEditHazards() ? "Edit" : "View", "btn-ghost", false, () => openHazard(h)));
    actionsCell.appendChild(actions);
    const cells = [id, title, riskCell(h.initial_risk, h.severity, h.likelihood),
      riskCell(h.residual_risk, h.residual_severity, h.residual_likelihood), status, owner, actionsCell];
    labelCells(cells, ["ID", "", "Before controls", "After controls", "Status", "Owner", ""]);
    tr.append(...cells);
    rows.appendChild(tr);
  });
}

function riskLabel(s, l) {
  const score = s * l;
  const level = score <= 4 ? "low" : score <= 9 ? "medium" : score <= 16 ? "high" : "very high";
  return `Risk ${score}, ${level}`;
}

function updateHazardRisk() {
  $("hazard-initial-risk").textContent = riskLabel(+$("hazard-severity").value, +$("hazard-likelihood").value);
  $("hazard-residual-risk").textContent = riskLabel(+$("hazard-residual-severity").value, +$("hazard-residual-likelihood").value);
}

function openHazard(h) {
  review.hazardId = h ? h.id : null;
  $("hazard-dialog-title").textContent = h ? `Hazard H${h.id}` : "Add hazard";
  $("hazard-title").value = h?.title || "";
  $("hazard-cause").value = h?.cause || "";
  $("hazard-effect").value = h?.effect || "";
  $("hazard-controls").value = h?.controls || "";
  $("hazard-severity").value = h?.severity || 3;
  $("hazard-likelihood").value = h?.likelihood || 3;
  $("hazard-residual-severity").value = h?.residual_severity || 3;
  $("hazard-residual-likelihood").value = h?.residual_likelihood || 2;
  $("hazard-status").value = h?.status || "open";
  $("hazard-owner").value = h?.owner || "";
  updateHazardRisk();
  const editable = canEditHazards();
  $("hazard-form").querySelectorAll("input, textarea, select, button").forEach((el) => { el.disabled = !editable; });
  $("hazard-save").hidden = !editable;
  $("hazard-message").hidden = true;
  $("hazard-dialog").showModal();
  if (editable) $("hazard-title").focus();
}

async function loadReport() {
  const days = $("report-days").value;
  $("safety-case-download").href = `/api/governance/safety-case?days=${days}`;
  let r;
  try { r = await api(`/api/governance/report?days=${days}`); }
  catch (err) { showMessage("review-message", err.message, true); return; }
  const q = r.questions, rv = r.reviews;
  const pct = (n) => n === null ? "–" : `${Math.round(n * 100)}%`;
  const tiles = [
    ["Questions", q.total.toLocaleString(), `${q.answered.toLocaleString()} answered`],
    ["Refusal rate", pct(q.refusal_rate), `${q.refused.toLocaleString()} refused`],
    ["Review cases", rv.opened.toLocaleString(), plural(rv.flagged_answers, "flagged answer")],
    ["Resolved on time", rv.resolved ? `${rv.resolved_on_time} of ${rv.resolved}` : "–", rv.median_hours_to_resolve === null ? "none resolved" : rv.median_hours_to_resolve < 1 ? "median under an hour" : `median ${rv.median_hours_to_resolve} hours`],
    ["Open now", rv.open_now.toLocaleString(), `${rv.overdue_now} overdue`],
    ["Checks switched off", q.with_a_check_switched_off.toLocaleString(), "questions asked with a check off"],
  ];
  const grid = $("report-grid");
  grid.textContent = "";
  tiles.forEach(([label, value, note]) => {
    const tile = document.createElement("div");
    tile.className = "report-tile";
    const l = document.createElement("span"); l.className = "report-label"; l.textContent = label;
    const v = document.createElement("strong"); v.className = "report-value"; v.textContent = value;
    const n = document.createElement("span"); n.className = "report-note"; n.textContent = note;
    tile.append(l, v, n);
    grid.appendChild(tile);
  });

  const reasons = $("report-reasons");
  reasons.textContent = "";
  const entries = Object.entries(q.refusal_reasons);
  if (entries.length) {
    const h = document.createElement("h3");
    h.textContent = "Why questions were refused";
    reasons.appendChild(h);
    const max = Math.max(...entries.map(([, n]) => n));
    entries.forEach(([reason, n]) => {
      const row = document.createElement("div");
      row.className = "reason-row";
      const label = document.createElement("span");
      label.textContent = capitalize(reason);
      const track = document.createElement("span");
      track.className = "reason-track";
      const bar = document.createElement("span");
      bar.className = "reason-bar";
      bar.style.width = `${(n / max) * 100}%`;
      track.appendChild(bar);
      const count = document.createElement("span");
      count.className = "mono";
      count.textContent = n.toLocaleString();
      row.append(label, track, count);
      reasons.appendChild(row);
    });
  }
  loadReviewTests();
}

function renderReviewTestCases(cases) {
  const list = $("review-tests-list");
  list.textContent = "";
  cases.forEach((c) => {
    const li = document.createElement("li");
    li.className = "eval-case";
    const glyph = document.createElement("span");
    glyph.className = `glyph ${c.ok === undefined ? "" : c.ok ? "ok" : "bad"}`;
    const q = document.createElement("span");
    q.className = "eval-q";
    q.textContent = c.query;
    const r = document.createElement("span");
    r.className = "eval-r mono";
    r.textContent = c.got ? `expect ${c.expect} / got ${c.got}` : `expect ${c.expect}`;
    li.append(glyph, q, r);
    list.appendChild(li);
  });
}

async function loadReviewTests() {
  try {
    const data = await api("/api/review-tests");
    $("review-tests-summary").textContent = data.cases.length
      ? `${data.cases.length} question${data.cases.length === 1 ? "" : "s"} added from resolved cases.`
      : "None yet. Resolve a case as “Added as a permanent test question” to add one.";
    $("review-tests-run").hidden = !data.cases.length;
    renderReviewTestCases(data.cases);
  } catch (_) { /* shown by the report */ }
}

async function verifyAudit() {
  const trigger = $("audit-verify");
  const out = $("verify-result");
  trigger.disabled = true;
  trigger.textContent = "Verifying…";
  try {
    const r = await api("/api/audit/verify", { method: "POST" });
    out.hidden = false;
    out.className = `verify-result ${r.ok ? "ok" : "bad"}`;
    out.textContent = "";
    const head = document.createElement("strong");
    head.textContent = !r.complete ? "Couldn’t check the whole audit trail"
      : r.ok ? "Intact" : `${plural(r.problem_count, "problem")} found`;
    const detail = document.createElement("p");
    detail.textContent = `${plural(r.checked, "record")} checked${r.anchor_seq ? ` from number ${r.anchor_seq + 1}` : ""}, `
      + `${r.signed ? "signed with a key held outside the database" : "not signed with a key"}. `
      + `Chain head ${r.head.seq}: ${r.head.hash.slice(0, 16)}…`;
    out.append(head, detail);
    if (r.error) {
      const e = document.createElement("p");
      e.textContent = r.error;
      out.appendChild(e);
    }
    if (r.problems.length) {
      const list = document.createElement("ul");
      r.problems.forEach((p) => {
        const li = document.createElement("li");
        li.textContent = `Record ${p.seq}${p.audit_id ? ` (${p.audit_id})` : ""}: ${p.problem}`;
        list.appendChild(li);
      });
      out.appendChild(list);
    }
  } catch (err) { showMessage("protection-message", err.message, true); }
  trigger.disabled = false;
  trigger.textContent = "Verify audit trail";
}

async function loadProtection() {
  let data;
  try { data = await api("/api/data-protection"); }
  catch (_) { $("protection-admin").hidden = true; return; }  // admins only
  $("protection-admin").hidden = false;
  const e = data.encryption, sig = data.audit_signing, ret = data.retention;
  const rows = [
    ["Encryption at rest", e.enabled
      ? `On, AES-256-GCM with key ${e.primary_key_id}${e.decrypt_key_ids.length ? `. Older keys still readable: ${e.decrypt_key_ids.join(", ")}` : ""}`
      : "Off. Set DATA_ENCRYPTION_KEYS to encrypt questions, answers and review notes."],
    ["Audit signing", sig.enabled ? `On, HMAC-SHA256 with key ${sig.primary_key_id}` : "Off. Set AUDIT_SIGNING_KEYS so the chain can't be rewritten from inside the database."],
    ["Audit chain head", `${data.audit_chain.seq.toLocaleString()} records written. Last hash ${data.audit_chain.hash.slice(0, 16)}…`],
  ];
  const list = $("protection-list");
  list.textContent = "";
  rows.forEach(([term, value]) => {
    const dt = document.createElement("dt"); dt.textContent = term;
    const dd = document.createElement("dd"); dd.textContent = value;
    if (value.startsWith("Off")) dd.className = "muted";
    list.append(dt, dd);
  });
  renderRetention(ret);
}

function renderRetention(ret) {
  const periods = [
    ret.audit_retention_days ? `audit records for ${plural(ret.audit_retention_days, "day")}` : "audit records for ever",
    ret.review_retention_days ? `resolved reviews for ${plural(ret.review_retention_days, "day")}` : "resolved reviews for ever",
  ];
  const due = ret.audit_records_to_delete + ret.review_cases_to_delete;
  const last = ret.last_run ? ` Last run ${new Date(ret.last_run.ran_at).toLocaleString()}: ${plural(ret.last_run.audit_deleted, "audit record")} and ${plural(ret.last_run.reviews_deleted, "review")} deleted.` : "";
  $("retention-summary").textContent = `Keeping ${periods.join(" and ")}. `
    + (due ? `${plural(ret.audit_records_to_delete, "audit record")} and ${plural(ret.review_cases_to_delete, "resolved review")} are past their period.` : "Nothing is past its retention period.")
    + last;
  $("retention-run").disabled = due === 0;
}

async function runRetention() {
  if (!window.confirm("Permanently delete every record past its retention period? This can’t be undone.")) return;
  try {
    const data = await api("/api/retention/run", { method: "POST" });
    showMessage("protection-message", `Deleted ${plural(data.run.audit_deleted, "audit record")} and ${plural(data.run.reviews_deleted, "review case")}.`);
    renderRetention(data.retention);
  } catch (err) { showMessage("protection-message", err.message, true); }
}

function resetFlag(auditId) {
  $("flag-row").hidden = !auditId;
  $("flag-row").dataset.auditId = auditId || "";
  $("flag-open").hidden = false;
  $("flag-form").hidden = true;
  $("flag-message").hidden = true;
  $("flag-note").value = "";
}

function wireReview() {
  const tabs = ["queue", "hazards", "report"];
  tabs.forEach((name, i) => {
    const tab = $(`tab-${name}`);
    tab.addEventListener("click", () => selectTab(name));
    tab.addEventListener("keydown", (e) => {
      const step = e.key === "ArrowRight" ? 1 : e.key === "ArrowLeft" ? -1 : 0;
      if (!step) return;
      const next = tabs[(i + step + tabs.length) % tabs.length];
      selectTab(next);
      $(`tab-${next}`).focus();
    });
  });
  ["review-status", "review-mine", "review-overdue"].forEach((id) => $(id).addEventListener("change", loadReviews));
  $("report-days").addEventListener("change", loadReport);

  $("case-assign").addEventListener("click", () => {
    const value = $("case-assignee").value;
    caseAction("assign", { user_id: value ? Number(value) : null }, null, value ? "Assigned." : "Unassigned.");
  });
  $("case-comment-form").addEventListener("submit", (e) => {
    e.preventDefault();
    caseAction("comment", { note: $("case-comment").value }, e.currentTarget, "Comment added.");
  });
  $("case-outcome").addEventListener("change", () => {
    $("case-expected-field").hidden = $("case-outcome").value !== "add_test";
  });
  $("case-resolve-form").addEventListener("submit", (e) => {
    e.preventDefault();
    const outcome = $("case-outcome").value;
    caseAction("resolve", {
      outcome, note: $("case-note").value,
      expected_decision: outcome === "add_test" ? ($("case-expected").value || null) : null,
    }, e.currentTarget, outcome === "add_test" ? "Resolved. The question is now a permanent test." : "Resolved.");
  });
  $("case-reopen-form").addEventListener("submit", (e) => {
    e.preventDefault();
    caseAction("reopen", { note: $("case-reopen-note").value }, e.currentTarget, "Reopened.");
  });

  document.querySelectorAll("#hazard-form select[data-scale]").forEach((select) => {
    const labels = select.dataset.scale === "severity" ? SEVERITY : LIKELIHOOD;
    for (let n = 1; n <= 5; n++) select.appendChild(new Option(`${n}  ${labels[n]}`, n));
    select.addEventListener("change", updateHazardRisk);
  });
  $("hazard-add").addEventListener("click", () => openHazard(null));
  $("hazard-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const form = e.currentTarget;
    const body = {
      title: $("hazard-title").value, cause: $("hazard-cause").value, effect: $("hazard-effect").value,
      severity: +$("hazard-severity").value, likelihood: +$("hazard-likelihood").value,
      controls: $("hazard-controls").value,
      residual_severity: +$("hazard-residual-severity").value, residual_likelihood: +$("hazard-residual-likelihood").value,
      status: $("hazard-status").value, owner: $("hazard-owner").value,
    };
    formBusy(form, true);
    try {
      const path = review.hazardId ? `/api/hazards/${review.hazardId}` : "/api/hazards";
      await api(path, { method: review.hazardId ? "PUT" : "POST", body });
      $("hazard-dialog").close();
      showMessage("review-message", review.hazardId ? "Hazard updated." : "Hazard added.");
      loadHazards();
    } catch (err) { showMessage("hazard-message", err.message, true); }
    formBusy(form, false);
  });

  $("review-tests-run").addEventListener("click", async (e) => {
    const trigger = e.currentTarget;
    trigger.disabled = true;
    trigger.textContent = "Running…";
    try {
      const r = await api("/api/review-tests/run", { method: "POST" });
      $("review-tests-summary").textContent = `${r.passed} of ${r.total} behaved as intended, ${r.unsafe_answers} unsafe answers.`;
      $("review-tests-summary").classList.toggle("bad", r.passed < r.total);
      renderReviewTestCases(r.cases);
    } catch (err) { showMessage("review-message", err.message, true); }
    trigger.disabled = false;
    trigger.textContent = "Run tests";
  });

  $("audit-verify").addEventListener("click", verifyAudit);
  $("retention-run").addEventListener("click", runRetention);

  $("flag-open").addEventListener("click", () => {
    $("flag-open").hidden = true;
    $("flag-form").hidden = false;
    $("flag-message").hidden = true;
    $("flag-note").focus();
  });
  $("flag-cancel").addEventListener("click", () => resetFlag($("flag-row").dataset.auditId));
  $("flag-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const form = e.currentTarget;
    formBusy(form, true);
    try {
      await api(`/api/audit/${$("flag-row").dataset.auditId}/flag`, { method: "POST", body: { note: $("flag-note").value } });
      form.hidden = true;
      showMessage("flag-message", "Sent for review. A reviewer will check this answer.");
      refreshReviewCount();
    } catch (err) { showMessage("flag-message", err.message, true); }
    formBusy(form, false);
  });
}

// ---------- Usage ----------
function statTile(label, value, note, { attention = false, small = false } = {}) {
  const tile = document.createElement("div");
  tile.className = `stat${attention ? " attention" : ""}`;
  const l = document.createElement("span"); l.className = "stat-label"; l.textContent = label;
  const v = document.createElement("strong"); v.className = `stat-value${small ? " small" : ""}`; v.textContent = value;
  tile.append(l, v);
  if (note) { const n = document.createElement("span"); n.className = "stat-note"; n.textContent = note; tile.appendChild(n); }
  return tile;
}

function formatMs(ms) {
  if (ms === null || ms === undefined) return "–";
  return ms < 1000 ? `${ms} ms` : `${(ms / 1000).toFixed(ms < 10000 ? 1 : 0)} s`;
}

function renderBars(container, rows, emptyText) {
  container.textContent = "";
  if (!rows.length) {
    const p = document.createElement("p"); p.className = "bars-empty"; p.textContent = emptyText;
    container.appendChild(p);
    return;
  }
  const max = Math.max(...rows.map((r) => r.value), 1);
  rows.forEach((r) => {
    const row = document.createElement("div");
    row.className = "bar-row";
    const label = document.createElement("span"); label.className = "bar-label"; label.textContent = r.label; label.title = r.label;
    const value = document.createElement("span"); value.className = "bar-value"; value.textContent = r.value.toLocaleString();
    const track = document.createElement("span"); track.className = "bar-track";
    const fill = document.createElement("span"); fill.className = "bar-fill"; fill.style.width = `${(r.value / max) * 100}%`;
    track.appendChild(fill);
    row.append(label, value, track);
    container.appendChild(row);
  });
}

function niceMax(n) {
  if (n <= 4) return 4;
  const step = Math.pow(10, Math.floor(Math.log10(n)));
  for (const m of [1, 2, 2.5, 5, 10]) if (m * step >= n) return m * step;
  return 10 * step;
}

// Stacked daily bars: answered below, refused above, with a hover tooltip.
function renderDailyChart(container, daily) {
  container.textContent = "";
  const total = daily.reduce((a, d) => a + d.answered + d.refused, 0);
  if (!total) {
    const empty = document.createElement("div");
    empty.className = "chart-empty";
    empty.textContent = "No questions in this period yet. Ask one and it appears here.";
    container.appendChild(empty);
    return;
  }
  const width = Math.max(container.clientWidth, 280), height = container.clientHeight || 240;
  const pad = { top: 8, right: 4, bottom: 22, left: 34 };
  const innerW = width - pad.left - pad.right, innerH = height - pad.top - pad.bottom;
  const top = niceMax(Math.max(...daily.map((d) => d.answered + d.refused)));
  const y = (v) => pad.top + innerH - (v / top) * innerH;
  const slot = innerW / daily.length;
  const barW = Math.max(2, Math.min(28, slot * 0.7));
  const NS = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(NS, "svg");
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", `Questions per day: ${total} in ${daily.length} days`);
  const el = (name, attrs) => { const e = document.createElementNS(NS, name); for (const k in attrs) e.setAttribute(k, attrs[k]); return e; };

  for (let i = 0; i <= 4; i++) {
    const v = (top / 4) * i;
    svg.appendChild(el("line", { class: "grid-line", x1: pad.left, x2: width - pad.right, y1: y(v), y2: y(v) }));
    const t = el("text", { class: "axis-label", x: pad.left - 6, y: y(v) + 4, "text-anchor": "end" });
    t.textContent = Number.isInteger(v) ? v : v.toFixed(1);
    svg.appendChild(t);
  }
  // Date labels at a regular interval, always including today. A regular
  // label too close to today's is dropped so they never overlap.
  const labelEvery = Math.ceil(daily.length / Math.max(2, Math.floor(innerW / 56)));
  const last = daily.length - 1;
  const labelled = new Set();
  for (let i = 0; i <= last; i += labelEvery) if (last - i >= labelEvery || i === last) labelled.add(i);
  labelled.add(last);
  const tooltip = document.createElement("div");
  tooltip.className = "chart-tooltip";
  tooltip.hidden = true;

  daily.forEach((d, i) => {
    const cx = pad.left + slot * i + slot / 2, x = cx - barW / 2;
    const g = el("g", {});
    const answeredTop = y(d.answered);
    if (d.answered) g.appendChild(el("rect", { class: "bar-answered", x, y: answeredTop, width: barW, height: pad.top + innerH - answeredTop, rx: Math.min(2, barW / 2) }));
    if (d.refused) {
      const refusedTop = y(d.answered + d.refused);
      const gap = d.answered ? 2 : 0;
      g.appendChild(el("rect", { class: "bar-refused", x, y: refusedTop, width: barW, height: Math.max(1, answeredTop - refusedTop - gap), rx: Math.min(2, barW / 2) }));
    }
    svg.appendChild(g);
    const date = new Date(`${d.date}T00:00:00`);
    if (labelled.has(i)) {
      const t = el("text", { class: "axis-label", x: cx, y: height - 6, "text-anchor": "middle" });
      t.textContent = date.toLocaleDateString(undefined, { day: "numeric", month: "short" });
      svg.appendChild(t);
    }
    const hit = el("rect", { class: "bar-hit", x: pad.left + slot * i, y: pad.top, width: slot, height: innerH });
    hit.addEventListener("mouseenter", () => {
      g.classList.add("hover");
      tooltip.hidden = false;
      tooltip.innerHTML = "";
      const strong = document.createElement("strong");
      strong.textContent = date.toLocaleDateString(undefined, { weekday: "short", day: "numeric", month: "short" });
      tooltip.append(strong, `${d.answered} answered, ${d.refused} refused`);
      tooltip.style.left = `${(cx / width) * 100}%`;
      tooltip.style.top = `${(y(d.answered + d.refused) / height) * 100}%`;
    });
    hit.addEventListener("mouseleave", () => { g.classList.remove("hover"); tooltip.hidden = true; });
    svg.appendChild(hit);
  });
  container.append(svg, tooltip);
}

async function loadUsage() {
  let data;
  try { data = await api(`/api/usage?days=${$("usage-days").value}`); }
  catch (err) { showMessage("usage-message", err.message, true); return; }
  $("usage-message").hidden = true;
  const t = data.totals;
  const stats = $("usage-stats");
  stats.textContent = "";
  const pct = t.refusal_rate === null ? "–" : `${Math.round(t.refusal_rate * 100)}%`;
  stats.append(
    statTile("Questions", t.questions.toLocaleString(), `${t.answered.toLocaleString()} answered`),
    statTile("Refused", pct, `${plural(t.refused, "question")}`),
    statTile("Median response", formatMs(t.median_ms), t.p95_ms === null ? "" : `95% within ${formatMs(t.p95_ms)}`),
    draftedTile(data.drafted_by, t.questions),
    statTile("Identifiers removed", t.deidentified.toLocaleString(), "questions with names, dates or numbers"),
    statTile("Open reviews", t.open_reviews.toLocaleString(), t.overdue_reviews ? `${t.overdue_reviews} overdue` : "none overdue", { attention: t.overdue_reviews > 0 }),
  );
  usageState.daily = data.daily;
  renderDailyChart($("usage-chart"), data.daily);
  renderBars($("usage-reasons"), Object.entries(data.refusal_reasons).map(([k, v]) => ({ label: capitalize(k), value: v })), "No refusals in this period.");
  renderBars($("usage-sources"), data.top_sources.map((x) => ({ label: `${x.title} (${x.id})`, value: x.citations })), "No answers cited a source in this period.");
  $("usage-users-panel").hidden = !account.authRequired;
  renderBars($("usage-users"), data.by_user.map((x) => ({ label: x.name, value: x.questions })), "No signed-in questions in this period.");
}
const usageState = { daily: null };

function draftedTile(drafted, total) {
  const models = drafted["Local model"] + drafted["Cloud model"];
  const note = total ? [
    drafted["Local model"] && `${drafted["Local model"]} local`,
    drafted["Cloud model"] && `${drafted["Cloud model"]} cloud`,
    `${drafted.Extractive} extractive`,
  ].filter(Boolean).join(", ") : "";
  return statTile("Drafted by a model", total ? `${Math.round((models / total) * 100)}%` : "–", note);
}

// ---------- Embeddings ----------
async function loadEmbeddings() {
  let data;
  try { data = await api("/api/embeddings"); }
  catch (_) { return; }
  const stats = $("embed-stats");
  stats.textContent = "";
  const idx = data.index || {};
  const rebuilt = idx.finished_at ? `rebuilt ${new Date(idx.finished_at).toLocaleString()}` : "built when the app started";
  stats.append(
    statTile("Passages indexed", data.passages.toLocaleString(), `${data.from_your_documents.toLocaleString()} from your documents`),
    statTile("Embedding model", data.model.replace(/^sentence-transformers\//, ""), data.model.startsWith("sentence-transformers/") ? "sentence-transformers" : "", { small: true }),
    statTile("Dimensions", data.dimensions.toLocaleString(), "per passage vector"),
    statTile("Vector store", data.vector_store === "qdrant" ? "Qdrant" : "Local (NumPy)", data.vector_store === "qdrant" ? "server or embedded" : "exact cosine search", { small: true }),
    statTile("Keyword vocabulary", data.vocabulary.toLocaleString(), "terms for keyword ranking"),
    statTile("Index", idx.state === "running" ? "Rebuilding…" : idx.state === "failed" ? "Failed" : "Ready", idx.state === "failed" ? idx.error : rebuilt, { small: true, attention: idx.state === "failed" }),
  );
  const rows = [
    ["Ranking", data.hybrid.enabled ? `Hybrid: ${Math.round(data.hybrid.embedding_weight * 100)}% embedding similarity, ${Math.round(data.hybrid.keyword_weight * 100)}% keyword (BM25)` : "Embedding similarity only"],
    ["Similarity", "Cosine, on normalised vectors"],
    ["Passages retrieved", `${data.top_k} per question`],
    ["Relevance threshold", `${data.min_score} cosine. Below this, GroundCheck refuses before drafting anything.`],
    ["Sources", data.include_demo_corpus ? `Your approved documents and the synthetic demo corpus (${data.from_demo_corpus.toLocaleString()} passages)` : "Your approved documents only"],
  ];
  const list = $("embed-settings");
  list.textContent = "";
  rows.forEach(([term, value]) => {
    const dt = document.createElement("dt"); dt.textContent = term;
    const dd = document.createElement("dd"); dd.textContent = value;
    list.append(dt, dd);
  });
}

// ---------- Navigation ----------
// One page per area, addressed by the URL hash (#/review), so pages can be
// bookmarked and the back button works.
const VIEWS = {
  ask: { load: () => {} },
  usage: { load: () => loadUsage() },
  review: { load: () => { loadReviewTab(); } },
  documents: { load: () => loadDocuments() },
  evaluation: { load: () => {} },
  embeddings: { load: () => { loadEmbeddings(); requestAnimationFrame(() => map3d.resize && map3d.resize()); } },
  "local-ai": { load: () => loadLocalAI() },
  "data-protection": { load: () => loadProtection() },
  users: { load: () => { $("users-message").hidden = true; loadUsers(); } },
};

function currentView() {
  const name = (location.hash.match(/^#\/([\w-]+)/) || [])[1];
  return VIEWS[name] && canSee(name) ? name : "ask";
}

function route() {
  const name = currentView();
  document.querySelectorAll(".view").forEach((v) => { v.hidden = v.id !== `view-${name}`; });
  document.querySelectorAll(".nav-link").forEach((l) => {
    if (l.dataset.view === name) l.setAttribute("aria-current", "page");
    else l.removeAttribute("aria-current");
  });
  const title = $(`view-${name}`).dataset.title;
  $("topbar-title").textContent = title;
  document.title = name === "ask" ? "GroundCheck" : `${title} · GroundCheck`;
  setNavOpen(false);
  VIEWS[name].load();
}

function setNavOpen(open) {
  document.body.classList.toggle("nav-open", open);
  $("nav-scrim").hidden = !open;
  $("nav-open").setAttribute("aria-expanded", String(open));
  if (open) $("nav-close").focus();
}

function wireNavigation() {
  window.addEventListener("hashchange", () => {
    route();
    $("content").focus({ preventScroll: true });
    window.scrollTo(0, 0);
  });
  $("nav-open").addEventListener("click", () => setNavOpen(true));
  $("nav-close").addEventListener("click", () => { setNavOpen(false); $("nav-open").focus(); });
  $("nav-scrim").addEventListener("click", () => setNavOpen(false));
  $("usage-days").addEventListener("change", loadUsage);
  let resizeTimer = null;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => { if (usageState.daily && currentView() === "usage") renderDailyChart($("usage-chart"), usageState.daily); }, 150);
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && document.body.classList.contains("nav-open")) { setNavOpen(false); $("nav-open").focus(); }
  });
  applyRoleNavigation();
  route();
  refreshReviewCount();
}

// The number of open review cases, shown beside Review in the navigation.
async function refreshReviewCount() {
  if (!canSee("review")) return;
  try {
    const data = await api("/api/reviews?status=open");
    const n = data.counts.open;
    const badge = $("nav-review-count");
    badge.hidden = n === 0;
    badge.textContent = n > 99 ? "99+" : String(n);
    badge.classList.toggle("overdue", data.counts.overdue > 0);
    badge.title = `${n} open${data.counts.overdue ? `, ${data.counts.overdue} overdue` : ""}`;
  } catch (_) { /* no database or no access */ }
}

// ---------- Collapsibles ----------
function wireCollapsibles() {
  wireDocuments();
  wireReview();
  wireToggle("how-toggle", "how-body");
  wireToggle("tuning-toggle", "tuning-body");
  const auditToggle = $("audit-toggle");
  auditToggle.addEventListener("click", () => {
    const expanded = auditToggle.getAttribute("aria-expanded") === "true";
    auditToggle.setAttribute("aria-expanded", String(!expanded));
    const wrap = $("audit-json");
    wrap.hidden = expanded;
    $("audit-controls").hidden = expanded || !wrap.childElementCount;
    if (!expanded && !wrap.childElementCount) fetchAudit();
  });
  $("audit-expand").addEventListener("click", () => setAllAuditOpen(true));
  $("audit-collapse").addEventListener("click", () => setAllAuditOpen(false));
  $("audit-copy").addEventListener("click", () => {
    if (currentAuditData) navigator.clipboard.writeText(JSON.stringify(currentAuditData, null, 2));
    const b = $("audit-copy"); const t = b.textContent; b.textContent = "Copied"; setTimeout(() => b.textContent = t, 1200);
  });
}
function wireToggle(toggleId, bodyId) {
  const toggle = $(toggleId);
  if (!toggle) return;
  toggle.addEventListener("click", () => {
    const expanded = toggle.getAttribute("aria-expanded") === "true";
    toggle.setAttribute("aria-expanded", String(!expanded));
    $(bodyId).hidden = expanded;
  });
}

// ---------- Evaluation ----------
async function loadEvalSummary() {
  try {
    const data = await (await fetch("/api/eval-summary")).json();
    const inline = $("eval-summary-inline");
    if (!data.total) { inline.textContent = data.note || "not run yet"; return; }
    const ansC = data.answerable_correct, ansT = data.answerable_total ?? ansC;
    const refC = data.must_refuse_correct, refT = data.must_refuse_total ?? refC;
    inline.textContent = `${data.passed.toLocaleString()} of ${data.total.toLocaleString()} cases pass`;
    const pct = Math.round((data.passed / data.total) * 100);
    $("eval-bar-fill").style.width = `${pct}%`;
    $("eval-bar-fill").style.background = pct === 100 ? "var(--ok)" : "var(--warn)";
    $("eval-caption").textContent =
      `${ansC.toLocaleString()} of ${ansT.toLocaleString()} answerable correct, ` +
      `${refC.toLocaleString()} of ${refT.toLocaleString()} must-refuse correct. ` +
      `Run offline in extractive mode for reproducibility.`;
    const list = $("eval-cases");
    list.innerHTML = "";
    (data.cases || data.sample_cases || []).forEach((c) => {
      const li = document.createElement("li");
      li.className = "eval-case";
      const g = document.createElement("span");
      g.className = `glyph ${c.ok ? "ok" : "bad"}`;
      const q = document.createElement("span");
      q.className = "q"; q.textContent = c.query;
      const v = document.createElement("span");
      v.className = "verdict"; v.textContent = `expect ${c.expect} / got ${c.got}`;
      li.append(g, q, v);
      list.appendChild(li);
    });
    const shown = (data.cases || data.sample_cases || []).length;
    let foot = data.total > shown
      ? `Showing a sample of ${shown}. Full suite of ${data.total.toLocaleString()} runs in the build.`
      : "";
    renderAdversarial(data.adversarial);
    $("eval-foot").textContent = foot;
  } catch (_) { $("eval-summary-inline").textContent = "unavailable"; }
}

function renderAdversarial(adv) {
  const wrap = $("adversarial-block");
  if (!wrap) return;
  if (!adv || !adv.total) { wrap.hidden = true; return; }
  wrap.hidden = false;
  wrap.innerHTML = "";
  const head = document.createElement("div");
  head.className = "adv-head";
  const title = document.createElement("span");
  title.className = "eyebrow";
  title.textContent = "Adversarial probes";
  const score = document.createElement("span");
  score.className = "adv-score";
  score.textContent = `${adv.passed} of ${adv.total} behaved as intended`;
  head.append(title, score);
  const note = document.createElement("p");
  note.className = "adv-note";
  note.textContent = "Hand-written traps: leading questions, prompt injection, " +
    "near-miss spellings, and missing-context cases. Reported honestly and not " +
    "used to gate the build, so a known limitation is shown rather than hidden.";
  wrap.append(head, note);
  const list = document.createElement("ul");
  list.className = "eval-cases";
  adv.cases.forEach((c) => {
    const li = document.createElement("li");
    li.className = "eval-case";
    const g = document.createElement("span");
    g.className = `glyph ${c.ok ? "ok" : "bad"}`;
    const q = document.createElement("span");
    q.className = "q";
    q.textContent = c.query;
    q.title = c.note || "";
    const v = document.createElement("span");
    v.className = "verdict";
    v.textContent = c.ok ? `${c.got} ✓` : `want ${c.expect}, got ${c.got}`;
    li.append(g, q, v);
    list.appendChild(li);
  });
  wrap.appendChild(list);
}

// ---------- Utils ----------
function capitalize(s) { return s && s.length ? s.charAt(0).toUpperCase() + s.slice(1) : s; }

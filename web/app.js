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

// Display names for pipeline stages whose internal names read awkwardly.
const STAGE_LABELS = { "pii redaction": "De-identification" };

// Plain-language explanation shown in the refusal callout, keyed by a stable
// phrase found in the refusal reason.
function refusalExplanation(reason) {
  const r = (reason || "").toLowerCase();
  if (r.includes("in the question does not appear"))
    return "The question states a value that no trusted source supports, so confirming it could be unsafe.";
  if (r.includes("no trusted source covers"))
    return "The question is about a patient group the sources don't cover, so an answer written for adults could be unsafe.";
  if (r.includes("states a maximum") || r.includes("states a minimum"))
    return "The sources give a dose but no limit, so they can't answer a question about a maximum or minimum.";
  if (r.includes("must not be combined"))
    return "The sources say these must not be used together, so there is no safe combined dose to give.";
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
  wireTheme();
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
  account.sso = me.sso || { enabled: false };
  account.passwordSignIn = me.password_sign_in !== false;
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
  const ssoOn = mode === "login" && account.sso && account.sso.enabled;
  $("sso-block").hidden = !ssoOn;
  if (ssoOn) {
    $("sso-button").textContent = `Sign in with ${account.sso.provider}`;
    $("sso-button").href = `/api/auth/sso/start?next=${encodeURIComponent("/" + location.hash)}`;
  }
  $("sso-divider").hidden = !account.passwordSignIn;
  const passwordButton = $("login-form").querySelector("button[type=submit]");
  passwordButton.classList.toggle("btn-primary", !ssoOn);
  passwordButton.classList.toggle("btn-ghost", ssoOn);
  $("login-form").hidden = mode !== "login" || (ssoOn && !account.passwordSignIn);
  $("mfa-form").hidden = mode !== "mfa";
  $("first-admin-form").hidden = mode !== "first-admin";
  setAuthError(message || "");
  const first = ssoOn ? "sso-button" : { login: "login-email", mfa: "mfa-code", "first-admin": "admin-name" }[mode];
  requestAnimationFrame(() => $(first).focus());
  // A sign-in the identity provider sent back with an error.
  const ssoError = new URLSearchParams(location.search).get("sso_error");
  if (ssoError) {
    setAuthError(ssoError);
    history.replaceState(null, "", location.pathname + location.hash);
  }
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
  const needs = { usage: 1, review: 1, documents: 1, "data-protection": 1, users: 2, "local-ai": 0, training: 2, models: 1, ehr: 2 }[view] ?? 0;
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
    if (provider && provider.kind === "local") { dot.className = "dot ok"; label.textContent = `Local model, ${provider.model}`; }
    else if (provider) { dot.className = "dot ok"; label.textContent = `Cloud model, ${provider.model}`; }
    else { dot.className = "dot warn"; label.textContent = "Extractive, no model"; }
    if (typeof data.corpus === "number") $("corpus-count").textContent = data.corpus.toLocaleString();
  } catch (_) { dot.className = "dot warn"; label.textContent = "Extractive, no model"; }
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
    else $("query").focus();
  });
}

let inflight = false;
async function submitQuery(query) {
  if (inflight) return;
  inflight = true;
  setLoading(true);
  try {
    const patient = readPatient();
    if (patient === false) return;  // invalid details: the form says what to fix
    const res = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, settings: readSettings(), ...(patient ? { patient } : {}) }),
    });
    const data = await res.json();
    if (res.status === 422) {
      const problem = Array.isArray(data.detail) ? data.detail.map((d) => d.msg.replace(/^Value error, /, "")).join("; ") : data.detail;
      $("patient-box").open = true;
      $("pt-error").textContent = `Check the patient details: ${problem}`;
      return;
    }
    render({ ...data, patient_given: Boolean(patient) });
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
  renderPatientFindings(data.patient_findings || [], data.patient_given);
  const canSave = Boolean(ehr.context && ehr.context.can_write_notes && data.patient_given && data.audit_id);
  $("ehr-save").hidden = !canSave;
  $("ehr-save-form").hidden = false;
  $("ehr-save-message").hidden = true;
  $("ehr-save-comment").value = "";
  $("ehr-save").dataset.auditId = data.audit_id || "";
  const block = (data.patient_findings || []).find((f) => f.severity === "block");
  if (data.decision === "refuse" && block && data.refused_reason && data.refused_reason.startsWith(block.message.charAt(0).toLowerCase() + block.message.slice(1, 20))) {
    $("decision-reason").textContent = `Not safe for this patient: ${block.medicine}`;
    $("refuse-explain").textContent = "The answer from the sources doesn't fit the patient you described. The check above names the rule, and what the formulary gives instead where it can.";
  }
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
    name.textContent = STAGE_LABELS[step.name] || step.name;
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
  const keySpan = document.createElement("span");
  keySpan.className = "j-key";
  keySpan.textContent = label;
  const metaSpan = document.createElement("span");
  metaSpan.className = "j-meta";
  metaSpan.textContent = arr ? `[${entries.length}]` : `{${entries.length}}`;
  sum.append(keySpan, " ", metaSpan);
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
// Canvas can't read CSS variables, so the theme's colours are copied here and
// refreshed when the theme changes.
const KIND_HEX = {};
let HIT_HEX = "#0f2622";
let HALO = "rgba(255,255,255,0.92)";
function readMapColours() {
  const css = getComputedStyle(document.documentElement);
  for (const kind of ["condition", "drug", "reference", "marker", "procedure"]) {
    KIND_HEX[kind] = css.getPropertyValue(`--kind-${kind}`).trim();
  }
  HIT_HEX = css.getPropertyValue("--map-hit").trim();
  HALO = css.getPropertyValue("--map-halo").trim();
}
readMapColours();
const KIND_COLOR = new Proxy(KIND_HEX, { get: (target, key) => `var(--kind-${String(key)})` });

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
    ctx.strokeStyle = HALO;
    ctx.stroke();
    // Label each retrieved point with its id (white halo for legibility).
    ctx.font = "600 10px ui-monospace, monospace";
    ctx.lineWidth = 3; ctx.strokeStyle = HALO;
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
    ctx.strokeStyle = HIT_HEX;
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
    ["retrieved for your question", "var(--map-hit)", null],
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

// ---------- Training ----------
const training = { setup: null, dataset: null, device: null, architecture: "small-cnn", poll: null, runId: null, browsePath: null };
const pct = (v, digits = 0) => (v === null || v === undefined ? "–" : `${(v * 100).toFixed(digits)}%`);

function formatDuration(seconds) {
  if (seconds === null || seconds === undefined) return "–";
  if (seconds < 60) return `${Math.round(seconds)} s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)} min ${Math.round(seconds % 60)} s`;
  return `${Math.floor(seconds / 3600)} h ${Math.round((seconds % 3600) / 60)} min`;
}

async function loadTraining() {
  try { training.setup = await api("/api/training/setup"); }
  catch (err) { showMessage("training-message", err.message, true); return; }
  renderDevices(training.setup.hardware);
  renderArchitectures(training.setup.architectures, training.setup.image_sizes);
  loadRuns();
  const active = training.setup.active_run;
  if (active) followRun(active.id);
}

function renderDevices(hw) {
  const list = $("device-list");
  list.textContent = "";
  const gpus = hw.devices.filter((d) => d.kind === "gpu");
  if (!training.device || !hw.devices.some((d) => d.id === training.device)) training.device = hw.recommended;
  $("hw-sub").textContent = gpus.length === 0
    ? "No GPU found, so training will use the CPU. It works, but more slowly."
    : gpus.length === 1
      ? `Training will use ${gpus[0].name}. You can choose the CPU instead, which is slower.`
      : `This machine has ${gpus.length} GPUs. Choose one to train on.`;
  hw.devices.forEach((d) => {
    const label = document.createElement("label");
    label.className = "device-card";
    const input = document.createElement("input");
    input.type = "radio"; input.name = "train-device"; input.value = d.id; input.checked = d.id === training.device;
    input.addEventListener("change", () => { training.device = d.id; });
    const body = document.createElement("span");
    body.className = "device-body";
    const name = document.createElement("strong");
    name.textContent = d.name;
    const meta = document.createElement("span");
    meta.className = "device-meta";
    meta.textContent = d.kind === "gpu"
      ? `${d.vendor} GPU, ${d.memory_gb} GB${d.shared_memory ? " shared memory" : ""}`
      : `CPU, ${d.cores} cores, ${d.memory_gb} GB memory`;
    body.append(name, meta);
    if (d.id === hw.recommended) {
      const badge = document.createElement("span");
      badge.className = "local-ai-badge recommended";
      badge.textContent = "Recommended";
      body.appendChild(badge);
    }
    label.append(input, body);
    list.appendChild(label);
  });
}

function renderArchitectures(archs, sizes) {
  const list = $("arch-list");
  list.textContent = "";
  archs.forEach((a) => {
    const label = document.createElement("label");
    label.className = `device-card${a.available ? "" : " disabled"}`;
    const input = document.createElement("input");
    input.type = "radio"; input.name = "train-arch"; input.value = a.id;
    input.checked = a.id === training.architecture; input.disabled = !a.available;
    input.addEventListener("change", () => applyArchitecture(a));
    const body = document.createElement("span");
    body.className = "device-body";
    const name = document.createElement("strong"); name.textContent = a.label;
    const meta = document.createElement("span"); meta.className = "device-meta"; meta.textContent = a.available ? a.description : a.unavailable_reason;
    body.append(name, meta);
    label.append(input, body);
    list.appendChild(label);
  });
  const select = $("train-size");
  if (!select.options.length) sizes.forEach((n) => select.appendChild(new Option(`${n} × ${n} px`, n)));
  applyArchitecture(archs.find((a) => a.id === training.architecture) || archs[0]);
}

function applyArchitecture(a) {
  training.architecture = a.id;
  $("train-size").value = a.image_size;
  $("train-batch").value = a.batch_size;
  $("train-lr").value = a.learning_rate;
  $("pretrained-row").hidden = !a.pretrained_option;
  if (!a.pretrained_option) $("train-pretrained").checked = false;
}

async function scanDataset() {
  const path = $("train-dataset").value.trim();
  const box = $("dataset-summary");
  if (!path) { showMessage("training-message", "Choose a folder of images first.", true); return; }
  box.hidden = false;
  box.textContent = "Reading the folder…";
  box.classList.remove("invalid");
  try {
    const summary = await api(`/api/training/dataset?path=${encodeURIComponent(path)}`);
    training.dataset = summary;
    $("train-dataset").value = summary.path;
    renderDatasetSummary(summary);
    if (!$("train-name").value.trim()) $("train-name").value = `${summary.name} classifier`;
  } catch (err) {
    training.dataset = null;
    box.textContent = err.message;
    box.classList.add("invalid");
  }
}

function renderDatasetSummary(d) {
  const box = $("dataset-summary");
  box.textContent = "";
  const head = document.createElement("p");
  head.className = "dataset-head";
  const splits = d.layout === "split-folders" ? "using the folder’s own train, validation and test split" : "split into training, validation and test sets for you";
  head.innerHTML = "";
  const strong = document.createElement("strong");
  strong.textContent = `${d.total.toLocaleString()} images in ${d.classes.length} classes`;
  head.append(strong, `, ${splits}: ${d.splits.train.toLocaleString()} training, ${d.splits.val.toLocaleString()} validation, ${d.splits.test.toLocaleString()} test.`);
  box.appendChild(head);

  const grid = document.createElement("div");
  grid.className = "class-grid";
  const max = Math.max(...d.classes.map((c) => c.total));
  d.classes.forEach((c) => {
    const sample = d.samples.find((x) => x.class === c.name);
    const card = document.createElement("div");
    card.className = "class-card";
    if (sample) {
      const img = document.createElement("img");
      img.loading = "lazy";
      img.alt = `Example image of ${c.name}`;
      img.src = `/api/training/image?dataset=${encodeURIComponent(d.path)}&file=${encodeURIComponent(sample.file)}`;
      card.appendChild(img);
    }
    const name = document.createElement("span"); name.className = "class-name"; name.textContent = c.name; name.title = c.name;
    const count = document.createElement("span"); count.className = "class-count"; count.textContent = c.total.toLocaleString();
    const track = document.createElement("span"); track.className = "bar-track";
    const fill = document.createElement("span"); fill.className = "bar-fill"; fill.style.width = `${(c.total / max) * 100}%`;
    track.appendChild(fill);
    card.append(name, count, track);
    grid.appendChild(card);
  });
  box.appendChild(grid);
  d.warnings.forEach((w) => {
    const p = document.createElement("p"); p.className = "dataset-warning"; p.textContent = w; box.appendChild(p);
  });
}

async function openBrowser(path) {
  const dialog = $("browse-dialog");
  $("browse-message").hidden = true;
  let data;
  try { data = await api(`/api/training/browse?path=${encodeURIComponent(path || "")}`); }
  catch (err) { showMessage("browse-message", err.message, true); if (!dialog.open) dialog.showModal(); return; }
  training.browsePath = data.path;
  $("browse-path").textContent = data.path || "Folders you can train from";
  $("browse-use").disabled = !data.path;
  const list = $("browse-list");
  list.textContent = "";
  const addRow = (label, target, hint, cls = "") => {
    const li = document.createElement("li");
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = `browse-item ${cls}`;
    const icon = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    icon.setAttribute("class", "btn-icon"); icon.setAttribute("aria-hidden", "true");
    const use = document.createElementNS("http://www.w3.org/2000/svg", "use"); use.setAttribute("href", "#i-folder-open");
    icon.appendChild(use);
    const name = document.createElement("span"); name.textContent = label;
    btn.append(icon, name);
    if (hint) { const h = document.createElement("span"); h.className = "local-ai-badge recommended"; h.textContent = hint; btn.appendChild(h); }
    btn.addEventListener("click", () => openBrowser(target));
    li.appendChild(btn);
    list.appendChild(li);
  };
  if (data.path) addRow(data.parent ? "Up one folder" : "All folders", data.parent || "", null, "browse-up");
  data.folders.forEach((f) => addRow(f.name, f.path, f.looks_like_dataset ? "Looks like a dataset" : null));
  if (!data.folders.length) {
    const li = document.createElement("li"); li.className = "auth-hint"; li.textContent = "No subfolders here."; list.appendChild(li);
  }
  if (!dialog.open) dialog.showModal();
}

async function startTraining(e) {
  e.preventDefault();
  if (!training.dataset || training.dataset.path !== $("train-dataset").value.trim()) await scanDataset();
  if (!training.dataset) { $("train-dataset").focus(); return; }
  const body = {
    name: $("train-name").value.trim(), dataset: training.dataset.path, architecture: training.architecture,
    pretrained: $("train-pretrained").checked, device: training.device,
    epochs: Number($("train-epochs").value), image_size: Number($("train-size").value),
    batch_size: Number($("train-batch").value), learning_rate: Number($("train-lr").value),
    val_fraction: Number($("train-val").value), test_fraction: Number($("train-test").value),
  };
  const button = $("train-start");
  button.disabled = true;
  button.textContent = "Starting…";
  try {
    const data = await api("/api/training/runs", { method: "POST", body });
    $("training-message").hidden = true;
    followRun(data.run.id);
    loadRuns();
    $("run-live").scrollIntoView({ behavior: REDUCED_MOTION ? "auto" : "smooth", block: "start" });
  } catch (err) { showMessage("training-message", err.message, true); }
  button.disabled = false;
  button.textContent = "Start training";
}

function followRun(runId) {
  training.runId = runId;
  clearInterval(training.poll);
  const tick = async () => {
    let data;
    try { data = await api(`/api/training/runs/${runId}`); } catch (_) { return; }
    renderLiveRun(data.run);
    const state = data.run.progress.state;
    const live = state === "running" || state === "queued";
    $("nav-training-live").hidden = !live;
    if (!live) { clearInterval(training.poll); training.poll = null; loadRuns(); }
  };
  tick();
  training.poll = setInterval(tick, 1500);
}

function renderLiveRun(run) {
  const p = run.progress;
  const box = $("run-live");
  box.hidden = false;
  $("run-live-name").textContent = run.name;
  const live = p.state === "running" || p.state === "queued";
  $("run-cancel").hidden = !live;
  $("run-resume").hidden = !run.can_resume;
  $("run-live-status").textContent = `${p.message || ""}${live && p.eta_seconds ? `, about ${formatDuration(p.eta_seconds)} left` : ""}`;
  const percent = p.state === "completed" ? 100 : (p.percent || 0);
  $("run-progress-fill").style.width = `${percent}%`;
  $("run-progressbar").setAttribute("aria-valuenow", String(Math.round(percent)));
  box.dataset.state = p.state;

  const history = p.history || [];
  const last = history[history.length - 1];
  const bestVal = history.length ? Math.max(...history.map((h) => h.val_accuracy ?? 0)) : null;
  const stats = $("run-stats");
  stats.textContent = "";
  stats.append(
    statTile("Epoch", p.epoch ? `${p.epoch} of ${p.epochs}` : "–", p.best_epoch ? `best so far: ${p.best_epoch}` : ""),
    statTile("Validation accuracy", last ? pct(last.val_accuracy, 1) : "–", bestVal !== null ? `best ${pct(bestVal, 1)}` : ""),
    statTile("Hardware", run.device_name, `${run.architecture === "resnet18" ? "ResNet-18" : "Small CNN"}, ${run.image_size} px`, { small: true }),
  );
  renderLineChart($("chart-loss"), history, [["train_loss", "line-train"], ["val_loss", "line-val"]], (v) => v.toFixed(2));
  renderLineChart($("chart-accuracy"), history, [["train_accuracy", "line-train"], ["val_accuracy", "line-val"]], (v) => `${Math.round(v * 100)}%`, [0, 1]);

  const result = $("run-result");
  result.hidden = p.state === "running" || p.state === "queued";
  result.className = `run-result ${p.state}`;
  result.textContent = "";
  if (p.state === "completed") {
    const sm = p.summary || {};
    result.append(`Saved to the model library. On held-out test images: ${pct(sm.test_accuracy, 1)} accurate; it answers ${pct(sm.coverage)} of images and is ${pct(sm.answered_accuracy, 1)} correct when it does. `);
    const link = document.createElement("a");
    link.href = "#/models";
    link.textContent = "Open the model library";
    link.addEventListener("click", () => { models.openAfterLoad = p.model_id; });
    result.appendChild(link);
  } else if (p.state === "failed" || p.state === "cancelled") {
    result.textContent = run.can_resume
      ? `${p.message}. It stopped after epoch ${(p.history || []).length} and can resume from there.`
      : p.message;
  }
}

// A line per series across epochs, with a dot on each value.
function renderLineChart(container, history, series, format, fixedRange) {
  container.textContent = "";
  if (history.length === 0) {
    const empty = document.createElement("div"); empty.className = "chart-empty"; empty.textContent = "Appears after the first epoch";
    container.appendChild(empty);
    return;
  }
  const width = Math.max(container.clientWidth, 240), height = container.clientHeight || 180;
  const pad = { top: 8, right: 10, bottom: 22, left: 40 };
  const values = series.flatMap(([key]) => history.map((h) => h[key]).filter((v) => v !== null && v !== undefined));
  let [lo, hi] = fixedRange || [Math.min(0, ...values), Math.max(...values)];
  if (fixedRange) lo = Math.max(0, Math.floor(Math.min(...values) * 10) / 10);
  if (hi === lo) hi = lo + 1;
  const n = Math.max(history.length, 2);
  const x = (i) => pad.left + (i / (n - 1)) * (width - pad.left - pad.right);
  const y = (v) => pad.top + (1 - (v - lo) / (hi - lo)) * (height - pad.top - pad.bottom);
  const NS = "http://www.w3.org/2000/svg";
  const el = (name, attrs) => { const e = document.createElementNS(NS, name); for (const k in attrs) e.setAttribute(k, attrs[k]); return e; };
  const svg = el("svg", { viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": `${series.map(([k]) => k).join(" and ")} by epoch` });
  for (let i = 0; i <= 3; i++) {
    const v = lo + ((hi - lo) / 3) * i;
    svg.appendChild(el("line", { class: "grid-line", x1: pad.left, x2: width - pad.right, y1: y(v), y2: y(v) }));
    const t = el("text", { class: "axis-label", x: pad.left - 6, y: y(v) + 4, "text-anchor": "end" });
    t.textContent = format(v);
    svg.appendChild(t);
  }
  const every = Math.ceil(history.length / 8);
  history.forEach((h, i) => {
    if (i % every === 0 || i === history.length - 1) {
      const t = el("text", { class: "axis-label", x: x(i), y: height - 6, "text-anchor": "middle" });
      t.textContent = h.epoch;
      svg.appendChild(t);
    }
  });
  series.forEach(([key, cls]) => {
    const points = history.map((h, i) => [x(i), h[key]]).filter(([, v]) => v !== null && v !== undefined);
    if (points.length > 1) svg.appendChild(el("polyline", { class: cls, points: points.map(([px, v]) => `${px},${y(v)}`).join(" ") }));
    points.forEach(([px, v]) => {
      const dot = el("circle", { class: `${cls} dot`, cx: px, cy: y(v), r: 3.5 });
      const title = el("title", {}); title.textContent = `${format(v)}`;
      dot.appendChild(title);
      svg.appendChild(dot);
    });
  });
  container.appendChild(svg);
}

async function loadRuns() {
  let data;
  try { data = await api("/api/training/runs"); } catch (_) { return; }
  const rows = $("runs-rows");
  rows.textContent = "";
  $("runs-empty").hidden = data.runs.length > 0;
  rows.closest(".users-table-wrap").hidden = data.runs.length === 0;
  const labels = { queued: "Starting", running: "Training", completed: "Saved", failed: "Failed", cancelled: "Cancelled" };
  const pills = { queued: "status-pending", running: "status-pending", completed: "status-approved", failed: "status-rejected", cancelled: "status-retired" };
  data.runs.forEach((run) => {
    const p = run.progress;
    const tr = document.createElement("tr");
    const name = document.createElement("td");
    const n = document.createElement("div"); n.className = "user-name"; n.textContent = run.name;
    const dsn = document.createElement("div"); dsn.className = "user-email"; dsn.textContent = `${run.dataset_summary.name}, ${run.dataset_summary.total.toLocaleString()} images`;
    name.append(n, dsn);
    const status = document.createElement("td");
    const pill = document.createElement("span"); pill.className = `status-pill ${pills[p.state] || ""}`; pill.textContent = labels[p.state] || p.state;
    status.appendChild(pill);
    if (p.state === "running") status.append(` ${Math.round(p.percent || 0)}%`);
    const hwCell = document.createElement("td"); hwCell.textContent = run.device_name;
    const started = document.createElement("td"); started.textContent = new Date(run.created_at).toLocaleString();
    const result = document.createElement("td");
    if (p.state === "completed" && p.summary) result.textContent = `${pct(p.summary.test_accuracy, 1)} accurate`;
    else if (p.state === "failed") result.textContent = p.message;
    const actions = document.createElement("div"); actions.className = "document-actions";
    if (p.state === "running" || p.state === "queued" || run.id !== training.runId) {
      actions.appendChild(button("View", "btn-ghost", false, () => { followRun(run.id); $("run-live").scrollIntoView({ block: "start" }); }));
    }
    if (run.can_resume) actions.appendChild(button("Resume", "btn-primary", false, () => resumeRun(run.id)));
    result.appendChild(actions);
    labelCells([name, status, hwCell, started, result], ["", "Status", "Hardware", "Started", "Result"]);
    tr.append(name, status, hwCell, started, result);
    rows.appendChild(tr);
  });
}

async function resumeRun(runId) {
  try {
    await api(`/api/training/runs/${runId}/resume`, { method: "POST" });
    $("training-message").hidden = true;
    followRun(runId);
    loadRuns();
    $("run-live").scrollIntoView({ behavior: REDUCED_MOTION ? "auto" : "smooth", block: "start" });
  } catch (err) { showMessage("training-message", err.message, true); }
}

function wireTraining() {
  $("run-resume").addEventListener("click", () => training.runId && resumeRun(training.runId));
  $("train-form").addEventListener("submit", startTraining);
  $("train-scan").addEventListener("click", scanDataset);
  $("train-dataset").addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); scanDataset(); } });
  $("train-browse").addEventListener("click", () => openBrowser($("train-dataset").value.trim() || training.browsePath || ""));
  $("browse-use").addEventListener("click", () => {
    $("train-dataset").value = training.browsePath;
    $("browse-dialog").close();
    scanDataset();
  });
  $("run-cancel").addEventListener("click", async () => {
    if (!training.runId || !window.confirm("Stop training? Nothing is saved to the library.")) return;
    try { await api(`/api/training/runs/${training.runId}/cancel`, { method: "POST" }); }
    catch (err) { showMessage("training-message", err.message, true); }
  });
}

// ---------- Model library ----------
const models = { list: [], current: null, openAfterLoad: null, target: 0.95 };

async function loadModels() {
  let data;
  try { data = await api("/api/models"); }
  catch (err) { showMessage("models-message", err.message, true); return; }
  models.list = data.models;
  models.target = data.target_accuracy;
  $("models-summary").textContent = data.models.length ? plural(data.models.length, "model") : "";
  $("models-empty").hidden = data.models.length > 0;
  $("models-train-link").hidden = !canSee("training");
  const grid = $("model-grid");
  grid.textContent = "";
  data.models.forEach((m) => {
    const card = document.createElement("article");
    card.className = "model-card";
    const head = document.createElement("div"); head.className = "model-card-head";
    const title = document.createElement("h2"); title.textContent = m.name;
    const meta = document.createElement("p"); meta.className = "model-meta";
    meta.textContent = `${m.architecture_label}, ${m.image_size} px. ${m.classes.length} classes from ${m.dataset}.`;
    head.append(title, meta);
    const classes = document.createElement("p"); classes.className = "model-classes";
    classes.textContent = m.classes.slice(0, 8).join(", ") + (m.classes.length > 8 ? ` and ${m.classes.length - 8} more` : "");
    const figures = document.createElement("dl"); figures.className = "model-figures";
    [["Accuracy", pct(m.accuracy, 1)], ["Balanced accuracy", pct(m.balanced_accuracy, 1)], ["Answers", pct(m.coverage)], ["Correct when it answers", pct(m.answered_accuracy, 1)]]
      .forEach(([k, v]) => { const d = document.createElement("div"); const dt = document.createElement("dt"); dt.textContent = k; const dd = document.createElement("dd"); dd.textContent = v; d.append(dt, dd); figures.appendChild(d); });
    const verdict = document.createElement("p");
    verdict.className = `model-verdict ${m.target_met === false ? "warn" : "ok"}`;
    verdict.textContent = m.target_met === false
      ? `Below the ${pct(m.target_accuracy)} target on its ${m.evaluated_on} images when it answers. Treat as experimental.`
      : `Met the ${pct(m.target_accuracy)} target on its ${m.evaluated_on} images when it answers.`;
    const use = document.createElement("p"); use.className = "model-meta";
    use.textContent = m.imaging
      ? `Used on ${m.imaging.modality} series${m.imaging.patch_mm ? `, ${Math.round(m.imaging.patch_mm)} mm regions` : ""}.`
      : "Not set up for CT or MRI series.";
    const foot = document.createElement("p"); foot.className = "model-meta";
    foot.textContent = `Trained ${new Date(m.created_at).toLocaleDateString()} on ${m.device_name} in ${formatDuration(m.training_seconds)}${m.created_by ? ` by ${m.created_by}` : ""}. ${m.size_mb} MB.`;
    const actions = document.createElement("div"); actions.className = "model-actions";
    actions.appendChild(button("Try it", "btn-primary", false, () => openModel(m.id)));
    if (canSee("training")) {
      const dl = document.createElement("a");
      dl.className = "btn btn-sm btn-ghost"; dl.href = `/api/models/${encodeURIComponent(m.id)}/download`; dl.textContent = "Download";
      actions.appendChild(dl);
      actions.appendChild(button("CT and MRI use", "btn-ghost", false, () => openModelImaging(m)));
      actions.appendChild(button("Delete", "btn-ghost", false, async () => {
        if (!window.confirm(`Delete “${m.name}” from the library? This can’t be undone.`)) return;
        try { await api(`/api/models/${encodeURIComponent(m.id)}`, { method: "DELETE" }); showMessage("models-message", "Deleted."); }
        catch (err) { showMessage("models-message", err.message, true); }
        loadModels();
      }));
    }
    card.append(head, classes, figures, verdict, use, foot, actions);
    grid.appendChild(card);
  });
  if (models.openAfterLoad) { const id = models.openAfterLoad; models.openAfterLoad = null; if (data.models.some((m) => m.id === id)) openModel(id); }
}

const IMAGING_WINDOWS = { CT: ["abdomen", "soft tissue", "lung", "bone", "brain"], MR: ["auto"] };

function fillImagingWindows(modality, chosen) {
  const select = $("mi-window");
  select.textContent = "";
  IMAGING_WINDOWS[modality].forEach((w) => {
    const o = document.createElement("option");
    o.value = w; o.textContent = w === "auto" ? "Automatic (1st to 99th percentile)" : w[0].toUpperCase() + w.slice(1);
    select.appendChild(o);
  });
  if (chosen && IMAGING_WINDOWS[modality].includes(chosen)) select.value = chosen;
}

function openModelImaging(m) {
  models.imagingFor = m;
  const s = m.imaging || {};
  $("model-imaging-title").textContent = `Use ${m.name} on CT and MRI`;
  $("mi-modality").value = s.modality || "CT";
  fillImagingWindows($("mi-modality").value, s.window);
  $("mi-orientation").value = s.orientation || "identity";
  $("mi-patch").value = s.patch_mm ?? "";
  $("mi-note").value = s.note || "";
  $("mi-remove").hidden = !m.imaging;
  $("model-imaging-message").hidden = true;
  $("model-imaging-dialog").showModal();
}

async function saveModelImaging(remove) {
  const m = models.imagingFor;
  const patch = $("mi-patch").value.trim();
  const body = remove ? null : { modality: $("mi-modality").value, window: $("mi-window").value, orientation: $("mi-orientation").value,
    patch_mm: patch === "" ? null : Number(patch), note: $("mi-note").value };
  try {
    await api(`/api/models/${encodeURIComponent(m.id)}/imaging`, remove ? { method: "DELETE" } : { method: "PUT", body });
    $("model-imaging-dialog").close();
    showMessage("models-message", remove ? `${m.name} is no longer used on series.` : `Saved. ${m.name} can now be run on ${body.modality} series.`);
    loadModels();
  } catch (err) {
    showMessage("model-imaging-message", err.message, true);
  }
}

async function openModel(id) {
  let data;
  try { data = await api(`/api/models/${encodeURIComponent(id)}`); }
  catch (err) { showMessage("models-message", err.message, true); return; }
  const m = data.model;
  models.current = m;
  $("model-dialog-title").textContent = m.name;
  $("model-meta").textContent = `${m.intended_use} ${m.classes.length} classes: ${m.classes.join(", ")}.`;
  $("try-result").textContent = "";
  $("try-preview").hidden = true;
  $("try-hint").hidden = false;
  $("try-file").value = "";
  const r = m.test || m.validation;
  const where = m.test ? "test" : "validation";
  const stats = $("model-stats");
  stats.textContent = "";
  stats.append(
    statTile("Accuracy", pct(r.accuracy, 1), `${r.images.toLocaleString()} ${where} images`),
    statTile("Balanced accuracy", pct(r.balanced_accuracy, 1), "average sensitivity across classes"),
    statTile("AUC", r.auc === null ? "–" : r.auc.toFixed(3), m.classes.length > 2 ? "average, one class against the rest" : ""),
    statTile("Confidence threshold", pct(m.threshold.threshold), m.threshold.met ? `chosen for ${pct(m.threshold.target)} accuracy on validation images` : "no share of images met the target: it abstains on all", { attention: !m.threshold.met }),
    statTile("Answers", pct(r.abstention.coverage), `${pct(r.abstention.answered_accuracy, 1)} correct when it does`, { attention: r.target_met === false }),
    statTile("Calibration error", r.calibration_error === null ? "–" : pct(r.calibration_error, 1), "gap between confidence and accuracy"),
  );
  const rows = $("model-classes");
  rows.textContent = "";
  r.per_class.forEach((c) => {
    const tr = document.createElement("tr");
    [c.name, c.support.toLocaleString(), pct(c.sensitivity, 1), pct(c.specificity, 1), pct(c.precision, 1), c.auc === null ? "–" : c.auc.toFixed(3)]
      .forEach((v, i) => { const td = document.createElement("td"); td.textContent = v; if (i) td.className = "mono"; tr.appendChild(td); });
    rows.appendChild(tr);
  });
  renderConfusion($("model-confusion"), m.classes, r.confusion);
  const tr = m.training;
  const items = [
    ["Dataset", `${m.dataset.name}: ${m.dataset.total.toLocaleString()} images (${m.dataset.splits.train.toLocaleString()} training, ${m.dataset.splits.val.toLocaleString()} validation, ${m.dataset.splits.test.toLocaleString()} test). Fingerprint ${m.dataset.fingerprint}.`],
    ["Data checks", m.dataset.checks ? [
      m.dataset.checks.duplicates.leaked ? `${m.dataset.checks.duplicates.leaked.toLocaleString()} ${m.dataset.checks.duplicates.leaked === 1 ? "copy" : "copies"} of training images left out of validation and test.` : "No training images copied into validation or test.",
      m.dataset.checks.duplicates.conflicting_labels ? `${plural(m.dataset.checks.duplicates.conflicting_labels, "identical image")} filed under different classes.` : "",
      m.dataset.checks.identifiers.file_names || m.dataset.checks.identifiers.metadata_files ? "Possible patient details found in file names or metadata." : "No patient details found in file names or sampled metadata.",
    ].filter(Boolean).join(" ") : "Not recorded for this model."],
    ["Architecture", `${m.architecture === "resnet18" ? "ResNet-18" : "Small CNN"}${m.pretrained ? ", from ImageNet weights" : ", trained from scratch"}. Input ${m.input.image_size} × ${m.input.image_size} px, ${m.input.channels === 1 ? "grayscale" : "colour"}.`],
    ["Training", `${m.history.length} of ${tr.epochs} epochs (best: ${m.best_epoch}), batch ${tr.batch_size}, learning rate ${tr.learning_rate}, on ${m.hardware.device_name} in ${formatDuration(m.training_seconds)}.`],
    ["Unfamiliar images", m.novelty ? `Abstains on images far from anything it was trained on (1% of validation images would be flagged${m.test && m.test.novelty_flagged !== undefined ? `; ${pct(m.test.novelty_flagged, 1)} of test images were` : ""}).` : "No check recorded for this model."],
    ["Created", `${new Date(m.created_at).toLocaleString()}${m.created_by ? ` by ${m.created_by}` : ""}. ID ${m.id}.`],
  ];
  const dl = $("model-training");
  dl.textContent = "";
  items.forEach(([k, v]) => { const dt = document.createElement("dt"); dt.textContent = k; const dd = document.createElement("dd"); dd.textContent = v; dl.append(dt, dd); });
  if (!$("model-dialog").open) $("model-dialog").showModal();
}

function renderConfusion(container, classes, matrix) {
  container.textContent = "";
  const table = document.createElement("table");
  table.className = "confusion";
  const max = Math.max(1, ...matrix.flat());
  const head = document.createElement("tr");
  head.appendChild(document.createElement("th"));
  classes.forEach((c) => { const th = document.createElement("th"); th.scope = "col"; th.textContent = c; th.title = `Predicted ${c}`; head.appendChild(th); });
  table.appendChild(head);
  matrix.forEach((row, i) => {
    const tr = document.createElement("tr");
    const th = document.createElement("th"); th.scope = "row"; th.textContent = classes[i]; tr.appendChild(th);
    const total = row.reduce((a, b) => a + b, 0) || 1;
    row.forEach((v, j) => {
      const td = document.createElement("td");
      td.textContent = v.toLocaleString();
      const share = v / total;
      td.style.setProperty("--share", Math.sqrt(v / max).toFixed(3));
      td.className = i === j ? `diag${Math.sqrt(v / max) >= 0.5 ? " strong" : ""}` : v ? "off" : "zero";
      td.title = `${classes[i]} predicted as ${classes[j]}: ${v} (${Math.round(share * 100)}% of ${classes[i]})`;
      tr.appendChild(td);
    });
    table.appendChild(tr);
  });
  container.appendChild(table);
}

async function tryModel(file) {
  if (!file || !models.current) return;
  const preview = $("try-preview");
  preview.src = URL.createObjectURL(file);
  preview.hidden = false;
  $("try-hint").hidden = true;
  const out = $("try-result");
  out.textContent = "Checking…";
  const body = new FormData();
  body.append("file", file);
  try {
    const res = await fetch(`/api/models/${encodeURIComponent(models.current.id)}/predict`, { method: "POST", body });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || `Request failed (${res.status}).`);
    out.textContent = "";
    const verdict = document.createElement("div");
    verdict.className = `try-verdict ${data.abstained ? "abstained" : "answered"}`;
    const chip = document.createElement("span");
    chip.className = `status-chip ${data.abstained ? "refuse" : "answer"}`;
    chip.textContent = data.abstained ? (data.abstain_reason === "unfamiliar" ? "UNFAMILIAR IMAGE" : "NOT CONFIDENT") : data.prediction.toUpperCase();
    const msg = document.createElement("p"); msg.textContent = data.message;
    verdict.append(chip, msg);
    out.appendChild(verdict);
    const bars = document.createElement("div"); bars.className = "bars";
    renderBars(bars, data.probabilities.slice(0, 6).map((p) => ({ label: p.class, value: Math.round(p.probability * 100) })), "");
    bars.querySelectorAll(".bar-value").forEach((v) => { v.textContent += "%"; });
    out.appendChild(bars);
  } catch (err) {
    out.textContent = err.message;
  }
}

function wireModels() {
  $("try-file").addEventListener("change", (e) => tryModel(e.target.files[0]));
  const drop = $("try-drop");
  ["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("dragging"); }));
  ["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("dragging"); }));
  drop.addEventListener("drop", (e) => tryModel(e.dataTransfer.files[0]));
}

// ---------- Patient ----------
const splitList = (value) => value.split(/[,;\n]/).map((s) => s.trim()).filter(Boolean);
const PATIENT_NUMBERS = [["pt-age", "age_years"], ["pt-weight", "weight_kg"], ["pt-egfr", "egfr"], ["pt-creatinine", "creatinine_umol_l"]];

// The patient details as the API expects them: null when nothing is entered,
// false when something entered can't be used.
function readPatient({ quiet = false } = {}) {
  if (!quiet) $("pt-error").textContent = "";
  const p = {};
  for (const [id, key] of PATIENT_NUMBERS) {
    const el = $(id);
    if (el.value === "") continue;
    if (!el.checkValidity()) {
      if (quiet) continue;
      $("patient-box").open = true;
      $("pt-error").textContent = `${el.labels[0].textContent.replace(/\s+/g, " ").trim()}: ${el.validationMessage}`;
      el.focus();
      return false;
    }
    p[key] = Number(el.value);
  }
  if ($("pt-sex").value) p.sex = $("pt-sex").value;
  if ($("pt-liver").value) p.child_pugh = $("pt-liver").value;
  if ($("pt-pregnant").checked) p.pregnant = true;
  if ($("pt-breastfeeding").checked) p.breastfeeding = true;
  for (const [id, key] of [["pt-allergies", "allergies"], ["pt-medicines", "medicines"], ["pt-conditions", "conditions"]]) {
    const items = splitList($(id).value);
    if (items.length) p[key] = items;
  }
  const labs = {};
  document.querySelectorAll("[data-lab]").forEach((el) => { if (el.value !== "") labs[el.dataset.lab] = Number(el.value); });
  if (Object.keys(labs).length) p.labs = labs;
  return Object.keys(p).length ? p : null;
}

function updatePatientStatus() {
  const p = readPatient({ quiet: true });
  const status = $("patient-status");
  if (!p) { status.textContent = "No patient details"; status.classList.remove("on"); return; }
  const bits = [];
  if (p.age_years !== undefined) bits.push(`${p.age_years} y`);
  if (p.sex) bits.push(p.sex);
  if (p.weight_kg !== undefined) bits.push(`${p.weight_kg} kg`);
  if (p.egfr !== undefined) bits.push(`eGFR ${p.egfr}`);
  if (p.pregnant) bits.push("pregnant");
  if (p.allergies) bits.push(`${p.allergies.length} ${p.allergies.length === 1 ? "allergy" : "allergies"}`);
  if (p.medicines) bits.push(plural(p.medicines.length, "medicine"));
  status.textContent = bits.join(", ") || "Details entered";
  status.classList.add("on");
}

const FINDING_LABEL = { block: "Unsafe for this patient", warn: "Check", info: "Note" };

function renderPatientFindings(findings, given) {
  const box = $("patient-findings");
  const list = $("patient-findings-list");
  list.textContent = "";
  box.hidden = !given;
  if (!given) return;
  if (!findings.length) {
    const li = document.createElement("li");
    li.className = "finding pass";
    li.textContent = "No problems found for this patient with the medicines in this answer.";
    list.appendChild(li);
    return;
  }
  const order = { block: 0, warn: 1, info: 2 };
  [...findings].sort((a, b) => order[a.severity] - order[b.severity]).forEach((f) => {
    const li = document.createElement("li");
    li.className = `finding ${f.severity}`;
    const tag = document.createElement("span");
    tag.className = "finding-tag";
    tag.textContent = FINDING_LABEL[f.severity];
    const text = document.createElement("span");
    text.className = "finding-text";
    const med = document.createElement("strong");
    med.textContent = `${f.medicine}: `;
    text.append(med, f.message);
    li.append(tag, text);
    if (f.rule_id || f.source) {
      const meta = document.createElement("span");
      meta.className = "finding-meta mono";
      meta.textContent = [f.rule_id, f.source].filter(Boolean).join(" · ");
      li.appendChild(meta);
    }
    list.appendChild(li);
  });
}

function wirePatient() {
  const box = $("patient-box");
  box.querySelectorAll("input, select").forEach((el) => el.addEventListener("input", updatePatientStatus));
  box.querySelectorAll("input[type=checkbox], select").forEach((el) => el.addEventListener("change", updatePatientStatus));
  $("pt-clear").addEventListener("click", () => {
    box.querySelectorAll("input").forEach((el) => { if (el.type === "checkbox") el.checked = false; else el.value = ""; });
    box.querySelectorAll("select").forEach((el) => { el.value = ""; });
    updatePatientStatus();
    $("pt-age").focus();
  });
  $("pt-sex").addEventListener("change", () => {
    const male = $("pt-sex").value === "male";
    ["pt-pregnant", "pt-breastfeeding"].forEach((id) => { $(id).disabled = male; if (male) $(id).checked = false; });
  });
  api("/api/formulary").then((data) => {
    const list = $("formulary-names");
    data.names.forEach((n) => { const o = document.createElement("option"); o.value = n; list.appendChild(o); });
  }).catch(() => {});
}

// ---------- Medicines ----------
let medicineTimer = null;
async function loadMedicines() {
  let data;
  try { data = await api(`/api/formulary?q=${encodeURIComponent($("medicine-search").value)}`); }
  catch (_) { return; }
  $("formulary-synthetic").hidden = !data.synthetic;
  $("medicines-summary").textContent = `${data.total.toLocaleString()} medicines, ${data.name} ${data.version}`;
  const rows = $("medicine-rows");
  rows.textContent = "";
  $("medicines-empty").hidden = data.medicines.length > 0;
  data.medicines.forEach((m) => {
    const tr = document.createElement("tr");
    const name = document.createElement("td");
    const n = document.createElement("div"); n.className = "user-name"; n.textContent = m.name;
    if (m.high_alert) { const b = document.createElement("span"); b.className = "status-pill status-rejected"; b.textContent = "High alert"; n.append(" ", b); }
    const c = document.createElement("div"); c.className = "user-email"; c.textContent = m.classes.join(", ");
    name.append(n, c);
    const dose = document.createElement("td");
    dose.textContent = m.adult_dose ? `${m.adult_dose.amount} ${m.adult_dose.unit} ${m.adult_dose.frequency}` : "By weight or not stated";
    const rules = document.createElement("td");
    const group = document.createElement("div"); group.className = "pill-group";
    const r = m.rules;
    [[r.allergies, "Allergy"], [r.interactions, "Interactions"], [r.kidney, "Kidney"], [r.liver, "Liver"], [r.children, "Children"],
     [r.weight, "Weight"], [r.labs, "Labs"], [r.pregnancy !== "no_data", `Pregnancy: ${r.pregnancy}`]]
      .filter(([on]) => on).forEach(([, label]) => { const s = document.createElement("span"); s.className = "status-pill"; s.textContent = label; group.appendChild(s); });
    rules.appendChild(group);
    const act = document.createElement("td");
    const actions = document.createElement("div"); actions.className = "document-actions";
    actions.appendChild(button("View rules", "btn-ghost", false, () => openMedicine(m.name)));
    act.appendChild(actions);
    labelCells([name, dose, rules, act], ["", "Adult dose", "Rules", ""]);
    tr.append(name, dose, rules, act);
    rows.appendChild(tr);
  });
}

async function openMedicine(name) {
  let data;
  try { data = await api(`/api/formulary/${encodeURIComponent(name)}`); } catch (_) { return; }
  const m = data.medicine;
  $("medicine-dialog-title").textContent = m.name;
  const doseText = (d) => d ? `${d.amount} ${d.unit} ${d.frequency}${d.max_daily ? `, at most ${d.max_daily} ${d.unit} a day` : ""}` : "";
  const weightText = (d) => d ? `${d.per_kg} ${d.unit}/kg ${d.frequency}${d.max_single ? `, at most ${d.max_single} ${d.unit} a dose` : ""}` : "";
  const rows = [
    ["Also called", m.aliases.join(", ")],
    ["Classes", m.classes.join(", ")],
    ["High alert", m.high_alert ? "Yes: use an independent double check" : ""],
    ["Adult dose", doseText(m.adult_dose) + (m.adult_dose ? ` (from age ${m.adult_min_age})` : "")],
    ["By weight", weightText(m.weight_dose)],
    ["Children", m.paediatric_dose ? `${weightText(m.paediatric_dose)}${m.paediatric_min_age != null ? `, from age ${m.paediatric_min_age}` : ""}` : "No paediatric dosing: questions about children are refused"],
    ["Allergies", m.allergy_groups.join(", ")],
    ["Conditions", m.contraindicated_conditions.join(", ")],
    ["Interactions", m.interactions.map((i) => `${i.with_medicine || i.with_class} (${i.severity})${i.note ? `: ${i.note}` : ""}`).join(" · ")],
    ["Kidney", m.renal.map((r) => `eGFR below ${r.egfr_below}: ${r.action}${r.dose ? ` to ${doseText(r.dose)}` : ""}`).join(" · ")],
    ["Liver", m.hepatic.map((r) => `Child-Pugh ${r.child_pugh}: ${r.action}`).join(" · ")],
    ["Pregnancy", { avoid: "Avoid", no_data: "No safety information: refused", compatible: "Compatible" }[m.pregnancy]],
    ["Breastfeeding", { avoid: "Avoid", no_data: "No safety information: warning", compatible: "Compatible" }[m.breastfeeding]],
    ["Lab results", m.labs.map((l) => `${l.lab} ${l.above != null ? `above ${l.above}` : `below ${l.below}`}: ${l.action}`).join(" · ")],
    ["Needs before dosing", m.requires.join(", ")],
    ["Sources", m.source_ids.join(", ")],
  ].filter(([, v]) => v);
  const dl = $("medicine-rules");
  dl.textContent = "";
  rows.forEach(([k, v]) => { const dt = document.createElement("dt"); dt.textContent = k; const dd = document.createElement("dd"); dd.textContent = v; dl.append(dt, dd); });
  $("medicine-dialog").showModal();
}

function renderPatientEval(p) {
  const wrap = $("patient-eval-block");
  if (!wrap || !p || !p.total) { if (wrap) wrap.hidden = true; return; }
  wrap.hidden = false;
  wrap.textContent = "";
  const head = document.createElement("div"); head.className = "adv-head";
  const title = document.createElement("span"); title.className = "eyebrow"; title.textContent = "Patient scenarios";
  const score = document.createElement("span"); score.className = "adv-score";
  score.textContent = `${p.passed.toLocaleString()} of ${p.total.toLocaleString()} correct, ${plural(p.unsafe_answers, "unsafe answer")}`;
  head.append(title, score);
  const note = document.createElement("p"); note.className = "adv-note";
  note.textContent = "The same questions asked for different patients: allergies, interactions, children, pregnancy, kidney and liver function, and missing details. Any unsafe answer fails the build.";
  wrap.append(head, note);
  if (p.failures && p.failures.length) {
    const list = document.createElement("ul"); list.className = "eval-cases";
    p.failures.forEach((f) => { const li = document.createElement("li"); li.className = "eval-case"; li.textContent = `${f.query}: expected ${f.expect}, got ${f.got}`; list.appendChild(li); });
    wrap.appendChild(list);
  }
}

// ---------- EHR ----------
const ehr = { config: null };
const CDS_EXAMPLE = {
  hook: "order-sign",
  hookInstance: "d1577c69-dfbe-44ad-ba6d-3e05e953b2ea",
  context: {
    userId: "Practitioner/example", patientId: "example",
    draftOrders: { resourceType: "Bundle", entry: [{ resource: {
      resourceType: "MedicationRequest", id: "order-1", status: "draft", intent: "order",
      medicationCodeableConcept: { text: "Caloradine" },
      dosageInstruction: [{ doseAndRate: [{ doseQuantity: { value: 15, unit: "mg" } }], timing: { repeat: { frequency: 1, period: 1, periodUnit: "d" } } }],
    } }] },
  },
  prefetch: {
    patient: { resourceType: "Patient", id: "example", birthDate: "1956-04-02", gender: "female" },
    observations: { resourceType: "Bundle", entry: [{ resource: {
      resourceType: "Observation", status: "final", code: { coding: [{ system: "http://loinc.org", code: "62238-1" }] },
      effectiveDateTime: new Date(Date.now() - 2 * 86400000).toISOString(), valueQuantity: { value: 41, unit: "mL/min/1.73m2" },
    } }] },
    allergies: { resourceType: "Bundle", entry: [] },
    medications: { resourceType: "Bundle", entry: [{ resource: { resourceType: "MedicationRequest", status: "active", intent: "order", medicationCodeableConcept: { text: "Mendel solution 5 mL" } } }] },
    conditions: { resourceType: "Bundle", entry: [] },
  },
};

function fillPatient(p) {
  const set = (id, v) => { $(id).value = v ?? ""; };
  set("pt-age", p.age_years); set("pt-sex", p.sex); set("pt-weight", p.weight_kg); set("pt-egfr", p.egfr);
  set("pt-creatinine", p.creatinine_umol_l); set("pt-liver", p.child_pugh);
  $("pt-pregnant").checked = !!p.pregnant; $("pt-breastfeeding").checked = !!p.breastfeeding;
  set("pt-allergies", (p.allergies || []).join(", ")); set("pt-medicines", (p.medicines || []).join(", "));
  set("pt-conditions", (p.conditions || []).join(", "));
  document.querySelectorAll("[data-lab]").forEach((el) => { el.value = (p.labs || {})[el.dataset.lab] ?? ""; });
  $("pt-sex").dispatchEvent(new Event("change"));
  updatePatientStatus();
}

function applyEhrContext(context) {
  const strip = $("ehr-strip");
  ehr.context = context;
  if (!context) { strip.hidden = true; return; }
  fillPatient(context.patient);
  const source = context.source || {};
  let host = source.server || "";
  try { host = new URL(source.server).host; } catch (_) { /* keep as is */ }
  $("ehr-source").textContent = `From ${source.system === "SMART on FHIR" ? "the EHR" : host}${source.patient ? `, ${source.patient}` : ""}`;
  const time = (iso) => new Date(iso).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  $("ehr-meta").textContent = ` Loaded ${time(context.loaded_at)}, kept until ${time(context.expires_at)}. Check the details before asking.`;
  const warnings = $("ehr-warnings");
  warnings.textContent = "";
  (context.warnings || []).forEach((w) => { const li = document.createElement("li"); li.textContent = w; warnings.appendChild(li); });
  strip.hidden = false;
  $("patient-box").open = true;
}

async function loadEhrForAsk() {
  try { ehr.config = await api("/api/ehr/config"); } catch (_) { return; }
  const servers = ehr.config.open_servers || [];
  $("ehr-load").hidden = servers.length === 0;
  const select = $("ehr-server");
  select.textContent = "";
  servers.forEach((s) => select.appendChild(new Option(s, s)));
  try {
    const data = await api("/api/ehr/context");
    if (data.context) applyEhrContext(data.context);
  } catch (_) { /* no database */ }
  const params = new URLSearchParams(location.search);
  if (params.get("ehr_error")) {
    $("patient-box").open = true;
    $("pt-error").textContent = params.get("ehr_error");
  }
  if (params.has("ehr") || params.has("ehr_error")) history.replaceState(null, "", location.pathname + location.hash);
}

async function loadEhrPage() {
  let c;
  try { c = await api("/api/ehr/config"); } catch (err) { return; }
  const list = (id, rows) => {
    const dl = $(id); dl.textContent = "";
    rows.forEach(([k, v]) => { const dt = document.createElement("dt"); dt.textContent = k; const dd = document.createElement("dd"); dd.textContent = v; dl.append(dt, dd); });
  };
  $("smart-status").textContent = c.smart_enabled
    ? "Set up. Register these addresses with your EHR."
    : "Not set up. Set SMART_CLIENT_ID and SMART_ALLOWED_ISSUERS, then register these addresses with your EHR.";
  list("smart-settings", [
    ["Launch URL", c.smart_launch_url], ["Redirect URL", c.smart_redirect_url],
    ["Client ID", c.smart_client_id || "Not set"], ["Allowed EHRs", c.smart_allowed_issuers.join(", ") || "None"],
    ["Scopes", c.smart_scopes], ["Open FHIR servers", c.open_servers.join(", ") || "None"],
    ["Patient kept for", `${c.context_minutes} minutes`],
  ]);
  $("cds-status").textContent = c.cds_trusted.length
    ? "Accepting signed calls from the EHRs below."
    : c.cds_unsigned ? "Accepting unsigned calls: for testing only." : "No EHR is trusted yet. Set CDS_HOOKS_TRUSTED.";
  list("cds-settings", [
    ["Discovery URL", c.cds_discovery_url], ["Hooks", "order-select, order-sign"],
    ["Trusted EHRs", c.cds_trusted.join(", ") || "None"], ["Unsigned calls", c.cds_unsigned ? "Allowed (testing)" : "Refused"],
  ]);
  if (!$("cds-request").value) $("cds-request").value = JSON.stringify(CDS_EXAMPLE, null, 2);
}

function renderCdsCards(cards) {
  const wrap = $("cds-cards");
  wrap.textContent = "";
  if (!cards.length) {
    const p = document.createElement("p"); p.className = "auth-hint"; p.textContent = "No cards: nothing to flag for this order.";
    wrap.appendChild(p);
    return;
  }
  cards.forEach((card) => {
    const el = document.createElement("article");
    el.className = `cds-card ${card.indicator}`;
    const head = document.createElement("div"); head.className = "cds-card-head";
    const tag = document.createElement("span"); tag.className = "finding-tag"; tag.textContent = { critical: "Critical", warning: "Warning", info: "Info" }[card.indicator];
    const summary = document.createElement("strong"); summary.textContent = card.summary;
    head.append(tag, summary);
    const detail = document.createElement("p"); detail.textContent = card.detail.replace(/\*\*/g, "");
    const source = document.createElement("p"); source.className = "finding-meta mono"; source.textContent = card.source.label;
    el.append(head, detail, source);
    if (card.overrideReasons && card.overrideReasons.length) {
      const reasons = document.createElement("p"); reasons.className = "auth-hint";
      reasons.textContent = `Override reasons offered: ${card.overrideReasons.map((r) => r.display).join("; ")}`;
      el.appendChild(reasons);
    }
    wrap.appendChild(el);
  });
}

function wireEhr() {
  $("ehr-load-button").addEventListener("click", async (e) => {
    const trigger = e.currentTarget;
    const patientId = $("ehr-patient-id").value.trim();
    if (!patientId) { $("pt-error").textContent = "Enter the patient's FHIR ID."; $("ehr-patient-id").focus(); return; }
    trigger.disabled = true; trigger.textContent = "Loading…";
    try {
      const data = await api("/api/ehr/load", { method: "POST", body: { server: $("ehr-server").value, patient_id: patientId } });
      $("pt-error").textContent = "";
      applyEhrContext(data.context);
    } catch (err) { $("pt-error").textContent = err.message; }
    trigger.disabled = false; trigger.textContent = "Load from EHR";
  });
  $("ehr-save-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const form = e.currentTarget;
    if (!window.confirm("Save this answer and the patient checks to the patient's record as a preliminary note?")) return;
    formBusy(form, true);
    try {
      const data = await api("/api/ehr/notes", { method: "POST", body: { audit_id: $("ehr-save").dataset.auditId, comment: $("ehr-save-comment").value } });
      form.hidden = true;
      showMessage("ehr-save-message", `Saved to the record as a preliminary note${data.reference ? ` (${data.reference})` : ""}.`);
    } catch (err) { showMessage("ehr-save-message", err.message, true); }
    formBusy(form, false);
  });
  $("ehr-forget").addEventListener("click", async () => {
    await api("/api/ehr/context", { method: "DELETE" }).catch(() => {});
    ehr.context = null;
    $("ehr-save").hidden = true;
    $("ehr-strip").hidden = true;
    $("pt-clear").click();
  });
  $("cds-run").addEventListener("click", async () => {
    $("cds-message").hidden = true;
    let body;
    try { body = JSON.parse($("cds-request").value); }
    catch (_) { showMessage("cds-message", "The request isn't valid JSON.", true); return; }
    const res = await fetch("/cds-services/groundcheck-medication-safety-order-sign", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    });
    const data = await res.json().catch(() => ({}));
    if (res.status === 401) {
      showMessage("cds-message", "This server only accepts calls signed by a trusted EHR. To try requests here, run a test server with CDS_HOOKS_ALLOW_UNSIGNED=true.", true);
      $("cds-cards").textContent = "";
      return;
    }
    if (!res.ok) { showMessage("cds-message", data.detail || `The service returned ${res.status}.`, true); return; }
    renderCdsCards(data.cards);
  });
}

// ---------- CT and MRI ----------
// A list of imported series, and a viewer for one (#/imaging/12): slices with
// windowing and the model's regions, the analysis, and the clinician's report.
const imaging = { home: null, series: null, index: 0, window: null, poll: null, files: [], editing: null };
const LABEL_COLOURS = 8;
const ORIENTATION = {
  axial: ["A", "P", "R", "L"], coronal: ["S", "I", "R", "L"], sagittal: ["S", "I", "A", "P"],
};
const AGREEMENT_LABELS = { agree: "Agrees with the model", partly: "Partly agrees with the model", disagree: "Disagrees with the model", not_used: "Model not used" };

function canDeleteSeries() {
  return !account.authRequired || ["reviewer", "admin"].includes(account.user?.role);
}

function imagingRoute() {
  const match = location.hash.match(/^#\/imaging\/(\d+)/);
  return match ? Number(match[1]) : null;
}

async function loadImaging() {
  clearTimeout(imaging.poll);
  const id = imagingRoute();
  $("imaging-list").hidden = id !== null;
  $("imaging-viewer").hidden = id === null;
  try { imaging.home = await api("/api/imaging"); }
  catch (err) { showMessage(id === null ? "imaging-message" : "viewer-message", err.message, true); return; }
  if (id === null) renderImagingList();
  else openSeries(id);
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = text;
  return node;
}

function shortDate(iso) {
  return iso ? new Date(iso).toLocaleString(undefined, { day: "numeric", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit" }) : "–";
}

function seriesTitle(s) {
  return s.label || s.description || `${s.modality} series ${s.id}`;
}

function renderImagingList() {
  const { series, pacs } = imaging.home;
  document.title = "CT and MRI · GroundCheck";
  $("imaging-summary").textContent = series.length ? plural(series.length, "series").replace(/seriess$/, "series") : "";
  $("pacs-off").hidden = pacs;
  $("pacs-search").hidden = !pacs;
  const rows = $("imaging-rows");
  rows.textContent = "";
  $("imaging-empty").hidden = series.length > 0;
  series.forEach((s) => {
    const tr = document.createElement("tr");
    const name = document.createElement("td");
    const link = el("a", "series-link");
    link.href = `#/imaging/${s.id}`;
    link.append(el("span", "modality", s.modality), seriesTitle(s));
    name.appendChild(link);
    const sub = [s.label && s.description ? s.description : "", s.plane && s.plane !== "unknown" ? s.plane : "", s.source === "pacs" ? "from the PACS" : ""].filter(Boolean).join(", ");
    if (sub) name.appendChild(el("span", "series-sub", sub));
    const images = el("td", "mono", `${s.slices} × ${s.columns}×${s.rows}`);
    images.dataset.label = "Images";
    const when = el("td", "", shortDate(s.created_at));
    when.dataset.label = "Imported";
    const model = document.createElement("td");
    model.dataset.label = "Model";
    model.appendChild(analysisPill(s.analysis));
    const report = document.createElement("td");
    report.dataset.label = "Report";
    report.appendChild(s.report === "signed" ? el("span", "status-pill status-approved", "Signed")
      : s.report === "draft" ? el("span", "status-pill status-pending", "Draft") : el("span", "muted", "None"));
    tr.append(name, images, when, model, report);
    rows.appendChild(tr);
  });
}

function analysisPill(a) {
  if (!a) return el("span", "muted", "Not run");
  if (a.status === "queued" || a.status === "running") return el("span", "status-pill status-running", "Running");
  if (a.status === "refused") return el("span", "status-pill status-rejected", "Refused");
  if (a.status === "failed") return el("span", "status-pill status-rejected", "Failed");
  if (a.abstained) return el("span", "status-pill status-abstained", "Abstained");
  return el("span", "status-pill status-approved", "Done");
}

// --- Import ---

function setImportTab(which) {
  const upload = which === "upload";
  $("imaging-tab-upload").setAttribute("aria-selected", String(upload));
  $("imaging-tab-pacs").setAttribute("aria-selected", String(!upload));
  $("imaging-tab-upload").tabIndex = upload ? 0 : -1;
  $("imaging-tab-pacs").tabIndex = upload ? -1 : 0;
  $("imaging-upload").hidden = !upload;
  $("imaging-pacs").hidden = upload;
}

function chooseDicomFiles(files) {
  imaging.files = [...files].filter((f) => !f.name.startsWith("."));
  const drop = $("dicom-drop");
  drop.classList.toggle("chosen", imaging.files.length > 0);
  const bytes = imaging.files.reduce((n, f) => n + f.size, 0);
  $("dicom-drop-sub").textContent = imaging.files.length
    ? `${plural(imaging.files.length, "file")} chosen, ${(bytes / 1024 ** 2).toFixed(1)} MB`
    : "CT and MR image storage. Up to 3,000 files.";
  $("dicom-upload").disabled = imaging.files.length === 0;
}

async function readDroppedEntries(items) {
  const out = [];
  const walk = async (entry) => {
    if (entry.isFile) {
      out.push(await new Promise((resolve, reject) => entry.file(resolve, reject)));
    } else if (entry.isDirectory) {
      const reader = entry.createReader();
      let batch;
      do {
        batch = await new Promise((resolve, reject) => reader.readEntries(resolve, reject));
        for (const child of batch) await walk(child);
      } while (batch.length);
    }
  };
  for (const item of items) {
    const entry = item.webkitGetAsEntry && item.webkitGetAsEntry();
    if (entry) await walk(entry);
    else if (item.getAsFile()) out.push(item.getAsFile());
  }
  return out;
}

function uploadDicom(event) {
  event.preventDefault();
  if (!imaging.files.length) return;
  const limit = imaging.home?.max_upload_mb || 1024;
  const bytes = imaging.files.reduce((n, f) => n + f.size, 0);
  if (imaging.files.length > 3000) { showMessage("imaging-message", "Choose up to 3,000 files, or upload a zip.", true); return; }
  if (bytes > limit * 1024 ** 2) { showMessage("imaging-message", `Uploads can be up to ${limit} MB.`, true); return; }
  const form = new FormData();
  imaging.files.forEach((f) => form.append("files", f, f.webkitRelativePath || f.name));
  form.append("label", $("dicom-label").value.trim());
  const xhr = new XMLHttpRequest();
  xhr.open("POST", "/api/imaging/upload");
  $("dicom-progress").hidden = false;
  $("dicom-upload").disabled = true;
  $("imaging-message").hidden = true;
  xhr.upload.onprogress = (e) => {
    if (!e.lengthComputable) return;
    const share = e.loaded / e.total;
    $("dicom-progress-fill").style.width = `${Math.round(share * 100)}%`;
    $("dicom-progress-text").textContent = share < 1 ? `Uploading, ${Math.round(share * 100)}%` : "De-identifying and reading the images";
  };
  xhr.onloadend = () => {
    $("dicom-progress").hidden = true;
    $("dicom-progress-fill").style.width = "0";
    let data = null;
    try { data = JSON.parse(xhr.responseText); } catch (_) { /* not JSON */ }
    if (xhr.status !== 200) {
      $("dicom-upload").disabled = false;
      showMessage("imaging-message", (data && typeof data.detail === "string") ? data.detail : `The upload failed (${xhr.status || "network error"}).`, true);
      return;
    }
    chooseDicomFiles([]);
    $("dicom-files").value = "";
    $("dicom-label").value = "";
    importFinished(data);
  };
  xhr.send(form);
}

function importFinished(data) {
  const added = data.added || [];
  const existing = data.existing || [];
  const parts = [];
  if (added.length) parts.push(`Imported ${plural(added.length, "series").replace(/seriess$/, "series")}, ${plural(added.reduce((n, s) => n + s.slices, 0), "image")}.`);
  if (existing.length) parts.push(`${existing.length === 1 ? "One series was" : `${existing.length} series were`} already imported.`);
  const warnings = added.flatMap((s) => s.warnings || []);
  showMessage("imaging-message", [...parts, ...warnings].join(" "));
  const only = added.length + existing.length === 1 ? (added[0] || existing[0]) : null;
  if (only) location.hash = `#/imaging/${only.id}`;
  else loadImaging();
}

async function searchPacs(event) {
  event.preventDefault();
  const params = new URLSearchParams({
    patient_id: $("pacs-patient-id").value, patient_name: $("pacs-patient-name").value,
    accession: $("pacs-accession").value, study_date: $("pacs-date").value, modality: $("pacs-modality").value,
  });
  const out = $("pacs-results");
  out.textContent = "";
  $("pacs-go").disabled = true;
  try {
    const { studies } = await api(`/api/imaging/pacs/studies?${params}`);
    if (!studies.length) out.appendChild(el("p", "auth-hint", "No studies match."));
    studies.forEach((study) => out.appendChild(pacsStudy(study)));
  } catch (err) {
    out.appendChild(el("p", "modal-message error", err.message));
  } finally {
    $("pacs-go").disabled = false;
  }
}

function pacsStudy(study) {
  const card = el("article", "pacs-study");
  const head = el("div", "pacs-study-head");
  head.append(el("strong", "", study.patient_name || "Unnamed patient"),
    el("span", "pacs-study-meta mono", [study.patient_id, study.study_date, study.accession].filter(Boolean).join("  ")));
  const desc = el("p", "pacs-study-meta", [study.description, Array.isArray(study.modalities) ? study.modalities.join(", ") : study.modalities].filter(Boolean).join(". "));
  const list = el("ul", "pacs-series");
  const show = button("Show series", "btn-ghost", false, async () => {
    show.disabled = true;
    try {
      const { series } = await api(`/api/imaging/pacs/studies/${encodeURIComponent(study.study_uid)}/series`);
      list.textContent = "";
      series.forEach((s) => {
        const li = document.createElement("li");
        const usable = ["CT", "MR"].includes(s.modality);
        li.append(el("span", "", `${s.modality || "?"} ${s.description || "Series " + (s.number ?? "")}${s.instances ? `, ${plural(Number(s.instances), "image")}` : ""}`));
        const importButton = button(usable ? "Import" : "Not CT or MR", usable ? "btn-primary" : "btn-ghost", !usable, async () => {
          importButton.disabled = true;
          importButton.textContent = "Retrieving…";
          try {
            importFinished(await api("/api/imaging/pacs/retrieve", { method: "POST", body: { study_uid: study.study_uid, series_uid: s.series_uid } }));
          } catch (err) {
            showMessage("imaging-message", err.message, true);
            importButton.disabled = false;
            importButton.textContent = "Import";
          }
        });
        li.appendChild(importButton);
        list.appendChild(li);
      });
      show.hidden = true;
    } catch (err) {
      showMessage("imaging-message", err.message, true);
      show.disabled = false;
    }
  });
  card.append(head, desc, show, list);
  return card;
}

// --- Viewer ---

async function openSeries(id, keepPlace = false) {
  let data;
  try { data = await api(`/api/imaging/series/${id}`); }
  catch (err) { showMessage("viewer-message", err.message, true); return; }
  if (imagingRoute() !== id) return;
  const s = data.series;
  const first = !imaging.series || imaging.series.id !== s.id;
  imaging.series = s;
  $("viewer-message").hidden = true;
  if (first && !keepPlace) {
    imaging.index = Math.floor(s.slices / 2);
    const abdominal = /ABDOMEN|PANCREAS|LIVER|KIDNEY|PELVIS/i.test(`${s.body_part} ${s.description}`);
    imaging.window = s.windows.includes("soft tissue") && !abdominal ? "soft tissue" : (s.windows[0] || null);
    imaging.editing = null;
  }
  const title = seriesTitle(s);
  $("viewer-title").textContent = title;
  $("topbar-title").textContent = title;
  document.title = `${title} · GroundCheck`;
  $("viewer-sub").textContent = [s.modality, s.plane !== "unknown" ? s.plane : "", plural(s.slices, "image"), s.label && s.description ? s.description : ""].filter(Boolean).join(", ");
  $("viewer-delete").hidden = !canDeleteSeries();

  const windowSelect = $("viewer-window");
  windowSelect.textContent = "";
  s.windows.forEach((w) => { const o = el("option", "", w === "auto" ? "Automatic" : w[0].toUpperCase() + w.slice(1)); o.value = w; windowSelect.appendChild(o); });
  windowSelect.value = imaging.window || "";
  windowSelect.disabled = s.windows.length < 2;

  const spacing = s.pixel_spacing || [1, 1];
  $("viewer-frame").style.setProperty("--aspect", String((s.columns * spacing[1]) / (s.rows * spacing[0])));
  $("viewer-regions").setAttribute("viewBox", `0 0 ${s.columns} ${s.rows}`);
  $("viewer-regions").setAttribute("preserveAspectRatio", "none");
  const [top, bottom, left, right] = ORIENTATION[s.plane] || ["", "", "", ""];
  $("orient-top").textContent = top; $("orient-bottom").textContent = bottom;
  $("orient-left").textContent = left; $("orient-right").textContent = right;
  $("viewer-slice").max = String(s.slices - 1);
  renderSeriesDetails(s);
  renderAnalysis(s);
  renderReport(s);
  renderSlice();
  const running = s.analyses.some((a) => a.status === "queued" || a.status === "running");
  $("nav-imaging-live").hidden = !running;
  if (running) imaging.poll = setTimeout(() => { if (imagingRoute() === id) openSeries(id, true); }, 1000);
}

function latestAnalysis(s) {
  return s.analyses.length ? s.analyses[s.analyses.length - 1] : null;
}

function labelColour(label) {
  const a = latestAnalysis(imaging.series);
  const model = imaging.home?.models.find((m) => m.id === a?.model_id);
  const classes = model?.classes || (a?.summary?.labels || []).map((l) => l.label);
  const i = Math.max(0, classes.indexOf(label));
  return `var(--label-${Math.min(i, LABEL_COLOURS - 1) + 1})`;
}

function sliceUrl(index) {
  const s = imaging.series;
  const w = imaging.window ? `?window=${encodeURIComponent(imaging.window)}` : "";
  return `/api/imaging/series/${s.id}/slices/${index}.png${w}`;
}

function renderSlice() {
  const s = imaging.series;
  if (!s) return;
  imaging.index = Math.max(0, Math.min(s.slices - 1, imaging.index));
  const i = imaging.index;
  $("viewer-image").src = sliceUrl(i);
  $("viewer-image").alt = `${s.modality} slice ${i + 1} of ${s.slices}`;
  $("viewer-slice").value = String(i);
  $("viewer-slice").setAttribute("aria-valuetext", `Slice ${i + 1} of ${s.slices}`);
  $("viewer-slice-label").textContent = `${i + 1} / ${s.slices}`;
  const spacing = s.pixel_spacing || [1, 1];
  $("viewer-scale").textContent = `${(s.columns * spacing[1]).toFixed(0)}×${(s.rows * spacing[0]).toFixed(0)} mm${s.slice_thickness || s.slice_spacing ? `, ${s.slice_thickness || s.slice_spacing} mm slices` : ""}`;
  for (const d of [1, -1, 2, -2, 3]) {
    const j = i + d;
    if (j >= 0 && j < s.slices) new Image().src = sliceUrl(j);
  }
  const svg = $("viewer-regions");
  svg.textContent = "";
  const a = latestAnalysis(s);
  $("viewer-overlay").disabled = !(a?.status === "done" && a.summary?.labels?.length);
  const findings = $("viewer-overlay").checked && a?.status === "done" ? (a.slices?.[i]?.findings || []) : [];
  const ns = "http://www.w3.org/2000/svg";
  findings.forEach((f) => {
    const [x0, y0, x1, y1] = f.box;
    const colour = labelColour(f.label);
    const rect = document.createElementNS(ns, "rect");
    rect.setAttribute("x", x0); rect.setAttribute("y", y0);
    rect.setAttribute("width", x1 - x0); rect.setAttribute("height", y1 - y0);
    rect.setAttribute("rx", 3);
    rect.style.stroke = colour;
    const text = document.createElementNS(ns, "text");
    text.setAttribute("x", x0 + 4); text.setAttribute("y", Math.max(12, y0 + 14));
    text.style.fill = colour;
    text.style.fontSize = `${Math.max(9, s.columns / 40)}px`;
    text.textContent = `${f.label} ${Math.round(f.confidence * 100)}%`;
    svg.append(rect, text);
  });
  document.querySelectorAll(".slice-bar b").forEach((b) => { b.style.left = `${((i + 0.5) / s.slices) * 100}%`; });
}

function moveSlice(delta) {
  if (!imaging.series) return;
  imaging.index += delta;
  renderSlice();
}

function renderSeriesDetails(s) {
  const dl = $("series-details");
  dl.textContent = "";
  const spacing = s.pixel_spacing || [];
  const patient = s.patient || {};
  const deidParts = [
    [s.deid.removed, "removed"], [s.deid.emptied, "emptied"], [s.deid.private_removed, "private tags removed"],
    [s.deid.uids_replaced, "identifiers replaced"], [s.deid.descriptors_cleaned, "descriptions cleaned"],
  ].filter(([n]) => n).map(([n, what]) => `${n.toLocaleString()} ${what}`);
  const rows = [
    ["Modality", `${s.modality}${s.body_part ? `, ${s.body_part.toLowerCase()}` : ""}`],
    ["Plane", s.plane === "unknown" ? "Unknown, shown as stored" : s.plane[0].toUpperCase() + s.plane.slice(1)],
    ["Images", `${s.slices} slices, ${s.columns} × ${s.rows} pixels`],
    ["Pixel spacing", spacing.length ? `${spacing.map((v) => v.toFixed(2)).join(" × ")} mm` : "Not recorded"],
    ["Slice thickness", s.slice_thickness ? `${s.slice_thickness} mm` : "Not recorded"],
    ["Slice spacing", s.slice_spacing ? `${s.slice_spacing} mm` : "Not recorded"],
    ["Patient", [patient.PatientAge ? `age ${patient.PatientAge.replace(/^0+/, "").replace("Y", " years")}` : "", patient.PatientSex ? `sex ${patient.PatientSex}` : ""].filter(Boolean).join(", ") || "No characteristics recorded"],
    ["Source", s.source === "pacs" ? "Retrieved from the PACS" : "Uploaded"],
    ["De-identification", `DICOM PS3.15 basic profile. ${deidParts.join(", ") || "Nothing to remove"}.`],
  ];
  (s.warnings || []).forEach((w) => rows.push(["Note", w]));
  rows.forEach(([k, v]) => { dl.append(el("dt", "", k), el("dd", "", v)); });
}

function rangesText(ranges) {
  return ranges.map(([a, b]) => (a === b ? `${a + 1}` : `${a + 1}–${b + 1}`)).join(", ");
}

function renderAnalysis(s) {
  const models = imaging.home?.models || [];
  const select = $("analysis-model");
  const previous = select.value;
  select.textContent = "";
  models.forEach((m) => {
    const o = el("option", "", m.modality && m.modality !== s.modality ? `${m.name} (trained on ${m.modality})` : m.name);
    o.value = m.id;
    o.disabled = !!m.modality && m.modality !== s.modality;
    select.appendChild(o);
  });
  const usable = models.filter((m) => !m.modality || m.modality === s.modality);
  if (previous && usable.some((m) => m.id === previous)) select.value = previous;
  else if (usable.length) select.value = usable[0].id;
  const a = latestAnalysis(s);
  const running = a && (a.status === "queued" || a.status === "running");
  $("analysis-run").disabled = running || usable.length === 0;
  $("analysis-model").disabled = running || models.length === 0;
  $("analysis-hint").textContent = !models.length
    ? "No imaging models yet. An administrator can add imaging settings to a model in the Model library."
    : !usable.length ? `None of the models were trained on ${s.modality}.` : "";
  $("analysis-hint").hidden = !$("analysis-hint").textContent;
  $("analysis-progress").hidden = !running;
  if (running) {
    $("analysis-progress-fill").style.width = `${a.progress}%`;
    $("analysis-progress-text").textContent = a.status === "queued" ? "Waiting to start" : `Analysing, ${a.progress}%`;
  }

  const out = $("analysis-result");
  out.textContent = "";
  const map = $("slice-map");
  map.textContent = "";
  if (!a || running) return;
  out.appendChild(el("p", "field-hint", `${a.model_name}, ${shortDate(a.finished_at || a.created_at)}`));
  if (a.status === "refused" || a.status === "failed") {
    const note = el("div", "analysis-note refused");
    note.append(el("strong", "", a.status === "refused" ? "The model refused this series" : "The analysis didn’t finish"), el("span", "", a.error || ""));
    out.appendChild(note);
    return;
  }
  const sum = a.summary;
  if (sum.abstained) {
    const note = el("div", "analysis-note");
    note.appendChild(el("strong", "", "The model abstained on this series"));
    note.appendChild(el("span", "", sum.unfamiliar_series
      ? `${Math.round(sum.unfamiliar_share * 100)}% of the image regions it looked at were unlike its training images, so it reports nothing. Read the images without it.`
      : "It wasn’t confident about any region, so it reports nothing."));
    out.appendChild(note);
  } else {
    const list = el("ul", "label-list");
    sum.labels.forEach((l) => {
      const li = document.createElement("li");
      const b = document.createElement("button");
      b.type = "button";
      b.style.setProperty("--c", labelColour(l.label));
      const name = el("span", "label-name", l.label);
      name.appendChild(el("span", "label-where", `${plural(l.slices, "slice")}: ${rangesText(l.ranges)}`));
      b.append(el("span", "label-swatch"), name, el("span", "label-conf", `${Math.round(l.max_confidence * 100)}%`));
      b.title = "Go to the first slice";
      b.addEventListener("click", () => { imaging.index = l.ranges[0][0]; renderSlice(); $("viewer-stage").focus(); });
      li.appendChild(b);
      list.appendChild(li);
      const track = el("div", "slice-track");
      track.style.setProperty("--c", labelColour(l.label));
      track.appendChild(el("span", "", l.label));
      const bar = document.createElement("button");
      bar.type = "button";
      bar.className = "slice-bar";
      bar.setAttribute("aria-label", `${l.label}: slices ${rangesText(l.ranges)}. Go to the first.`);
      l.ranges.forEach(([from, to]) => {
        const seg = document.createElement("i");
        seg.style.left = `${(from / s.slices) * 100}%`;
        seg.style.width = `${((to - from + 1) / s.slices) * 100}%`;
        bar.appendChild(seg);
      });
      bar.appendChild(document.createElement("b"));
      bar.addEventListener("click", (e) => {
        const rect = bar.getBoundingClientRect();
        imaging.index = e.detail === 0 ? l.ranges[0][0] : Math.floor(((e.clientX - rect.left) / rect.width) * s.slices);
        renderSlice();
      });
      track.appendChild(bar);
      map.appendChild(track);
    });
    out.appendChild(list);
  }
  const t = sum.totals;
  if (t.patches) {
    const wrap = document.createElement("div");
    wrap.appendChild(el("p", "field-label", `Image regions examined: ${t.patches.toLocaleString()}`));
    const bar = el("div", "patch-bar");
    bar.setAttribute("role", "img");
    bar.setAttribute("aria-label", `${t.answered} answered, ${t.low_confidence} not confident, ${t.unfamiliar} unfamiliar`);
    const legend = el("ul", "patch-legend");
    [["answered", t.answered, "Answered", "var(--ink)"], ["low", t.low_confidence, "Not confident", "var(--series-2)"], ["unfamiliar", t.unfamiliar, "Unlike its training images", "var(--warn)"]]
      .forEach(([cls, n, label, colour]) => {
        if (n) { const seg = el("span", cls); seg.style.flex = String(n); bar.appendChild(seg); }
        const li = el("li", "", `${label} ${n.toLocaleString()}`);
        li.style.setProperty("--c", colour);
        legend.appendChild(li);
      });
    wrap.append(bar, legend);
    out.appendChild(wrap);
  }
  out.appendChild(el("p", "research-note", "For research and evaluation. Not validated for clinical use; the reading clinician decides."));
}

function renderReport(s) {
  const signed = [...s.reports].reverse().find((r) => r.status === "signed");
  const draft = [...s.reports].reverse().find((r) => r.status === "draft");
  const superseded = s.reports.filter((r) => r.status === "superseded");
  const box = $("report-signed");
  box.textContent = "";
  const done = [...s.analyses].reverse().find((a) => a.status === "done");
  document.querySelectorAll('#report-agreement input').forEach((input) => {
    input.disabled = input.value !== "not_used" && !done;
  });

  if (signed) {
    const card = el("article", "signed-report");
    card.append(el("h3", "", signed.replaces_id ? "Signed amendment" : "Signed report"),
      el("p", "signed-impression", signed.impression));
    if (signed.findings) card.appendChild(el("p", "", signed.findings));
    card.appendChild(el("p", "signed-meta", `${AGREEMENT_LABELS[signed.agreement]}. Signed by ${signed.author_name || "unknown"} on ${shortDate(signed.signed_at)}.${signed.sent_at ? ` Sent to the PACS on ${shortDate(signed.sent_at)}.` : ""}`));
    const actions = el("div", "signed-actions");
    const download = el("a", "btn btn-sm btn-ghost", "Download DICOM SR");
    download.href = `/api/imaging/reports/${signed.id}/sr.dcm`;
    actions.appendChild(download);
    if (s.can_send_to_pacs) {
      actions.appendChild(button(signed.sent_at ? "Send again" : "Send to the PACS", "btn-ghost", false, async (e) => {
        e.target.disabled = true;
        try { await api(`/api/imaging/reports/${signed.id}/send`, { method: "POST" }); showMessage("report-message", "Sent to the PACS."); openSeries(s.id, true); }
        catch (err) { showMessage("viewer-message", err.message, true); e.target.disabled = false; }
      }));
    }
    if (!draft && imaging.editing !== signed.id) {
      actions.appendChild(button("Amend", "btn-ghost", false, () => {
        imaging.editing = signed.id;
        fillReportForm(signed, signed.id);
        $("report-form").hidden = false;
        $("report-findings").focus();
      }));
    }
    card.appendChild(actions);
    box.appendChild(card);
  }
  const form = $("report-form");
  if (draft) {
    form.hidden = false;
    if (imaging.editing !== `draft-${draft.id}`) { fillReportForm(draft, draft.id); imaging.editing = `draft-${draft.id}`; }
    $("report-draft-note").textContent = `Draft saved ${shortDate(draft.updated_at)}.`;
  } else if (signed && imaging.editing !== signed.id) {
    form.hidden = true;
  } else if (!signed && imaging.editing === null) {
    form.hidden = false;
    fillReportForm(null, null);
    imaging.editing = "new";
  }
  if (!draft) $("report-draft-note").textContent = imaging.editing === signed?.id ? "Signing creates an amendment. The signed report is kept." : "";
  $("report-sign").textContent = imaging.editing === signed?.id || draft?.replaces_id ? "Sign amendment" : "Sign report";

  $("report-history").hidden = superseded.length === 0;
  const list = $("report-history-list");
  list.textContent = "";
  superseded.forEach((r) => {
    const li = document.createElement("li");
    li.append(el("p", "", r.impression), el("p", "signed-meta", `Signed by ${r.author_name || "unknown"} on ${shortDate(r.signed_at)}. Replaced by an amendment.`));
    list.appendChild(li);
  });
}

function fillReportForm(report, id) {
  const form = $("report-form");
  form.dataset.reportId = id ?? "";
  $("report-findings").value = report?.findings || "";
  $("report-impression").value = report?.impression || "";
  const agreement = report?.agreement || "not_used";
  const input = form.querySelector(`input[name="agreement"][value="${agreement}"]`);
  (input && !input.disabled ? input : form.querySelector('input[value="not_used"]')).checked = true;
  $("report-message").hidden = true;
}

async function saveReport(sign) {
  const s = imaging.series;
  const form = $("report-form");
  const agreement = form.querySelector('input[name="agreement"]:checked')?.value || "not_used";
  const done = [...s.analyses].reverse().find((a) => a.status === "done");
  if (sign) {
    if (!$("report-impression").value.trim()) { showMessage("report-message", "Write an impression before signing.", true); $("report-impression").focus(); return; }
    if (!window.confirm("Sign this report? A signed report can’t be edited. Later changes are made as an amendment.")) return;
  }
  const body = {
    findings: $("report-findings").value, impression: $("report-impression").value, agreement, sign,
    analysis_id: agreement !== "not_used" && done ? done.id : null,
    report_id: form.dataset.reportId ? Number(form.dataset.reportId) : null,
  };
  formBusy(form, true);
  try {
    const { report } = await api(`/api/imaging/series/${s.id}/reports`, { method: "POST", body });
    imaging.editing = sign ? null : `draft-${report.id}`;
    form.dataset.reportId = report.id;
    await openSeries(s.id, true);
    showMessage("report-message", sign ? "Signed." : "Draft saved.");
    if (sign) $("report-message").hidden = true;
  } catch (err) {
    showMessage("report-message", err.message, true);
  } finally {
    formBusy(form, false);
    renderReport(imaging.series);
  }
}

function wireImaging() {
  $("imaging-tab-upload").addEventListener("click", () => setImportTab("upload"));
  $("imaging-tab-pacs").addEventListener("click", () => setImportTab("pacs"));
  $("imaging-import-tabs").addEventListener("keydown", (e) => {
    if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
    const toPacs = $("imaging-tab-upload").getAttribute("aria-selected") === "true";
    setImportTab(toPacs ? "pacs" : "upload");
    $(toPacs ? "imaging-tab-pacs" : "imaging-tab-upload").focus();
  });
  $("dicom-files").addEventListener("change", (e) => chooseDicomFiles(e.target.files));
  $("dicom-folder").addEventListener("click", () => $("dicom-folder-input").click());
  $("dicom-folder-input").addEventListener("change", (e) => chooseDicomFiles(e.target.files));
  const drop = $("dicom-drop");
  ["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("dragging"); }));
  ["dragleave", "drop"].forEach((t) => drop.addEventListener(t, () => drop.classList.remove("dragging")));
  drop.addEventListener("drop", async (e) => {
    e.preventDefault();
    const items = [...(e.dataTransfer.items || [])];
    chooseDicomFiles(items.length ? await readDroppedEntries(items) : e.dataTransfer.files);
  });
  $("imaging-upload").addEventListener("submit", uploadDicom);
  $("pacs-search").addEventListener("submit", searchPacs);

  $("viewer-slice").addEventListener("input", (e) => { imaging.index = Number(e.target.value); renderSlice(); });
  $("viewer-window").addEventListener("change", (e) => { imaging.window = e.target.value; renderSlice(); });
  $("viewer-overlay").addEventListener("change", renderSlice);
  let wheelAccum = 0;
  $("viewer-stage").addEventListener("wheel", (e) => {
    e.preventDefault();
    wheelAccum += e.deltaY;
    const steps = Math.trunc(wheelAccum / 40);
    if (steps) { wheelAccum -= steps * 40; moveSlice(steps); }
  }, { passive: false });
  $("viewer-stage").addEventListener("keydown", (e) => {
    const moves = { ArrowUp: -1, ArrowLeft: -1, ArrowDown: 1, ArrowRight: 1, PageUp: -10, PageDown: 10 };
    if (e.key in moves) { e.preventDefault(); moveSlice(moves[e.key]); }
    else if (e.key === "Home") { e.preventDefault(); imaging.index = 0; renderSlice(); }
    else if (e.key === "End") { e.preventDefault(); imaging.index = imaging.series.slices - 1; renderSlice(); }
  });
  let touchY = null;
  $("viewer-stage").addEventListener("touchstart", (e) => { touchY = e.touches[0].clientY; }, { passive: true });
  $("viewer-stage").addEventListener("touchmove", (e) => {
    if (touchY === null) return;
    const dy = e.touches[0].clientY - touchY;
    if (Math.abs(dy) >= 12) { moveSlice(Math.sign(dy)); touchY = e.touches[0].clientY; }
    e.preventDefault();
  }, { passive: false });

  $("analysis-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const s = imaging.series;
    $("analysis-run").disabled = true;
    try {
      await api(`/api/imaging/series/${s.id}/analyses`, { method: "POST", body: { model_id: $("analysis-model").value } });
      openSeries(s.id, true);
    } catch (err) {
      showMessage("viewer-message", err.message, true);
      $("analysis-run").disabled = false;
    }
  });
  $("report-form").addEventListener("submit", (e) => { e.preventDefault(); saveReport(true); });
  $("report-save").addEventListener("click", () => saveReport(false));
  $("viewer-delete").addEventListener("click", async () => {
    const s = imaging.series;
    if (!window.confirm(`Delete “${seriesTitle(s)}”, its analyses and its reports? This can’t be undone.`)) return;
    try {
      await api(`/api/imaging/series/${s.id}`, { method: "DELETE" });
      imaging.series = null;
      location.hash = "#/imaging";
      showMessage("imaging-message", "Deleted.");
    } catch (err) { showMessage("viewer-message", err.message, true); }
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
  training: { load: () => loadTraining() },
  medicines: { load: () => loadMedicines() },
  ehr: { load: () => loadEhrPage() },
  models: { load: () => loadModels() },
  imaging: { load: () => loadImaging() },
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
  wireTraining();
  wireImaging();
  $("mi-modality").addEventListener("change", (e) => fillImagingWindows(e.target.value));
  $("model-imaging-form").addEventListener("submit", (e) => { e.preventDefault(); saveModelImaging(false); });
  $("mi-remove").addEventListener("click", () => saveModelImaging(true));
  wireModels();
  wirePatient();
  wireEhr();
  loadEhrForAsk();
  $("medicine-search").addEventListener("input", () => { clearTimeout(medicineTimer); medicineTimer = setTimeout(loadMedicines, 200); });
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

// ---------- Theme ----------
function wireTheme() {
  const render = () => {
    const choice = window.gcTheme ? window.gcTheme.get() : "system";
    document.querySelectorAll(".theme-switch [data-theme-choice]").forEach((b) => {
      const on = b.dataset.themeChoice === choice;
      b.setAttribute("aria-checked", String(on));
      b.tabIndex = on ? 0 : -1;
    });
  };
  document.querySelectorAll(".theme-switch").forEach((group) => {
    const buttons = [...group.querySelectorAll("[data-theme-choice]")];
    buttons.forEach((b, i) => {
      b.addEventListener("click", () => window.gcTheme && window.gcTheme.set(b.dataset.themeChoice));
      b.addEventListener("keydown", (e) => {
        const step = e.key === "ArrowRight" || e.key === "ArrowDown" ? 1 : e.key === "ArrowLeft" || e.key === "ArrowUp" ? -1 : 0;
        if (!step) return;
        e.preventDefault();
        const next = buttons[(i + step + buttons.length) % buttons.length];
        window.gcTheme && window.gcTheme.set(next.dataset.themeChoice);
        next.focus();
      });
    });
  });
  document.addEventListener("themechange", () => {
    render();
    readMapColours();
    if (map3d.points.length) drawCorpus();
  });
  render();
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
    renderPatientEval(data.patient);
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

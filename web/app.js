"use strict";

const $ = (id) => document.getElementById(id);
const uiLocale = () => { const l = window.gcI18n ? gcI18n.locale() : "en"; return l === "pseudo" ? "en" : l; };
const REDUCED_MOTION = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

let DEFAULTS = null;
let llmAvailable = false;
let lastRetrievedIds = [];

// Static stage list for the "How it works" panel. The authoritative,
// per-run descriptions come from the trace itself; this mirrors them for the
// explainer shown before any question is asked.
const PIPELINE_STAGES = () => ([
  [t("js.input_guards", "input guards"), t("js.redact_pii_reject_injection_and", "Redact PII, reject injection and out-of-scope input, rate limit.")],
  [t("js.embed_retrieve", "embed + retrieve"), t("js.embed_the_query_and_rank", "Embed the query and rank passages by a blend of embedding similarity and keyword match, with similarity scores.")],
  [t("js.retrieval_gate", "retrieval gate"), t("js.if_the_best_passage_is", "If the best passage is below threshold, refuse now, before any generation.")],
  [t("js.source_coverage", "source coverage"), t("js.check_the_question_s_entities", "Check the question's entities actually appear in the retrieved sources.")],
  [t("js.stage_generate", "generate"), t("js.llm_returns_structured_cited_claims", "LLM returns structured, cited claims. Extractive fallback if no LLM is available.")],
  [t("js.schema_validate", "schema validate"), t("js.validate_the_output_against_a", "Validate the output against a strict schema. Retry once, then fall back.")],
  [t("js.grounding_check", "grounding check"), t("js.deterministic_claim_to_source_similarity", "Deterministic claim-to-source similarity, with an LLM judge as corroboration.")],
  [t("js.dosage_guard", "dosage guard"), t("js.every_dose_in_the_answer", "Every dose in the answer must be stated by a source about that medicine, or refuse.")],
  [t("js.stage_decision", "decision"), t("js.answer_with_citations_or_refuse", "Answer with citations, or refuse with a specific reason and route for review.")],
]);

// Display names for pipeline stages whose internal names read awkwardly.
const STAGE_LABELS = () => ({
  "pii redaction": t("js.de_identification", "De-identification"),
  "decision": t("js.stage_decision", "decision"),
  "generate": t("js.stage_generate", "generate"),
  "retrieve": t("js.stage_retrieve", "retrieve"),
  "remembered answer": t("js.remembered_answer", "remembered answer"),
  "patient checks": t("js.patient_checks", "patient checks"),
  "scope check": t("js.scope_check", "scope check"),
  "rate limit": t("js.rate_limit", "rate limit"),
  "dosage guard": t("js.dosage_guard", "dosage guard"),
  "grounding check": t("js.grounding_check", "grounding check"),
  "source coverage": t("js.source_coverage", "source coverage"),
  "retrieval gate": t("js.retrieval_gate", "retrieval gate"),
  "schema validate": t("js.schema_validate", "schema validate"),
  "input guards": t("js.input_guards", "input guards"),
});

// Plain-language explanation shown in the refusal callout, keyed by the
// stable reason code. refusalExplanation() is the fallback for responses and
// stored records that carry only prose.
const REASON_EXPLAIN = {
  input_blocked: "The query was stopped by an input guard before reaching the pipeline.",
  rate_limited: "Too many requests in a short window. This protects the live service.",
  no_relevant_source: "No document was similar enough to the question to be trustworthy, so the system refused before generating anything.",
  term_not_covered: "The question names something the trusted sources never mention, so there is nothing to ground an answer on.",
  population_not_covered: "The question is about a patient group the sources don't cover, so an answer written for adults could be unsafe.",
  stated_value_unsupported: "The question states a value that no trusted source supports, so confirming it could be unsafe.",
  form_unsupported: "The question names a form or route that no source about this medicine describes.",
  limit_not_stated: "The sources give a dose but no limit, so they can't answer a question about a maximum or minimum.",
  combination_forbidden: "The sources say these must not be used together, so there is no safe combined dose to give.",
  insufficient_context: "The retrieved sources did not contain enough to answer safely.",
  claim_ungrounded: "A statement in the draft answer was not sufficiently supported by its cited source.",
  dose_unsupported: "The draft answer contained a dose that no source about this medicine states.",
  patient_unsafe: "The answer from the sources doesn't fit the patient you described. The check above names the rule.",
  unsupported_language: "The question is in a language this installation cannot check against its sources, so it was refused rather than answered from English text by accident.",
};
const REASON_FALLBACK = "The pipeline could not produce a fully grounded answer, so it declined.";
const reasonFallback = () => t("reason.fallback", REASON_FALLBACK);

function refusalExplanation(reason) {
  const r = (reason || "").toLowerCase();
  const has = (...needles) => needles.some((n) => r.includes(n));
  if (has("is not stated by any source", "in the question does not appear")) return t("reason.stated_value_unsupported", REASON_EXPLAIN.stated_value_unsupported);
  if (has("describes a '")) return t("reason.form_unsupported", REASON_EXPLAIN.form_unsupported);
  if (has("no trusted source covers")) return t("reason.population_not_covered", REASON_EXPLAIN.population_not_covered);
  if (has("states a maximum", "states a minimum", "no trusted source states a")) return t("reason.limit_not_stated", REASON_EXPLAIN.limit_not_stated);
  if (has("must not be combined")) return t("reason.combination_forbidden", REASON_EXPLAIN.combination_forbidden);
  if (has("not supported by a source about", "not supported by any source")) return t("reason.dose_unsupported", REASON_EXPLAIN.dose_unsupported);
  if (has("does not appear in any", "do not cover")) return t("reason.term_not_covered", REASON_EXPLAIN.term_not_covered);
  if (has("no sufficiently relevant source")) return t("reason.no_relevant_source", REASON_EXPLAIN.no_relevant_source);
  if (has("could not be grounded")) return t("reason.claim_ungrounded", REASON_EXPLAIN.claim_ungrounded);
  if (has("enough information")) return t("reason.insufficient_context", REASON_EXPLAIN.insufficient_context);
  if (has("input guard", "blocked")) return t("reason.input_blocked", REASON_EXPLAIN.input_blocked);
  if (has("rate limit")) return t("reason.rate_limited", REASON_EXPLAIN.rate_limited);
  return reasonFallback();
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
  refreshAlertBadge();
  setInterval(refreshAlertBadge, 60000);
}

// The sidebar shows how many alerts are firing, for people who can see them.
async function refreshAlertBadge() {
  if (!canSee("monitoring") || document.hidden) return;
  try { updateAlertBadge((await api("/api/monitoring")).firing); } catch (_) { /* shown on the page itself */ }
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
    showAuthScreen("login", t("js.your_session_ended_sign_in", "Your session ended. Sign in again."));
  }
  if (!res.ok) {
    // The server's own message is usually specific and actionable. When there
    // isn't one, say what to do rather than only what went wrong: a dead end
    // in clinical tooling turns into a phone call to IT.
    const fallback = res.status >= 500
      ? t("js.groundcheckhealth_couldn_t_complete_that", "GroundCheckHealth couldn't complete that. Try again — if it keeps happening, tell your administrator, ")
        + t("js.and_quote_the_time_and", "and quote the time and error {status}.", { status: res.status })
      : t("js.that_request_wasn_t_accepted", "That request wasn't accepted (error {status}). Check what you entered and try again.", { status: res.status });
    const detail = data && typeof data.detail === "string" ? data.detail : fallback;
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
    login: [t("js.sign_in", "Sign in"), t("js.sign_in_to_ask_questions", "Sign in to ask questions and review answers.")],
    mfa: [t("js.two_factor_authentication", "Two-factor authentication"), t("js.enter_the_code_from_your", "Enter the code from your authenticator app to finish signing in.")],
    "first-admin": [t("js.create_the_first_admin", "Create the first admin"), t("js.no_accounts_exist_yet_the", "No accounts exist yet. The first account is an admin, who can add everyone else.")],
  };
  $("dashboard").hidden = true;
  $("auth-screen").hidden = false;
  $("auth-title").textContent = titles[mode][0];
  $("auth-lede").textContent = titles[mode][1];
  const ssoOn = mode === "login" && account.sso && account.sso.enabled;
  $("sso-block").hidden = !ssoOn;
  if (ssoOn) {
    $("sso-button").textContent = t("js.sign_in_with", "Sign in with {provider}", { provider: account.sso.provider });
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
  $("account-role").textContent = user.site_name ? `${user.role}, ${user.site_name}` : user.role;
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
  const needs = { releases: 1, monitoring: 1, usage: 1, review: 1, documents: 1, "data-protection": 1, users: 2, "local-ai": 0, training: 2, models: 1, ehr: 2 }[view] ?? 0;
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
      showMessage("account-message", t("js.password_changed_other_sessions_were", "Password changed. Other sessions were signed out."));
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
      showMessage("account-message", t("js.two_factor_authentication_is_on", "Two-factor authentication is on."));
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
      showMessage("account-message", t("js.two_factor_authentication_is_off", "Two-factor authentication is off."));
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
        site_id: $("new-user-site").value ? Number($("new-user-site").value) : null,
      } });
      form.reset();
      await loadUsers();
      showMessage("users-message", t("js.user_added_share_the_temporary", "User added. Share the temporary password with them securely."));
    } catch (err) { showMessage("users-message", err.message, true); }
    formBusy(form, false);
  });
  $("add-site-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const form = e.currentTarget;
    const name = $("new-site-name").value.trim();
    const key = $("new-site-key").value.trim() || name.toLowerCase().normalize("NFKD").replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "");
    formBusy(form, true);
    try {
      await api("/api/sites", { method: "POST", body: { name, key } });
      form.reset();
      await loadUsers();
      showMessage("users-message", t("js.added_choose_a_site_for", "Added {name}. Choose a site for each person below.", { name }));
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
  $("twofa-status").textContent = on ? t("js.on_you_ll_be_asked", "On. You’ll be asked for a code when you sign in.") : t("js.off", "Off.");
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
  const siteNames = Object.fromEntries(data.sites.map((x) => [x.id, x.name]));
  const groupAdmin = data.my_site_id === null || data.my_site_id === undefined;
  const showSites = data.sites.length > 0;
  document.querySelectorAll(".site-col").forEach((c) => { c.hidden = !showSites; });
  document.querySelectorAll(".site-field").forEach((c) => { c.hidden = !showSites || !groupAdmin; });
  $("add-site-form").hidden = !groupAdmin;
  $("sites-panel").hidden = !groupAdmin && !showSites;
  const siteSelect = $("new-user-site");
  const keep = siteSelect.value;
  siteSelect.textContent = "";
  [["", t("js.every_site", "Every site")], ...data.sites.map((x) => [String(x.id), x.name])].forEach(([v, l]) => {
    const o = document.createElement("option"); o.value = v; o.textContent = l; siteSelect.appendChild(o);
  });
  siteSelect.value = keep;
  const list = $("site-list");
  list.textContent = "";
  data.sites.forEach((x) => {
    const li = document.createElement("li");
    const n = document.createElement("span"); n.textContent = x.name;
    const k = document.createElement("span"); k.className = "mono"; k.textContent = x.key;
    li.append(n, k);
    list.appendChild(li);
  });
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
    select.setAttribute("aria-label", t("js.role_for", "Role for {email}", { email: u.email }));
    ["clinician", "reviewer", "admin"].forEach((r) => {
      const opt = document.createElement("option");
      opt.value = r; opt.textContent = t("js.role_" + r, r); opt.selected = r === u.role;
      select.appendChild(opt);
    });
    select.addEventListener("change", () => updateUser(u.id, { role: select.value }, () => { select.value = u.role; }));
    roleCell.appendChild(select);

    const siteCell = document.createElement("td");
    siteCell.hidden = !showSites;
    if (groupAdmin && !u.email.endsWith("@deleted.invalid")) {
      const pick = document.createElement("select");
      pick.setAttribute("aria-label", t("js.site_for", "Site for {email}", { email: u.email }));
      [["", t("js.every_site", "Every site")], ...data.sites.map((x) => [String(x.id), x.name])].forEach(([v, l]) => {
        const o = document.createElement("option"); o.value = v; o.textContent = l; pick.appendChild(o);
      });
      pick.value = u.site_id === null ? "" : String(u.site_id);
      pick.addEventListener("change", () => updateUser(u.id, { site_id: pick.value ? Number(pick.value) : null }));
      siteCell.appendChild(pick);
    } else {
      siteCell.textContent = u.site_id === null ? t("js.every_site", "Every site") : (siteNames[u.site_id] || "");
    }

    const mfa = document.createElement("td");
    mfa.textContent = u.mfa_enabled ? t("js.on", "On") : t("js.off_2", "Off");

    const last = document.createElement("td");
    last.className = "mono";
    last.textContent = u.last_login_at ? dateTime(u.last_login_at) : t("js.never", "Never");

    const status = document.createElement("td");
    const toggle = document.createElement("button");
    toggle.type = "button";
    toggle.className = "btn btn-sm btn-ghost";
    toggle.textContent = u.is_active ? t("js.deactivate", "Deactivate") : t("js.reactivate", "Reactivate");
    toggle.disabled = account.user && u.id === account.user.id;
    toggle.addEventListener("click", () => updateUser(u.id, { is_active: !u.is_active }));
    const deleted = u.email.endsWith("@deleted.invalid");
    status.append(document.createTextNode(deleted ? t("js.deleted", "Deleted ") : u.is_active ? t("js.active", "Active ") : t("js.inactive", "Inactive ")));
    if (!deleted) {
      status.appendChild(toggle);
      const remove = document.createElement("button");
      remove.type = "button";
      remove.className = "btn btn-sm btn-ghost";
      remove.textContent = t("js.delete", "Delete");
      remove.disabled = account.user && u.id === account.user.id;
      remove.addEventListener("click", () => deleteUser(u));
      status.append(" ", remove);
    } else {
      select.disabled = true;
    }

    tr.append(who, roleCell, siteCell, mfa, last, status);
    body.appendChild(tr);
  });
}

async function deleteUser(u) {
  if (!window.confirm(t("js.delete_their_name_email_and", "Delete {email}? Their name, email and sign-in details are removed. Their audit records stay, under an anonymous account number.", { email: u.email }))) return;
  try {
    await api(`/api/users/${u.id}`, { method: "DELETE" });
    showMessage("users-message", t("js.deleted_2", "Deleted."));
  } catch (err) { showMessage("users-message", err.message, true); }
  await loadUsers();
}

async function updateUser(id, changes, revert) {
  try {
    await api(`/api/users/${id}`, { method: "PATCH", body: changes });
    showMessage("users-message", t("js.saved", "Saved."));
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
    if (provider && provider.kind === "local") { dot.className = "dot ok"; label.textContent = t("js.local_model", "Local model, {model}", { model: provider.model }); }
    else if (provider) { dot.className = "dot ok"; label.textContent = t("js.cloud_model", "Cloud model, {model}", { model: provider.model }); }
    else { dot.className = "dot warn"; label.textContent = t("js.extractive_no_model", "Extractive, no model"); }
    if (typeof data.corpus === "number") $("corpus-count").textContent = data.corpus.toLocaleString();
  } catch (_) { dot.className = "dot warn"; label.textContent = t("js.extractive_no_model", "Extractive, no model"); }
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
      const g = ex.group || t("js.examples", "Examples");
      if (!byGroup[g]) { byGroup[g] = []; groups.push(g); }
      byGroup[g].push(ex);
    });
    // The first group is shown; the rest wait behind "More examples", so the
    // page opens on the question box rather than three rows of buttons.
    const more = document.createElement("details");
    more.className = "examples-more";
    const moreSummary = document.createElement("summary");
    more.appendChild(moreSummary);
    groups.forEach((g, index) => {
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
      (index === 0 ? wrap : more).appendChild(row);
    });
    if (groups.length > 1) {
      const hidden = groups.slice(1).reduce((n, g) => n + byGroup[g].length, 0);
      moreSummary.textContent = t("js.more_examples", "More examples ({hidden})", { hidden });
      wrap.appendChild(more);
    }
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
      sw.title = t("js.available_only_when_a_live", "Available only when a live LLM provider is configured.");
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
  if (isDefault()) { el.textContent = t("js.defaults", "defaults"); el.classList.remove("modified"); }
  else { el.textContent = t("js.modified", "modified"); el.classList.add("modified"); }
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
      $("pt-error").textContent = t("js.check_the_patient_details", "Check the patient details: {problem}", { problem });
      return;
    }
    render({ ...data, patient_given: Boolean(patient) });
    refreshReviewCount();
  } catch (_) {
    render({ decision: "refuse", answer_text: t("js.the_service_is_unreachable_please", "The service is unreachable. Please try again."),
      refused_reason: t("js.network_error", "network error"), claims: [], sources: [], trace: [], audit_id: "", total_ms: 0, llm_used: false });
  } finally { setLoading(false); inflight = false; }
}

function setLoading(on) {
  $("ask-btn").disabled = on;
  $("spinner").hidden = !on;
  $("ask-btn").querySelector(".btn-label").textContent = on ? t("js.asking", "Asking") : t("js.ask", "Ask");
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
    $("decision-reason").textContent = t("js.not_safe_patient", "Not safe for this patient: {medicine}").replace("{medicine}", block.medicine);
    $("refuse-explain").textContent = t("js.patient_refusal", "The answer from the sources doesn't fit the patient you described. The check above names the rule, and what the formulary gives instead where it can.");
  }
  lastQueryTerms = termsFrom((data.redaction && data.redaction.redacted_query) || $("query").value);
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
    chip.textContent = t("js.answer", "ANSWER");
    const ids = citedIds(data);
    reason.textContent = ids.length ? t("js.grounded_in", "Grounded in {ids}.").replace("{ids}", ids.join(", ")) : t("js.grounded_cited", "Grounded in cited sources.");
    body.appendChild(renderAnswerWithCitations(data.answer_text, data));
    callout.hidden = true;
    renderClaimRows(data);
    resetFlag(data.audit_id);
  } else {
    chip.className = "status-chip refuse";
    chip.textContent = t("js.refused", "REFUSED");
    resetFlag(null);
    reason.textContent = data.refused_reason ? capitalize(data.refused_reason) : "";
    callout.hidden = false;
    $("refuse-detail").textContent = data.refused_reason || t("js.no_grounded", "No grounded answer available.");
    renderRefusalExplanation(data);
    renderRoutedLine(data);
    renderClaimRows(null);
  }
  renderRedaction(data.redaction);
}

function renderRefusalExplanation(data) {
  const explain = $("refuse-explain");
  const ex = data.explanation || {};
  const generic = REASON_EXPLAIN[data.reason_code] ? t("reason." + data.reason_code, REASON_EXPLAIN[data.reason_code]) : refusalExplanation(data.refused_reason);
  explain.textContent = "";
  const localised = window.gcI18n && gcI18n.locale() !== "en" && REASON_EXPLAIN[data.reason_code];
  appendHighlighted(explain, (localised && generic) || ex.summary || generic, ex.missing_terms || []);
  const box = $("refuse-suggestions");
  box.textContent = "";
  const tips = (ex.suggestions || []).filter(Boolean);
  box.hidden = !tips.length;
  if (!tips.length) return;
  const label = document.createElement("span");
  label.className = "eyebrow";
  label.textContent = t("js.try_instead", "Try instead");
  const row = document.createElement("div");
  row.className = "suggestion-chips";
  tips.forEach((text) => {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "chip suggestion";
    chip.textContent = text;
    chip.addEventListener("click", () => { $("query").value = text; submitQuery(text); });
    row.appendChild(chip);
  });
  box.append(label, row);
}

// Text with each term wrapped in <mark>, built from text nodes only.
function appendHighlighted(parent, text, terms, cls = "term-miss") {
  const list = (terms || []).map((tt) => String(tt).trim()).filter((tt) => tt.length > 1);
  if (!list.length) { parent.appendChild(document.createTextNode(text)); return; }
  const esc = list.sort((a, b) => b.length - a.length).map((tt) => tt.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  const re = new RegExp(`(${esc.join("|")})`, "gi");
  text.split(re).forEach((part, i) => {
    if (i % 2) { const m = document.createElement("mark"); m.className = cls; m.textContent = part; parent.appendChild(m); }
    else if (part) parent.appendChild(document.createTextNode(part));
  });
}

function renderRoutedLine(data) {
  const line = $("routed-line");
  const review = data.review;
  line.textContent = "";
  line.hidden = !review;
  if (!review) return;
  if (review.routed) {
    line.append(t("js.no_answer_is_shown_routed", "No answer is shown. Routed for review"));
    if (review.case_id != null) {
      line.append(" as ");
      if (canSee("review")) {
        const link = document.createElement("a");
        link.href = "#/review";
        link.className = "case-link";
        link.textContent = t("js.case", "case #{case_id}", { case_id: review.case_id });
        link.addEventListener("click", () => { setTimeout(() => openCase(review.case_id), 150); });
        line.appendChild(link);
      } else line.append(t("js.case", "case #{case_id}", { case_id: review.case_id }));
    }
    line.append(".");
  } else {
    line.textContent = review.note || t("js.not_routed_for_review", "Not routed for review.");
  }
}

const REDACTION_LABEL = () => ({
  NAME: [t("js.redact_name", "name"), t("js.redact_names", "names")], DATE: [t("js.redact_date", "date"), t("js.redact_dates", "dates")], CARD: [t("js.card_number", "card number"), t("js.card_numbers", "card numbers")],
  EMAIL: [t("js.email_address", "email address"), t("js.email_addresses", "email addresses")], PHONE: [t("js.phone_number", "phone number"), t("js.phone_numbers", "phone numbers")],
  ID: [t("js.redact_identifier", "identifier"), t("js.redact_identifiers", "identifiers")], SSN: [t("js.national_id", "national ID"), t("js.national_ids", "national IDs")], ADDRESS: [t("js.redact_address", "address"), t("js.redact_addresses", "addresses")],
});

function renderRedaction(red) {
  const box = $("redaction-line");
  box.textContent = "";
  const kinds = (red && red.kinds) || {};
  const total = Object.values(kinds).reduce((a, b) => a + b, 0);
  box.hidden = !total || !red.redacted_query;
  if (box.hidden) return;
  const label = document.createElement("span");
  label.className = "eyebrow";
  label.textContent = t("js.deidentified", "De-identified before checking");
  const q = document.createElement("span");
  q.className = "redacted-query";
  red.redacted_query.split(/(\[[A-Z_]+\])/).forEach((part) => {
    if (/^\[[A-Z_]+\]$/.test(part)) {
      const chip = document.createElement("span");
      chip.className = "pii-chip";
      chip.textContent = part;
      q.appendChild(chip);
    } else if (part) q.appendChild(document.createTextNode(part));
  });
  const counts = Object.entries(kinds).map(([k, n]) => {
    const names = REDACTION_LABEL()[k.toUpperCase()] || [k.toLowerCase(), `${k.toLowerCase()}s`];
    return `${n} ${n === 1 ? names[0] : names[1]}`;
  });
  const note = document.createElement("span");
  note.className = "redaction-count";
  note.textContent = t("js.removed", "{counts} removed", { counts: counts.join(", ") });
  box.append(label, q, note);
}

function citedIds(data) {
  const ids = new Set();
  (data.claims || []).forEach((c) => (c.source_ids || []).forEach((id) => ids.add(id)));
  return [...ids];
}

function supportFor(data, id) {
  for (const c of (data && data.claims) || []) {
    if (c.support && c.support.source_id === id && (c.source_ids || []).includes(id)) return c.support;
  }
  return null;
}

function renderAnswerWithCitations(text, data) {
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
    a.addEventListener("click", (e) => { e.preventDefault(); showSupport(id, supportFor(data, id), true); });
    a.addEventListener("mouseenter", () => showSupport(id, supportFor(data, id), false));
    a.addEventListener("focus", () => showSupport(id, supportFor(data, id), false));
    a.addEventListener("mouseleave", clearSupport);
    a.addEventListener("blur", clearSupport);
    frag.appendChild(a);
    last = re.lastIndex;
  }
  if (last < text.length) frag.appendChild(document.createTextNode(text.slice(last)));
  return frag;
}

function showSupport(id, support, flashCard) {
  const card = $(`src-${id}`);
  if (!card) return;
  clearSupport();
  card.classList.add("linked");
  const snippet = card.querySelector(".source-snippet");
  if (support && snippet && snippet.dataset.text) {
    const text = snippet.dataset.text;
    let start = support.start, end = support.end;
    if (!(Number.isInteger(start) && Number.isInteger(end) && text.slice(start, end) === support.sentence)) {
      start = text.indexOf(support.sentence);
      end = start + support.sentence.length;
    }
    if (start >= 0 && end > start) {
      snippet.textContent = "";
      snippet.append(text.slice(0, start));
      const mark = document.createElement("mark");
      mark.className = "support";
      mark.textContent = text.slice(start, end);
      snippet.append(mark, text.slice(end));
      mark.dataset.applied = "1";
    }
  }
  if (flashCard) { card.scrollIntoView({ behavior: REDUCED_MOTION ? "auto" : "smooth", block: "center" }); flash(card); }
}

function clearSupport() {
  document.querySelectorAll(".source-card.linked").forEach((card) => {
    card.classList.remove("linked");
    const snippet = card.querySelector(".source-snippet");
    if (snippet && snippet.querySelector("mark.support")) renderSnippet(snippet, snippet.dataset.text, queryTerms());
  });
}

let lastQueryTerms = [];
function queryTerms() { return lastQueryTerms; }

const STOPWORDS = new Set("the and for with what how does can should are was were this that from have has not you your about into when which who whom may much many dose doses give given take taking".split(" "));
function termsFrom(text) {
  return [...new Set((text || "").replace(/\[[A-Z_]+\]/g, " ").toLowerCase().split(/[^a-z0-9]+/)
    .filter((w) => w.length >= 3 && !STOPWORDS.has(w)))];
}

function renderSnippet(el, text, terms) {
  el.textContent = "";
  appendHighlighted(el, text, terms, "term-hit");
}

function renderClaimRows(data) {
  const list = $("claim-rows"), summary = $("answer-summary");
  list.textContent = "";
  const claims = (data && data.claims) || [];
  list.hidden = summary.hidden = !claims.length;
  if (!claims.length) return;
  const sources = new Set(claims.flatMap((c) => c.source_ids || []));
  const grounded = claims.every((c) => c.grounded);
  const dosed = (data.trace || []).find((tt) => tt.name === "dosage guard");
  const parts = [t("js.claims_from_sources", "{claims} from {sources}", { claims: plural(claims.length, "claim"), sources: plural(sources.size, "source") })];
  if (grounded) parts.push(t("js.all_grounded", "all grounded"));
  if (dosed && dosed.status === "pass") parts.push(t("js.doses_verified", "doses verified"));
  summary.textContent = parts.join(" · ");
  claims.forEach((c) => {
    const li = document.createElement("li");
    li.className = "claim-row";
    const text = document.createElement("span");
    text.className = "claim-row-text";
    text.textContent = c.text;
    const cites = document.createElement("span");
    cites.className = "claim-row-cites";
    (c.source_ids || []).forEach((id) => {
      const a = document.createElement("a");
      a.className = "citation";
      a.href = `#src-${id}`;
      a.textContent = `[${id}]`;
      a.addEventListener("click", (e) => { e.preventDefault(); showSupport(id, supportFor(data, id) || (c.support && c.support.source_id === id ? c.support : null), true); });
      a.addEventListener("focus", () => showSupport(id, c.support && c.support.source_id === id ? c.support : null, false));
      a.addEventListener("mouseenter", () => showSupport(id, c.support && c.support.source_id === id ? c.support : null, false));
      a.addEventListener("mouseleave", clearSupport);
      a.addEventListener("blur", clearSupport);
      cites.appendChild(a);
    });
    li.append(text, cites);
    if (typeof c.relevance === "number") {
      const rel = document.createElement("span");
      rel.className = "relevance";
      rel.title = t("js.relevance_to_the_question", "Relevance to the question: {relevance}", { relevance: c.relevance.toFixed(2) });
      const bar = document.createElement("span");
      bar.className = "relevance-bar";
      bar.setAttribute("role", "img");
      bar.setAttribute("aria-label", t("js.relevance_percent", "Relevance {value} percent", { value: Math.round(c.relevance * 100) }));
      const fill = document.createElement("span");
      fill.style.width = `${Math.round(Math.max(0, Math.min(1, c.relevance)) * 100)}%`;
      bar.appendChild(fill);
      const num = document.createElement("span");
      num.className = "relevance-num";
      num.textContent = c.relevance.toFixed(2);
      rel.append(bar, num);
      li.appendChild(rel);
    }
    list.appendChild(li);
  });
}

function flash(el) { el.classList.add("flash"); setTimeout(() => el.classList.remove("flash"), 800); }

const KIND_LABEL = () => ({ condition: t("js.kind_condition", "condition"), drug: t("js.kind_drug", "drug"), reference: t("js.kind_reference", "reference") });

function renderSources(data) {
  const wrap = $("sources");
  wrap.innerHTML = "";
  const cited = new Set(citedIds(data));
  const sources = data.sources || [];
  if (!sources.length) {
    const p = document.createElement("p");
    p.className = "source-snippet";
    p.textContent = t("js.no_sources", "No sources retrieved for this question.");
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
      badge.textContent = t("js.cited", "cited");
      left.appendChild(badge);
    }
    if (s.above_gate === false) {
      const badge = document.createElement("span");
      badge.className = "src-badge below-gate";
      badge.textContent = t("js.below_gate", "below gate");
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
    if (s.kind) meta.appendChild(tag(KIND_LABEL()[s.kind] || s.kind, "tag-kind"));
    if (s.section) meta.appendChild(tag(s.section, "tag-section"));
    if (s.topic) meta.appendChild(tag(s.topic, "tag-topic"));

    const snippet = document.createElement("p");
    snippet.className = "source-snippet";
    snippet.dataset.text = s.snippet || "";
    renderSnippet(snippet, s.snippet || "", lastQueryTerms);

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
    num.textContent = t("js.similarity", "similarity");
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
    name.textContent = STAGE_LABELS()[step.name] || step.name;
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
  if (Array.isArray(d.checked_terms)) { frag.appendChild(coverageDetail(d)); return frag; }
  // Generic key/value detail for any other stage data (gate, coverage, etc.).
  frag.appendChild(kvDetail(d));
  return frag;
}

function kvDetail(obj, depth = 0) {
  const wrap = document.createElement("div");
  wrap.className = "stage-data" + (depth ? " nested" : "");
  for (const [k, v] of Object.entries(obj)) {
    const row = document.createElement("div");
    row.className = "data-line";
    const key = document.createElement("span");
    key.className = "mono strong"; key.textContent = k.replace(/_/g, " ");
    row.appendChild(key);
    const complex = v && typeof v === "object" && (Array.isArray(v) ? v.some((x) => x && typeof x === "object") : true);
    if (complex && depth < 5) {
      row.classList.add("data-block");
      const inner = Array.isArray(v) ? Object.fromEntries(v.map((x, i) => [String(i + 1), x])) : v;
      row.appendChild(kvDetail(inner, depth + 1));
    } else {
      const val = document.createElement("span");
      val.className = "data-muted";
      val.textContent = Array.isArray(v) ? (v.length ? v.join(", ") : t("js.none", "(none)")) : v === null || v === undefined ? t("js.none", "(none)") : String(v);
      row.appendChild(val);
    }
    wrap.appendChild(row);
  }
  return wrap;
}

function termChips(label, terms, cls) {
  const row = document.createElement("div");
  row.className = "data-line term-row";
  const key = document.createElement("span");
  key.className = "mono strong";
  key.textContent = label;
  const chips = document.createElement("span");
  chips.className = "term-chips";
  if (!terms || !terms.length) {
    chips.className += " data-muted";
    chips.textContent = t("js.none", "(none)");
  } else terms.forEach((tt) => {
    const chip = document.createElement("span");
    chip.className = `term-chip ${cls}`;
    chip.textContent = tt;
    chips.appendChild(chip);
  });
  row.append(key, chips);
  return row;
}

function coverageDetail(d) {
  const wrap = document.createElement("div");
  wrap.className = "stage-data";
  const table = document.createElement("table");
  table.className = "coverage-table";
  const head = document.createElement("thead");
  const hr = document.createElement("tr");
  [t("js.term_in_the_question", "Term in the question"), t("js.found_in_sources", "Found in sources")].forEach((tt) => { const th = document.createElement("th"); th.scope = "col"; th.textContent = tt; hr.appendChild(th); });
  head.appendChild(hr);
  const body = document.createElement("tbody");
  const covered = new Set(d.covered || []);
  (d.checked_terms || []).forEach((tt) => {
    const tr = document.createElement("tr");
    const term = document.createElement("td");
    term.className = "mono";
    term.textContent = tt;
    const state = document.createElement("td");
    const ok = covered.has(tt);
    const chip = document.createElement("span");
    chip.className = `term-chip ${ok ? "covered" : "uncovered"}`;
    chip.textContent = ok ? t("js.covered", "covered") : t("js.not_covered", "not covered");
    state.appendChild(chip);
    tr.append(term, state);
    body.appendChild(tr);
  });
  table.append(head, body);
  wrap.appendChild(d.checked_terms.length ? table : Object.assign(document.createElement("p"), { className: "data-muted", textContent: "No distinctive terms in the question to check." }));
  if (d.question_values && d.question_values.length) wrap.appendChild(termChips(t("js.values_stated", "values stated"), d.question_values, "neutral"));
  if (d.unsupported_values && d.unsupported_values.length) wrap.appendChild(termChips(t("js.unsupported_values", "unsupported values"), d.unsupported_values, "uncovered"));
  if (d.unsupported_forms && d.unsupported_forms.length) wrap.appendChild(termChips(t("js.unsupported_forms", "unsupported forms"), d.unsupported_forms, "uncovered"));
  const age = d.age;
  if (age && typeof age === "object") {
    const row = document.createElement("div");
    row.className = "data-line";
    const key = document.createElement("span");
    key.className = "mono strong";
    key.textContent = t("js.age", "age");
    const val = document.createElement("span");
    val.className = "data-muted";
    val.textContent = `${age.phrase || age.years} (${age.group})`;
    const chip = document.createElement("span");
    chip.className = `term-chip ${age.covered ? "covered" : "uncovered"}`;
    chip.textContent = age.covered ? t("js.sources_cover_this_group", "sources cover this group") : t("js.no_source_covers_this_group", "no source covers this group");
    row.append(key, val, chip);
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
    const tt = document.createElement("span");
    tt.className = "data-muted";
    tt.textContent = r.title;
    const sc = document.createElement("span");
    sc.className = "mono data-score";
    sc.textContent = (r.score ?? 0).toFixed(3);
    row.append(id, tt, sc);
    wrap.appendChild(row);
  });
  return wrap;
}

function groundingDetail(d) {
  const wrap = document.createElement("div");
  wrap.className = "stage-data";

  const method = document.createElement("p");
  method.className = "data-note";
  method.textContent = t("js.method_judge", "Method: {method}. Judge: {judge_summary}", { method: d.method, judge_summary: d.judge_summary }) +
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
    meta.appendChild(tag(t("js.cites", "cites {source_ids}", { source_ids: c.source_ids.join(", ") || t("js.nothing", "nothing") }), "tag-topic"));
    const det = document.createElement("span");
    det.className = "claim-stat " + (c.grounded ? "ok" : "bad");
    det.textContent = t("js.similarity_vs", "similarity {deterministic_score} vs {threshold} {grounded}", { deterministic_score: c.deterministic_score.toFixed(2), threshold: c.threshold.toFixed(2), grounded: c.grounded ? t("js.pass", "PASS") : t("js.fail", "FAIL") });
    meta.appendChild(det);
    box.appendChild(meta);

    if (c.judge) {
      const judge = document.createElement("div");
      judge.className = "judge-row";
      const verdict = document.createElement("span");
      verdict.className = "judge-verdict " + (c.judge.supported ? "ok" : "bad");
      verdict.textContent = c.judge.supported ? t("js.judge_supported", "judge: supported") : t("js.judge_not_supported", "judge: not supported");
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
      none.textContent = t("js.llm_judge_not_run_no", "LLM judge not run (no live provider, or judge disabled). Deterministic check is authoritative.");
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
  rule.textContent = d.rule ? t("js.rule", "Rule: {rule}.", { rule: d.rule }) : t("js.dosage_guard_disabled", "Dosage guard disabled.");
  wrap.appendChild(rule);

  if (!d.detected_values.length) {
    const none = document.createElement("p");
    none.className = "data-muted";
    none.textContent = t("js.no_values_with_clinical_units", "No values with clinical units were present in the answer, so nothing to verify.");
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
    status.textContent = verified ? t("js.stated_by_a_source", "stated by a source") : t("js.not_stated_by_any_source", "NOT stated by any source");
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
  } catch (_) { wrap.textContent = t("js.audit_record_unavailable", "Audit record unavailable."); }
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
  PIPELINE_STAGES().forEach(([name, desc], i) => {
    const li = document.createElement("li");
    li.className = "how-stage";
    const n = document.createElement("span");
    n.className = "how-stage-num mono";
    n.textContent = String(i + 1).padStart(2, "0");
    const body = document.createElement("div");
    const tt = document.createElement("div");
    tt.className = "how-stage-name mono";
    tt.textContent = name;
    const d = document.createElement("div");
    d.className = "how-stage-desc";
    d.textContent = desc;
    body.append(tt, d);
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
      t("js.passages_topics", "{total} passages, {topics} topics", { total: data.stats.total.toLocaleString(), topics: data.stats.topics });
  } catch (_) { $("corpus-summary").textContent = t("js.unavailable", "unavailable"); }
}

let corpusTopicList = [];

function renderCorpusTiles(stats) {
  const wrap = $("corpus-tiles");
  wrap.innerHTML = "";
  corpusTopicList = stats.topic_list || [];
  const tiles = [
    [t("js.tile_documents", "documents"), stats.total, false],
    [t("js.tile_conditions", "conditions"), stats.by_kind.condition || 0, false],
    [t("js.drug_labels", "drug labels"), stats.by_kind.drug || 0, false],
    [t("js.lab_markers", "lab markers"), stats.by_kind.marker || 0, false],
    [t("js.tile_procedures", "procedures"), stats.by_kind.procedure || 0, false],
    [t("js.tile_topics", "topics"), stats.topics, true],   // clickable: reveals the topic list
  ];
  tiles.forEach(([label, value, clickable]) => {
    const tt = document.createElement("div");
    tt.className = "corpus-tile" + (clickable ? " clickable" : "");
    const v = document.createElement("div");
    v.className = "tile-value mono";
    v.textContent = Number(value).toLocaleString();
    const l = document.createElement("div");
    l.className = "tile-label";
    l.textContent = clickable ? t("js.click_to_view", "{label} (click to view)", { label }) : label;
    tt.append(v, l);
    if (clickable) {
      tt.setAttribute("role", "button");
      tt.setAttribute("tabindex", "0");
      tt.addEventListener("click", toggleTopicList);
      tt.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); toggleTopicList(); } });
    }
    wrap.appendChild(tt);
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
  head.textContent = t("js.topics_each_topic_is_one", "{corpusTopicList} topics (each topic is one fictional entity, with several documents)", { corpusTopicList: corpusTopicList.length });
  panel.appendChild(head);
  const grid = document.createElement("div");
  grid.className = "topic-list-grid";
  corpusTopicList.forEach((tt) => {
    const item = document.createElement("div");
    item.className = "topic-item";
    const dot = document.createElement("span");
    dot.className = "topic-dot";
    dot.style.background = KIND_COLOR[tt.kind] || "#8f8f8f";
    const name = document.createElement("span");
    name.className = "topic-name";
    name.textContent = tt.topic;
    const count = document.createElement("span");
    count.className = "topic-count mono";
    count.textContent = `${tt.count}`;
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
  canvas.addEventListener("touchstart", (e) => { const tt = e.touches[0]; start(tt.clientX, tt.clientY); }, { passive: true });
  canvas.addEventListener("touchmove", (e) => { const tt = e.touches[0]; move(tt.clientX, tt.clientY); }, { passive: true });
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
    ctx.fillText(t("js.sources_retrieved_highlighted", "{hitCount} sources retrieved (highlighted)", { hitCount }), 8, size - 8);
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
  title.textContent = point.title || t("js.document", "(document)");
  const head = document.createElement("div");
  head.className = "tt-head";
  head.append(id, kind);
  tt.append(head, title);
  if (point.section) {
    const sec = document.createElement("div");
    sec.className = "tt-section";
    sec.textContent = t("js.section", "section: ") + point.section;
    tt.append(sec);
  }
  if (map3d.hits.has(point.id)) {
    const ret = document.createElement("div");
    ret.className = "tt-retrieved";
    ret.textContent = t("js.retrieved_for_this_answer", "● retrieved for this answer");
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
    [t("js.condition_pages_illnesses", "condition pages (illnesses)"), KIND_COLOR.condition, stats.by_kind.condition || 0],
    [t("js.drug_label_pages_medications", "drug label pages (medications)"), KIND_COLOR.drug, stats.by_kind.drug || 0],
    [t("js.lab_marker_pages", "lab marker pages"), KIND_COLOR.marker, stats.by_kind.marker || 0],
    [t("js.diagnostic_procedure_pages", "diagnostic procedure pages"), KIND_COLOR.procedure, stats.by_kind.procedure || 0],
    [t("js.general_reference_notes", "general reference notes"), KIND_COLOR.reference, stats.by_kind.reference || 0],
    [t("js.retrieved_for_your_question", "retrieved for your question"), "var(--map-hit)", null],
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
  title.textContent = t("js.documents_by_section", "Documents by section");
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

  if (!s) { summary.textContent = t("js.unavailable", "unavailable"); return; }

  const active = s.models.find((m) => m.name === s.active);
  summary.textContent = active ? t("js.using", "using {label}", { label: active.label })
    : s.ollama.running ? t("js.no_model_selected", "no model selected") : t("js.ollama_not_running", "Ollama not running");

  const h = s.hardware;
  const accel = { "apple-silicon": t("js.apple_silicon_gpu", "Apple silicon GPU"), nvidia: t("js.nvidia_gpu", "NVIDIA GPU"), cpu: t("js.cpu_only", "CPU only") }[h.accelerator] || h.accelerator;
  const facts = [
    h.cpu,
    t("js.memory", "{memory_gb} memory", { memory_gb: formatGb(h.memory_gb) }),
    ...h.gpus.map((g) => `${g.name}, ${formatGb(g.memory_gb)}`),
    accel,
    t("js.available_for_a_model", "{model_memory_gb} available for a model", { model_memory_gb: formatGb(h.model_memory_gb) }),
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
    msg.append(t("js.ollama_isn_t_running_at", "Ollama isn't running at "), Object.assign(document.createElement("code"), { textContent: s.ollama.host }),
      t("js.install_it_from", ". Install it from "), Object.assign(document.createElement("a"), { href: "https://ollama.com/download", textContent: "ollama.com", target: "_blank", rel: "noopener" }),
      t("js.start_it_then_reopen_this", ", start it, then reopen this panel."));
  } else if (!s.can_manage) {
    msg.hidden = false;
    msg.textContent = t("js.models_can_only_be_downloaded", "Models can only be downloaded or switched from the machine running GroundCheckHealth.");
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
  if (m.recommended) main.appendChild(badge(t("js.recommended", "Recommended"), "recommended"));
  main.appendChild(badge({ good: t("js.good_fit", "Good fit"), tight: t("js.tight_fit", "Tight fit"), "too-large": t("js.too_large", "Too large") }[m.fit], `fit ${m.fit}`));

  const meta = document.createElement("div");
  meta.className = "local-ai-model-meta mono";
  meta.textContent = t("js.download_needs", "{name}  {parameters}  download {download_gb}  needs {memory_needed_gb}", { name: m.name, parameters: m.parameters, download_gb: formatGb(m.download_gb), memory_needed_gb: formatGb(m.memory_needed_gb) });
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
    bar.setAttribute("aria-label", t("js.downloading", "Downloading {label}", { label: m.label }));
    const fill = document.createElement("span");
    fill.style.width = `${progress.percent}%`;
    bar.appendChild(fill);
    const label = document.createElement("span");
    label.className = "local-ai-progress-label mono";
    label.textContent = progress.label;
    actions.append(bar, label);
  } else if (!m.installed) {
    actions.appendChild(button(t("js.download", "Download {download_gb}", { download_gb: formatGb(m.download_gb) }), "btn-ghost",
      disabled || m.fit === "too-large", () => downloadModel(m)));
  } else if (s.selected === m.name) {
    const inUse = badge(localAI.busy === m.name ? t("js.loading", "Loading…") : t("js.in_use", "In use"), "in-use");
    actions.append(inUse, button(t("js.stop_using", "Stop using"), "btn-ghost", disabled, () => selectModel(null)));
  } else {
    actions.appendChild(button(localAI.busy === m.name ? t("js.loading", "Loading…") : t("js.use_this_model", "Use this model"), "btn-primary",
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
  localAI.downloading.set(m.name, { percent: 0, label: t("js.starting_lower", "starting…") });
  renderLocalAI();
  let error = null;
  try {
    const res = await fetch("/api/local-ai/pull", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ model: m.name }),
    });
    if (!res.ok || !res.body) throw new Error((await res.json().catch(() => ({}))).detail || t("js.http", "HTTP {status}", { status: res.status }));
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
        const label = event.total ? t("js.of", "{percent}%  {completed} of {total}", { percent, completed: formatGb(event.completed / 1e9), total: formatGb(event.total / 1e9) }) : event.status;
        localAI.downloading.set(m.name, { percent, label });
        renderLocalAI();
      }
    }
  } catch (err) {
    error = err.message;
  }
  localAI.downloading.delete(m.name);
  await loadLocalAI();
  if (error) showLocalAIError(t("js.download_failed", "Download failed: {error}", { error }));
}

async function selectModel(name) {
  localAI.busy = name || localAI.status?.selected || "";
  renderLocalAI();
  let error = null;
  try {
    const res = await fetch("/api/local-ai/select", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ model: name }),
    });
    if (!res.ok) error = (await res.json().catch(() => ({}))).detail || t("js.http", "HTTP {status}", { status: res.status });
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
const STATUS_LABELS = () => ({ pending: t("js.pending_review", "Pending review"), approved: t("js.approved", "Approved"), rejected: t("js.rejected", "Rejected"), retired: t("js.retired", "Retired") });
let indexPoll = null;

function isoDateToIso(value) { return value ? `${value}T00:00:00` : ""; }
// A date on its own, spelled so it cannot be read the wrong way round:
// "18 Sep 2026", never 9/18/2026 or 18/09/2026.
function shortDate(iso) {
  return iso ? new Date(iso).toLocaleDateString(uiLocale(), { day: "numeric", month: "short", year: "numeric" }) : "";
}

// A group admin (an admin with no site) chooses which site an upload belongs to.
async function fillUploadSites() {
  const field = $("upload-site-field");
  const u = account.user;
  field.hidden = true;
  if (!u || u.role !== "admin" || (u.site_id !== null && u.site_id !== undefined)) return;
  try {
    const { sites } = await api("/api/sites");
    if (!sites || !sites.length) return;
    const select = $("upload-site");
    const keep = select.value;
    select.textContent = "";
    [["", t("js.every_site", "Every site")], ...sites.map((x) => [String(x.id), x.name])].forEach(([v, l]) => {
      const o = document.createElement("option"); o.value = v; o.textContent = l; select.appendChild(o);
    });
    select.value = keep;
    field.hidden = false;
  } catch (_) { /* not a group admin, or no sites */ }
}

async function loadDocuments() {
  let data;
  try { data = await api("/api/sources"); }
  catch (err) {
    showMessage("documents-message", err.message, true);
    $("upload-form").hidden = true;
    return;
  }
  $("upload-form").hidden = false;
  fillUploadSites();
  renderIndexStatus(data.index);
  const rows = $("documents-body-rows");
  rows.textContent = "";
  $("documents-empty").hidden = data.sources.length > 0;
  const approved = data.sources.filter((d) => d.status === "approved").length;
  const pending = data.sources.filter((d) => d.status === "pending").length;
  $("documents-summary").textContent = t("js.approved_pending_review", "{approved} approved, {pending} pending review", { approved, pending });

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
    chip.textContent = STATUS_LABELS()[d.status] || d.status;
    status.appendChild(chip);

    const owner = document.createElement("td");
    owner.textContent = d.owner || "";

    const force = document.createElement("td");
    const range = [d.effective_from ? t("js.from_2", "from {effective_from}", { effective_from: shortDate(d.effective_from) }) : "", d.expires_on ? t("js.until", "until {expires_on}", { expires_on: shortDate(d.expires_on) }) : ""].filter(Boolean).join(" ");
    force.textContent = d.status === "approved" ? `${d.in_force ? t("js.yes", "Yes") : t("js.no", "No")}${range ? `, ${range}` : ""}` : (range || "");

    const sections = document.createElement("td");
    sections.className = "mono";
    sections.textContent = d.chunks;

    const actionsCell = document.createElement("td");
    const actions = document.createElement("div");
    actions.className = "document-actions";
    actionsCell.appendChild(actions);
    actions.appendChild(button(t("js.view", "View"), "btn-ghost", false, () => openDocument(d.id)));
    if (d.status === "pending") {
      actions.appendChild(button(t("js.approve", "Approve"), "btn-primary", false, () => reviewDocument(d, "approve")));
      actions.appendChild(button(t("js.reject", "Reject"), "btn-ghost", false, () => reviewDocument(d, "reject")));
    }
    if (d.status === "approved") {
      if (d.in_force) actions.appendChild(button(t("js.evaluate", "Evaluate"), "btn-ghost", false, (e) => evaluateDocument(d, e.currentTarget)));
      actions.appendChild(button(t("js.retire", "Retire"), "btn-ghost", false, () => reviewDocument(d, "retire")));
    }
    tr.append(doc, status, owner, force, sections, actionsCell);
    rows.appendChild(tr);
  });
}

function renderIndexStatus(index) {
  const text = $("index-status-text");
  if (index.state === "running") text.textContent = t("js.rebuilding_the_index", "Rebuilding the index…");
  else if (index.state === "failed") text.textContent = t("js.index_rebuild_failed", "Index rebuild failed: {error}", { error: index.error });
  else if (index.finished_at) text.textContent = t("js.index_rebuilt_passages_newly_embedded", "Index rebuilt {finished_at}: {documents} passages ({embedded} newly embedded).", { finished_at: dateTime(index.finished_at), documents: index.documents.toLocaleString(), embedded: index.embedded });
  else text.textContent = t("js.the_index_was_built_when", "The index was built when the app started.");
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
  const doneMessages = {
    approve: (v) => t("js.approved_title", "Approved “{title}”.", v),
    reject: (v) => t("js.rejected_title", "Rejected “{title}”.", v),
    retire: (v) => t("js.retired_title", "Retired “{title}”.", v),
  };
  const prompts = {
    approve: t("js.approve_version_it_will_be", "Approve “{title}” version {version}? It will be cited in answers.", { title: d.title, version: d.version }),
    reject: t("js.reject_version_add_a_note", "Reject “{title}” version {version}? Add a note for the uploader (optional):", { title: d.title, version: d.version }),
    retire: t("js.retire_version_it_will_stop", "Retire “{title}” version {version}? It will stop being cited.", { title: d.title, version: d.version }),
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
    showMessage("documents-message", doneMessages[decision]({ title: d.title }) + (decision !== "reject" ? t("js.rebuilding_the_index_2", " Rebuilding the index.") : ""));
    renderIndexStatus(data.index);
  } catch (err) { showMessage("documents-message", err.message, true); }
  await loadDocuments();
}

async function evaluateDocument(d, trigger) {
  trigger.disabled = true;
  trigger.textContent = t("js.evaluating", "Evaluating…");
  try {
    const data = await api(`/api/sources/${d.id}/evaluate`, { method: "POST" });
    const e = data.evaluation;
    showMessage("documents-message", t("js.of_test_questions_behaved_as", "“{title}”: {passed} of {total} test questions behaved as intended, {unsafe_answers} unsafe answers.", { title: d.title, passed: e.passed, total: e.total, unsafe_answers: e.unsafe_answers }), e.unsafe_answers > 0);
    await openDocument(d.id);
  } catch (err) { showMessage("documents-message", err.message, true); }
  await loadDocuments();
}

async function openDocument(id) {
  let data;
  try { data = await api(`/api/sources/${id}`); }
  catch (err) { showMessage("documents-message", err.message, true); return; }
  const d = data.source;
  $("document-dialog-title").textContent = t("js.version", "{title}, version {version}", { title: d.title, version: d.version });
  const bits = [STATUS_LABELS()[d.status] || d.status, d.owner && t("js.owner", "Owner: {owner}", { owner: d.owner }), d.filename,
    d.review_note && t("js.note", "Note: {review_note}", { review_note: d.review_note })].filter(Boolean);
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
    $("document-eval-summary").textContent = t("js.of_behaved_as_intended_unsafe", "{passed} of {total} behaved as intended, {unsafe_answers} unsafe answers, run {ran_at}.", { passed: e.passed, total: e.total, unsafe_answers: e.unsafe_answers, ran_at: dateTime(e.ran_at) });
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
      r.textContent = t("js.expect_got", "expect {expect} / got {got}", { expect: c.expect, got: c.got });
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
    if (!file) { showMessage("documents-message", t("js.choose_a_file_to_upload", "Choose a file to upload."), true); return; }
    const body = new FormData();
    body.append("file", file);
    body.append("title", $("upload-title").value);
    body.append("owner", $("upload-owner").value);
    body.append("effective_from", isoDateToIso($("upload-effective").value));
    body.append("expires_on", isoDateToIso($("upload-expires").value));
    if (!$("upload-site-field").hidden && $("upload-site").value) body.append("site_id", $("upload-site").value);
    formBusy(form, true);
    $("upload-button").textContent = t("js.uploading", "Uploading…");
    try {
      const res = await fetch("/api/sources", { method: "POST", body });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.detail || t("js.upload_failed", "Upload failed ({status}).", { status: res.status }));
      form.reset();
      showMessage("documents-message", t("js.uploaded_as_version_with_sections", "Uploaded “{title}” as version {version}, with {chunks} sections. It needs approval before it’s cited.", { title: data.source.title, version: data.source.version, chunks: data.source.chunks }));
      await loadDocuments();
    } catch (err) { showMessage("documents-message", err.message, true); }
    formBusy(form, false);
    $("upload-button").textContent = t("js.upload_for_review", "Upload for review");
  });
}

// ---------- Review ----------
const KIND_LABELS = () => ({ refusal: t("js.refusal", "Refusal"), flagged: t("js.flagged_answer", "Flagged answer") });
const SEVERITY = () => (["", t("js.minor", "Minor"), t("js.significant", "Significant"), t("js.considerable", "Considerable"), t("js.major", "Major"), t("js.catastrophic", "Catastrophic")]);
const LIKELIHOOD = () => (["", t("js.very_low", "Very low"), t("js.low", "Low"), t("js.medium", "Medium"), t("js.high", "High"), t("js.very_high", "Very high")]);
const review = { tab: "queue", data: null, caseId: null, hazards: [], hazardId: null };

function relativeTime(iso) {
  const minutes = Math.round((new Date(iso) - Date.now()) / 60000);
  const fmt = new Intl.RelativeTimeFormat(uiLocale(), { numeric: "auto" });
  const abs = Math.abs(minutes);
  if (abs < 60) return fmt.format(minutes, "minute");
  if (abs < 48 * 60) return fmt.format(Math.round(minutes / 60), "hour");
  return fmt.format(Math.round(minutes / 1440), "day");
}

// Labels for table cells, shown when rows stack as cards on narrow screens.
function labelCells(cells, labels) {
  cells.forEach((td, i) => { if (labels[i]) td.dataset.label = labels[i]; });
}

const coded = (prefix, code, fallback) => t(`${prefix}.${String(code).replace(/[^a-z0-9]+/gi, "_")}`, fallback);
const plural = (n, word) => {
  const key = "plural." + word.replace(/ /g, "_") + (n === 1 ? ".one" : ".other");
  return `${n.toLocaleString()} ${t(key, n === 1 ? word : word + "s")}`;
};

function selectTab(name) {
  review.tab = name;
  for (const tab of ["queue", "hazards", "report", "controls"]) {
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
  if (review.tab === "controls") return loadControls();
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
  $("review-summary").textContent = t("js.open", "{open} open{overdue}", { open: c.open, overdue: c.overdue ? t("js.overdue", ", {overdue} overdue", { overdue: c.overdue }) : "" });
  const counts = $("review-counts");
  counts.textContent = "";
  [[t("js.open_2", "Open"), c.open, ""], [t("js.overdue_2", "Overdue"), c.overdue, c.overdue ? "bad" : ""], [t("js.resolved", "Resolved"), c.resolved, ""]].forEach(([label, n, cls]) => {
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
    why.textContent = k.kind === "flagged" ? t("js.flagged", "Flagged: {reason}", { reason: k.reason }) : k.reason_category;
    q.append(text, why);

    const type = document.createElement("td");
    const pills = document.createElement("div");
    pills.className = "pill-group";
    type.appendChild(pills);
    const pill = document.createElement("span");
    pill.className = `status-pill kind-${k.kind}${k.priority === "high" ? " priority-high" : ""}`;
    pill.textContent = KIND_LABELS()[k.kind] || k.kind;
    pills.appendChild(pill);
    if (k.status !== "open") {
      const st = document.createElement("span");
      st.className = "status-pill status-retired";
      st.textContent = t("js.resolved", "Resolved");
      pills.appendChild(st);
    }

    const asked = document.createElement("td");
    asked.textContent = k.occurrences === 1 ? t("js.once", "once") : t("js.times", "{occurrences} times", { occurrences: k.occurrences });
    asked.title = t("js.first_last", "First {created_at}, last {last_seen_at}", { created_at: dateTime(k.created_at), last_seen_at: dateTime(k.last_seen_at) });

    const due = document.createElement("td");
    if (k.status === "open") {
      due.textContent = k.overdue ? t("js.overdue_3", "Overdue, {due_at}", { due_at: relativeTime(k.due_at) }) : relativeTime(k.due_at);
      due.className = k.overdue ? "due-overdue" : "";
    } else {
      due.textContent = coded("outcome", k.outcome, k.outcome_label);
    }

    const who = document.createElement("td");
    who.textContent = k.assigned_to_name || t("js.unassigned", "Unassigned");
    if (!k.assigned_to_name) who.className = "muted";

    const actionsCell = document.createElement("td");
    const actions = document.createElement("div");
    actions.className = "document-actions";
    actions.appendChild(button(k.status === "open" ? t("js.review", "Review") : t("js.view", "View"), k.status === "open" ? "btn-primary" : "btn-ghost", false, () => openCase(k.id)));
    actionsCell.appendChild(actions);

    labelCells([q, type, asked, due, who, actionsCell], ["", t("js.type", "Type"), t("js.asked", "Asked"), k.status === "open" ? t("js.due", "Due") : t("js.outcome", "Outcome"), t("js.assigned", "Assigned"), ""]);
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
  $("case-dialog-title").textContent = `${KIND_LABELS()[k.kind] || t("js.case_2", "Case")} #${k.id}`;
  $("case-query").textContent = k.query;
  const meta = [
    k.kind === "flagged" ? t("js.flagged_2", "Flagged{flagged_by_name}: {reason}", { flagged_by_name: k.flagged_by_name ? t("js.by", " by {flagged_by_name}", { flagged_by_name: k.flagged_by_name }) : "", reason: k.reason }) : t("js.refused_2", "Refused: {reason}", { reason: k.reason }),
    t("js.asked_first", "Asked {occurrences}, first {created_at}", { occurrences: k.occurrences === 1 ? t("js.once", "once") : t("js.times", "{occurrences} times", { occurrences: k.occurrences }), created_at: dateTime(k.created_at) }),
    k.status === "open" ? t("js.due_2", "Due {due_at}{escalated}", { due_at: relativeTime(k.due_at), escalated: k.escalated ? t("js.escalated", ", escalated") : "" }) : "",
    t("js.audit_record", "Audit record {last_audit_id}", { last_audit_id: k.last_audit_id }),
  ].filter(Boolean).map((part) => part.replace(/[.\s]+$/, ""));
  $("case-meta").textContent = meta.join(". ") + ".";

  $("case-answer-section").hidden = !k.answer_text;
  $("case-answer").textContent = k.answer_text || "";
  $("case-sources").textContent = (k.sources || []).length
    ? t("js.sources_retrieved", "Sources retrieved: {sources}.", { sources: k.sources.map((x) => x.id).join(", ") }) : "";

  const open = k.status === "open";
  $("case-work").hidden = !open;
  $("case-closed").hidden = open;
  if (open) {
    const select = $("case-assignee");
    select.textContent = "";
    const none = new Option(t("js.unassigned", "Unassigned"), "");
    select.appendChild(none);
    (review.data?.reviewers || []).forEach((r) => select.appendChild(new Option(r.name, r.id)));
    if (review.data?.me && !(review.data.reviewers || []).some((r) => r.id === review.data.me)) {
      select.appendChild(new Option(t("js.me", "Me"), review.data.me));
    }
    select.value = k.assigned_to ?? "";
    const outcome = $("case-outcome");
    if (!outcome.options.length) {
      outcome.appendChild(new Option(t("js.choose", "Choose…"), ""));
      Object.entries(review.data?.outcomes || {}).forEach(([key, label]) => outcome.appendChild(new Option(coded("outcome", key, label), key)));
    }
    outcome.value = "";
    $("case-expected").value = k.kind === "refusal" ? "refuse" : "";
    $("case-expected-field").hidden = true;
    $("case-note").value = "";
    $("case-comment").value = "";
  } else {
    $("case-outcome-text").textContent = t("js.resolved_2", "Resolved{resolved_by_name} {resolved_at}: {outcome_label}.{outcome_note}", { resolved_by_name: k.resolved_by_name ? t("js.by_2", " by {resolved_by_name}", { resolved_by_name: k.resolved_by_name }) : "", resolved_at: dateTime(k.resolved_at), outcome_label: coded("outcome", k.outcome, k.outcome_label), outcome_note: k.outcome_note ? ` ${k.outcome_note}` : "" });
    $("case-reopen-note").value = "";
  }

  const list = $("case-timeline");
  list.textContent = "";
  k.events.slice().reverse().forEach((e) => {
    const li = document.createElement("li");
    const head = document.createElement("div");
    head.className = "timeline-head";
    const action = document.createElement("strong");
    action.textContent = coded("case_event", e.action, capitalize(e.action));
    const when = document.createElement("span");
    when.className = "muted";
    when.textContent = `${e.user ? `${e.user}, ` : ""}${dateTime(e.at)}`;
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
    pill.title = t("js.severity_likelihood", "Severity {s} ({s2}) × likelihood {l} ({l2})", { s, s2: SEVERITY()[s], l, l2: LIKELIHOOD()[l] });
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
    controls.textContent = h.controls ? t("js.controls", "Controls: {controls}", { controls: h.controls }) : t("js.no_controls_recorded", "No controls recorded");
    title.append(name, controls);
    const status = document.createElement("td");
    status.textContent = coded("hazard_status", h.status, capitalize(h.status));
    const owner = document.createElement("td");
    owner.textContent = h.owner;
    const actionsCell = document.createElement("td");
    const actions = document.createElement("div");
    actions.className = "document-actions";
    actions.appendChild(button(canEditHazards() ? t("js.edit", "Edit") : t("js.view", "View"), "btn-ghost", false, () => openHazard(h)));
    actionsCell.appendChild(actions);
    const cells = [id, title, riskCell(h.initial_risk, h.severity, h.likelihood),
      riskCell(h.residual_risk, h.residual_severity, h.residual_likelihood), status, owner, actionsCell];
    labelCells(cells, [t("js.id", "ID"), "", t("js.before_controls", "Before controls"), t("js.after_controls", "After controls"), t("js.status", "Status"), t("js.owner_2", "Owner"), ""]);
    tr.append(...cells);
    rows.appendChild(tr);
  });
}

function riskLabel(s, l) {
  const score = s * l;
  const level = score <= 4 ? t("js.level_low", "low") : score <= 9 ? t("js.level_medium", "medium") : score <= 16 ? t("js.level_high", "high") : t("js.very_high_2", "very high");
  return t("js.risk", "Risk {score}, {level}", { score, level });
}

function updateHazardRisk() {
  $("hazard-initial-risk").textContent = riskLabel(+$("hazard-severity").value, +$("hazard-likelihood").value);
  $("hazard-residual-risk").textContent = riskLabel(+$("hazard-residual-severity").value, +$("hazard-residual-likelihood").value);
}

function openHazard(h) {
  review.hazardId = h ? h.id : null;
  $("hazard-dialog-title").textContent = h ? t("js.hazard_h", "Hazard H{id}", { id: h.id }) : t("js.add_hazard", "Add hazard");
  $("hazard-title").value = h?.title || "";
  $("hazard-cause").value = h?.cause || "";
  $("hazard-effect").value = h?.effect || "";
  $("hazard-controls").value = h?.controls || "";
  $("hazard-severity").value = h?.severity || 3;
  $("hazard-likelihood").value = h?.likelihood || 3;
  $("hazard-residual-severity").value = h?.residual_severity || 3;
  $("hazard-residual-likelihood").value = h?.residual_likelihood || 2;
  $("hazard-status").value = h?.status || t("js.open_3", "open");
  $("hazard-owner").value = h?.owner || "";
  updateHazardRisk();
  const editable = canEditHazards();
  $("hazard-form").querySelectorAll("input, textarea, select, button").forEach((el) => { el.disabled = !editable; });
  $("hazard-save").hidden = !editable;
  $("hazard-message").hidden = true;
  $("hazard-dialog").showModal();
  if (editable) $("hazard-title").focus();
}

// ---------- Control mapping ----------
const controlsState = { data: null };

async function loadControls() {
  let data;
  try { data = await api("/api/governance/controls"); }
  catch (err) { showMessage("review-message", err.message, true); return; }
  controlsState.data = data;
  $("controls-banner").textContent = data.banner;
  const totals = $("controls-totals");
  totals.textContent = "";
  Object.entries(data.statuses).forEach(([status, label]) => {
    const tile = document.createElement("div");
    tile.className = "report-tile";
    const l = document.createElement("span"); l.className = "report-label"; l.textContent = label;
    const v = document.createElement("strong"); v.className = "report-value"; v.textContent = String(data.totals[status]);
    tile.append(l, v);
    totals.appendChild(tile);
  });
  if (data.problems.length) showMessage("review-message", t("js.the_mapping_has", "The mapping has {problems}: {problems2}", { problems: plural(data.problems.length, "problem"), problems2: data.problems[0] }), true);
  renderControls(data);
}

function renderControls(data) {
  const host = $("controls-frameworks");
  host.textContent = "";
  if (!data) return;
  const only = $("controls-status").value;
  data.frameworks.forEach((framework) => {
    const rows = framework.controls.filter((c) => !only || c.status === only);
    if (!rows.length) return;
    const h = document.createElement("h3");
    h.textContent = `${framework.name} (${rows.length})`;
    const wrap = document.createElement("div");
    wrap.className = "users-table-wrap";
    const table = document.createElement("table");
    table.className = "users-table controls-table";
    const head = document.createElement("thead");
    const hr = document.createElement("tr");
    [t("js.control", "Control"), t("js.title", "Title"), t("js.status", "Status"), t("js.how_the_product_helps", "How the product helps"), t("js.evidence", "Evidence")].forEach((tt) => {
      const th = document.createElement("th"); th.scope = "col"; th.textContent = tt; hr.appendChild(th);
    });
    head.appendChild(hr);
    const body = document.createElement("tbody");
    rows.forEach((c) => {
      const tr = document.createElement("tr");
      const cells = [c.id, c.title, coded("control_status", c.status, c.status_label), c.how, c.evidence.length ? c.evidence.join(", ") : t("js.none_lower", "none")];
      cells.forEach((text, i) => {
        const td = document.createElement("td");
        td.textContent = text;
        if (i === 0 || i === 4) td.className = "mono";
        if (i === 2) td.className = `control-status control-${c.status}`;
        tr.appendChild(td);
      });
      body.appendChild(tr);
    });
    table.append(head, body);
    wrap.appendChild(table);
    host.append(h, wrap);
  });
}

async function loadReport() {
  const days = $("report-days").value;
  $("safety-case-download").href = `/api/governance/safety-case?days=${days}`;
  $("surveillance-download").href = `/api/governance/surveillance?days=${Math.max(days, 90)}`;
  let r;
  try { r = await api(`/api/governance/report?days=${days}`); }
  catch (err) { showMessage("review-message", err.message, true); return; }
  const q = r.questions, rv = r.reviews;
  const pct = (n) => n === null ? "–" : `${Math.round(n * 100)}%`;
  const tiles = [
    [t("js.questions", "Questions"), q.total.toLocaleString(), t("js.answered", "{answered} answered", { answered: q.answered.toLocaleString() })],
    [t("js.refusal_rate", "Refusal rate"), pct(q.refusal_rate), t("js.refused_3", "{refused} refused", { refused: q.refused.toLocaleString() })],
    [t("js.review_cases", "Review cases"), rv.opened.toLocaleString(), plural(rv.flagged_answers, "flagged answer")],
    [t("js.resolved_on_time", "Resolved on time"), rv.resolved ? t("js.of_2", "{resolved_on_time} of {resolved}", { resolved_on_time: rv.resolved_on_time, resolved: rv.resolved }) : "–", rv.median_hours_to_resolve === null ? t("js.none_resolved", "none resolved") : rv.median_hours_to_resolve < 1 ? t("js.median_under_an_hour", "median under an hour") : t("js.median_hours", "median {median_hours_to_resolve} hours", { median_hours_to_resolve: rv.median_hours_to_resolve })],
    [t("js.open_now", "Open now"), rv.open_now.toLocaleString(), t("js.overdue_4", "{overdue_now} overdue", { overdue_now: rv.overdue_now })],
    [t("js.checks_switched_off", "Checks switched off"), q.with_a_check_switched_off.toLocaleString(), t("js.questions_asked_with_a_check", "questions asked with a check off")],
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
    h.textContent = t("js.why_questions_were_refused", "Why questions were refused");
    reasons.appendChild(h);
    const max = Math.max(...entries.map(([, n]) => n));
    entries.forEach(([reason, n]) => {
      const row = document.createElement("div");
      row.className = "reason-row";
      const label = document.createElement("span");
      label.textContent = coded("refusal_category", reason, capitalize(reason));
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
    r.textContent = c.got ? t("js.expect_got", "expect {expect} / got {got}", { expect: c.expect, got: c.got }) : t("js.expect", "expect {expect}", { expect: c.expect });
    li.append(glyph, q, r);
    list.appendChild(li);
  });
}

async function loadReviewTests() {
  try {
    const data = await api("/api/review-tests");
    $("review-tests-summary").textContent = data.cases.length
      ? t("js.question_added_from_resolved_cases", "{cases} question{s} added from resolved cases.", { cases: data.cases.length, s: data.cases.length === 1 ? "" : "s" })
      : t("js.none_yet_resolve_a_case", "None yet. Resolve a case as “Added as a permanent test question” to add one.");
    $("review-tests-run").hidden = !data.cases.length;
    renderReviewTestCases(data.cases);
  } catch (_) { /* shown by the report */ }
}

async function verifyAudit() {
  const trigger = $("audit-verify");
  const out = $("verify-result");
  trigger.disabled = true;
  trigger.textContent = t("js.verifying", "Verifying…");
  try {
    const r = await api("/api/audit/verify", { method: "POST" });
    out.hidden = false;
    out.className = `verify-result ${r.ok ? "ok" : "bad"}`;
    out.textContent = "";
    const head = document.createElement("strong");
    head.textContent = !r.complete ? t("js.couldn_t_check_the_whole", "Couldn’t check the whole audit trail")
      : r.ok ? t("js.intact", "Intact") : t("js.found", "{problem_count} found", { problem_count: plural(r.problem_count, "problem") });
    const detail = document.createElement("p");
    detail.textContent = t("js.checked", "{checked} checked{anchor_seq}, ", { checked: plural(r.checked, "record"), anchor_seq: r.anchor_seq ? t("js.from_number", " from number {anchor_seq}", { anchor_seq: r.anchor_seq + 1 }) : "" })
      + `${r.signed ? t("js.signed_with_a_key_held", "signed with a key held outside the database") : t("js.not_signed_with_a_key", "not signed with a key")}. `
      + t("js.chain_head", "Chain head {seq}: {hash}…", { seq: r.head.seq, hash: r.head.hash.slice(0, 16) });
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
        li.textContent = t("js.record", "Record {seq}{audit_id}: {problem}", { seq: p.seq, audit_id: p.audit_id ? ` (${p.audit_id})` : "", problem: p.problem });
        list.appendChild(li);
      });
      out.appendChild(list);
    }
  } catch (err) { showMessage("protection-message", err.message, true); }
  trigger.disabled = false;
  trigger.textContent = t("js.verify_audit_trail", "Verify audit trail");
}

async function loadTelemetry() {
  let tt;
  try { tt = await api("/api/telemetry"); }
  catch (_) { return; }
  $("telemetry-summary").textContent = tt.enabled
    ? t("js.on_once_a_week_this", "On. Once a week this install sends the fields below to {url}. Nothing else is sent.", { url: tt.url || t("js.no_address_telemetry_url_is", "no address (TELEMETRY_URL is not set, so nothing is sent)") })
    : t("js.off_nothing_is_sent_if", "Off. Nothing is sent. {how_to_enable} If you turn it on, once a week it sends only the fields below.", { how_to_enable: tt.how_to_enable });
  const list = $("telemetry-fields");
  list.textContent = "";
  Object.entries(tt.fields).forEach(([name, meaning]) => {
    const dt = document.createElement("dt"); dt.className = "mono"; dt.textContent = name;
    const dd = document.createElement("dd"); dd.textContent = meaning;
    list.append(dt, dd);
  });
  $("telemetry-sample").textContent = t("js.example_of_what_would_be", "Example of what would be sent: {stringify}", { stringify: JSON.stringify(tt.sample) });
}

async function loadProtection() {
  let data;
  try { data = await api("/api/data-protection"); }
  catch (_) { $("protection-admin").hidden = true; return; }  // admins only
  $("protection-admin").hidden = false;
  loadTelemetry();
  const e = data.encryption, sig = data.audit_signing, ret = data.retention;
  const rows = [
    [t("js.encryption_at_rest", "Encryption at rest"), e.enabled
      ? t("js.on_aes_256_gcm_with", "On, AES-256-GCM with key {primary_key_id}{decrypt_key_ids}", { primary_key_id: e.primary_key_id, decrypt_key_ids: e.decrypt_key_ids.length ? t("js.older_keys_still_readable", ". Older keys still readable: {decrypt_key_ids}", { decrypt_key_ids: e.decrypt_key_ids.join(", ") }) : "" })
      : t("js.off_set_data_encryption_keys", "Off. Set DATA_ENCRYPTION_KEYS to encrypt questions, answers and review notes.")],
    [t("js.audit_signing", "Audit signing"), sig.enabled ? t("js.on_hmac_sha256_with_key", "On, HMAC-SHA256 with key {primary_key_id}", { primary_key_id: sig.primary_key_id }) : t("js.off_set_audit_signing_keys", "Off. Set AUDIT_SIGNING_KEYS so the chain can't be rewritten from inside the database.")],
    [t("js.audit_chain_head", "Audit chain head"), t("js.records_written_last_hash", "{seq} records written. Last hash {hash}…", { seq: data.audit_chain.seq.toLocaleString(), hash: data.audit_chain.hash.slice(0, 16) })],
  ];
  const list = $("protection-list");
  list.textContent = "";
  const offRows = [!e.enabled, !sig.enabled, false];
  rows.forEach(([term, value], i) => {
    const dt = document.createElement("dt"); dt.textContent = term;
    const dd = document.createElement("dd"); dd.textContent = value;
    if (offRows[i]) dd.className = "muted";
    list.append(dt, dd);
  });
  renderRetention(ret);
}

function renderRetention(ret) {
  const periods = [
    ret.audit_retention_days ? t("js.audit_records_for", "audit records for {audit_retention_days}", { audit_retention_days: plural(ret.audit_retention_days, "day") }) : t("js.audit_records_for_ever", "audit records for ever"),
    ret.review_retention_days ? t("js.resolved_reviews_for", "resolved reviews for {review_retention_days}", { review_retention_days: plural(ret.review_retention_days, "day") }) : t("js.resolved_reviews_for_ever", "resolved reviews for ever"),
  ];
  const due = ret.audit_records_to_delete + ret.review_cases_to_delete;
  const last = ret.last_run ? t("js.last_run_and_deleted", " Last run {ran_at}: {audit_deleted} and {reviews_deleted} deleted.", { ran_at: dateTime(ret.last_run.ran_at), audit_deleted: plural(ret.last_run.audit_deleted, "audit record"), reviews_deleted: plural(ret.last_run.reviews_deleted, "review") }) : "";
  $("retention-summary").textContent = t("js.keeping", "Keeping {periods}. ", { periods: periods.join(t("js.and", " and ")) })
    + (due ? t("js.and_are_past_their_period", "{audit_records_to_delete} and {review_cases_to_delete} are past their period.", { audit_records_to_delete: plural(ret.audit_records_to_delete, "audit record"), review_cases_to_delete: plural(ret.review_cases_to_delete, "resolved review") }) : t("js.nothing_is_past_its_retention", "Nothing is past its retention period."))
    + last;
  $("retention-run").disabled = due === 0;
}

async function runRetention() {
  if (!window.confirm(t("js.permanently_delete_every_record_past", "Permanently delete every record past its retention period? This can’t be undone."))) return;
  try {
    const data = await api("/api/retention/run", { method: "POST" });
    showMessage("protection-message", t("js.deleted_and", "Deleted {audit_deleted} and {reviews_deleted}.", { audit_deleted: plural(data.run.audit_deleted, "audit record"), reviews_deleted: plural(data.run.reviews_deleted, "review case") }));
    renderRetention(data.retention);
  } catch (err) { showMessage("protection-message", err.message, true); }
}

// ---------- Feedback on an answer ----------
function resetFeedback(auditId) {
  const row = $("feedback-row");
  row.hidden = !auditId;
  row.dataset.auditId = auditId || "";
  $("feedback-form").hidden = true;
  $("feedback-message").hidden = true;
  $("feedback-reason").value = "";
  $("feedback-comment").value = "";
  setRated(null);
  if (!auditId) return;
  api(`/api/audit/${auditId}/feedback`).then((data) => {
    if (row.dataset.auditId === auditId && data.feedback) setRated(data.feedback.rating);
  }).catch(() => { /* feedback is optional */ });
}

function setRated(rating) {
  $("feedback-up").setAttribute("aria-pressed", String(rating === "up"));
  $("feedback-down").setAttribute("aria-pressed", String(rating === "down"));
}

async function sendFeedback(body) {
  const auditId = $("feedback-row").dataset.auditId;
  const sent = await api(`/api/audit/${auditId}/feedback`, { method: "POST", body });
  setRated(sent.feedback.rating);
  $("feedback-form").hidden = true;
  showMessage("feedback-message", sent.feedback.case_id
    ? t("js.thank_you_this_answer_has", "Thank you. This answer has been sent to a reviewer.")
    : t("js.thank_you_your_feedback_was", "Thank you. Your feedback was saved."));
  if (sent.feedback.case_id) refreshReviewCount();
}

function wireFeedback() {
  $("feedback-up").addEventListener("click", async () => {
    $("feedback-form").hidden = true;
    try { await sendFeedback({ rating: "up" }); }
    catch (err) { showMessage("feedback-message", err.message, true); }
  });
  $("feedback-down").addEventListener("click", () => {
    $("feedback-form").hidden = false;
    $("feedback-message").hidden = true;
    $("feedback-reason").focus();
  });
  $("feedback-cancel").addEventListener("click", () => { $("feedback-form").hidden = true; });
  $("feedback-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const form = e.currentTarget;
    formBusy(form, true);
    try {
      await sendFeedback({ rating: "down", reason: $("feedback-reason").value || null, comment: $("feedback-comment").value || null });
    } catch (err) { showMessage("feedback-message", err.message, true); }
    formBusy(form, false);
  });
}

function resetFlag(auditId) {
  resetFeedback(auditId);
  $("flag-row").hidden = !auditId;
  $("flag-row").dataset.auditId = auditId || "";
  $("flag-open").hidden = false;
  $("flag-form").hidden = true;
  $("flag-message").hidden = true;
  $("flag-note").value = "";
}

function wireReview() {
  const tabs = ["queue", "hazards", "report", "controls"];
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
  $("controls-status").addEventListener("change", () => renderControls(controlsState.data));

  $("case-assign").addEventListener("click", () => {
    const value = $("case-assignee").value;
    caseAction("assign", { user_id: value ? Number(value) : null }, null, value ? t("js.assigned_2", "Assigned.") : t("js.unassigned_2", "Unassigned."));
  });
  $("case-comment-form").addEventListener("submit", (e) => {
    e.preventDefault();
    caseAction("comment", { note: $("case-comment").value }, e.currentTarget, t("js.comment_added", "Comment added."));
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
    }, e.currentTarget, outcome === "add_test" ? t("js.resolved_the_question_is_now", "Resolved. The question is now a permanent test.") : t("js.resolved_3", "Resolved."));
  });
  $("case-reopen-form").addEventListener("submit", (e) => {
    e.preventDefault();
    caseAction("reopen", { note: $("case-reopen-note").value }, e.currentTarget, t("js.reopened", "Reopened."));
  });

  document.querySelectorAll("#hazard-form select[data-scale]").forEach((select) => {
    const labels = select.dataset.scale === "severity" ? SEVERITY() : LIKELIHOOD();
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
      showMessage("review-message", review.hazardId ? t("js.hazard_updated", "Hazard updated.") : t("js.hazard_added", "Hazard added."));
      loadHazards();
    } catch (err) { showMessage("hazard-message", err.message, true); }
    formBusy(form, false);
  });

  $("review-tests-run").addEventListener("click", async (e) => {
    const trigger = e.currentTarget;
    trigger.disabled = true;
    trigger.textContent = t("js.running", "Running…");
    try {
      const r = await api("/api/review-tests/run", { method: "POST" });
      $("review-tests-summary").textContent = t("js.of_behaved_as_intended_unsafe_2", "{passed} of {total} behaved as intended, {unsafe_answers} unsafe answers.", { passed: r.passed, total: r.total, unsafe_answers: r.unsafe_answers });
      $("review-tests-summary").classList.toggle("bad", r.passed < r.total);
      renderReviewTestCases(r.cases);
    } catch (err) { showMessage("review-message", err.message, true); }
    trigger.disabled = false;
    trigger.textContent = t("js.run_tests", "Run tests");
  });

  $("audit-verify").addEventListener("click", verifyAudit);
  $("retention-run").addEventListener("click", runRetention);

  wireFeedback();
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
      showMessage("flag-message", t("js.sent_for_review_a_reviewer", "Sent for review. A reviewer will check this answer."));
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
    empty.textContent = t("js.no_questions_in_this_period", "No questions in this period yet. Ask one and it appears here.");
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
  svg.setAttribute("aria-label", t("js.questions_per_day_in_days", "Questions per day: {total} in {daily} days", { total, daily: daily.length }));
  const el = (name, attrs) => { const e = document.createElementNS(NS, name); for (const k in attrs) e.setAttribute(k, attrs[k]); return e; };

  for (let i = 0; i <= 4; i++) {
    const v = (top / 4) * i;
    svg.appendChild(el("line", { class: "grid-line", x1: pad.left, x2: width - pad.right, y1: y(v), y2: y(v) }));
    const tt = el("text", { class: "axis-label", x: pad.left - 6, y: y(v) + 4, "text-anchor": "end" });
    tt.textContent = Number.isInteger(v) ? v : v.toFixed(1);
    svg.appendChild(tt);
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
      const tt = el("text", { class: "axis-label", x: cx, y: height - 6, "text-anchor": "middle" });
      tt.textContent = date.toLocaleDateString(uiLocale(), { day: "numeric", month: "short" });
      svg.appendChild(tt);
    }
    const hit = el("rect", { class: "bar-hit", x: pad.left + slot * i, y: pad.top, width: slot, height: innerH });
    hit.addEventListener("mouseenter", () => {
      g.classList.add("hover");
      tooltip.hidden = false;
      tooltip.innerHTML = "";
      const strong = document.createElement("strong");
      strong.textContent = date.toLocaleDateString(uiLocale(), { weekday: "short", day: "numeric", month: "short" });
      tooltip.append(strong, t("js.answered_refused", "{answered} answered, {refused} refused", { answered: d.answered, refused: d.refused }));
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
  const tt = data.totals;
  const stats = $("usage-stats");
  stats.textContent = "";
  const pct = tt.refusal_rate === null ? "–" : `${Math.round(tt.refusal_rate * 100)}%`;
  stats.append(
    statTile(t("js.questions", "Questions"), tt.questions.toLocaleString(), t("js.answered", "{answered} answered", { answered: tt.answered.toLocaleString() })),
    statTile(t("js.refused_4", "Refused"), pct, `${plural(tt.refused, "question")}`),
    statTile(t("js.median_response", "Median response"), formatMs(tt.median_ms), tt.p95_ms === null ? "" : t("js.95_within", "95% within {p95_ms}", { p95_ms: formatMs(tt.p95_ms) })),
    draftedTile(data.drafted_by, tt.questions),
    statTile(t("js.identifiers_removed", "Identifiers removed"), tt.deidentified.toLocaleString(), t("js.questions_with_names_dates_or", "questions with names, dates or numbers")),
    statTile(t("js.open_reviews", "Open reviews"), tt.open_reviews.toLocaleString(), tt.overdue_reviews ? t("js.overdue_5", "{overdue_reviews} overdue", { overdue_reviews: tt.overdue_reviews }) : t("js.none_overdue", "none overdue"), { attention: tt.overdue_reviews > 0 }),
  );
  usageState.daily = data.daily;
  renderDailyChart($("usage-chart"), data.daily);
  renderBars($("usage-reasons"), Object.entries(data.refusal_reasons).map(([k, v]) => ({ label: coded("refusal_category", k, capitalize(k)), value: v })), t("js.no_refusals_in_this_period", "No refusals in this period."));
  renderBars($("usage-sources"), data.top_sources.map((x) => ({ label: `${x.title} (${x.id})`, value: x.citations })), t("js.no_answers_cited_a_source", "No answers cited a source in this period."));
  $("usage-users-panel").hidden = !account.authRequired;
  renderBars($("usage-users"), data.by_user.map((x) => ({ label: x.name, value: x.questions })), t("js.no_signed_in_questions_in", "No signed-in questions in this period."));
  loadAdoption();
}

async function loadAdoption() {
  const days = $("usage-days").value;
  const panel = $("adoption-panel");
  $("pilot-report-download").href = `/api/governance/pilot-report?days=${days}`;
  let m;
  try { m = await api(`/api/metrics/adoption?days=${days}`); }
  catch (_) { panel.hidden = true; return; }  // admins only
  panel.hidden = false;
  $("adoption-message").hidden = true;
  const pct = (n) => n === null ? "–" : `${Math.round(n * 1000) / 10}%`;
  const c = m.clinicians, q = m.questions, f = m.feedback, r = m.reviews;
  const stats = $("adoption-stats");
  stats.textContent = "";
  stats.append(
    statTile(t("js.active_clinicians", "Active clinicians"), c.active_in_period.toLocaleString(), t("js.in_the_last_7_days", "{weekly_active} in the last 7 days", { weekly_active: c.weekly_active.toLocaleString() })),
    statTile(t("js.questions_per_clinician", "Questions per clinician"), c.questions_per_active_clinician === null ? "–" : String(c.questions_per_active_clinician), t("js.signed_in_questions_only", "signed-in questions only")),
    statTile(t("js.answer_rate", "Answer rate"), pct(q.answer_rate), t("js.of_answered", "{answered} of {total} answered", { answered: q.answered.toLocaleString(), total: q.total.toLocaleString() })),
    statTile(t("js.feedback", "Feedback"), f.total ? pct(f.positive_rate) : "–", f.total ? t("js.useful_not", "{up} useful, {down} not", { up: f.up, down: f.down }) : t("js.none_yet", "none yet")),
    statTile(t("js.time_to_close_a_review", "Time to close a review"), r.median_hours_to_close === null ? "–" : `${r.median_hours_to_close} h`, t("js.closed_open_now", "{resolved} closed, {open_now} open now", { resolved: r.resolved, open_now: r.open_now })),
    statTile(t("js.added_to_the_test_set", "Added to the test set"), String(m.tests_added), t("js.approved_2", "{documents_approved} approved", { documents_approved: plural(m.documents_approved, "document") })),
  );
  renderBars($("adoption-weekly"), c.weekly.map((w) => ({ label: t("js.week_of", "Week of {week_start}", { week_start: w.week_start }), value: w.active_clinicians })), t("js.no_activity_in_this_period", "No activity in this period."));
  renderBars($("adoption-feedback"), Object.entries(f.by_reason).map(([k, v]) => ({ label: coded("feedback_reason", k, capitalize(k.replace("_", " "))), value: v })), t("js.no_thumbs_down_in_this", "No thumbs down in this period."));
}
const usageState = { daily: null };

function draftedTile(drafted, total) {
  const models = drafted["Local model"] + drafted["Cloud model"];
  const note = total ? [
    drafted["Local model"] && t("js.local", "{drafted} local", { drafted: drafted["Local model"] }),
    drafted["Cloud model"] && t("js.cloud", "{drafted} cloud", { drafted: drafted["Cloud model"] }),
    t("js.extractive", "{Extractive} extractive", { Extractive: drafted.Extractive }),
  ].filter(Boolean).join(", ") : "";
  return statTile(t("js.drafted_by_a_model", "Drafted by a model"), total ? `${Math.round((models / total) * 100)}%` : "–", note);
}

// ---------- Embeddings ----------
async function loadEmbeddings() {
  let data;
  try { data = await api("/api/embeddings"); }
  catch (_) { return; }
  const stats = $("embed-stats");
  stats.textContent = "";
  const idx = data.index || {};
  const rebuilt = idx.finished_at ? t("js.rebuilt", "rebuilt {finished_at}", { finished_at: dateTime(idx.finished_at) }) : t("js.built_when_the_app_started", "built when the app started");
  stats.append(
    statTile(t("js.passages_indexed", "Passages indexed"), data.passages.toLocaleString(), t("js.from_your_documents", "{from_your_documents} from your documents", { from_your_documents: data.from_your_documents.toLocaleString() })),
    statTile(t("js.embedding_model", "Embedding model"), data.model.replace(/^sentence-transformers\//, ""), data.model.startsWith("sentence-transformers/") ? "sentence-transformers" : "", { small: true }),
    statTile(t("js.dimensions", "Dimensions"), data.dimensions.toLocaleString(), t("js.per_passage_vector", "per passage vector")),
    statTile(t("js.vector_store", "Vector store"), data.vector_store === "qdrant" ? "Qdrant" : t("js.local_numpy", "Local (NumPy)"), data.vector_store === "qdrant" ? t("js.server_or_embedded", "server or embedded") : t("js.exact_cosine_search", "exact cosine search"), { small: true }),
    statTile(t("js.keyword_vocabulary", "Keyword vocabulary"), data.vocabulary.toLocaleString(), t("js.terms_for_keyword_ranking", "terms for keyword ranking")),
    statTile(t("js.index", "Index"), idx.state === "running" ? t("js.rebuilding", "Rebuilding…") : idx.state === "failed" ? t("js.failed", "Failed") : t("js.ready", "Ready"), idx.state === "failed" ? idx.error : rebuilt, { small: true, attention: idx.state === "failed" }),
  );
  const rows = [
    [t("js.ranking", "Ranking"), data.hybrid.enabled ? t("js.hybrid_embedding_similarity_keyword_bm25", "Hybrid: {embedding}% embedding similarity, {keyword}% keyword (BM25)", { embedding: Math.round(data.hybrid.embedding_weight * 100), keyword: Math.round(data.hybrid.keyword_weight * 100) }) : t("js.embedding_similarity_only", "Embedding similarity only")],
    [t("js.similarity_2", "Similarity"), t("js.cosine_on_normalised_vectors", "Cosine, on normalised vectors")],
    [t("js.passages_retrieved", "Passages retrieved"), t("js.per_question", "{top_k} per question", { top_k: data.top_k })],
    [t("js.relevance_threshold", "Relevance threshold"), t("js.cosine_below_this_groundcheckhealth_refuses", "{min_score} cosine. Below this, GroundCheckHealth refuses before drafting anything.", { min_score: data.min_score })],
    [t("js.sources", "Sources"), data.include_demo_corpus ? t("js.your_approved_documents_and_the", "Your approved documents and the synthetic demo corpus ({from_demo_corpus} passages)", { from_demo_corpus: data.from_demo_corpus.toLocaleString() }) : t("js.your_approved_documents_only", "Your approved documents only")],
  ];
  const list = $("embed-settings");
  list.textContent = "";
  rows.forEach(([term, value], i) => {
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
  if (seconds < 3600) return t("js.duration_min_s", "{minutes} min {seconds} s", { minutes: Math.floor(seconds / 60), seconds: Math.round(seconds % 60) });
  return t("js.duration_h_min", "{hours} h {minutes} min", { hours: Math.floor(seconds / 3600), minutes: Math.round((seconds % 3600) / 60) });
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
    ? t("js.no_gpu_found_so_training", "No GPU found, so training will use the CPU. It works, but more slowly.")
    : gpus.length === 1
      ? t("js.training_will_use_you_can", "Training will use {name}. You can choose the CPU instead, which is slower.", { name: gpus[0].name })
      : t("js.this_machine_has_gpus_choose", "This machine has {gpus} GPUs. Choose one to train on.", { gpus: gpus.length });
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
      ? t("js.gpu_gb", "{vendor} GPU, {memory_gb} GB{shared_memory}", { vendor: d.vendor, memory_gb: d.memory_gb, shared_memory: d.shared_memory ? t("js.shared_memory", " shared memory") : "" })
      : t("js.cpu_cores_gb_memory", "CPU, {cores} cores, {memory_gb} GB memory", { cores: d.cores, memory_gb: d.memory_gb });
    body.append(name, meta);
    if (d.id === hw.recommended) {
      const badge = document.createElement("span");
      badge.className = "local-ai-badge recommended";
      badge.textContent = t("js.recommended", "Recommended");
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
  if (!select.options.length) sizes.forEach((n) => select.appendChild(new Option(t("js.px", "{n} × {n2} px", { n, n2: n }), n)));
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
  if (!path) { showMessage("training-message", t("js.choose_a_folder_of_images", "Choose a folder of images first."), true); return; }
  box.hidden = false;
  box.textContent = t("js.reading_the_folder", "Reading the folder…");
  box.classList.remove("invalid");
  try {
    const summary = await api(`/api/training/dataset?path=${encodeURIComponent(path)}`);
    training.dataset = summary;
    $("train-dataset").value = summary.path;
    renderDatasetSummary(summary);
    if (!$("train-name").value.trim()) $("train-name").value = t("js.classifier", "{name} classifier", { name: summary.name });
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
  const splits = d.layout === "split-folders" ? t("js.using_the_folder_s_own", "using the folder’s own train, validation and test split") : t("js.split_into_training_validation_and", "split into training, validation and test sets for you");
  head.innerHTML = "";
  const strong = document.createElement("strong");
  strong.textContent = t("js.images_in_classes", "{total} images in {classes} classes", { total: d.total.toLocaleString(), classes: d.classes.length });
  head.append(strong, t("js.training_validation_test", ", {splits}: {train} training, {val} validation, {test} test.", { splits, train: d.splits.train.toLocaleString(), val: d.splits.val.toLocaleString(), test: d.splits.test.toLocaleString() }));
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
      img.alt = t("js.example_image_of", "Example image of {name}", { name: c.name });
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
  $("browse-path").textContent = data.path || t("js.folders_you_can_train_from", "Folders you can train from");
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
  if (data.path) addRow(data.parent ? t("js.up_one_folder", "Up one folder") : t("js.all_folders", "All folders"), data.parent || "", null, "browse-up");
  data.folders.forEach((f) => addRow(f.name, f.path, f.looks_like_dataset ? t("js.looks_like_a_dataset", "Looks like a dataset") : null));
  if (!data.folders.length) {
    const li = document.createElement("li"); li.className = "auth-hint"; li.textContent = t("js.no_subfolders_here", "No subfolders here."); list.appendChild(li);
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
  button.textContent = t("js.starting", "Starting…");
  try {
    const data = await api("/api/training/runs", { method: "POST", body });
    $("training-message").hidden = true;
    followRun(data.run.id);
    loadRuns();
    $("run-live").scrollIntoView({ behavior: REDUCED_MOTION ? "auto" : "smooth", block: "start" });
  } catch (err) { showMessage("training-message", err.message, true); }
  button.disabled = false;
  button.textContent = t("js.start_training", "Start training");
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
  $("run-live-status").textContent = `${p.message || ""}${live && p.eta_seconds ? t("js.about_left", ", about {eta_seconds} left", { eta_seconds: formatDuration(p.eta_seconds) }) : ""}`;
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
    statTile(t("js.epoch", "Epoch"), p.epoch ? t("js.of_3", "{epoch} of {epochs}", { epoch: p.epoch, epochs: p.epochs }) : "–", p.best_epoch ? t("js.best_so_far", "best so far: {best_epoch}", { best_epoch: p.best_epoch }) : ""),
    statTile(t("js.validation_accuracy", "Validation accuracy"), last ? pct(last.val_accuracy, 1) : "–", bestVal !== null ? t("js.best", "best {bestVal}", { bestVal: pct(bestVal, 1) }) : ""),
    statTile(t("js.hardware", "Hardware"), run.device_name, t("js.px_2", "{architecture}, {image_size} px", { architecture: run.architecture === "resnet18" ? "ResNet-18" : "Small CNN", image_size: run.image_size }), { small: true }),
  );
  renderLineChart($("chart-loss"), history, [["train_loss", "line-train"], ["val_loss", "line-val"]], (v) => v.toFixed(2));
  renderLineChart($("chart-accuracy"), history, [["train_accuracy", "line-train"], ["val_accuracy", "line-val"]], (v) => `${Math.round(v * 100)}%`, [0, 1]);

  const result = $("run-result");
  result.hidden = p.state === "running" || p.state === "queued";
  result.className = `run-result ${p.state}`;
  result.textContent = "";
  if (p.state === "completed") {
    const sm = p.summary || {};
    result.append(t("js.saved_to_the_model_library", "Saved to the model library. On held-out test images: {test_accuracy} accurate; it answers {coverage} of images and is {answered_accuracy} correct when it does. ", { test_accuracy: pct(sm.test_accuracy, 1), coverage: pct(sm.coverage), answered_accuracy: pct(sm.answered_accuracy, 1) }));
    const link = document.createElement("a");
    link.href = "#/models";
    link.textContent = t("js.open_the_model_library", "Open the model library");
    link.addEventListener("click", () => { models.openAfterLoad = p.model_id; });
    result.appendChild(link);
  } else if (p.state === "failed" || p.state === "cancelled") {
    result.textContent = run.can_resume
      ? t("js.it_stopped_after_epoch_and", "{message}. It stopped after epoch {history} and can resume from there.", { message: p.message, history: (p.history || []).length })
      : p.message;
  }
}

// A line per series across epochs, with a dot on each value.
function renderLineChart(container, history, series, format, fixedRange) {
  container.textContent = "";
  if (history.length === 0) {
    const empty = document.createElement("div"); empty.className = "chart-empty"; empty.textContent = t("js.appears_after_the_first_epoch", "Appears after the first epoch");
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
  const svg = el("svg", { viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": t("js.by_epoch", "{series} by epoch", { series: series.map(([k]) => k).join(t("js.and", " and ")) }) });
  for (let i = 0; i <= 3; i++) {
    const v = lo + ((hi - lo) / 3) * i;
    svg.appendChild(el("line", { class: "grid-line", x1: pad.left, x2: width - pad.right, y1: y(v), y2: y(v) }));
    const tt = el("text", { class: "axis-label", x: pad.left - 6, y: y(v) + 4, "text-anchor": "end" });
    tt.textContent = format(v);
    svg.appendChild(tt);
  }
  const every = Math.ceil(history.length / 8);
  history.forEach((h, i) => {
    if (i % every === 0 || i === history.length - 1) {
      const tt = el("text", { class: "axis-label", x: x(i), y: height - 6, "text-anchor": "middle" });
      tt.textContent = h.epoch;
      svg.appendChild(tt);
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
  const labels = { queued: t("js.starting_2", "Starting"), running: t("js.training", "Training"), completed: t("js.saved_2", "Saved"), failed: t("js.failed", "Failed"), cancelled: t("js.cancelled", "Cancelled") };
  const pills = { queued: "status-pending", running: "status-pending", completed: "status-approved", failed: "status-rejected", cancelled: "status-retired" };
  data.runs.forEach((run) => {
    const p = run.progress;
    const tr = document.createElement("tr");
    const name = document.createElement("td");
    const n = document.createElement("div"); n.className = "user-name"; n.textContent = run.name;
    const dsn = document.createElement("div"); dsn.className = "user-email"; dsn.textContent = t("js.images", "{name}, {total} images", { name: run.dataset_summary.name, total: run.dataset_summary.total.toLocaleString() });
    name.append(n, dsn);
    const status = document.createElement("td");
    const pill = document.createElement("span"); pill.className = `status-pill ${pills[p.state] || ""}`; pill.textContent = labels[p.state] || p.state;
    status.appendChild(pill);
    if (p.state === "running") status.append(` ${Math.round(p.percent || 0)}%`);
    const hwCell = document.createElement("td"); hwCell.textContent = run.device_name;
    const started = document.createElement("td"); started.textContent = dateTime(run.created_at);
    const result = document.createElement("td");
    if (p.state === "completed" && p.summary) result.textContent = t("js.accurate", "{test_accuracy} accurate", { test_accuracy: pct(p.summary.test_accuracy, 1) });
    else if (p.state === "failed") result.textContent = p.message;
    const actions = document.createElement("div"); actions.className = "document-actions";
    if (p.state === "running" || p.state === "queued" || run.id !== training.runId) {
      actions.appendChild(button(t("js.view", "View"), "btn-ghost", false, () => { followRun(run.id); $("run-live").scrollIntoView({ block: "start" }); }));
    }
    if (run.can_resume) actions.appendChild(button(t("js.resume", "Resume"), "btn-primary", false, () => resumeRun(run.id)));
    result.appendChild(actions);
    labelCells([name, status, hwCell, started, result], ["", t("js.status", "Status"), t("js.hardware", "Hardware"), t("js.started", "Started"), t("js.result", "Result")]);
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
    if (!training.runId || !window.confirm(t("js.stop_training_nothing_is_saved", "Stop training? Nothing is saved to the library."))) return;
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
    meta.textContent = t("js.px_classes_from", "{architecture_label}, {image_size} px. {classes} classes from {dataset}.", { architecture_label: m.architecture_label, image_size: m.image_size, classes: m.classes.length, dataset: m.dataset });
    head.append(title, meta);
    const classes = document.createElement("p"); classes.className = "model-classes";
    classes.textContent = m.classes.slice(0, 8).join(", ") + (m.classes.length > 8 ? t("js.and_more", " and {classes} more", { classes: m.classes.length - 8 }) : "");
    const figures = document.createElement("dl"); figures.className = "model-figures";
    [[t("js.accuracy", "Accuracy"), pct(m.accuracy, 1)], [t("js.balanced_accuracy", "Balanced accuracy"), pct(m.balanced_accuracy, 1)], [t("js.answers", "Answers"), pct(m.coverage)], [t("js.correct_when_it_answers", "Correct when it answers"), pct(m.answered_accuracy, 1)]]
      .forEach(([k, v]) => { const d = document.createElement("div"); const dt = document.createElement("dt"); dt.textContent = k; const dd = document.createElement("dd"); dd.textContent = v; d.append(dt, dd); figures.appendChild(d); });
    const verdict = document.createElement("p");
    verdict.className = `model-verdict ${m.target_met === false ? "warn" : "ok"}`;
    verdict.textContent = m.target_met === false
      ? t("js.below_the_target_on_its", "Below the {target_accuracy} target on its {evaluated_on} images when it answers. Treat as experimental.", { target_accuracy: pct(m.target_accuracy), evaluated_on: m.evaluated_on })
      : t("js.met_the_target_on_its", "Met the {target_accuracy} target on its {evaluated_on} images when it answers.", { target_accuracy: pct(m.target_accuracy), evaluated_on: m.evaluated_on });
    const use = document.createElement("p"); use.className = "model-meta";
    use.textContent = m.imaging
      ? t("js.used_on_series", "Used on {modality} series{patch_mm}.", { modality: m.imaging.modality, patch_mm: m.imaging.patch_mm ? t("js.mm_regions", ", {value} mm regions", { value: Math.round(m.imaging.patch_mm) }) : "" })
      : t("js.not_set_up_for_ct", "Not set up for CT or MRI series.");
    const foot = document.createElement("p"); foot.className = "model-meta";
    foot.textContent = t("js.trained_on_in_mb", "Trained {created_at} on {device_name} in {training_seconds}{created_by}. {size_mb} MB.", { created_at: new Date(m.created_at).toLocaleDateString(uiLocale()), device_name: m.device_name, training_seconds: formatDuration(m.training_seconds), created_by: m.created_by ? t("js.by_3", " by {created_by}", { created_by: m.created_by }) : "", size_mb: m.size_mb });
    const actions = document.createElement("div"); actions.className = "model-actions";
    actions.appendChild(button(t("js.try_it", "Try it"), "btn-primary", false, () => openModel(m.id)));
    if (canSee("training")) {
      const dl = document.createElement("a");
      dl.className = "btn btn-sm btn-ghost"; dl.href = `/api/models/${encodeURIComponent(m.id)}/download`; dl.textContent = t("js.download_2", "Download");
      actions.appendChild(dl);
      actions.appendChild(button(t("js.ct_and_mri_use", "CT and MRI use"), "btn-ghost", false, () => openModelImaging(m)));
      actions.appendChild(button(t("js.delete", "Delete"), "btn-ghost", false, async () => {
        if (!window.confirm(t("js.delete_from_the_library_this", "Delete “{name}” from the library? This can’t be undone.", { name: m.name }))) return;
        try { await api(`/api/models/${encodeURIComponent(m.id)}`, { method: "DELETE" }); showMessage("models-message", t("js.deleted_2", "Deleted.")); }
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
    o.value = w; o.textContent = w === "auto" ? t("js.automatic_1st_to_99th_percentile", "Automatic (1st to 99th percentile)") : w[0].toUpperCase() + w.slice(1);
    select.appendChild(o);
  });
  if (chosen && IMAGING_WINDOWS[modality].includes(chosen)) select.value = chosen;
}

function openModelImaging(m) {
  models.imagingFor = m;
  const s = m.imaging || {};
  $("model-imaging-title").textContent = t("js.use_on_ct_and_mri", "Use {name} on CT and MRI", { name: m.name });
  $("mi-modality").value = s.modality || t("js.ct", "CT");
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
    showMessage("models-message", remove ? t("js.is_no_longer_used_on", "{name} is no longer used on series.", { name: m.name }) : t("js.saved_can_now_be_run", "Saved. {name} can now be run on {modality} series.", { name: m.name, modality: body.modality }));
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
  $("model-meta").textContent = t("js.classes", "{intended_use} {classes} classes: {classes2}.", { intended_use: m.intended_use, classes: m.classes.length, classes2: m.classes.join(", ") });
  $("try-result").textContent = "";
  $("try-preview").hidden = true;
  $("try-hint").hidden = false;
  $("try-file").value = "";
  const r = m.test || m.validation;
  const where = m.test ? "test" : "validation";
  const stats = $("model-stats");
  stats.textContent = "";
  stats.append(
    statTile(t("js.accuracy", "Accuracy"), pct(r.accuracy, 1), t("js.images_2", "{images} {where} images", { images: r.images.toLocaleString(), where })),
    statTile(t("js.balanced_accuracy", "Balanced accuracy"), pct(r.balanced_accuracy, 1), t("js.average_sensitivity_across_classes", "average sensitivity across classes")),
    statTile("AUC", r.auc === null ? "–" : r.auc.toFixed(3), m.classes.length > 2 ? t("js.average_one_class_against_the", "average, one class against the rest") : ""),
    statTile(t("js.confidence_threshold", "Confidence threshold"), pct(m.threshold.threshold), m.threshold.met ? t("js.chosen_for_accuracy_on_validation", "chosen for {target} accuracy on validation images", { target: pct(m.threshold.target) }) : t("js.no_share_of_images_met", "no share of images met the target: it abstains on all"), { attention: !m.threshold.met }),
    statTile(t("js.answers", "Answers"), pct(r.abstention.coverage), t("js.correct_when_it_does", "{answered_accuracy} correct when it does", { answered_accuracy: pct(r.abstention.answered_accuracy, 1) }), { attention: r.target_met === false }),
    statTile(t("js.calibration_error", "Calibration error"), r.calibration_error === null ? "–" : pct(r.calibration_error, 1), t("js.gap_between_confidence_and_accuracy", "gap between confidence and accuracy")),
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
    [t("js.dataset", "Dataset"), t("js.images_training_validation_test_fingerprint", "{name}: {total} images ({train} training, {val} validation, {test} test). Fingerprint {fingerprint}.", { name: m.dataset.name, total: m.dataset.total.toLocaleString(), train: m.dataset.splits.train.toLocaleString(), val: m.dataset.splits.val.toLocaleString(), test: m.dataset.splits.test.toLocaleString(), fingerprint: m.dataset.fingerprint })],
    ["Data checks", m.dataset.checks ? [
      m.dataset.checks.duplicates.leaked ? `${m.dataset.checks.duplicates.leaked.toLocaleString()} ${m.dataset.checks.duplicates.leaked === 1 ? t("js.copy_one", "copy") : t("js.copy_many", "copies")} ${t("js.of_training_images_left_out", "of training images left out of validation and test.")}` : t("js.no_training_images_copied", "No training images copied into validation or test."),
      m.dataset.checks.duplicates.conflicting_labels ? `${plural(m.dataset.checks.duplicates.conflicting_labels, "identical image")} filed under different classes.` : "",
      m.dataset.checks.identifiers.file_names || m.dataset.checks.identifiers.metadata_files ? "Possible patient details found in file names or metadata." : "No patient details found in file names or sampled metadata.",
    ].filter(Boolean).join(" ") : t("js.not_recorded_for_this_model", "Not recorded for this model.")],
    [t("js.architecture", "Architecture"), t("js.input_px", "{architecture}{pretrained}. Input {image_size} × {image_size2} px, {channels}.", { architecture: m.architecture === "resnet18" ? "ResNet-18" : "Small CNN", pretrained: m.pretrained ? t("js.from_imagenet_weights", ", from ImageNet weights") : t("js.trained_from_scratch", ", trained from scratch"), image_size: m.input.image_size, image_size2: m.input.image_size, channels: m.input.channels === 1 ? "grayscale" : "colour" })],
    [t("js.training", "Training"), t("js.of_epochs_best_batch_learning", "{history} of {epochs} epochs (best: {best_epoch}), batch {batch_size}, learning rate {learning_rate}, on {device_name} in {training_seconds}.", { history: m.history.length, epochs: tr.epochs, best_epoch: m.best_epoch, batch_size: tr.batch_size, learning_rate: tr.learning_rate, device_name: m.hardware.device_name, training_seconds: formatDuration(m.training_seconds) })],
    [t("js.unfamiliar_images", "Unfamiliar images"), m.novelty ? t("js.abstains_on_images_far_from", "Abstains on images far from anything it was trained on (1% of validation images would be flagged{test}).", { test: m.test && m.test.novelty_flagged !== undefined ? t("js.of_test_images_were", "; {novelty_flagged} of test images were", { novelty_flagged: pct(m.test.novelty_flagged, 1) }) : "" }) : t("js.no_check_recorded_for_this", "No check recorded for this model.")],
    [t("js.created", "Created"), `${dateTime(m.created_at)}${m.created_by ? t("js.by_3", " by {created_by}", { created_by: m.created_by }) : ""}${t("js.id_suffix", ". ID {id}.", { id: m.id })}`],
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
  classes.forEach((c) => { const th = document.createElement("th"); th.scope = "col"; th.textContent = c; th.title = t("js.predicted", "Predicted {c}", { c }); head.appendChild(th); });
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
      td.title = t("js.predicted_as_of", "{classes} predicted as {classes2}: {v} ({value}% of {classes3})", { classes: classes[i], classes2: classes[j], v, value: Math.round(share * 100), classes3: classes[i] });
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
  out.textContent = t("js.checking", "Checking…");
  const body = new FormData();
  body.append("file", file);
  try {
    const res = await fetch(`/api/models/${encodeURIComponent(models.current.id)}/predict`, { method: "POST", body });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || t("js.request_failed", "Request failed ({status}).", { status: res.status }));
    out.textContent = "";
    const verdict = document.createElement("div");
    verdict.className = `try-verdict ${data.abstained ? "abstained" : "answered"}`;
    const chip = document.createElement("span");
    chip.className = `status-chip ${data.abstained ? "refuse" : "answer"}`;
    chip.textContent = data.abstained ? (data.abstain_reason === "unfamiliar" ? t("js.unfamiliar_image", "UNFAMILIAR IMAGE") : t("js.not_confident", "NOT CONFIDENT")) : data.prediction.toUpperCase();
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
  ["dragenter", "dragover"].forEach((tt) => drop.addEventListener(tt, (e) => { e.preventDefault(); drop.classList.add("dragging"); }));
  ["dragleave", "drop"].forEach((tt) => drop.addEventListener(tt, (e) => { e.preventDefault(); drop.classList.remove("dragging"); }));
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
  if (!p) { status.textContent = t("js.no_patient_details", "No patient details"); status.classList.remove("on"); return; }
  const bits = [];
  if (p.age_years !== undefined) bits.push(`${p.age_years} y`);
  if (p.sex) bits.push(p.sex);
  if (p.weight_kg !== undefined) bits.push(`${p.weight_kg} kg`);
  if (p.egfr !== undefined) bits.push(t("js.egfr", "eGFR {egfr}", { egfr: p.egfr }));
  if (p.pregnant) bits.push(t("js.pregnant_lower", "pregnant"));
  if (p.allergies) bits.push(`${p.allergies.length} ${p.allergies.length === 1 ? t("js.allergy_one", "allergy") : t("js.allergy_many", "allergies")}`);
  if (p.medicines) bits.push(plural(p.medicines.length, "medicine"));
  status.textContent = bits.join(", ") || t("js.details_entered", "Details entered");
  status.classList.add("on");
}

const FINDING_LABEL = () => ({ block: t("js.unsafe_for_this_patient", "Unsafe for this patient"), warn: t("js.check", "Check"), info: t("js.note_2", "Note") });

function renderPatientFindings(findings, given) {
  const box = $("patient-findings");
  const list = $("patient-findings-list");
  list.textContent = "";
  box.hidden = !given;
  if (!given) return;
  if (!findings.length) {
    const li = document.createElement("li");
    li.className = "finding pass";
    li.textContent = t("js.no_problems_found_for_this", "No problems found for this patient with the medicines in this answer.");
    list.appendChild(li);
    return;
  }
  const order = { block: 0, warn: 1, info: 2 };
  [...findings].sort((a, b) => order[a.severity] - order[b.severity]).forEach((f) => {
    const li = document.createElement("li");
    li.className = `finding ${f.severity}`;
    const tag = document.createElement("span");
    tag.className = "finding-tag";
    tag.textContent = FINDING_LABEL()[f.severity];
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
  $("medicines-summary").textContent = t("js.medicines", "{total} medicines, {name} {version}", { total: data.total.toLocaleString(), name: data.name, version: data.version });
  const rows = $("medicine-rows");
  rows.textContent = "";
  $("medicines-empty").hidden = data.medicines.length > 0;
  data.medicines.forEach((m) => {
    const tr = document.createElement("tr");
    const name = document.createElement("td");
    const n = document.createElement("div"); n.className = "user-name"; n.textContent = m.name;
    if (m.high_alert) { const b = document.createElement("span"); b.className = "status-pill status-rejected"; b.textContent = t("js.high_alert", "High alert"); n.append(" ", b); }
    const c = document.createElement("div"); c.className = "user-email"; c.textContent = m.classes.join(", ");
    name.append(n, c);
    const dose = document.createElement("td");
    dose.textContent = m.adult_dose ? `${m.adult_dose.amount} ${m.adult_dose.unit} ${m.adult_dose.frequency}` : t("js.by_weight_or_not_stated", "By weight or not stated");
    const rules = document.createElement("td");
    const group = document.createElement("div"); group.className = "pill-group";
    const r = m.rules;
    [[r.allergies, t("js.allergy", "Allergy")], [r.interactions, t("js.interactions", "Interactions")], [r.kidney, t("js.kidney", "Kidney")], [r.liver, t("js.liver", "Liver")], [r.children, t("js.children", "Children")],
     [r.weight, t("js.weight", "Weight")], [r.labs, t("js.labs", "Labs")], [r.pregnancy !== "no_data", t("js.pregnancy", "Pregnancy: {pregnancy}", { pregnancy: r.pregnancy })]]
      .filter(([on]) => on).forEach(([, label]) => { const s = document.createElement("span"); s.className = "status-pill"; s.textContent = label; group.appendChild(s); });
    rules.appendChild(group);
    const act = document.createElement("td");
    const actions = document.createElement("div"); actions.className = "document-actions";
    actions.appendChild(button(t("js.view_rules", "View rules"), "btn-ghost", false, () => openMedicine(m.name)));
    act.appendChild(actions);
    labelCells([name, dose, rules, act], ["", t("js.adult_dose", "Adult dose"), t("js.rules", "Rules"), ""]);
    tr.append(name, dose, rules, act);
    rows.appendChild(tr);
  });
}

async function openMedicine(name) {
  let data;
  try { data = await api(`/api/formulary/${encodeURIComponent(name)}`); } catch (_) { return; }
  const m = data.medicine;
  $("medicine-dialog-title").textContent = m.name;
  const doseText = (d) => d ? `${d.amount} ${d.unit} ${d.frequency}${d.max_daily ? t("js.at_most_a_day", ", at most {max_daily} {unit} a day", { max_daily: d.max_daily, unit: d.unit }) : ""}` : "";
  const weightText = (d) => d ? `${d.per_kg} ${d.unit}/kg ${d.frequency}${d.max_single ? t("js.at_most_a_dose", ", at most {max_single} {unit} a dose", { max_single: d.max_single, unit: d.unit }) : ""}` : "";
  const rows = [
    [t("js.also_called", "Also called"), m.aliases.join(", ")],
    [t("js.classes_2", "Classes"), m.classes.join(", ")],
    [t("js.high_alert", "High alert"), m.high_alert ? t("js.yes_use_an_independent_double", "Yes: use an independent double check") : ""],
    [t("js.adult_dose", "Adult dose"), doseText(m.adult_dose) + (m.adult_dose ? t("js.from_age", " (from age {adult_min_age})", { adult_min_age: m.adult_min_age }) : "")],
    [t("js.by_weight", "By weight"), weightText(m.weight_dose)],
    [t("js.children", "Children"), m.paediatric_dose ? `${weightText(m.paediatric_dose)}${m.paediatric_min_age != null ? t("js.from_age_2", ", from age {paediatric_min_age}", { paediatric_min_age: m.paediatric_min_age }) : ""}` : t("js.no_paediatric_dosing_questions_about", "No paediatric dosing: questions about children are refused")],
    [t("js.allergies", "Allergies"), m.allergy_groups.join(", ")],
    [t("js.conditions", "Conditions"), m.contraindicated_conditions.join(", ")],
    [t("js.interactions", "Interactions"), m.interactions.map((i) => `${i.with_medicine || i.with_class} (${i.severity})${i.note ? `: ${i.note}` : ""}`).join(" · ")],
    [t("js.kidney", "Kidney"), m.renal.map((r) => t("js.egfr_below", "eGFR below {egfr_below}: {action}{dose}", { egfr_below: r.egfr_below, action: r.action, dose: r.dose ? t("js.to", " to {dose}", { dose: doseText(r.dose) }) : "" })).join(" · ")],
    [t("js.liver", "Liver"), m.hepatic.map((r) => t("js.child_pugh", "Child-Pugh {child_pugh}: {action}", { child_pugh: r.child_pugh, action: r.action })).join(" · ")],
    [t("js.pregnancy_2", "Pregnancy"), { avoid: t("js.avoid", "Avoid"), no_data: t("js.no_safety_information_refused", "No safety information: refused"), compatible: t("js.compatible", "Compatible") }[m.pregnancy]],
    [t("js.breastfeeding", "Breastfeeding"), { avoid: t("js.avoid", "Avoid"), no_data: t("js.no_safety_information_warning", "No safety information: warning"), compatible: t("js.compatible", "Compatible") }[m.breastfeeding]],
    [t("js.lab_results", "Lab results"), m.labs.map((l) => `${l.lab} ${l.above != null ? t("js.above", "above {above}", { above: l.above }) : t("js.below", "below {below}", { below: l.below })}: ${l.action}`).join(" · ")],
    [t("js.needs_before_dosing", "Needs before dosing"), m.requires.join(", ")],
    [t("js.sources", "Sources"), m.source_ids.join(", ")],
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
  const title = document.createElement("span"); title.className = "eyebrow"; title.textContent = t("js.patient_scenarios", "Patient scenarios");
  const score = document.createElement("span"); score.className = "adv-score";
  score.textContent = t("js.of_correct", "{passed} of {total} correct, {unsafe_answers}", { passed: p.passed.toLocaleString(), total: p.total.toLocaleString(), unsafe_answers: plural(p.unsafe_answers, "unsafe answer") });
  head.append(title, score);
  const note = document.createElement("p"); note.className = "adv-note";
  note.textContent = t("js.the_same_questions_asked_for", "The same questions asked for different patients: allergies, interactions, children, pregnancy, kidney and liver function, and missing details. Any unsafe answer fails the build.");
  wrap.append(head, note);
  if (p.failures && p.failures.length) {
    const list = document.createElement("ul"); list.className = "eval-cases";
    p.failures.forEach((f) => { const li = document.createElement("li"); li.className = "eval-case"; li.textContent = t("js.expected_got", "{query}: expected {expect}, got {got}", { query: f.query, expect: f.expect, got: f.got }); list.appendChild(li); });
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
  $("ehr-source").textContent = t("js.from_3", "From {system}{patient}", { system: source.system === "SMART on FHIR" ? t("js.the_ehr", "the EHR") : host, patient: source.patient ? `, ${source.patient}` : "" });
  const time = (iso) => new Date(iso).toLocaleTimeString(uiLocale(), { hour: "2-digit", minute: "2-digit" });
  $("ehr-meta").textContent = t("js.loaded_kept_until_check_the", " Loaded {loaded_at}, kept until {expires_at}. Check the details before asking.", { loaded_at: time(context.loaded_at), expires_at: time(context.expires_at) });
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
    ? t("js.set_up_register_these_addresses", "Set up. Register these addresses with your EHR.")
    : t("js.not_set_up_yet_whoever", "Not set up yet. Whoever runs this server has to add the EHR's client ID and the EHRs it may be launched ")
      + t("js.from_smart_client_id_and", "from (SMART_CLIENT_ID and SMART_ALLOWED_ISSUERS), and then these addresses get registered with the EHR.");
  list("smart-settings", [
    [t("js.launch_url", "Launch URL"), c.smart_launch_url], [t("js.redirect_url", "Redirect URL"), c.smart_redirect_url],
    [t("js.client_id", "Client ID"), c.smart_client_id || t("js.not_set", "Not set")], [t("js.allowed_ehrs", "Allowed EHRs"), c.smart_allowed_issuers.join(", ") || t("js.none_2", "None")],
    [t("js.scopes", "Scopes"), c.smart_scopes], [t("js.open_fhir_servers", "Open FHIR servers"), c.open_servers.join(", ") || t("js.none_2", "None")],
    [t("js.patient_kept_for", "Patient kept for"), t("js.minutes", "{context_minutes} minutes", { context_minutes: c.context_minutes })],
  ]);
  $("cds-status").textContent = c.cds_trusted.length
    ? t("js.accepting_signed_calls_from_the", "Accepting signed calls from the EHRs below.")
    : c.cds_unsigned ? t("js.accepting_unsigned_calls_for_testing", "Accepting unsigned calls: for testing only.")
      : t("js.no_ehr_is_trusted_yet", "No EHR is trusted yet. Whoever runs this server has to list each EHR that may call in, with the address its ")
        + t("js.signing_keys_are_published_at", "signing keys are published at (CDS_HOOKS_TRUSTED).");
  list("cds-settings", [
    [t("js.discovery_url", "Discovery URL"), c.cds_discovery_url], [t("js.hooks", "Hooks"), "order-select, order-sign"],
    [t("js.trusted_ehrs", "Trusted EHRs"), c.cds_trusted.join(", ") || t("js.none_2", "None")], [t("js.unsigned_calls", "Unsigned calls"), c.cds_unsigned ? t("js.allowed_testing", "Allowed (testing)") : t("js.refused_4", "Refused")],
  ]);
  if (!$("cds-request").value) $("cds-request").value = JSON.stringify(CDS_EXAMPLE, null, 2);
}

function renderCdsCards(cards) {
  const wrap = $("cds-cards");
  wrap.textContent = "";
  if (!cards.length) {
    const p = document.createElement("p"); p.className = "auth-hint"; p.textContent = t("js.no_cards_nothing_to_flag", "No cards: nothing to flag for this order.");
    wrap.appendChild(p);
    return;
  }
  cards.forEach((card) => {
    const el = document.createElement("article");
    el.className = `cds-card ${card.indicator}`;
    const head = document.createElement("div"); head.className = "cds-card-head";
    const tag = document.createElement("span"); tag.className = "finding-tag"; tag.textContent = { critical: t("js.critical", "Critical"), warning: t("js.warning", "Warning"), info: t("js.info", "Info") }[card.indicator];
    const summary = document.createElement("strong"); summary.textContent = card.summary;
    head.append(tag, summary);
    // The detail repeats the summary after naming the order, for EHRs that
    // show only one of them. Here both are shown, so name the order once.
    const plain = card.detail.replace(/\*\*/g, "");
    const detail = document.createElement("p");
    detail.textContent = plain.endsWith(card.summary) ? t("js.order", "Order: {plain}", { plain: plain.slice(0, -card.summary.length).replace(/[:\s]+$/, "") }) : plain;
    const source = document.createElement("p"); source.className = "finding-meta mono"; source.textContent = card.source.label;
    el.append(head, detail, source);
    if (card.overrideReasons && card.overrideReasons.length) {
      const reasons = document.createElement("p"); reasons.className = "auth-hint";
      reasons.textContent = t("js.override_reasons_offered", "Override reasons offered: {overrideReasons}", { overrideReasons: card.overrideReasons.map((r) => r.display).join("; ") });
      el.appendChild(reasons);
    }
    wrap.appendChild(el);
  });
}

function wireEhr() {
  $("ehr-load-button").addEventListener("click", async (e) => {
    const trigger = e.currentTarget;
    const patientId = $("ehr-patient-id").value.trim();
    if (!patientId) { $("pt-error").textContent = t("js.enter_the_patient_s_fhir", "Enter the patient's FHIR ID."); $("ehr-patient-id").focus(); return; }
    trigger.disabled = true; trigger.textContent = t("js.loading", "Loading…");
    try {
      const data = await api("/api/ehr/load", { method: "POST", body: { server: $("ehr-server").value, patient_id: patientId } });
      $("pt-error").textContent = "";
      applyEhrContext(data.context);
    } catch (err) { $("pt-error").textContent = err.message; }
    trigger.disabled = false; trigger.textContent = t("js.load_from_ehr", "Load from EHR");
  });
  $("ehr-save-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const form = e.currentTarget;
    if (!window.confirm(t("js.save_this_answer_and_the", "Save this answer and the patient checks to the patient's record as a preliminary note?"))) return;
    formBusy(form, true);
    try {
      const data = await api("/api/ehr/notes", { method: "POST", body: { audit_id: $("ehr-save").dataset.auditId, comment: $("ehr-save-comment").value } });
      form.hidden = true;
      showMessage("ehr-save-message", t("js.saved_to_the_record_as", "Saved to the record as a preliminary note{reference}.", { reference: data.reference ? ` (${data.reference})` : "" }));
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
    catch (_) { showMessage("cds-message", t("js.the_request_isn_t_valid", "The request isn't valid JSON."), true); return; }
    const res = await fetch("/cds-services/groundcheck-medication-safety-order-sign", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    });
    const data = await res.json().catch(() => ({}));
    if (res.status === 401) {
      showMessage("cds-message", t("js.this_server_only_accepts_calls", "This server only accepts calls signed by a trusted EHR. To try requests here, run a test server with CDS_HOOKS_ALLOW_UNSIGNED=true."), true);
      $("cds-cards").textContent = "";
      return;
    }
    if (!res.ok) { showMessage("cds-message", data.detail || t("js.the_service_returned", "The service returned {status}.", { status: res.status }), true); return; }
    renderCdsCards(data.cards);
  });
}

// ---------- Releases ----------
const RELEASE_STATUS = () => ({ live: [t("js.live", "Live"), "status-live"], ready: [t("js.ready_to_go_live", "Ready to go live"), "status-ready"], checking: [t("js.checking_2", "Checking"), "status-checking"],
  failed: [t("js.blocked", "Blocked"), "status-failed"], retired: [t("js.earlier", "Earlier"), "status-earlier"], rolled_back: [t("js.rolled_back", "Rolled back"), "status-earlier"] });
let releasePoll = null;

function canAdmin() {
  return !account.authRequired || account.user?.role === "admin";
}

function checkText(check) {
  if (!check) return [t("js.not_checked_yet", "Not checked yet."), "skip"];
  if (check.skipped) return [t("js.not_checked_release_checks_are", "Not checked: release checks are turned off, or this was the first release."), "skip"];
  if (check.error) return [check.error, "bad"];
  const g = check.groups || {};
  const parts = [];
  if (g.golden) parts.push(t("js.golden_set_questions", "{total} golden-set questions", { total: g.golden.total.toLocaleString() }));
  if (g.patient) parts.push(t("js.patient_scenarios_2", "{total} patient scenarios", { total: g.patient.total.toLocaleString() }));
  if (g.review_tests) parts.push(t("js.review_test", "{total} review test{s}", { total: g.review_tests.total.toLocaleString(), s: g.review_tests.total === 1 ? "" : "s" }));
  const what = parts.length ? parts.join(", ") : t("js.no_questions", "no questions");
  if (!check.passed) return [t("js.blocked_answered_question_that_must", "Blocked: answered {unsafe} question{s} that must be refused, of {what}.", { unsafe: check.unsafe, s: check.unsafe === 1 ? "" : "s", what }), "bad"];
  return [t("js.passed_nothing_that_must_be", "Passed: nothing that must be refused was answered, across {what}, in {seconds} s.", { what, seconds: check.seconds }), ""];
}

function changeText(r) {
  if (!r.changes) return "";
  const parts = [];
  if (r.changes.added.length) parts.push(t("js.adds", "adds {added}", { added: r.changes.added.join(", ") }));
  if (r.changes.updated.length) parts.push(t("js.updates", "updates {updated}", { updated: r.changes.updated.join(", ") }));
  if (r.changes.removed.length) parts.push(t("js.removes", "removes {removed}", { removed: r.changes.removed.join(", ") }));
  return parts.length ? t("js.compared_with_live", "Compared with live: {parts}.", { parts: parts.join("; ") }) : t("js.same_documents_as_live", "Same documents as live.");
}

async function loadReleases() {
  clearTimeout(releasePoll);
  let data;
  try { data = await api("/api/releases"); }
  catch (err) { showMessage("releases-message", err.message, true); return; }
  const admin = canAdmin();
  const live = data.releases.find((r) => r.status === "live");
  const earlier = live && data.releases.find((r) => r.number < live.number && r.live_at && ["retired", "rolled_back"].includes(r.status) && r.snapshot);
  $("release-rebuild").hidden = !admin;
  $("release-rollback").hidden = !admin || !earlier;
  if (earlier) $("release-rollback").textContent = t("js.roll_back_to_r", "Roll back to R{number}", { number: earlier.number });
  const running = data.index.state === "running";
  $("release-rebuild").disabled = running;

  const banner = $("live-release");
  banner.textContent = "";
  banner.className = `live-release${running ? " checking" : ""}`;
  if (live) {
    const body = document.createElement("div");
    body.append(el("h2", "", running ? t("js.live_r_a_new_release", "Live: R{number}. A new release is being built and checked.", { number: live.number }) : t("js.live_r", "Live: R{number}", { number: live.number })),
      el("p", "", t("js.live_since", "{reason}. Live since {live_at}{live_by}.", { reason: live.reason, live_at: dateTime(live.live_at), live_by: live.live_by ? t("js.by_4", ", by {live_by}", { live_by: live.live_by }) : "" })));
    const figures = el("div", "figures");
    figures.append(el("div", "", t("js.passages", "{passages} passages", { passages: live.passages.toLocaleString() })),
      el("div", "", t("js.of_your_document", "{documents} of your document{s}", { documents: live.documents.length, s: live.documents.length === 1 ? "" : "s" })),
      el("div", "", live.demo_passages ? t("js.from_the_demo_corpus", "{demo_passages} from the demo corpus", { demo_passages: live.demo_passages.toLocaleString() }) : t("js.no_demo_corpus", "No demo corpus")));
    banner.append(el("span", "release-badge", `R${live.number}`), body, figures);
  }

  const list = $("release-list");
  list.textContent = "";
  data.releases.forEach((r) => {
    const li = el("li", "release-item");
    const body = el("div", "release-body");
    const [label, cls] = RELEASE_STATUS()[r.status] || [r.status, ""];
    const h = el("h3");
    h.append(r.reason || t("js.index_rebuilt", "Index rebuilt"), el("span", `status-pill ${cls}`, label));
    body.append(h, el("p", "release-meta", t("js.built_by_passages_document", "Built {created_at} by {created_by}. {passages} passages, {documents} document{s}.{snapshot}", { created_at: dateTime(r.created_at), created_by: r.created_by || "GroundCheckHealth", passages: r.passages.toLocaleString(), documents: r.documents.length, s: r.documents.length === 1 ? "" : "s", snapshot: r.snapshot ? "" : t("js.snapshot_removed", " Snapshot removed.") })));
    const changes = changeText(r);
    if (changes) body.appendChild(el("p", "", changes));
    const [text, state] = checkText(r.check);
    body.appendChild(el("p", `release-check ${state}`, text));
    const examples = [...(r.check?.unsafe_examples || []).map((e) => t("js.answered_must_refuse", "Answered, must refuse: {query}", { query: e.query })),
      ...(r.check?.over_refused_examples || []).map((q) => t("js.now_refused", "Now refused: {q}", { q }))];
    if (examples.length) {
      const details = el("details", "release-details");
      details.appendChild(el("summary", "", t("js.questions", "Questions")));
      const ul = document.createElement("ul");
      examples.forEach((tt) => ul.appendChild(el("li", "", tt)));
      details.appendChild(ul);
      body.appendChild(details);
    }
    const actions = el("div", "release-actions");
    if (admin && r.snapshot && ["ready", "retired", "rolled_back"].includes(r.status)) {
      const verb = r.status === "ready" ? t("js.put_live", "Put live") : t("js.make_live_again", "Make live again");
      actions.appendChild(button(verb, r.status === "ready" ? "btn-primary" : "btn-ghost", running, async (e) => {
        if (!window.confirm(t("js.make_r_live_answers_will", "Make R{number} live? Answers will come from its documents straight away.", { number: r.number }))) return;
        e.target.disabled = true;
        try {
          const { release } = await api(`/api/releases/${r.id}/promote`, { method: "POST" });
          showMessage("releases-message", t("js.r_is_live", "R{number} is live.", { number: release.number }));
        } catch (err) { showMessage("releases-message", err.message, true); }
        loadReleases();
      }));
    }
    li.append(el("span", "release-name", `R${r.number}`), body, actions);
    list.appendChild(li);
  });
  $("release-policy").textContent = t("js.the_last_snapshots_are_kept", "{checks} {auto_promote} The last {keep} snapshots are kept.", { checks: data.checks ? t("js.checks_are_on", "Checks are on.") : t("js.checks_are_off_release_checks", "Checks are off (RELEASE_CHECKS=false)."), auto_promote: data.auto_promote ? t("js.a_release_that_passes_goes", "A release that passes goes live straight away.") : t("js.a_release_that_passes_waits", "A release that passes waits for an admin to put it live."), keep: data.keep });
  if (running && currentView() === "releases") releasePoll = setTimeout(loadReleases, 2000);
}

function wireReleases() {
  $("release-rebuild").addEventListener("click", async (e) => {
    e.target.disabled = true;
    try { await api("/api/index/rebuild", { method: "POST" }); showMessage("releases-message", t("js.rebuilding_and_checking_this_takes", "Rebuilding and checking. This takes about a minute.")); }
    catch (err) { showMessage("releases-message", err.message, true); }
    loadReleases();
  });
  $("release-rollback").addEventListener("click", async (e) => {
    if (!window.confirm(t("js.go_back_to_the_previous", "Go back to the previous release? Answers will come from its documents straight away."))) return;
    e.target.disabled = true;
    try {
      const { release } = await api("/api/releases/rollback", { method: "POST" });
      showMessage("releases-message", t("js.rolled_back_r_is_live", "Rolled back. R{number} is live.", { number: release.number }));
    } catch (err) { showMessage("releases-message", err.message, true); }
    e.target.disabled = false;
    loadReleases();
  });
}

// ---------- Incidents ----------
const incidentState = { labels: null, people: [], canManage: false, current: null, links: {} };
const INCIDENT_CATEGORIES = () => ({ answer: t("js.an_answer_or_refusal", "An answer or refusal"), patient_check: t("js.a_patient_safety_check", "A patient safety check"), imaging: t("js.imaging", "Imaging"), ehr: t("js.ehr_integration", "EHR integration"),
  data_protection: t("js.data_protection_or_security", "Data protection or security"), availability: t("js.outage_or_slowness", "Outage or slowness"), other: t("js.something_else", "Something else") });
const INCIDENT_HARM = () => ({ none: t("js.no_harm_or_near_miss", "No harm or near miss"), low: t("js.low_harm", "Low harm"), moderate: t("js.moderate_harm", "Moderate harm"), severe: t("js.severe_harm", "Severe harm"), death: t("js.death", "Death") });
const INCIDENT_STATUS = () => ({ open: t("js.open_2", "Open"), investigating: t("js.investigating", "Investigating"), closed: t("js.closed", "Closed") });

function fillSelect(select, options, chosen) {
  const keep = chosen ?? select.value;
  select.textContent = "";
  Object.entries(options).forEach(([value, label]) => { const o = el("option", "", label); o.value = value; select.appendChild(o); });
  if (keep !== undefined && keep !== null && keep in options) select.value = keep;
}

function harmPill(harm) {
  return el("span", `harm harm-${harm}`, INCIDENT_HARM()[harm]);
}

function statusPill(status) {
  return el("span", `status-pill ${status === "closed" ? "status-retired" : status === "investigating" ? "status-investigating" : "status-pending"}`, INCIDENT_STATUS()[status]);
}

async function loadIncidents() {
  const params = new URLSearchParams({ status: $("incident-status-filter").value, category: $("incident-category-filter").value });
  let data;
  try { data = await api(`/api/incidents?${params}`); }
  catch (err) { showMessage("incidents-message", err.message, true); return; }
  incidentState.canManage = data.can_manage;
  incidentState.people = data.people || [];
  $("incidents-export").hidden = !data.can_manage;
  $("incidents-sub").textContent = data.can_manage
    ? t("js.report_anything_that_went_wrong", "Report anything that went wrong, or nearly did. Investigate, record the root cause and the actions taken, and close it.")
    : t("js.report_anything_that_went_wrong_2", "Report anything that went wrong, or nearly did. You can follow the incidents you reported here.");
  const category = $("incident-category-filter");
  if (category.options.length === 1) Object.entries(INCIDENT_CATEGORIES()).forEach(([v, l]) => { const o = el("option", "", l); o.value = v; category.appendChild(o); });

  const counts = $("incident-counts");
  counts.textContent = "";
  if (data.can_manage) {
    [[t("js.open_2", "Open"), data.counts.open], [t("js.investigating", "Investigating"), data.counts.investigating], [t("js.closed", "Closed"), data.counts.closed], [t("js.moderate_harm_or_worse_not", "Moderate harm or worse, not closed"), data.serious_active]]
      .forEach(([label, n], i) => {
        const c = el("span", `review-count${i === 3 && n ? " bad" : ""}`);
        c.append(el("strong", "", String(n)), ` ${label}`);
        counts.appendChild(c);
      });
  }
  const rows = $("incident-rows");
  rows.textContent = "";
  $("incident-empty").hidden = data.incidents.length > 0;
  data.incidents.forEach((i) => {
    const tr = document.createElement("tr");
    if (["moderate", "severe", "death"].includes(i.harm) && i.status !== "closed") tr.className = "serious";
    const name = document.createElement("td");
    const open = el("button", "incident-open", i.title);
    open.type = "button";
    open.addEventListener("click", () => openIncident(i.id));
    name.append(el("span", "incident-ref", `${i.reference}, ${(INCIDENT_CATEGORIES()[i.category] || i.category_label).toLowerCase()}`), open);
    const harm = document.createElement("td"); harm.dataset.label = t("js.harm", "Harm"); harm.appendChild(harmPill(i.harm));
    const status = document.createElement("td"); status.dataset.label = t("js.status", "Status"); status.appendChild(statusPill(i.status));
    const reported = el("td", "", `${dateTime(i.reported_at)}, ${i.reported_by}`); reported.dataset.label = t("js.reported", "Reported");
    const owner = el("td", i.owner ? "" : "muted", i.owner || t("js.nobody_yet", "Nobody yet")); owner.dataset.label = t("js.owner_2", "Owner");
    tr.append(name, harm, status, reported, owner);
    rows.appendChild(tr);
  });
}

function openIncidentReport(links = {}) {
  incidentState.links = links;
  const form = $("incident-report-form");
  form.reset();
  fillSelect($("ir-category"), INCIDENT_CATEGORIES(), links.category || "answer");
  fillSelect($("ir-harm"), INCIDENT_HARM(), "none");
  $("ir-audit").value = links.audit_id || "";
  const described = [links.alert_id && t("js.alert", "alert: {alert_title}", { alert_title: links.alert_title }), links.imaging_series_id && t("js.imaging_series", "imaging series {imaging_series_id}", { imaging_series_id: links.imaging_series_id }),
    links.review_case_id && t("js.review_case", "review case {review_case_id}", { review_case_id: links.review_case_id })].filter(Boolean);
  $("ir-links").hidden = described.length === 0;
  $("ir-links").textContent = described.length ? t("js.linked_to_the", "Linked to the {described}.", { described: described.join(t("js.and_the", " and the ")) }) : "";
  if (links.title) $("ir-title").value = links.title;
  $("ir-message").hidden = true;
  $("incident-report-dialog").showModal();
  $("ir-title").focus();
}

async function submitIncidentReport(event) {
  event.preventDefault();
  const title = $("ir-title").value.trim();
  const description = $("ir-description").value.trim();
  if (!title) { showMessage("ir-message", t("js.add_a_short_title", "Add a short title."), true); $("ir-title").focus(); return; }
  if (!description) { showMessage("ir-message", t("js.describe_what_happened", "Describe what happened."), true); $("ir-description").focus(); return; }
  const occurred = $("ir-occurred").value;
  const body = {
    title, description, category: $("ir-category").value, harm: $("ir-harm").value,
    occurred_at: occurred ? new Date(occurred).toISOString() : null,
    audit_id: $("ir-audit").value.trim() || null,
    alert_id: incidentState.links.alert_id || null, imaging_series_id: incidentState.links.imaging_series_id || null,
    review_case_id: incidentState.links.review_case_id || null,
  };
  const form = $("incident-report-form");
  formBusy(form, true);
  try {
    const { incident } = await api("/api/incidents", { method: "POST", body });
    $("incident-report-dialog").close();
    if (currentView() === "incidents") loadIncidents();
    const message = t("js.reported_as", "Reported as {reference}.", { reference: incident.reference });
    if (currentView() === "incidents") showMessage("incidents-message", message);
    else if (currentView() === "monitoring") showMessage("monitoring-message", message);
    else showMessage("flag-message", message);
  } catch (err) {
    showMessage("ir-message", err.message, true);
  } finally {
    formBusy(form, false);
  }
}

async function openIncident(id) {
  let data;
  try { data = await api(`/api/incidents/${id}`); }
  catch (err) { showMessage("incidents-message", err.message, true); return; }
  renderIncident(data.incident, data.can_manage);
  if (!$("incident-dialog").open) $("incident-dialog").showModal();
}

function renderIncident(i, canManage) {
  incidentState.current = i;
  $("incident-dialog-title").textContent = `${i.reference}: ${i.title}`;
  $("incident-meta").textContent = t("js.reported_by_on", "{category_label}. {harm_label}. {status}. Reported by {reported_by} on {reported_at}{occurred_at}.", { category_label: INCIDENT_CATEGORIES()[i.category] || i.category_label, harm_label: INCIDENT_HARM()[i.harm] || i.harm_label, status: INCIDENT_STATUS()[i.status], reported_by: i.reported_by, reported_at: dateTime(i.reported_at), occurred_at: i.occurred_at ? t("js.happened", "; happened {occurred_at}", { occurred_at: dateTime(i.occurred_at) }) : "" });
  const deadlines = $("incident-deadlines");
  deadlines.textContent = "";
  i.deadlines.forEach((d) => {
    const p = el("p", `deadline${d.done ? " done" : ""}`);
    p.append(el("strong", "", d.done ? t("js.recorded", "Recorded") : d.due_at ? t("js.due_by", "Due by {due_at}", { due_at: dateTime(d.due_at) }) : t("js.decision_needed", "Decision needed")), d.text);
    deadlines.appendChild(p);
  });
  $("incident-description").textContent = i.description;
  const links = [i.links.audit_id && t("js.audit_record_2", "audit record {audit_id}", { audit_id: i.links.audit_id }), i.links.review_case_id && t("js.review_case", "review case {review_case_id}", { review_case_id: i.links.review_case_id }),
    i.links.alert_id && t("js.alert_2", "alert {alert_id}", { alert_id: i.links.alert_id }), i.links.imaging_series_id && t("js.imaging_series", "imaging series {imaging_series_id}", { imaging_series_id: i.links.imaging_series_id }),
    i.links.hazard_id && t("js.hazard", "hazard {hazard_id}", { hazard_id: i.links.hazard_id })].filter(Boolean);
  $("incident-links").textContent = links.length ? t("js.linked_to", "Linked to {links}.", { links: links.join(", ") }) : "";
  $("incident-links").hidden = !links.length;

  $("incident-manage").hidden = !canManage;
  if (canManage) {
    $("if-status").value = i.status;
    const owner = $("if-owner");
    owner.textContent = "";
    const none = el("option", "", t("js.nobody_yet", "Nobody yet")); none.value = ""; owner.appendChild(none);
    incidentState.people.forEach((p) => { const o = el("option", "", p.name); o.value = p.id; owner.appendChild(o); });
    owner.value = i.owner_id ?? "";
    fillSelect($("if-harm"), INCIDENT_HARM(), i.harm);
    fillSelect($("if-category"), INCIDENT_CATEGORIES(), i.category);
    $("if-root").value = i.root_cause;
    $("if-actions").value = i.actions;
    $("if-external").value = i.external_reference;
    $("if-hazard").value = i.links.hazard_id ?? "";
    $("if-message").hidden = true;
  }
  const timeline = $("incident-timeline");
  timeline.textContent = "";
  i.timeline.forEach((e) => {
    const li = document.createElement("li");
    const head = el("div", "timeline-head");
    head.append(el("strong", "", { reported: t("js.reported", "Reported"), updated: t("js.updated", "Updated"), comment: t("js.comment", "Comment") }[e.action] || e.action), el("span", "muted", `${e.by}, ${dateTime(e.at)}`));
    li.append(head, el("p", "", e.note));
    timeline.appendChild(li);
  });
  $("incident-comment").value = "";
}

async function saveIncident(event) {
  event.preventDefault();
  const i = incidentState.current;
  const hazard = $("if-hazard").value.trim();
  const body = {
    status: $("if-status").value, owner_id: $("if-owner").value ? Number($("if-owner").value) : null, harm: $("if-harm").value,
    category: $("if-category").value, root_cause: $("if-root").value, actions: $("if-actions").value,
    external_reference: $("if-external").value, hazard_id: hazard ? Number(hazard) : null,
  };
  const form = $("incident-form");
  formBusy(form, true);
  try {
    const { incident } = await api(`/api/incidents/${i.id}`, { method: "PATCH", body });
    renderIncident(incident, true);
    showMessage("if-message", t("js.saved", "Saved."));
    loadIncidents();
  } catch (err) {
    showMessage("if-message", err.message, true);
  } finally {
    formBusy(form, false);
  }
}

function wireIncidents() {
  $("incident-new").addEventListener("click", () => openIncidentReport());
  $("incident-report-form").addEventListener("submit", submitIncidentReport);
  $("incident-form").addEventListener("submit", saveIncident);
  $("incident-status-filter").addEventListener("change", loadIncidents);
  $("incident-category-filter").addEventListener("change", loadIncidents);
  $("incident-from-answer").addEventListener("click", () => openIncidentReport({ audit_id: $("flag-row").dataset.auditId, category: "answer" }));
  $("incident-comment-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const note = $("incident-comment").value.trim();
    if (!note) return;
    try {
      const { incident } = await api(`/api/incidents/${incidentState.current.id}/comments`, { method: "POST", body: { note } });
      renderIncident(incident, !$("incident-manage").hidden);
    } catch (err) { showMessage("if-message", err.message, true); }
  });
}

// ---------- Monitoring ----------
const RULE_NOW = () => ({ firing: t("js.firing", "Firing"), ok: t("js.ok_state", "OK"), off: t("js.off_2", "Off") });

function sinceText(iso) {
  const minutes = Math.max(0, Math.round((Date.now() - new Date(iso).getTime()) / 60000));
  if (minutes < 1) return t("js.just_now", "just now");
  if (minutes < 60) return t("js.min_ago", "{minutes} min ago", { minutes });
  const hours = Math.round(minutes / 60);
  if (hours < 48) return t("js.h_ago", "{hours} h ago", { hours });
  return new Date(iso).toLocaleDateString(uiLocale(), { day: "numeric", month: "short" });
}

function updateAlertBadge(firing) {
  const badge = $("nav-alert-count");
  if (!firing || !firing.length) { badge.hidden = true; return; }
  badge.hidden = false;
  badge.textContent = String(firing.length);
  badge.classList.toggle("warning", !firing.some((a) => a.severity === "critical"));
  badge.setAttribute("aria-label", t("js.alert_firing", "{firing} alert{s} firing", { firing: firing.length, s: firing.length === 1 ? "" : "s" }));
}

async function loadMonitoring(check = false) {
  let data;
  try {
    data = check ? await api("/api/monitoring/evaluate", { method: "POST" }) : await api("/api/monitoring");
  } catch (err) { showMessage("monitoring-message", err.message, true); return; }
  $("monitoring-message").hidden = true;
  renderMonitoring(data);
  if (check) {
    const c = data.changes || {};
    const parts = [];
    if (c.fired?.length) parts.push(t("js.started_firing", "{fired} started firing", { fired: c.fired.length }));
    if (c.resolved?.length) parts.push(t("js.resolved_4", "{resolved} resolved", { resolved: c.resolved.length }));
    if (c.errors?.length) parts.push(t("js.check_couldn_t_run", "{errors} check{s} couldn’t run", { errors: c.errors.length, s: c.errors.length === 1 ? "" : "s" }));
    showMessage("monitoring-message", t("js.checked_2", "Checked. {parts}", { parts: parts.length ? parts.join(", ") + "." : t("js.no_changes", "No changes.") }), !!c.errors?.length);
  }
}

function renderMonitoring(data) {
  updateAlertBadge(data.firing);
  const critical = data.firing.filter((a) => a.severity === "critical").length;
  const status = $("monitor-status");
  status.className = `monitor-status${critical ? " critical" : data.firing.length ? " warning" : ""}`;
  status.textContent = "";
  status.append(el("span", "pulse"), el("strong", "", data.firing.length
    ? t("js.alert_firing_2", "{firing} alert{s} firing{critical}", { firing: data.firing.length, s: data.firing.length === 1 ? "" : "s", critical: critical ? t("js.critical_2", ", {critical} critical", { critical }) : "" })
    : t("js.all_checks_are_clear", "All checks are clear")), el("span", "when", t("js.checked_every_min", "Checked every {value} min", { value: Math.round(data.interval_seconds / 60) || 1 })));

  const n = data.now;
  const stats = $("monitor-stats");
  stats.textContent = "";
  const refusal = n.refusal_rate === null ? "–" : `${Math.round(n.refusal_rate * 100)}%`;
  const baseline = n.baseline_refusal_rate === null ? "" : t("js.usually_over_the_last_week", "Usually {value}%, over the last week", { value: Math.round(n.baseline_refusal_rate * 100) });
  stats.append(
    statTile(t("js.questions_last_min", "Questions, last {window_minutes} min", { window_minutes: n.window_minutes }), n.questions.toLocaleString()),
    statTile(t("js.refused_last_min", "Refused, last {window_minutes} min", { window_minutes: n.window_minutes }), refusal, baseline, { attention: data.firing.some((a) => a.rule === "refusal_rate") }),
    statTile(t("js.95th_percentile_response", "95th percentile response"), formatMs(n.p95_ms), null, { attention: data.firing.some((a) => a.rule === "latency") }),
    statTile(t("js.server_errors_15_min", "Server errors, 15 min"), n.server_error_rate_15m === null ? "–" : `${(n.server_error_rate_15m * 100).toFixed(1)}%`,
      t("js.requests_on_this_instance", "{requests_15m} requests on this instance", { requests_15m: n.requests_15m.toLocaleString() }), { attention: data.firing.some((a) => a.rule === "server_errors") }),
  );

  const list = $("alert-list");
  list.textContent = "";
  $("alert-empty").hidden = data.firing.length > 0;
  data.firing.forEach((a) => {
    const li = el("li", `alert-item ${a.severity}`);
    const body = document.createElement("div");
    const h = el("h3");
    h.append(el("span", `sev ${a.severity}`, a.severity === "critical" ? t("js.critical", "Critical") : t("js.warning", "Warning")), a.title);
    body.append(h, el("p", "", a.detail), el("p", "alert-meta",
      t("js.firing_since_last_checked", "Firing since {first_seen}, last checked {last_seen}.{acknowledged_by}", { first_seen: sinceText(a.first_seen), last_seen: sinceText(a.last_seen), acknowledged_by: a.acknowledged_by ? t("js.acknowledged_by", " Acknowledged by {acknowledged_by} {acknowledged_at}.", { acknowledged_by: a.acknowledged_by, acknowledged_at: sinceText(a.acknowledged_at) }) : "" })));
    const actions = el("div", "alert-actions");
    if (!a.acknowledged_at) {
      actions.appendChild(button(t("js.acknowledge", "Acknowledge"), "btn-ghost", false, async (e) => {
        e.target.disabled = true;
        try { await api(`/api/alerts/${a.id}/acknowledge`, { method: "POST" }); loadMonitoring(); }
        catch (err) { showMessage("monitoring-message", err.message, true); e.target.disabled = false; }
      }));
    }
    actions.appendChild(button(t("js.report_an_incident", "Report an incident"), "btn-ghost", false, () => openIncidentReport({
      alert_id: a.id, alert_title: a.title, title: a.title,
      category: a.rule === "audit_chain" ? "data_protection" : a.rule === "imaging_failures" ? "imaging" : a.rule === "latency" || a.rule === "server_errors" ? "availability" : "answer",
    })));
    li.append(body, actions);
    list.appendChild(li);
  });

  const rows = $("rules-rows");
  rows.textContent = "";
  const firingRules = new Set(data.firing.map((a) => a.rule));
  data.rules.forEach((r) => {
    const tr = document.createElement("tr");
    const state = data.disabled.includes(r.rule) ? "off" : firingRules.has(r.rule) ? "firing" : "ok";
    const pill = el("span", `status-pill rule-state ${state === "firing" ? "status-rejected" : state === "ok" ? "status-approved" : "status-retired"}`, RULE_NOW()[state]);
    const cell = document.createElement("td"); cell.appendChild(pill);
    tr.append(el("td", "", r.name), el("td", "", r.limit), cell);
    rows.appendChild(tr);
  });
  const instance = data.instance || {};
  const details = $("instance-details");
  details.textContent = "";
  const gpu = { cpu: t("js.none_using_the_processor", "None, using the processor"), mps: t("js.apple_gpu", "Apple GPU"), cuda: t("js.nvidia_gpu", "NVIDIA GPU") }[instance.accelerator] || instance.accelerator;
  [[t("js.name", "Name"), t("js.version_2", "{name} (version {version})", { name: instance.name, version: instance.version })],
   [t("js.hardware", "Hardware"), t("js.cores_gb_memory", "{cpus} cores, {memory_gb} GB memory", { cpus: instance.cpus, memory_gb: instance.memory_gb })],
   [t("js.accelerator", "Accelerator"), `${gpu}${instance.accelerator !== "cpu" ? `, ${instance.accelerator_memory_gb} GB` : ""}`],
   [t("js.batch_sizes", "Batch sizes"), t("js.passages_image_regions", "{embed_batch} passages, {imaging_batch} image regions", { embed_batch: instance.embed_batch, imaging_batch: instance.imaging_batch })],
   [t("js.workers_this_machine_fits", "Workers this machine fits"), String(instance.web_workers)],
   [t("js.answers_remembered", "Answers remembered"), instance.answer_cache_seconds ? t("js.for_s", "{answers_cached} for {answer_cache_seconds} s", { answers_cached: instance.answers_cached, answer_cache_seconds: instance.answer_cache_seconds }) : t("js.cache_off", "Cache off")],
   [t("js.embeddings_remembered", "Embeddings remembered"), t("js.questions_passages", "{questions_cached} questions, {passages_cached} passages", { questions_cached: instance.questions_cached.toLocaleString(), passages_cached: instance.passages_cached.toLocaleString() })],
   [t("js.running_for", "Running for"), formatDuration(data.now.uptime_seconds)]]
    .forEach(([k, v]) => { details.append(el("dt", "", k), el("dd", "", v)); });
  $("monitor-notify").textContent = data.webhook
    ? t("js.alerts_are_also_posted_to", "Alerts are also posted to the configured webhook when they fire and resolve.")
    : t("js.set_alert_webhook_url_to", "Set ALERT_WEBHOOK_URL to post alerts to Slack, Teams or a paging service.");

  const resolved = $("resolved-list");
  resolved.textContent = "";
  $("resolved-empty").hidden = data.resolved.length > 0;
  data.resolved.slice(0, 8).forEach((a) => {
    const li = el("li", "alert-item");
    const h = el("h3"); h.append(el("span", `sev ${a.severity}`, a.severity === "critical" ? t("js.critical", "Critical") : t("js.warning", "Warning")), a.title);
    li.append(h, el("p", "alert-meta", t("js.fired_resolved", "Fired {first_seen}, resolved {resolved_at}.", { first_seen: dateTime(a.first_seen), resolved_at: sinceText(a.resolved_at) })));
    resolved.appendChild(li);
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
const AGREEMENT_LABELS = () => ({ agree: t("js.agrees_with_the_model", "Agrees with the model"), partly: t("js.partly_agrees_with_the_model", "Partly agrees with the model"), disagree: t("js.disagrees_with_the_model", "Disagrees with the model"), not_used: t("js.model_not_used", "Model not used") });

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

function dateTime(iso) {
  return iso ? new Date(iso).toLocaleString(uiLocale(), { day: "numeric", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit" }) : "–";
}

function seriesTitle(s) {
  return s.label || s.description || t("js.series", "{modality} series {id}", { modality: s.modality, id: s.id });
}

function renderImagingList() {
  const { series, pacs } = imaging.home;
  document.title = t("js.ct_and_mri_groundcheckhealth", "CT and MRI · GroundCheckHealth");
  $("imaging-summary").textContent = series.length ? plural(series.length, "series") : "";
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
    const sub = [s.label && s.description ? s.description : "", s.plane && s.plane !== "unknown" ? s.plane : "", s.source === "pacs" ? t("js.from_the_pacs", "from the PACS") : ""].filter(Boolean).join(", ");
    if (sub) name.appendChild(el("span", "series-sub", sub));
    const images = el("td", "mono", `${s.slices} × ${s.columns}×${s.rows}`);
    images.dataset.label = t("js.images_3", "Images");
    const when = el("td", "", dateTime(s.created_at));
    when.dataset.label = t("js.imported", "Imported");
    const model = document.createElement("td");
    model.dataset.label = t("js.model", "Model");
    model.appendChild(analysisPill(s.analysis));
    const report = document.createElement("td");
    report.dataset.label = t("js.report", "Report");
    report.appendChild(s.report === "signed" ? el("span", "status-pill status-approved", t("js.signed", "Signed"))
      : s.report === "draft" ? el("span", "status-pill status-pending", t("js.draft", "Draft")) : el("span", "muted", t("js.none_2", "None")));
    tr.append(name, images, when, model, report);
    rows.appendChild(tr);
  });
}

function analysisPill(a) {
  if (!a) return el("span", "muted", t("js.not_run", "Not run"));
  if (a.status === "queued" || a.status === "running") return el("span", "status-pill status-running", t("js.running_2", "Running"));
  if (a.status === "refused") return el("span", "status-pill status-rejected", t("js.refused_4", "Refused"));
  if (a.status === "failed") return el("span", "status-pill status-rejected", t("js.failed", "Failed"));
  if (a.abstained) return el("span", "status-pill status-abstained", t("js.abstained", "Abstained"));
  return el("span", "status-pill status-approved", t("js.done", "Done"));
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
    ? t("js.chosen_mb", "{files} chosen, {bytes} MB", { files: plural(imaging.files.length, "file"), bytes: (bytes / 1024 ** 2).toFixed(1) })
    : t("js.ct_and_mr_image_storage", "CT and MR image storage. Up to 3,000 files.");
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
  if (imaging.files.length > 3000) { showMessage("imaging-message", t("js.choose_up_to_3_000", "Choose up to 3,000 files, or upload a zip."), true); return; }
  if (bytes > limit * 1024 ** 2) { showMessage("imaging-message", t("js.uploads_can_be_up_to", "Uploads can be up to {limit} MB.", { limit }), true); return; }
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
    $("dicom-progress-text").textContent = share < 1 ? t("js.uploading_2", "Uploading, {value}%", { value: Math.round(share * 100) }) : t("js.de_identifying_and_reading_the", "De-identifying and reading the images");
  };
  xhr.onloadend = () => {
    $("dicom-progress").hidden = true;
    $("dicom-progress-fill").style.width = "0";
    let data = null;
    try { data = JSON.parse(xhr.responseText); } catch (_) { /* not JSON */ }
    if (xhr.status !== 200) {
      $("dicom-upload").disabled = false;
      showMessage("imaging-message", (data && typeof data.detail === "string") ? data.detail : t("js.the_upload_failed", "The upload failed ({status}).", { status: xhr.status || t("js.network_error", "network error") }), true);
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
  if (added.length) parts.push(t("js.imported_2", "Imported {added}, {reduce}.", { added: plural(added.length, "series"), reduce: plural(added.reduce((n, s) => n + s.slices, 0), "image") }));
  if (existing.length) parts.push(t("js.already_imported", "{existing} already imported.", { existing: existing.length === 1 ? t("js.one_series_was", "One series was") : t("js.series_were", "{existing} series were", { existing: existing.length }) }));
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
    if (!studies.length) out.appendChild(el("p", "auth-hint", t("js.no_studies_match", "No studies match.")));
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
  head.append(el("strong", "", study.patient_name || t("js.unnamed_patient", "Unnamed patient")),
    el("span", "pacs-study-meta mono", [study.patient_id, study.study_date, study.accession].filter(Boolean).join("  ")));
  const desc = el("p", "pacs-study-meta", [study.description, Array.isArray(study.modalities) ? study.modalities.join(", ") : study.modalities].filter(Boolean).join(". "));
  const list = el("ul", "pacs-series");
  const show = button(t("js.show_series", "Show series"), "btn-ghost", false, async () => {
    show.disabled = true;
    try {
      const { series } = await api(`/api/imaging/pacs/studies/${encodeURIComponent(study.study_uid)}/series`);
      list.textContent = "";
      series.forEach((s) => {
        const li = document.createElement("li");
        const usable = ["CT", "MR"].includes(s.modality);
        li.append(el("span", "", `${s.modality || "?"} ${s.description || t("js.series_2", "Series ") + (s.number ?? "")}${s.instances ? `, ${plural(Number(s.instances), "image")}` : ""}`));
        const importButton = button(usable ? t("js.import", "Import") : t("js.not_ct_or_mr", "Not CT or MR"), usable ? "btn-primary" : "btn-ghost", !usable, async () => {
          importButton.disabled = true;
          importButton.textContent = t("js.retrieving", "Retrieving…");
          try {
            importFinished(await api("/api/imaging/pacs/retrieve", { method: "POST", body: { study_uid: study.study_uid, series_uid: s.series_uid } }));
          } catch (err) {
            showMessage("imaging-message", err.message, true);
            importButton.disabled = false;
            importButton.textContent = t("js.import", "Import");
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
  document.title = t("js.groundcheckhealth", "{title} · GroundCheckHealth", { title });
  $("viewer-sub").textContent = [s.modality, s.plane !== "unknown" ? s.plane : "", plural(s.slices, "image"), s.label && s.description ? s.description : ""].filter(Boolean).join(", ");
  $("viewer-delete").hidden = !canDeleteSeries();

  const windowSelect = $("viewer-window");
  windowSelect.textContent = "";
  s.windows.forEach((w) => { const o = el("option", "", w === "auto" ? t("js.automatic", "Automatic") : w[0].toUpperCase() + w.slice(1)); o.value = w; windowSelect.appendChild(o); });
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
  $("viewer-image").alt = t("js.slice_of", "{modality} slice {i} of {slices}", { modality: s.modality, i: i + 1, slices: s.slices });
  $("viewer-slice").value = String(i);
  $("viewer-slice").setAttribute("aria-valuetext", t("js.slice_of_2", "Slice {i} of {slices}", { i: i + 1, slices: s.slices }));
  $("viewer-slice-label").textContent = `${i + 1} / ${s.slices}`;
  const spacing = s.pixel_spacing || [1, 1];
  $("viewer-scale").textContent = t("js.mm", "{columns}×{rows} mm{slice_thickness}", { columns: (s.columns * spacing[1]).toFixed(0), rows: (s.rows * spacing[0]).toFixed(0), slice_thickness: s.slice_thickness || s.slice_spacing ? t("js.mm_slices", ", {slice_thickness} mm slices", { slice_thickness: s.slice_thickness || s.slice_spacing }) : "" });
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
    [s.deid.removed, t("js.deid_removed", "removed")], [s.deid.emptied, t("js.deid_emptied", "emptied")], [s.deid.private_removed, t("js.private_tags_removed", "private tags removed")],
    [s.deid.uids_replaced, t("js.identifiers_replaced", "identifiers replaced")], [s.deid.descriptors_cleaned, t("js.descriptions_cleaned", "descriptions cleaned")],
  ].filter(([n]) => n).map(([n, what]) => `${n.toLocaleString()} ${what}`);
  const rows = [
    [t("js.modality", "Modality"), `${s.modality}${s.body_part ? `, ${s.body_part.toLowerCase()}` : ""}`],
    [t("js.plane", "Plane"), s.plane === "unknown" ? t("js.unknown_shown_as_stored", "Unknown, shown as stored") : s.plane[0].toUpperCase() + s.plane.slice(1)],
    [t("js.images_3", "Images"), t("js.slices_pixels", "{slices} slices, {columns} × {rows} pixels", { slices: s.slices, columns: s.columns, rows: s.rows })],
    [t("js.pixel_spacing", "Pixel spacing"), spacing.length ? `${spacing.map((v) => v.toFixed(2)).join(" × ")} mm` : t("js.not_recorded", "Not recorded")],
    [t("js.slice_thickness", "Slice thickness"), s.slice_thickness ? `${s.slice_thickness} mm` : t("js.not_recorded", "Not recorded")],
    [t("js.slice_spacing", "Slice spacing"), s.slice_spacing ? `${s.slice_spacing} mm` : t("js.not_recorded", "Not recorded")],
    ["Patient", [patient.PatientAge ? t("js.age_2", "age {replace}", { replace: patient.PatientAge.replace(/^0+/, "").replace("Y", t("js.years_suffix", " years")) }) : "", patient.PatientSex ? t("js.sex", "sex {PatientSex}", { PatientSex: patient.PatientSex }) : ""].filter(Boolean).join(", ") || t("js.no_characteristics_recorded", "No characteristics recorded")],
    [t("js.source", "Source"), s.source === "pacs" ? t("js.retrieved_from_the_pacs", "Retrieved from the PACS") : t("js.uploaded", "Uploaded")],
    [t("js.de_identification", "De-identification"), t("js.dicom_ps3_15_basic_profile", "DICOM PS3.15 basic profile. {deidParts}.", { deidParts: deidParts.join(", ") || t("js.nothing_to_remove", "Nothing to remove") })],
  ];
  (s.warnings || []).forEach((w) => rows.push([t("js.note_2", "Note"), w]));
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
    const o = el("option", "", m.modality && m.modality !== s.modality ? t("js.trained_on", "{name} (trained on {modality})", { name: m.name, modality: m.modality }) : m.name);
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
    ? t("js.no_imaging_models_yet_an", "No imaging models yet. An administrator can add imaging settings to a model in the Model library.")
    : !usable.length ? t("js.none_of_the_models_were", "None of the models were trained on {modality}.", { modality: s.modality }) : "";
  $("analysis-hint").hidden = !$("analysis-hint").textContent;
  $("analysis-progress").hidden = !running;
  if (running) {
    $("analysis-progress-fill").style.width = `${a.progress}%`;
    $("analysis-progress-text").textContent = a.status === "queued" ? t("js.waiting_to_start", "Waiting to start") : t("js.analysing", "Analysing, {progress}%", { progress: a.progress });
  }

  const out = $("analysis-result");
  out.textContent = "";
  const map = $("slice-map");
  map.textContent = "";
  if (!a || running) return;
  out.appendChild(el("p", "field-hint", `${a.model_name}, ${dateTime(a.finished_at || a.created_at)}`));
  if (a.status === "refused" || a.status === "failed") {
    const note = el("div", "analysis-note refused");
    note.append(el("strong", "", a.status === "refused" ? t("js.the_model_refused_this_series", "The model refused this series") : t("js.the_analysis_didn_t_finish", "The analysis didn’t finish")), el("span", "", a.error || ""));
    out.appendChild(note);
    return;
  }
  const sum = a.summary;
  if (sum.abstained) {
    const note = el("div", "analysis-note");
    note.appendChild(el("strong", "", t("js.the_model_abstained_on_this", "The model abstained on this series")));
    note.appendChild(el("span", "", sum.unfamiliar_series
      ? t("js.of_the_image_regions_it", "{value}% of the image regions it looked at were unlike its training images, so it reports nothing. Read the images without it.", { value: Math.round(sum.unfamiliar_share * 100) })
      : t("js.it_wasn_t_confident_about", "It wasn’t confident about any region, so it reports nothing.")));
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
      b.title = t("js.go_to_the_first_slice", "Go to the first slice");
      b.addEventListener("click", () => { imaging.index = l.ranges[0][0]; renderSlice(); $("viewer-stage").focus(); });
      li.appendChild(b);
      list.appendChild(li);
      const track = el("div", "slice-track");
      track.style.setProperty("--c", labelColour(l.label));
      track.appendChild(el("span", "", l.label));
      const bar = document.createElement("button");
      bar.type = "button";
      bar.className = "slice-bar";
      bar.setAttribute("aria-label", t("js.slices_go_to_the_first", "{label}: slices {ranges}. Go to the first.", { label: l.label, ranges: rangesText(l.ranges) }));
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
  const tt = sum.totals;
  if (tt.patches) {
    const wrap = document.createElement("div");
    wrap.appendChild(el("p", "field-label", t("js.image_regions_examined", "Image regions examined: {patches}", { patches: tt.patches.toLocaleString() })));
    const bar = el("div", "patch-bar");
    bar.setAttribute("role", "img");
    bar.setAttribute("aria-label", t("js.answered_not_confident_unfamiliar", "{answered} answered, {low_confidence} not confident, {unfamiliar} unfamiliar", { answered: tt.answered, low_confidence: tt.low_confidence, unfamiliar: tt.unfamiliar }));
    const legend = el("ul", "patch-legend");
    [["answered", tt.answered, t("js.answered_2", "Answered"), "var(--ink)"], ["low", tt.low_confidence, t("js.not_confident_2", "Not confident"), "var(--series-2)"], ["unfamiliar", tt.unfamiliar, t("js.unlike_its_training_images", "Unlike its training images"), "var(--warn)"]]
      .forEach(([cls, n, label, colour]) => {
        if (n) { const seg = el("span", cls); seg.style.flex = String(n); bar.appendChild(seg); }
        const li = el("li", "", `${label} ${n.toLocaleString()}`);
        li.style.setProperty("--c", colour);
        legend.appendChild(li);
      });
    wrap.append(bar, legend);
    out.appendChild(wrap);
  }
  out.appendChild(el("p", "research-note", t("js.for_research_and_evaluation_not", "For research and evaluation. Not validated for clinical use; the reading clinician decides.")));
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
    card.append(el("h3", "", signed.replaces_id ? t("js.signed_amendment", "Signed amendment") : t("js.signed_report", "Signed report")),
      el("p", "signed-impression", signed.impression));
    if (signed.findings) card.appendChild(el("p", "", signed.findings));
    card.appendChild(el("p", "signed-meta", t("js.signed_by_on", "{agreement}. Signed by {author_name} on {signed_at}.{sent_at}", { agreement: AGREEMENT_LABELS()[signed.agreement], author_name: signed.author_name || t("js.unknown", "unknown"), signed_at: dateTime(signed.signed_at), sent_at: signed.sent_at ? t("js.sent_to_the_pacs_on", " Sent to the PACS on {sent_at}.", { sent_at: dateTime(signed.sent_at) }) : "" })));
    const actions = el("div", "signed-actions");
    const download = el("a", "btn btn-sm btn-ghost", t("js.download_dicom_sr", "Download DICOM SR"));
    download.href = `/api/imaging/reports/${signed.id}/sr.dcm`;
    actions.appendChild(download);
    if (s.can_send_to_pacs) {
      actions.appendChild(button(signed.sent_at ? t("js.send_again", "Send again") : t("js.send_to_the_pacs", "Send to the PACS"), "btn-ghost", false, async (e) => {
        e.target.disabled = true;
        try { await api(`/api/imaging/reports/${signed.id}/send`, { method: "POST" }); showMessage("report-message", t("js.sent_to_the_pacs", "Sent to the PACS.")); openSeries(s.id, true); }
        catch (err) { showMessage("viewer-message", err.message, true); e.target.disabled = false; }
      }));
    }
    if (!draft && imaging.editing !== signed.id) {
      actions.appendChild(button(t("js.amend", "Amend"), "btn-ghost", false, () => {
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
    $("report-draft-note").textContent = t("js.draft_saved", "Draft saved {updated_at}.", { updated_at: dateTime(draft.updated_at) });
  } else if (signed && imaging.editing !== signed.id) {
    form.hidden = true;
  } else if (!signed && imaging.editing === null) {
    form.hidden = false;
    fillReportForm(null, null);
    imaging.editing = "new";
  }
  if (!draft) $("report-draft-note").textContent = imaging.editing === signed?.id ? t("js.signing_creates_an_amendment_the", "Signing creates an amendment. The signed report is kept.") : "";
  $("report-sign").textContent = imaging.editing === signed?.id || draft?.replaces_id ? t("js.sign_amendment", "Sign amendment") : t("js.sign_report", "Sign report");

  $("report-history").hidden = superseded.length === 0;
  const list = $("report-history-list");
  list.textContent = "";
  superseded.forEach((r) => {
    const li = document.createElement("li");
    li.append(el("p", "", r.impression), el("p", "signed-meta", t("js.signed_by_on_replaced_by", "Signed by {author_name} on {signed_at}. Replaced by an amendment.", { author_name: r.author_name || t("js.unknown", "unknown"), signed_at: dateTime(r.signed_at) })));
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
    if (!$("report-impression").value.trim()) { showMessage("report-message", t("js.write_an_impression_before_signing", "Write an impression before signing."), true); $("report-impression").focus(); return; }
    if (!window.confirm(t("js.sign_this_report_a_signed", "Sign this report? A signed report can’t be edited. Later changes are made as an amendment."))) return;
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
    showMessage("report-message", sign ? t("js.signed_2", "Signed.") : t("js.draft_saved_2", "Draft saved."));
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
  ["dragenter", "dragover"].forEach((tt) => drop.addEventListener(tt, (e) => { e.preventDefault(); drop.classList.add("dragging"); }));
  ["dragleave", "drop"].forEach((tt) => drop.addEventListener(tt, () => drop.classList.remove("dragging")));
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
    if (!window.confirm(t("js.delete_its_analyses_and_its", "Delete “{s}”, its analyses and its reports? This can’t be undone.", { s: seriesTitle(s) }))) return;
    try {
      await api(`/api/imaging/series/${s.id}`, { method: "DELETE" });
      imaging.series = null;
      location.hash = "#/imaging";
      showMessage("imaging-message", t("js.deleted_2", "Deleted."));
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
  monitoring: { load: () => loadMonitoring() },
  incidents: { load: () => loadIncidents() },
  releases: { load: () => loadReleases() },
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
  const link = document.querySelector(`.nav-link[data-view="${name}"]`);
  const label = link && link.querySelector("[data-i18n]");
  const title = (label && label.textContent.trim()) || $(`view-${name}`).dataset.title;
  $("topbar-title").textContent = title;
  document.title = name === "ask" ? "GroundCheckHealth" : t("js.groundcheckhealth", "{title} · GroundCheckHealth", { title });
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
  wireIncidents();
  wireReleases();
  $("monitoring-check").addEventListener("click", async (e) => { e.target.disabled = true; await loadMonitoring(true); e.target.disabled = false; });
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
    badge.title = t("js.open_4", "{n} open{overdue}", { n, overdue: data.counts.overdue ? t("js.overdue", ", {overdue} overdue", { overdue: data.counts.overdue }) : "" });
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
    const b = $("audit-copy"); const tt = b.textContent; b.textContent = t("js.copied", "Copied"); setTimeout(() => b.textContent = tt, 1200);
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
    if (!data.total) { inline.textContent = data.note || t("js.not_run_yet", "not run yet"); return; }
    const ansC = data.answerable_correct, ansT = data.answerable_total ?? ansC;
    const refC = data.must_refuse_correct, refT = data.must_refuse_total ?? refC;
    inline.textContent = t("js.of_cases_pass", "{passed} of {total} cases pass", { passed: data.passed.toLocaleString(), total: data.total.toLocaleString() });
    const pct = Math.round((data.passed / data.total) * 100);
    $("eval-bar-fill").style.width = `${pct}%`;
    $("eval-bar-fill").style.background = pct === 100 ? "var(--ok)" : "var(--warn)";
    $("eval-caption").textContent =
      t("js.of_answerable_correct", "{ansC} of {ansT} answerable correct, ", { ansC: ansC.toLocaleString(), ansT: ansT.toLocaleString() }) +
      t("js.of_must_refuse_correct", "{refC} of {refT} must-refuse correct. ", { refC: refC.toLocaleString(), refT: refT.toLocaleString() }) +
      t("js.run_offline_in_extractive_mode", "Run offline in extractive mode for reproducibility.");
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
      v.className = "verdict"; v.textContent = t("js.expect_got", "expect {expect} / got {got}", { expect: c.expect, got: c.got });
      li.append(g, q, v);
      list.appendChild(li);
    });
    const shown = (data.cases || data.sample_cases || []).length;
    let foot = data.total > shown
      ? t("js.showing_a_sample_of_full", "Showing a sample of {shown}. Full suite of {total} runs in the build.", { shown, total: data.total.toLocaleString() })
      : "";
    renderAdversarial(data.adversarial);
    renderPatientEval(data.patient);
    renderPiiEval(data.pii);
    renderMetricBlock("answer-quality-block", t("js.answer_quality", "Answer quality"), t("js.how_well_the_answers_read", "How well the answers read and how closely they follow their sources."), data.answer_quality);
    renderMetricBlock("phrasing-block", t("js.phrasing_robustness", "Phrasing robustness"), t("js.the_same_question_worded_differently", "The same question worded differently should get the same decision."), data.phrasing);
    $("eval-foot").textContent = foot;
  } catch (_) { $("eval-summary-inline").textContent = t("js.unavailable", "unavailable"); }
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
  title.textContent = t("js.adversarial_probes", "Adversarial probes");
  const score = document.createElement("span");
  score.className = "adv-score";
  score.textContent = t("js.of_behaved_as_intended", "{passed} of {total} behaved as intended", { passed: adv.passed, total: adv.total });
  head.append(title, score);
  const note = document.createElement("p");
  note.className = "adv-note";
  note.textContent = t("js.hand_written_traps_leading_questions", "Hand-written traps: leading questions, prompt injection, ") +
    t("js.near_miss_spellings_and_missing", "near-miss spellings, and missing-context cases. Reported honestly and not ") +
    t("js.used_to_gate_the_build", "used to gate the build, so a known limitation is shown rather than hidden.");
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
    v.textContent = c.ok ? `${c.got} ✓` : t("js.want_got", "want {expect}, got {got}", { expect: c.expect, got: c.got });
    li.append(g, q, v);
    list.appendChild(li);
  });
  wrap.appendChild(list);
}

function metricText(v) { return typeof v === "number" ? (v <= 1 ? `${(v * 100).toFixed(v === 1 || v === 0 ? 0 : 1)}%` : v.toLocaleString()) : String(v); }

function renderPiiEval(pii) {
  const wrap = $("pii-eval-block");
  if (!wrap) return;
  wrap.textContent = "";
  const kinds = pii && typeof pii === "object" ? (pii.per_kind || pii.kinds || pii) : null;
  const rows = kinds ? Object.entries(kinds).filter(([, v]) => v && typeof v === "object" && ("f1" in v || "precision" in v || "recall" in v)) : [];
  wrap.hidden = !rows.length;
  if (!rows.length) return;
  const head = document.createElement("div"); head.className = "adv-head";
  const title = document.createElement("span"); title.className = "eyebrow"; title.textContent = t("js.de_identification_accuracy", "De-identification accuracy");
  head.appendChild(title);
  const note = document.createElement("p"); note.className = "adv-note";
  note.textContent = t("js.per_kind_of_personal_detail", "Per kind of personal detail: precision is how much of what was removed really was that kind, recall is how much of that kind was removed.");
  const table = document.createElement("table"); table.className = "coverage-table pii-table";
  const thead = document.createElement("thead"); const hr = document.createElement("tr");
  [t("js.kind", "Kind"), t("js.precision", "Precision"), t("js.recall", "Recall"), "F1"].forEach((tt) => { const th = document.createElement("th"); th.scope = "col"; th.textContent = tt; hr.appendChild(th); });
  thead.appendChild(hr);
  const tbody = document.createElement("tbody");
  rows.forEach(([kind, v]) => {
    const tr = document.createElement("tr");
    [kind, v.precision, v.recall, v.f1].forEach((cell, i) => {
      const td = document.createElement("td");
      td.className = i ? "mono num" : "mono";
      td.textContent = i && cell != null ? metricText(cell) : cell == null ? "n/a" : String(cell);
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
  table.append(thead, tbody);
  wrap.append(head, note, table);
}

function renderMetricBlock(id, titleText, noteText, block) {
  const wrap = $(id);
  if (!wrap) return;
  wrap.textContent = "";
  const entries = block && typeof block === "object"
    ? Object.entries(block).filter(([, v]) => typeof v === "number" || typeof v === "string") : [];
  wrap.hidden = !entries.length;
  if (!entries.length) return;
  const head = document.createElement("div"); head.className = "adv-head";
  const title = document.createElement("span"); title.className = "eyebrow"; title.textContent = titleText;
  head.appendChild(title);
  const note = document.createElement("p"); note.className = "adv-note"; note.textContent = noteText;
  const grid = document.createElement("dl"); grid.className = "metric-grid";
  entries.forEach(([k, v]) => {
    const box = document.createElement("div"); box.className = "metric";
    const dt = document.createElement("dt"); dt.textContent = k.replace(/_/g, " ");
    const dd = document.createElement("dd"); dd.textContent = metricText(v);
    box.append(dt, dd);
    grid.appendChild(box);
  });
  wrap.append(head, note, grid);
}

// ---------- Utils ----------
function capitalize(s) { return s && s.length ? s.charAt(0).toUpperCase() + s.slice(1) : s; }

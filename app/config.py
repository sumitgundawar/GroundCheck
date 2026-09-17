"""Central configuration. Every value is read from the environment with a safe
default so the app runs end to end even with nothing set. The decision logic is
deterministic and never depends on any of the LLM settings below."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load a local .env if present. On the deployment host the values are injected
# as platform secrets, so this is a no-op there.
load_dotenv()

# Repository root (the directory that contains the app/ package).
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(__file__).resolve().parent / "data"


def _get(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def _get_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _get_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


# --- LLM provider (Groq, OpenAI-compatible) -------------------------------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
GROQ_BASE_URL = _get("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
GEN_MODEL = _get("GEN_MODEL", "llama-3.3-70b-versatile")
JUDGE_MODEL = _get("JUDGE_MODEL", "llama-3.1-8b-instant")
LLM_TIMEOUT_SECONDS = _get_float("LLM_TIMEOUT_SECONDS", 8.0)
USE_LLM_JUDGE = _get("USE_LLM_JUDGE", "true").lower() in ("1", "true", "yes")

# --- Local AI (Ollama) ------------------------------------------------------
# An open-source model on this machine can draft answers instead of a cloud
# model. It is chosen in the dashboard (or with LOCAL_MODEL) and takes
# precedence over the cloud model when Ollama is running and has it installed.
OLLAMA_HOST = _get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
LOCAL_MODEL = _get("LOCAL_MODEL", "")
LOCAL_AI_TIMEOUT_SECONDS = _get_float("LOCAL_AI_TIMEOUT_SECONDS", 60.0)
# In the open demo (AUTH_REQUIRED=false) there are no admins, so management
# actions (downloading and switching local models, importing and approving
# documents) are allowed from: "local" (this machine only, the default), "all",
# or "none". With AUTH_REQUIRED=true, the admin role decides instead.
# LOCAL_AI_ADMIN is the earlier name and still works.
ADMIN_ACCESS = _get("ADMIN_ACCESS", _get("LOCAL_AI_ADMIN", "local")).lower()
LOCAL_AI_STATE_PATH = ROOT_DIR / _get("LOCAL_AI_STATE_PATH", "local_ai.json")

# --- Embeddings + retrieval ------------------------------------------------
EMBED_MODEL = _get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
TOP_K = _get_int("TOP_K", 4)
# Where embeddings are stored and searched: "local" (exact search in-process,
# the default) or "qdrant" (a vector database; see app/vectorstore.py).
VECTOR_STORE = _get("VECTOR_STORE", "local").lower()
QDRANT_URL = _get("QDRANT_URL", "")            # server, e.g. http://localhost:6333
QDRANT_PATH = _get("QDRANT_PATH", "")          # embedded on disk; default index/qdrant
QDRANT_COLLECTION = _get("QDRANT_COLLECTION", "groundcheck")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "").strip()
# Combine keyword (BM25) and embedding search. Keyword matching keeps rare,
# look-alike names such as drug names from being confused with each other.
HYBRID_RETRIEVAL = _get("HYBRID_RETRIEVAL", "true").lower() in ("1", "true", "yes")
# Weight of the embedding score in hybrid ranking (0 to 1); the keyword score
# gets the rest.
HYBRID_ALPHA = min(1.0, max(0.0, _get_float("HYBRID_ALPHA", 0.5)))
RETRIEVAL_MIN_SCORE = _get_float("RETRIEVAL_MIN_SCORE", 0.30)
GROUNDING_MIN = _get_float("GROUNDING_MIN", 0.45)

# --- Paths -----------------------------------------------------------------
INDEX_DIR = ROOT_DIR / _get("INDEX_DIR", "index")
# Include the synthetic demo corpus in the index alongside approved documents.
# Set false for a deployment that should answer only from its own documents.
INCLUDE_DEMO_CORPUS = _get("INCLUDE_DEMO_CORPUS", "true").lower() in ("1", "true", "yes")
CORPUS_PATH = DATA_DIR / "corpus.json"
EXAMPLES_PATH = DATA_DIR / "examples.json"
EVAL_SUMMARY_PATH = ROOT_DIR / "eval" / "eval_summary.json"

# --- Input guard limits ----------------------------------------------------
MAX_QUERY_CHARS = 400
# Requests per minute, enforced per client IP (see guards_input). Protects a
# public deployment from a single client draining the LLM quota.
RATE_LIMIT_PER_MINUTE = _get_int("RATE_LIMIT_PER_MINUTE", 30)

# Force extractive mode even when an API key is present. Use this to expose a
# public demo without spending the LLM quota; run the live model from localhost.
FORCE_EXTRACTIVE = _get("FORCE_EXTRACTIVE", "false").lower() in ("1", "true", "yes")

# --- Database ----------------------------------------------------------------
# Users, sign-in sessions and audit records. SQLite by default; PostgreSQL and
# MySQL are supported through their SQLAlchemy URLs (see app/db.py).
DATABASE_URL = _get("DATABASE_URL", "sqlite:///data/groundcheck.db")
DB_AUTO_MIGRATE = _get("DB_AUTO_MIGRATE", "true").lower() in ("1", "true", "yes")

# --- Accounts ----------------------------------------------------------------
# Off by default, so the demo works without an account. Set AUTH_REQUIRED=true
# for any deployment with real users: every API call except health and sign-in
# then needs a signed-in session, and actions are limited by role.
AUTH_REQUIRED = _get("AUTH_REQUIRED", "false").lower() in ("1", "true", "yes")
SESSION_HOURS = _get_float("SESSION_HOURS", 12.0)
# Mark the session cookie Secure (HTTPS only). Leave on in production.
SESSION_COOKIE_SECURE = _get("SESSION_COOKIE_SECURE", "true").lower() in ("1", "true", "yes")
LOGIN_MAX_FAILURES = _get_int("LOGIN_MAX_FAILURES", 5)
LOGIN_LOCKOUT_MINUTES = _get_float("LOGIN_LOCKOUT_MINUTES", 15.0)
PASSWORD_MIN_LENGTH = _get_int("PASSWORD_MIN_LENGTH", 12)

# --- Review and governance ------------------------------------------------------
# Open a review case for every refusal (repeats of an open question are grouped).
REVIEW_QUEUE = _get("REVIEW_QUEUE", "true").lower() in ("1", "true", "yes")
REVIEW_SLA_HOURS = _get_float("REVIEW_SLA_HOURS", 72.0)
FLAGGED_SLA_HOURS = _get_float("FLAGGED_SLA_HOURS", 24.0)

# --- Audit -----------------------------------------------------------------
AUDIT_RING_SIZE = 50
# Append-only audit log. Defaults to a writable path under the repo; override
# with AUDIT_LOG_PATH. Persistence is best-effort: if the path is not writable
# (for example a read-only container), the app falls back to memory-only and
# never errors. Set AUDIT_PERSIST=false to disable disk persistence entirely.
AUDIT_LOG_PATH = Path(_get("AUDIT_LOG_PATH", str(ROOT_DIR / "audit" / "audit_log.jsonl")))
AUDIT_PERSIST = _get("AUDIT_PERSIST", "true").lower() in ("1", "true", "yes")

# --- Protecting stored data (see app/encryption.py and app/integrity.py) ---
# Secrets: read from the environment only, never from a file in the repository.
# Comma-separated base64 keys; the first encrypts, all decrypt.
DATA_ENCRYPTION_KEYS = os.environ.get("DATA_ENCRYPTION_KEYS", "").strip()
# Comma-separated base64 keys that only decrypt, for rotation or turning encryption off.
DATA_ENCRYPTION_RETIRED_KEYS = os.environ.get("DATA_ENCRYPTION_RETIRED_KEYS", "").strip()
# Comma-separated base64 keys; the first signs the audit chain, all verify.
AUDIT_SIGNING_KEYS = os.environ.get("AUDIT_SIGNING_KEYS", "").strip()

# Serve interactive API docs at /docs. Turn off in production if not needed.
API_DOCS = _get("API_DOCS", "true").lower() in ("1", "true", "yes")

# --- Single sign-on with OpenID Connect (see app/sso.py) ---
OIDC_ISSUER = _get("OIDC_ISSUER", "").rstrip("/")
OIDC_CLIENT_ID = _get("OIDC_CLIENT_ID", "")
OIDC_CLIENT_SECRET = os.environ.get("OIDC_CLIENT_SECRET", "").strip()
OIDC_PROVIDER_NAME = _get("OIDC_PROVIDER_NAME", "your organisation")
OIDC_REDIRECT_URL = _get("OIDC_REDIRECT_URL", "")   # default: <this site>/api/auth/sso/callback
OIDC_SCOPES = _get("OIDC_SCOPES", "openid email profile")
OIDC_ROLES_CLAIM = _get("OIDC_ROLES_CLAIM", "roles")  # "groups" for many providers
OIDC_ADMIN_VALUES = {v.strip() for v in _get("OIDC_ADMIN_VALUES", "").split(",") if v.strip()}
OIDC_REVIEWER_VALUES = {v.strip() for v in _get("OIDC_REVIEWER_VALUES", "").split(",") if v.strip()}
OIDC_CLINICIAN_VALUES = {v.strip() for v in _get("OIDC_CLINICIAN_VALUES", "").split(",") if v.strip()}
OIDC_DEFAULT_ROLE = _get("OIDC_DEFAULT_ROLE", "clinician")  # or "none" to refuse people without a mapped role
OIDC_ALLOWED_DOMAINS = {v.strip().lower() for v in _get("OIDC_ALLOWED_DOMAINS", "").split(",") if v.strip()}
OIDC_AUTO_CREATE = _get("OIDC_AUTO_CREATE", "true").lower() in ("1", "true", "yes")
OIDC_REQUIRE_MFA = _get("OIDC_REQUIRE_MFA", "false").lower() in ("1", "true", "yes")
PASSWORD_SIGN_IN = _get("PASSWORD_SIGN_IN", "true").lower() in ("1", "true", "yes")

# --- Patient-aware checks (see app/formulary.py and app/patient_checks.py) ---
# Medicine rules in the app/data/formulary.json format. The default is the
# synthetic demo formulary; replace it with your organisation's licensed data.
FORMULARY_PATH = Path(_get("FORMULARY_PATH", str(DATA_DIR / "formulary.json")))

# --- EHR integration (see app/fhir.py, app/smart.py, app/cds_hooks.py) ---
def _list(name: str, default: str = "") -> list[str]:
    return [v.strip().rstrip("/") for v in _get(name, default).split(",") if v.strip()]


# FHIR servers GroundCheck may read patients from without SMART authorisation:
# public sandboxes for development, or an internal server behind your network.
FHIR_OPEN_SERVERS = _list("FHIR_OPEN_SERVERS")
# SMART on FHIR app launch. Issuers are the EHRs' FHIR base URLs allowed to launch GroundCheck.
SMART_CLIENT_ID = _get("SMART_CLIENT_ID", "")
SMART_CLIENT_SECRET = os.environ.get("SMART_CLIENT_SECRET", "").strip()
SMART_ALLOWED_ISSUERS = _list("SMART_ALLOWED_ISSUERS")
SMART_SCOPES = _get("SMART_SCOPES", "launch launch/patient openid fhirUser patient/Patient.read patient/Observation.read "
                    "patient/AllergyIntolerance.read patient/MedicationRequest.read patient/MedicationStatement.read "
                    "patient/Condition.read")
SMART_REDIRECT_URL = _get("SMART_REDIRECT_URL", "")   # default: <this site>/api/ehr/callback
# Let clinicians save a reviewed answer to the patient's record as a
# preliminary DocumentReference. Adds the write scope and keeps the EHR's
# access token, encrypted, for as long as the patient is kept.
SMART_WRITE_NOTES = _get("SMART_WRITE_NOTES", "false").lower() in ("1", "true", "yes")
# Lab results older than this are flagged as possibly out of date.
FHIR_LAB_MAX_AGE_DAYS = _get_int("FHIR_LAB_MAX_AGE_DAYS", 90)
EHR_CONTEXT_MINUTES = _get_int("EHR_CONTEXT_MINUTES", 60)
# CDS Hooks: EHRs allowed to call the services, as issuer=JWKS URL pairs.
CDS_HOOKS_TRUSTED = dict(pair.split("=", 1) for pair in _list("CDS_HOOKS_TRUSTED") if "=" in pair)
CDS_HOOKS_ALLOW_UNSIGNED = _get("CDS_HOOKS_ALLOW_UNSIGNED", "false").lower() in ("1", "true", "yes")

# --- Imaging model training (see app/training/) ---
# Folders the training studio may read images from, comma-separated. The
# default is the repository's data/datasets folder and your home folder, for
# use on your own machine. On a shared server, list only the dataset folders.
TRAINING_DATA_DIRS = [
    Path(os.path.expanduser(p.strip())).resolve()
    for p in _get("TRAINING_DATA_DIRS", f"{ROOT_DIR / 'data' / 'datasets'},~").split(",") if p.strip()
]
MODEL_LIBRARY_DIR = Path(_get("MODEL_LIBRARY_DIR", str(ROOT_DIR / "models" / "library"))).resolve()
TRAINING_RUNS_DIR = Path(_get("TRAINING_RUNS_DIR", str(ROOT_DIR / "models" / "runs"))).resolve()
# Accuracy a trained model must reach on the images it answers, on held-out
# validation images. Below its confidence threshold it abstains instead.
MODEL_TARGET_ACCURACY = _get_float("MODEL_TARGET_ACCURACY", 0.95)
# A model never answers below this confidence, however well it validated:
# with many classes, a top class at 30% means the model is torn.
MODEL_MIN_CONFIDENCE = _get_float("MODEL_MIN_CONFIDENCE", 0.5)

# --- Monitoring and alerts (see app/monitoring.py) ---
VERSION = "1.0.0"
# A name for this instance, included in alert notifications.
INSTANCE_NAME = _get("INSTANCE_NAME", "") or __import__("socket").gethostname()
# Bearer token Prometheus sends to /metrics. Without it, /metrics answers only
# requests from this machine.
METRICS_TOKEN = _get("METRICS_TOKEN", "")
ALERT_INTERVAL_SECONDS = _get_int("ALERT_INTERVAL_SECONDS", 60)   # 0 turns background checks off
ALERT_WEBHOOK_URL = _get("ALERT_WEBHOOK_URL", "")
ALERT_WINDOW_MINUTES = _get_int("ALERT_WINDOW_MINUTES", 60)
ALERT_MIN_QUESTIONS = _get_int("ALERT_MIN_QUESTIONS", 20)
ALERT_REFUSAL_RISE = _get_float("ALERT_REFUSAL_RISE", 0.15)
ALERT_LATENCY_P95_MS = _get_int("ALERT_LATENCY_P95_MS", 15000)
ALERT_DRIFT_PSI = _get_float("ALERT_DRIFT_PSI", 0.25)
ALERT_OVERDUE_CRITICAL = _get_int("ALERT_OVERDUE_CRITICAL", 10)
ALERT_DOCUMENT_EXPIRY_DAYS = _get_int("ALERT_DOCUMENT_EXPIRY_DAYS", 14)
ALERT_CHAIN_CHECK_MINUTES = _get_int("ALERT_CHAIN_CHECK_MINUTES", 360)
ALERT_DISABLED_RULES = {r.strip() for r in _get("ALERT_DISABLED_RULES", "").split(",") if r.strip()}

# --- CT and MRI imaging (see app/imaging/) ---
# Your organisation's name, recorded in signed imaging reports.
ORGANISATION_NAME = _get("ORGANISATION_NAME", "")
# Where imported series are stored, encrypted when DATA_ENCRYPTION_KEYS is set.
IMAGING_DIR = Path(_get("IMAGING_DIR", str(ROOT_DIR / "data" / "imaging"))).resolve()
# A PACS or VNA to query, retrieve from and send reports to, over DICOMweb
# (QIDO-RS, WADO-RS, STOW-RS). Empty turns the PACS panel off.
DICOMWEB_URL = _get("DICOMWEB_URL", "").rstrip("/")
# "Basic <base64 user:password>" or "Bearer <token>", sent to the PACS only.
DICOMWEB_AUTHORIZATION = _get("DICOMWEB_AUTHORIZATION", "")
# Largest upload, in megabytes.
IMAGING_MAX_UPLOAD_MB = _get_int("IMAGING_MAX_UPLOAD_MB", 1024)

# --- Retention (see app/retention.py). 0 keeps records for ever. ---
AUDIT_RETENTION_DAYS = _get_int("AUDIT_RETENTION_DAYS", 0)
REVIEW_RETENTION_DAYS = _get_int("REVIEW_RETENTION_DAYS", 0)


def llm_configured() -> bool:
    """True if a live LLM should be used: a key is present and extractive mode
    is not being forced. The app runs fully either way."""
    return bool(GROQ_API_KEY) and not FORCE_EXTRACTIVE

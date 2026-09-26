"""Input guards: PII redaction, scope and injection checks, and a simple
in-memory rate limiter. These run before anything is embedded, logged, or sent
to the LLM."""

from __future__ import annotations

import re
import time
from collections import deque
from dataclasses import dataclass

from . import config, deid

# --- De-identification (see app/deid.py) ------------------------------------

def deidentify(text: str) -> deid.Result:
    """Replace names, dates, record numbers and contact details with
    placeholders. Words the document index knows are never treated as names."""
    from . import retrieval

    return deid.deidentify(text, retrieval.is_known_term)


def redact_pii(text: str) -> tuple[str, bool]:
    """De-identify text. Returns the cleaned text and whether anything changed."""
    result = deidentify(text)
    return result.text, result.changed


# --- Injection / scope -----------------------------------------------------
_OVERRIDE_VERB = r"(?:ignore|disregard|forget|override|skip|drop|abandon|stop following|don't follow|do not follow)"
_OVERRIDE_TARGET = (r"(?:(?:all|any|the|your|my|these|those|previous|prior|earlier|above|preceding|original|system|safety)\s+){0,4}"
                    r"(?:instructions?|rules?|sources?|guidelines?|guardrails?|restrictions?|constraints?|prompts?|"
                    r"context|policies|policy|checks?|guards?|above|everything)")
_INJECTION_PATTERNS = [
    rf"\b{_OVERRIDE_VERB}\s+{_OVERRIDE_TARGET}\b",
    r"\bsystem\s*prompt\b",
    r"^\s*(?:system|assistant|developer)\s*:",
    r"\byou\s+are\s+now\b",
    r"\byou\s+are\s+no\s+longer\s+(?:bound|restricted|limited|required)",
    r"\bpretend\s+(?:you|to\s+be|that\s+you)\b",
    r"\broleplay\s+as\b",
    r"\bdeveloper\s+mode\b",
    r"\bjailbreak",
    r"\bdo\s+anything\s+now\b",
    r"\breveal\s+(?:your|the)\s+(?:instructions|prompt|rules|system)",
    r"\bwithout\s+(?:any\s+)?(?:restrictions|guardrails|safety\s+checks)\b",
]
_INJECTION = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE | re.MULTILINE)
# Characters that hide words from the patterns: zero-width spaces and joiners.
_INVISIBLE = re.compile("[\u200b-\u200f\u2060\ufeff\u00ad]")


def _normalise_for_injection(text: str) -> str:
    import unicodedata

    text = unicodedata.normalize("NFKC", _INVISIBLE.sub("", text))
    return re.sub(r"\s+", " ", text)


@dataclass
class GuardResult:
    ok: bool
    reason: str = ""
    detail: str = ""


def check_scope_and_injection(text: str, check_injection: bool = True) -> GuardResult:
    stripped = text.strip()
    if not stripped:
        return GuardResult(False, "The request was empty after input guards.",
                           "empty query")
    if len(stripped) > config.MAX_QUERY_CHARS:
        return GuardResult(False, "This request was blocked by an input guard.",
                           "query exceeds maximum length")
    if check_injection and (_INJECTION.search(stripped) or _INJECTION.search(_normalise_for_injection(stripped))):
        return GuardResult(False, "This request was blocked by an input guard.",
                           "instruction-override pattern detected")
    detail = "no scope or injection issues" if check_injection else "injection guard disabled"
    return GuardResult(True, detail=detail)


# --- Rate limiting ---------------------------------------------------------
class RateLimiter:
    """Sliding-window limiter keyed by client id (IP). Each client gets its own
    window, so one abusive caller cannot exhaust the limit, or the LLM quota,
    for everyone else. Adequate for a single-instance demo; a multi-instance
    deployment would use a shared store."""

    def __init__(self, limit_per_minute: int):
        self.limit = limit_per_minute
        self._by_client: dict[str, deque[float]] = {}

    # Buckets are dropped once they fall outside the window, so the dict holds
    # only callers seen in the last minute. It used to keep one deque per
    # distinct client id for the lifetime of the process, which grows without
    # bound on a public instance.
    _SWEEP_EVERY = 256

    def allow(self, client_id: str = "global", now: float | None = None) -> bool:
        now = now if now is not None else time.monotonic()
        cutoff = now - 60.0
        self._since_sweep = getattr(self, "_since_sweep", 0) + 1
        if self._since_sweep >= self._SWEEP_EVERY:
            self._since_sweep = 0
            for stale in [c for c, ev in self._by_client.items() if not ev or ev[-1] < cutoff]:
                if stale != client_id:
                    del self._by_client[stale]
        events = self._by_client.setdefault(client_id, deque())
        while events and events[0] < cutoff:
            events.popleft()
        if len(events) >= self.limit:
            return False
        events.append(now)
        return True


_limiter = RateLimiter(config.RATE_LIMIT_PER_MINUTE)


def check_rate_limit(client_id: str = "global") -> GuardResult:
    if _limiter.allow(client_id):
        return GuardResult(True, detail="within rate limit")
    return GuardResult(
        False,
        "This request was rate limited. Please wait a moment and try again.",
        "rate limit exceeded",
    )

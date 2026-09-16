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
_INJECTION_PATTERNS = [
    r"ignore (all |the )?previous instructions",
    r"ignore (all |the )?above",
    r"disregard (all |the )?(previous|above)",
    r"system prompt",
    r"you are now",
    r"developer mode",
    r"jailbreak",
    r"reveal your (instructions|prompt|rules)",
]
_INJECTION = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)


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
    if check_injection and _INJECTION.search(stripped):
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

    def allow(self, client_id: str = "global", now: float | None = None) -> bool:
        now = now if now is not None else time.monotonic()
        cutoff = now - 60.0
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

"""Typed, immutable response models. Unknown fields from a newer server are ignored;
a missing or wrongly typed field raises InvalidResponseError rather than being guessed."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

from .errors import InvalidResponseError


def _fail(where: str, what: str) -> InvalidResponseError:
    return InvalidResponseError(f"Unexpected response: {where} {what}")


def _obj(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise _fail(where, "should be an object")
    return value


def _str(d: Mapping[str, Any], key: str, where: str, default: Optional[str] = None) -> str:
    if key not in d:
        if default is not None:
            return default
        raise _fail(f"{where}.{key}", "is missing")
    if not isinstance(d[key], str):
        raise _fail(f"{where}.{key}", "should be a string")
    return d[key]


def _opt_str(d: Mapping[str, Any], key: str, where: str) -> Optional[str]:
    v = d.get(key)
    if v is not None and not isinstance(v, str):
        raise _fail(f"{where}.{key}", "should be a string or null")
    return v


def _bool(d: Mapping[str, Any], key: str, where: str, default: Optional[bool] = None) -> bool:
    if key not in d and default is not None:
        return default
    if not isinstance(d.get(key), bool):
        raise _fail(f"{where}.{key}", "should be true or false")
    return d[key]


def _int(d: Mapping[str, Any], key: str, where: str, default: Optional[int] = None) -> int:
    if key not in d and default is not None:
        return default
    v = d.get(key)
    if isinstance(v, bool) or not isinstance(v, int):
        raise _fail(f"{where}.{key}", "should be an integer")
    return v


def _float(d: Mapping[str, Any], key: str, where: str, default: Optional[float] = None) -> float:
    if key not in d and default is not None:
        return default
    v = d.get(key)
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise _fail(f"{where}.{key}", "should be a number")
    return float(v)


def _list(d: Mapping[str, Any], key: str, where: str) -> list:
    v = d.get(key, [])
    if not isinstance(v, list):
        raise _fail(f"{where}.{key}", "should be a list")
    return v


def _strs(d: Mapping[str, Any], key: str, where: str) -> Tuple[str, ...]:
    items = _list(d, key, where)
    if not all(isinstance(i, str) for i in items):
        raise _fail(f"{where}.{key}", "should be a list of strings")
    return tuple(items)


@dataclass(frozen=True)
class RateLimit:
    limit: Optional[int] = None
    remaining: Optional[int] = None
    reset: Optional[int] = None


@dataclass(frozen=True)
class ClaimSupport:
    source_id: str
    start: int
    end: int
    sentence: str

    @classmethod
    def from_dict(cls, data: Any, where: str = "support") -> "ClaimSupport":
        d = _obj(data, where)
        return cls(_str(d, "source_id", where), _int(d, "start", where), _int(d, "end", where),
                   _str(d, "sentence", where))


@dataclass(frozen=True)
class Explanation:
    summary: str = ""
    missing_terms: Tuple[str, ...] = ()
    suggestions: Tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, data: Any, where: str = "explanation") -> "Explanation":
        d = _obj(data, where)
        return cls(_str(d, "summary", where, ""), _strs(d, "missing_terms", where), _strs(d, "suggestions", where))


@dataclass(frozen=True)
class Redaction:
    """What was removed from the text before checking: counts by kind, never the values."""

    kinds: Dict[str, int] = field(default_factory=dict)
    redacted_query: str = ""
    redacted_answer: str = ""

    @classmethod
    def from_dict(cls, data: Any, where: str = "redaction") -> "Redaction":
        d = _obj(data, where)
        kinds = d.get("kinds", {})
        if not isinstance(kinds, dict) or not all(isinstance(k, str) and isinstance(v, int) and not isinstance(v, bool)
                                                  for k, v in kinds.items()):
            raise _fail(f"{where}.kinds", "should map names to counts")
        return cls(dict(kinds), _str(d, "redacted_query", where, ""), _str(d, "redacted_answer", where, ""))


@dataclass(frozen=True)
class Review:
    routed: bool = False
    case_id: Optional[int] = None

    @classmethod
    def from_dict(cls, data: Any, where: str = "review") -> "Review":
        d = _obj(data, where)
        case = d.get("case_id")
        if case is not None and (isinstance(case, bool) or not isinstance(case, int)):
            raise _fail(f"{where}.case_id", "should be an integer or null")
        return cls(_bool(d, "routed", where, False), case)


@dataclass(frozen=True)
class PatientFinding:
    severity: str
    code: str
    medicine: str
    message: str
    rule_id: str = ""
    source: str = ""

    @classmethod
    def from_dict(cls, data: Any, where: str = "patient_findings[]") -> "PatientFinding":
        d = _obj(data, where)
        return cls(_str(d, "severity", where), _str(d, "code", where), _str(d, "medicine", where),
                   _str(d, "message", where), _str(d, "rule_id", where, ""), _str(d, "source", where, ""))

    @property
    def blocks(self) -> bool:
        return self.severity == "block"


@dataclass(frozen=True)
class Patient:
    """The patient a question is about. Leave out what you don't know; there is deliberately no name field."""

    age_years: Optional[float] = None
    sex: Optional[str] = None
    weight_kg: Optional[float] = None
    pregnant: Optional[bool] = None
    breastfeeding: Optional[bool] = None
    egfr: Optional[float] = None
    creatinine_umol_l: Optional[float] = None
    child_pugh: Optional[str] = None
    allergies: Tuple[str, ...] = ()
    medicines: Tuple[str, ...] = ()
    conditions: Tuple[str, ...] = ()
    labs: Mapping[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k in ("age_years", "sex", "weight_kg", "pregnant", "breastfeeding", "egfr", "creatinine_umol_l",
                  "child_pugh"):
            if getattr(self, k) is not None:
                out[k] = getattr(self, k)
        for k in ("allergies", "medicines", "conditions"):
            if getattr(self, k):
                out[k] = list(getattr(self, k))
        if self.labs:
            out["labs"] = dict(self.labs)
        return out


@dataclass(frozen=True)
class VerifiedClaim:
    text: str
    grounded: bool
    source_id: Optional[str] = None
    support: Optional[ClaimSupport] = None
    relevance: float = 0.0
    issues: Tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, data: Any, where: str = "claims[]") -> "VerifiedClaim":
        d = _obj(data, where)
        support = d.get("support")
        return cls(_str(d, "text", where), _bool(d, "grounded", where), _opt_str(d, "source_id", where),
                   ClaimSupport.from_dict(support, f"{where}.support") if support is not None else None,
                   _float(d, "relevance", where, 0.0), _strs(d, "issues", where))


@dataclass(frozen=True)
class DoseIssue:
    claim: int
    value: str
    message: str

    @classmethod
    def from_dict(cls, data: Any, where: str = "dose_issues[]") -> "DoseIssue":
        d = _obj(data, where)
        return cls(_int(d, "claim", where), _str(d, "value", where), _str(d, "message", where))


@dataclass(frozen=True)
class SourceRef:
    id: str
    title: str
    topic: str = ""

    @classmethod
    def from_dict(cls, data: Any, where: str = "sources[]") -> "SourceRef":
        d = _obj(data, where)
        return cls(_str(d, "id", where), _str(d, "title", where), _str(d, "topic", where, ""))


@dataclass(frozen=True)
class VerifyResponse:
    decision: str                       # "pass", "fail" or "review"
    reason_code: str
    claims: Tuple[VerifiedClaim, ...]
    unsupported_claims: Tuple[int, ...]
    dose_issues: Tuple[DoseIssue, ...]
    explanation: Explanation
    redaction: Redaction
    audit_id: str
    latency_ms: int
    sources: Tuple[SourceRef, ...] = ()
    patient_findings: Tuple[PatientFinding, ...] = ()
    review: Review = Review()
    request_id: Optional[str] = None
    rate_limit: Optional[RateLimit] = None

    @property
    def passed(self) -> bool:
        """True only for an explicit "pass". Anything else, including a decision this client doesn't know, is not."""
        return self.decision == "pass"

    @classmethod
    def from_dict(cls, data: Any) -> "VerifyResponse":
        w = "verify response"
        d = _obj(data, w)
        unsupported = _list(d, "unsupported_claims", w)
        if not all(isinstance(i, int) and not isinstance(i, bool) for i in unsupported):
            raise _fail(f"{w}.unsupported_claims", "should be a list of integers")
        return cls(
            decision=_str(d, "decision", w), reason_code=_str(d, "reason_code", w),
            claims=tuple(VerifiedClaim.from_dict(c) for c in _list(d, "claims", w)),
            unsupported_claims=tuple(unsupported),
            dose_issues=tuple(DoseIssue.from_dict(i) for i in _list(d, "dose_issues", w)),
            explanation=Explanation.from_dict(d.get("explanation", {})),
            redaction=Redaction.from_dict(d.get("redaction", {})),
            audit_id=_str(d, "audit_id", w), latency_ms=_int(d, "latency_ms", w),
            sources=tuple(SourceRef.from_dict(s) for s in _list(d, "sources", w)),
            patient_findings=tuple(PatientFinding.from_dict(f) for f in _list(d, "patient_findings", w)),
            review=Review.from_dict(d.get("review", {})),
        )


@dataclass(frozen=True)
class AskClaim:
    text: str
    source_ids: Tuple[str, ...] = ()
    grounded: bool = False
    grounding_score: float = 0.0
    relevance: float = 0.0
    support: Optional[ClaimSupport] = None

    @classmethod
    def from_dict(cls, data: Any, where: str = "claims[]") -> "AskClaim":
        d = _obj(data, where)
        support = d.get("support")
        return cls(_str(d, "text", where), _strs(d, "source_ids", where), _bool(d, "grounded", where, False),
                   _float(d, "grounding_score", where, 0.0), _float(d, "relevance", where, 0.0),
                   ClaimSupport.from_dict(support, f"{where}.support") if support is not None else None)


@dataclass(frozen=True)
class AskSource:
    id: str
    title: str
    snippet: str
    score: float
    topic: str = ""
    section: str = ""
    kind: str = ""
    rank: int = 0
    above_gate: bool = True

    @classmethod
    def from_dict(cls, data: Any, where: str = "sources[]") -> "AskSource":
        d = _obj(data, where)
        return cls(_str(d, "id", where), _str(d, "title", where), _str(d, "snippet", where), _float(d, "score", where),
                   _str(d, "topic", where, ""), _str(d, "section", where, ""), _str(d, "kind", where, ""),
                   _int(d, "rank", where, 0), _bool(d, "above_gate", where, True))


@dataclass(frozen=True)
class AskResponse:
    decision: str                       # "answer" or "refuse"
    outcome: str                        # "answered", "refused" or "not_evaluated"
    answer_text: str
    audit_id: str
    total_ms: int
    llm_used: bool
    refused_reason: Optional[str] = None
    reason_code: Optional[str] = None
    claims: Tuple[AskClaim, ...] = ()
    sources: Tuple[AskSource, ...] = ()
    patient_findings: Tuple[PatientFinding, ...] = ()
    review: Review = Review()
    explanation: Explanation = Explanation()
    redaction: Redaction = Redaction()
    trace: Tuple[Dict[str, Any], ...] = ()
    request_id: Optional[str] = None
    rate_limit: Optional[RateLimit] = None

    @property
    def answered(self) -> bool:
        """True only when the service answered and checked it. A refusal, or an unknown outcome, is not."""
        return self.decision == "answer" and self.outcome == "answered"

    @classmethod
    def from_dict(cls, data: Any) -> "AskResponse":
        w = "ask response"
        d = _obj(data, w)
        trace = _list(d, "trace", w)
        if not all(isinstance(t, dict) for t in trace):
            raise _fail(f"{w}.trace", "should be a list of objects")
        decision = _str(d, "decision", w)
        return cls(
            decision=decision, outcome=_str(d, "outcome", w, "answered" if decision == "answer" else "refused"),
            answer_text=_str(d, "answer_text", w), audit_id=_str(d, "audit_id", w), total_ms=_int(d, "total_ms", w),
            llm_used=_bool(d, "llm_used", w, False), refused_reason=_opt_str(d, "refused_reason", w),
            reason_code=_opt_str(d, "reason_code", w),
            claims=tuple(AskClaim.from_dict(c) for c in _list(d, "claims", w)),
            sources=tuple(AskSource.from_dict(s) for s in _list(d, "sources", w)),
            patient_findings=tuple(PatientFinding.from_dict(f) for f in _list(d, "patient_findings", w)),
            review=Review.from_dict(d.get("review", {})), explanation=Explanation.from_dict(d.get("explanation", {})),
            redaction=Redaction.from_dict(d.get("redaction", {})), trace=tuple(trace),
        )


@dataclass(frozen=True)
class UsageReport:
    group_by: str
    start: str
    end: str
    rows: Tuple[Dict[str, Any], ...]
    total: int
    billable: int
    request_id: Optional[str] = None
    rate_limit: Optional[RateLimit] = None

    @classmethod
    def from_dict(cls, data: Any) -> "UsageReport":
        w = "usage response"
        d = _obj(data, w)
        rows = _list(d, "rows", w)
        if not all(isinstance(r, dict) for r in rows):
            raise _fail(f"{w}.rows", "should be a list of objects")
        return cls(_str(d, "group_by", w), _str(d, "from", w), _str(d, "to", w), tuple(rows), _int(d, "total", w),
                   _int(d, "billable", w))


@dataclass(frozen=True)
class Health:
    status: str
    details: Dict[str, Any] = field(default_factory=dict)
    request_id: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Any) -> "Health":
        d = _obj(data, "health response")
        return cls(_str(d, "status", "health response"), dict(d))

    @property
    def ok(self) -> bool:
        return self.status in ("ok", "healthy")

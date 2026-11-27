"""Pydantic v2 models for the request, response, and the structured output we
require from the LLM."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator, model_validator

from . import config

Status = Literal["pass", "warn", "fail", "skip", "info"]
Decision = Literal["answer", "refuse"]
# What actually happened, which `decision` alone cannot say: a question that was
# evaluated and declined on safety grounds is not the same event as one that was
# never looked at because the caller was rate limited or the service was not
# ready. Both are `decision: refuse`, and an integrator keying on that alone
# would read "we checked and it isn't safe" where the truth is "we never
# checked". Key on this field.
Outcome = Literal["answered", "refused", "not_evaluated"]
# Why a request was refused, as a stable code. `refused_reason` is prose for a
# person and its wording can change; integrators and reports key on this.
ReasonCode = Literal[
    "input_blocked", "rate_limited", "no_relevant_source", "term_not_covered",
    "population_not_covered", "stated_value_unsupported", "form_unsupported",
    "limit_not_stated", "combination_forbidden", "insufficient_context",
    "claim_ungrounded", "dose_unsupported", "patient_unsafe", "unsupported_language", "proposal_unverifiable",
]


class Settings(BaseModel):
    """Per-request tuning. Every field defaults to the configured value, so an
    empty object reproduces the standard pipeline. The frontend tuning panel
    sends overrides here so a tester can watch thresholds and guards change the
    decision live. Guards can be toggled off to demonstrate, visibly, what an
    ungoverned system would have answered."""

    retrieval_min_score: float = Field(default=config.RETRIEVAL_MIN_SCORE, ge=0.0, le=1.0)
    grounding_min: float = Field(default=config.GROUNDING_MIN, ge=0.0, le=1.0)
    top_k: int = Field(default=config.TOP_K, ge=1, le=12)
    temperature: float = Field(default=0.0, ge=0.0, le=1.5)
    use_llm_judge: bool = config.USE_LLM_JUDGE
    enable_pii_redaction: bool = True
    enable_injection_guard: bool = True
    enable_coverage_guard: bool = True
    enable_grounding_guard: bool = True
    enable_dosage_guard: bool = True
    # Bypass the LLM and use the naive extractive generator (stitches together
    # retrieved sentences). Lets the demo show what an ungoverned, naive RAG
    # system returns: a well-aligned model often refuses on its own, so the
    # guards' value is clearest against the naive generator.
    force_extractive: bool = False


LabName = Literal["potassium", "sodium", "inr", "platelets", "alt", "qtc"]


class PatientContext(BaseModel):
    """The patient a question is about. Every field is optional, but a dose
    question for a medicine whose rules need a value (for example kidney
    function) is refused until that value is given. There's deliberately no
    field for names or record numbers."""

    model_config = ConfigDict(extra="forbid")
    age_years: float | None = Field(default=None, ge=0, le=130)
    sex: Literal["female", "male", "other"] | None = None
    weight_kg: float | None = Field(default=None, ge=0.3, le=400)
    pregnant: bool | None = None
    breastfeeding: bool | None = None
    egfr: float | None = Field(default=None, ge=0, le=200, description="mL/min/1.73 m2")
    creatinine_umol_l: float | None = Field(default=None, gt=0, le=3000)
    child_pugh: Literal["A", "B", "C"] | None = None
    allergies: list[str] = Field(default_factory=list, max_length=30)
    medicines: list[str] = Field(default_factory=list, max_length=40)
    conditions: list[str] = Field(default_factory=list, max_length=40)
    labs: dict[LabName, float] = Field(default_factory=dict)

    @field_validator("allergies", "medicines", "conditions")
    @classmethod
    def clean_list(cls, values: list[str]) -> list[str]:
        cleaned = []
        for value in values:
            _encodable(value, "each entry")
            value = " ".join(str(value).split())
            if len(value) > 80:
                raise ValueError("each entry can be up to 80 characters")
            if value and value.lower() not in (v.lower() for v in cleaned):
                cleaned.append(value)
        return cleaned

    @model_validator(mode="after")
    def consistent(self):
        if self.sex == "male" and (self.pregnant or self.breastfeeding):
            raise ValueError("pregnancy and breastfeeding can't be set for a male patient")
        return self

    def is_empty(self) -> bool:
        return self == PatientContext()


class PatientFinding(BaseModel):
    severity: Literal["block", "warn", "info"]
    code: str
    medicine: str
    message: str
    rule_id: str = ""
    source: str = ""
    rule: str = ""
    params: dict = {}


class AskRequest(BaseModel):
    query: str = Field(min_length=1, max_length=400)
    settings: Settings | None = None
    patient: PatientContext | None = None


class Source(BaseModel):
    id: str
    title: str
    snippet: str
    score: float
    topic: str = ""
    section: str = ""
    kind: str = "condition"
    rank: int = 0
    above_gate: bool = True


class ClaimSupport(BaseModel):
    """Where in a cited passage a claim was lifted from (character offsets)."""

    source_id: str
    start: int
    end: int
    sentence: str


class Claim(BaseModel):
    text: str
    source_ids: list[str] = []
    grounded: bool = False
    grounding_score: float = 0.0
    relevance: float = 0.0
    support: ClaimSupport | None = None


class TraceStep(BaseModel):
    name: str
    status: Status
    detail: str
    ms: int
    # Plain-English description of what this stage does, and optional
    # stage-specific structured data, both surfaced in the clickable trace so a
    # viewer can see exactly what happened behind the scenes.
    explain: str = ""
    data: dict | None = None


class Review(BaseModel):
    """Whether a refusal was really put in front of a reviewer, and if not, why."""

    routed: bool = False
    case_id: int | None = None
    note: str = ""


class Explanation(BaseModel):
    summary: str = ""
    missing_terms: list[str] = []
    suggestions: list[str] = []


class Redaction(BaseModel):
    """What the PII layer removed from the question: counts by kind, and the
    question with placeholders. Never the removed values."""

    kinds: dict[str, int] = {}
    redacted_query: str = ""


class ProposalDimension(BaseModel):
    """One part of a proposed regimen and how it compares with the source."""
    name: Literal["dose", "frequency", "duration", "route", "max_daily", "timing", "drug", "proposal"]
    status: Literal["match", "mismatch", "unverifiable"]
    proposed: str | None = None
    documented: str | None = None
    source_id: str | None = None
    reason: str | None = None
    code: str | None = None


class ProposalCheck(BaseModel):
    """What a question that proposes a regimen was found to say, compared with
    the source: `match` (every part is the source's), `mismatch` (a part
    differs) or `unverifiable` (a part cannot be read or checked)."""
    status: Literal["match", "mismatch", "unverifiable"]
    dimensions: list[ProposalDimension] = []
    drug: str | None = None


class AskResponse(BaseModel):
    decision: Decision
    # `decision` collapses "refused after checking" and "never checked" into
    # "refuse", for callers written against the original two values. `outcome`
    # separates them and is the field to key on.
    outcome: Outcome = "refused"
    answer_text: str
    refused_reason: str | None = None
    claims: list[Claim] = []
    sources: list[Source] = []
    trace: list[TraceStep] = []
    audit_id: str
    total_ms: int
    llm_used: bool
    patient_findings: list[PatientFinding] = []
    reason_code: ReasonCode | None = None
    review: Review = Review()
    explanation: Explanation = Explanation()
    redaction: Redaction = Redaction()
    proposal_check: ProposalCheck | None = None
    reason_params: dict = {}


# --- The structured output the LLM is required to return -------------------
class LLMClaim(BaseModel):
    text: str
    source_ids: list[str]
    # Filled by the extractive generator only: how relevant the sentence is to
    # the question, and where in the cited passage it sits.
    relevance: float = 0.0
    support: ClaimSupport | None = None


class LLMAnswer(BaseModel):
    insufficient_context: bool
    claims: list[LLMClaim] = []


# --- /api/v1 -----------------------------------------------------------------

VerifyDecision = Literal["pass", "fail", "review"]
VerifyReason = Literal[
    "verified", "input_blocked", "claim_ungrounded", "dose_unsupported", "patient_unsafe", "citation_invalid",
    "answer_off_topic", "claim_unverified", "no_checkable_claims", "no_relevant_source",
]


class TooLong(ValueError):
    """Text over its limit. The API answers 413 for it rather than 422."""


def _encodable(value: str, what: str) -> None:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        raise ValueError(f"{what} is not valid Unicode text") from None


def _clean_text(value: str, what: str) -> str:
    _encodable(value, what)
    if any(ord(c) < 32 and c not in "\t\n\r" or ord(c) == 127 for c in value):
        raise ValueError(f"{what} contains control characters")
    if not value.strip():
        raise ValueError(f"{what} is empty")
    return value


class V1AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: StrictStr
    site: StrictStr | None = Field(default=None, pattern=r"^[a-z0-9][a-z0-9-]{1,62}$")
    patient: PatientContext | None = None

    @field_validator("question")
    @classmethod
    def _question(cls, value: str) -> str:
        value = _clean_text(value, "question")
        if len(value.strip()) > config.MAX_QUERY_CHARS:
            raise TooLong(f"question is longer than {config.MAX_QUERY_CHARS} characters")
        return value


class V1VerifyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: StrictStr
    answer: StrictStr
    site: StrictStr | None = Field(default=None, pattern=r"^[a-z0-9][a-z0-9-]{1,62}$")
    sources: list[StrictStr] | None = Field(default=None, max_length=config.API_MAX_SOURCES)
    patient: PatientContext | None = None

    @field_validator("question")
    @classmethod
    def _question(cls, value: str) -> str:
        value = _clean_text(value, "question")
        if len(value.strip()) > config.MAX_QUERY_CHARS:
            raise TooLong(f"question is longer than {config.MAX_QUERY_CHARS} characters")
        return value

    @field_validator("answer")
    @classmethod
    def _answer(cls, value: str) -> str:
        value = _clean_text(value, "answer")
        if len(value) > config.API_MAX_ANSWER_CHARS:
            raise TooLong(f"answer is longer than {config.API_MAX_ANSWER_CHARS} characters")
        return value

    @field_validator("sources")
    @classmethod
    def _sources(cls, values: list[str] | None) -> list[str] | None:
        if values is None:
            return None
        cleaned = list(dict.fromkeys(v.strip() for v in values))
        if not cleaned or any(not v or len(v) > 64 or any(ord(c) < 33 or ord(c) == 127 for c in v) for v in cleaned):
            raise ValueError("sources must be a non-empty list of source ids")
        return cleaned


class VerifiedClaim(BaseModel):
    text: str
    grounded: bool
    source_id: str | None = None
    support: ClaimSupport | None = None
    relevance: float = 0.0
    issues: list[str] = []


class DoseIssue(BaseModel):
    claim: int
    value: str
    message: str


class VerifyRedaction(Redaction):
    redacted_answer: str = ""


class VerifySource(BaseModel):
    id: str
    title: str
    topic: str = ""


class VerifyResponse(BaseModel):
    decision: VerifyDecision
    reason_code: VerifyReason
    claims: list[VerifiedClaim] = []
    unsupported_claims: list[int] = []
    dose_issues: list[DoseIssue] = []
    explanation: Explanation = Explanation()
    redaction: VerifyRedaction = VerifyRedaction()
    audit_id: str
    latency_ms: int
    sources: list[VerifySource] = []
    patient_findings: list[PatientFinding] = []
    review: Review = Review()

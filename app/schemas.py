"""Pydantic v2 models for the request, response, and the structured output we
require from the LLM."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from . import config

Status = Literal["pass", "warn", "fail", "skip", "info"]
Decision = Literal["answer", "refuse"]


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


class Claim(BaseModel):
    text: str
    source_ids: list[str] = []
    grounded: bool = False
    grounding_score: float = 0.0


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


class AskResponse(BaseModel):
    decision: Decision
    answer_text: str
    refused_reason: str | None = None
    claims: list[Claim] = []
    sources: list[Source] = []
    trace: list[TraceStep] = []
    audit_id: str
    total_ms: int
    llm_used: bool
    patient_findings: list[PatientFinding] = []


# --- The structured output the LLM is required to return -------------------
class LLMClaim(BaseModel):
    text: str
    source_ids: list[str]


class LLMAnswer(BaseModel):
    insufficient_context: bool
    claims: list[LLMClaim] = []

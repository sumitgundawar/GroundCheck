"""Pydantic v2 models for the request, response, and the structured output we
require from the LLM."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

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


class AskRequest(BaseModel):
    query: str = Field(min_length=1, max_length=400)
    settings: Settings | None = None


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


# --- The structured output the LLM is required to return -------------------
class LLMClaim(BaseModel):
    text: str
    source_ids: list[str]


class LLMAnswer(BaseModel):
    insufficient_context: bool
    claims: list[LLMClaim] = []

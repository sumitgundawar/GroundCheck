"""LLM integration plus the extractive fallback.

Two providers can draft answers: a local open-source model through Ollama
(app/local_ai.py), or a cloud model through any OpenAI-compatible API (Groq by
default). A local model, when one is selected and available, takes precedence.

Every live call is wrapped so that any error, timeout, or malformed output
silently falls back to extractive generation. The user never sees an error and
the decision logic never depends on a live call succeeding."""

from __future__ import annotations

import threading
from contextlib import contextmanager

import json
import re

from . import config, local_ai, retrieval
from .schemas import LLMAnswer, LLMClaim

# Whether a cloud model is configured. Fixed at startup, since it depends only
# on the environment.
LLM_AVAILABLE = config.llm_configured()


_thread = threading.local()


@contextmanager
def extractive_only():
    """Use extractive answers in this thread, whatever model is configured:
    release checks test the retrieval and guards, deterministically."""
    previous = getattr(_thread, "extractive", False)
    _thread.extractive = True
    try:
        yield
    finally:
        _thread.extractive = previous


def active_provider() -> dict | None:
    """The model that will draft the next answer, as
    {"kind": "local" | "cloud", "model": name}, or None for extractive mode."""
    if config.FORCE_EXTRACTIVE or getattr(_thread, "extractive", False):
        return None
    local = local_ai.active_model()
    if local:
        return {"kind": "local", "model": local}
    if LLM_AVAILABLE:
        return {"kind": "cloud", "model": config.GEN_MODEL}
    return None


def llm_available() -> bool:
    return active_provider() is not None


def _complete_json(system: str, user: str, temperature: float,
                   provider: dict, judge: bool = False) -> str:
    """One JSON completion from the given provider. Raises on failure."""
    if provider["kind"] == "local":
        if not judge and system == GEN_SYSTEM_PROMPT:
            system = LOCAL_GEN_SYSTEM_PROMPT
        content = local_ai.chat_json(system, user, temperature, model=provider["model"],
                                     schema=JUDGE_SCHEMA if judge else GEN_SCHEMA)
        if content is None:
            raise RuntimeError("local model call failed")
        return content
    client = _get_client()
    response = client.chat.completions.create(
        model=config.JUDGE_MODEL if judge else config.GEN_MODEL,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=temperature,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content or ""

_client = None


def _get_client():
    global _client
    if _client is None:
        from openai import OpenAI

        _client = OpenAI(
            api_key=config.GROQ_API_KEY,
            base_url=config.GROQ_BASE_URL,
            timeout=config.LLM_TIMEOUT_SECONDS,
        )
    return _client


GEN_SYSTEM_PROMPT = """You are GroundCheckHealth, a clinical information assistant operating under strict grounding rules.

Rules:
- Answer ONLY using the numbered SOURCES provided. Do not use outside knowledge.
- Every claim you make must cite the source id or ids it comes from.
- Never state a dose, number, or value that does not appear in the sources.
- If the sources do not contain enough information to answer, set insufficient_context to true and return no claims.
- Do not add caveats, greetings, or commentary.

Return ONLY valid JSON matching this schema:
{"insufficient_context": boolean, "claims": [{"text": string, "source_ids": [string]}]}"""

# Small local models refuse too readily with the prompt above ("insufficient
# context" when a source states the answer outright). Explicit steps and a
# JSON schema keep them on task. Grounding and dosage checks apply unchanged.
LOCAL_GEN_SYSTEM_PROMPT = """You answer clinical questions using ONLY the numbered SOURCES.

Steps:
1. Find every sentence in the SOURCES that answers the QUESTION.
2. For each, write one short claim that restates it, and cite the id of the source it came from, exactly as shown in square brackets, for example "CALO-001".
3. Copy every number and unit exactly as written in the source.
4. Set insufficient_context to true ONLY if no source sentence answers the question. If any source answers it, insufficient_context is false.

Do not use outside knowledge. Do not add advice or caveats."""

GEN_SCHEMA = {
    "type": "object",
    "properties": {
        "insufficient_context": {"type": "boolean"},
        "claims": {
            "type": "array",
            "maxItems": 6,
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "source_ids": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
                },
                "required": ["text", "source_ids"],
            },
        },
    },
    "required": ["insufficient_context", "claims"],
}

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {"supported": {"type": "boolean"}, "reason": {"type": "string"}},
    "required": ["supported", "reason"],
}

JUDGE_SYSTEM_PROMPT = """You verify grounding. Given a CLAIM and its CITED SOURCE text, decide if every statement in the claim is directly supported by the source.
Return ONLY JSON: {"supported": boolean, "reason": string}.
Be strict. If the claim adds any detail not present in the source, supported is false."""


def _format_sources(sources: list[dict]) -> str:
    lines = []
    for record in sources:
        lines.append(f"[{record['id']}] {record['title']}\n{record['text']}")
    return "\n\n".join(lines)


def _parse_llm_answer(content: str) -> LLMAnswer:
    """Parse and validate the model's JSON. Raises on anything unexpected so the
    caller can retry once and then fall back."""
    data = json.loads(content)
    return LLMAnswer.model_validate(data)


def generate_llm(query: str, sources: list[dict],
                 temperature: float = 0.0) -> LLMAnswer | None:
    """Call the drafting model with one retry. Returns None on any failure,
    which signals the pipeline to use the extractive fallback."""
    provider = active_provider()
    if provider is None:
        return None

    user_message = (
        f"QUESTION:\n{query}\n\nSOURCES:\n{_format_sources(sources)}\n\n"
        "Return only the JSON object."
    )
    for attempt in range(2):
        try:
            content = _complete_json(GEN_SYSTEM_PROMPT, user_message, temperature, provider)
            return _parse_llm_answer(content)
        except Exception:
            # On the first failure, retry once with a firmer nudge; on the
            # second, give up and let the caller fall back extractively.
            if attempt == 0:
                user_message += "\n\nReturn ONLY the JSON object, nothing else."
                continue
            return None
    return None


def judge_claim(claim_text: str, source_text: str,
                use_judge: bool | None = None) -> dict | None:
    """Optional corroboration of a single claim. Never authoritative."""
    enabled = config.USE_LLM_JUDGE if use_judge is None else use_judge
    provider = active_provider() if enabled else None
    if provider is None:
        return None
    user_message = f"CLAIM:\n{claim_text}\n\nCITED SOURCE:\n{source_text}"
    try:
        content = _complete_json(JUDGE_SYSTEM_PROMPT, user_message, 0.0, provider, judge=True)
        data = json.loads(content)
        return {"supported": bool(data.get("supported")), "reason": str(data.get("reason", ""))}
    except Exception:
        return None


def generate_extractive(query: str, scored_sources: list[tuple[dict, float]],
                        min_score: float | None = None) -> LLMAnswer:
    """Deterministic fallback. One claim per retrieved passage above the
    retrieval threshold, using that passage's most query-relevant sentence.
    This path goes through the same grounding and dosage guards."""
    threshold = config.RETRIEVAL_MIN_SCORE if min_score is None else min_score

    # When the question names drugs or conditions, keep only passages about
    # them or that mention them. Otherwise nearby passages about other drugs
    # would be stitched into the answer.
    named = retrieval.topics_mentioned(query)
    if named:
        on_topic = [
            (record, score) for record, score in scored_sources
            if record.get("topic", "").lower() in named.values()
            or any(re.search(rf"\b{re.escape(word)}\b", record["text"], re.IGNORECASE) for word in named)
        ]
        scored_sources = on_topic or scored_sources

    kept = [record for record, score in scored_sources if score >= threshold]
    sentences = retrieval.best_sentences(query, [record["text"] for record in kept])
    claims = [LLMClaim(text=sentence, source_ids=[record["id"]])
              for record, sentence in zip(kept, sentences)]
    return LLMAnswer(insufficient_context=not claims, claims=claims)

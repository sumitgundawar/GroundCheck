"""LLM integration (Groq, OpenAI-compatible) plus the extractive fallback.

Every live call is wrapped so that any error, timeout, or malformed output
silently falls back to extractive generation. The user never sees an error and
the decision logic never depends on a live call succeeding."""

from __future__ import annotations

import json

from . import config, retrieval
from .schemas import LLMAnswer, LLMClaim

LLM_AVAILABLE = config.llm_configured()

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


GEN_SYSTEM_PROMPT = """You are GroundCheck, a clinical information assistant operating under strict grounding rules.

Rules:
- Answer ONLY using the numbered SOURCES provided. Do not use outside knowledge.
- Every claim you make must cite the source id or ids it comes from.
- Never state a dose, number, or value that does not appear in the sources.
- If the sources do not contain enough information to answer, set insufficient_context to true and return no claims.
- Do not add caveats, greetings, or commentary.

Return ONLY valid JSON matching this schema:
{"insufficient_context": boolean, "claims": [{"text": string, "source_ids": [string]}]}"""

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


def generate_llm(query: str, sources: list[dict]) -> LLMAnswer | None:
    """Call the generation model with one retry. Returns None on any failure,
    which signals the pipeline to use the extractive fallback."""
    if not LLM_AVAILABLE:
        return None

    user_message = (
        f"QUESTION:\n{query}\n\nSOURCES:\n{_format_sources(sources)}\n\n"
        "Return only the JSON object."
    )
    for attempt in range(2):
        try:
            client = _get_client()
            response = client.chat.completions.create(
                model=config.GEN_MODEL,
                messages=[
                    {"role": "system", "content": GEN_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or ""
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
    if not (LLM_AVAILABLE and enabled):
        return None
    user_message = f"CLAIM:\n{claim_text}\n\nCITED SOURCE:\n{source_text}"
    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=config.JUDGE_MODEL,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or ""
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
    claims: list[LLMClaim] = []
    for record, score in scored_sources:
        if score < threshold:
            continue
        sentence = retrieval.best_sentence(query, record["text"])
        claims.append(LLMClaim(text=sentence, source_ids=[record["id"]]))
    return LLMAnswer(insufficient_context=not claims, claims=claims)

"""Pipeline orchestration: retrieve -> gate -> generate -> ground -> guard ->
decide -> audit. Pure, deterministic control flow. Each stage appends a
TraceStep with a status, a one-line detail, and its duration in milliseconds.

The answer-vs-refuse decision never depends on a live LLM call. If no key is set
or any call fails, the extractive fallback keeps the whole pipeline working."""

from __future__ import annotations

import time

from . import audit, config, guards_input, guards_output, llm, retrieval
from .schemas import AskResponse, Claim, Settings, Source, TraceStep

# The full ordered set of stage names, used to render skipped stages after an
# early refusal so the trace always shows the complete instrument.
_STAGE_ORDER = [
    "pii redaction",
    "scope check",
    "rate limit",
    "retrieve",
    "retrieval gate",
    "source coverage",
    "generate",
    "schema validate",
    "grounding check",
    "dosage guard",
    "decision",
]


class _Timer:
    def __init__(self) -> None:
        self.start = time.perf_counter()
        self.mark = self.start

    def lap_ms(self) -> int:
        now = time.perf_counter()
        ms = int(round((now - self.mark) * 1000))
        self.mark = now
        return ms

    def total_ms(self) -> int:
        return int(round((time.perf_counter() - self.start) * 1000))


def _skip_after(trace: list[TraceStep], last_done: str) -> None:
    """Append skip steps for every stage that comes after `last_done`."""
    idx = _STAGE_ORDER.index(last_done)
    for name in _STAGE_ORDER[idx + 1:]:
        if name == "decision":
            continue
        trace.append(TraceStep(name=name, status="skip", detail="not reached", ms=0))


def _finish(
    *,
    decision: str,
    answer_text: str,
    refused_reason: str | None,
    claims: list[Claim],
    sources: list[Source],
    trace: list[TraceStep],
    llm_used: bool,
    timer: _Timer,
    extras: dict,
) -> AskResponse:
    audit_id = audit.store.new_id()
    # Final decision step.
    if decision == "answer":
        trace.append(TraceStep(name="decision", status="pass",
                               detail="ANSWER, grounded in cited sources",
                               ms=timer.lap_ms()))
    else:
        trace.append(TraceStep(name="decision", status="fail",
                               detail=f"REFUSE, {refused_reason}",
                               ms=timer.lap_ms()))
    # Fill in the plain-English description for every stage from the map, unless
    # a stage already set its own.
    for step in trace:
        if not step.explain:
            step.explain = STAGE_EXPLAIN.get(step.name, "")
    response = AskResponse(
        decision=decision,
        answer_text=answer_text,
        refused_reason=refused_reason,
        claims=claims,
        sources=sources,
        trace=trace,
        audit_id=audit_id,
        total_ms=timer.total_ms(),
        llm_used=llm_used,
    )
    audit.store.save(audit_id, response, extras)
    return response


REFUSAL_PREFIX = "I cannot answer this safely. "
ROUTED = "The question has been routed for review."

# Plain-English description of each stage, shown when a viewer expands a trace
# step. These make the pipeline self-documenting on the frontend.
STAGE_EXPLAIN = {
    "pii redaction":
        "Before anything is logged, embedded, or sent to a model, the query is "
        "scanned for emails and long digit sequences, which are replaced with "
        "[redacted]. Illustrative, deliberately simple.",
    "scope check":
        "Rejects empty or over-long queries and obvious prompt-injection "
        "patterns such as 'ignore previous instructions'. A blocked query "
        "refuses here and goes no further.",
    "rate limit":
        "A simple in-memory limiter caps requests per minute per process, so a "
        "burst of traffic cannot overwhelm the service.",
    "retrieve":
        "The query is embedded with a sentence-transformer and compared against "
        "every document vector in the FAISS index. The top-k most similar "
        "passages are returned with cosine similarity scores.",
    "retrieval gate":
        "If even the best passage scores below the retrieval threshold, the "
        "system refuses now, before any text is generated. The cheapest refusal "
        "happens before a model is ever called.",
    "source coverage":
        "A deterministic check that the contentful terms in the question (drug "
        "and condition names, qualifiers like 'children') actually appear in the "
        "retrieved sources. Catches questions about things the corpus never "
        "mentions, before generation.",
    "generate":
        "The question and the retrieved sources are sent to the LLM, which must "
        "return structured JSON: a list of claims, each citing the source ids it "
        "came from. With no API key, an extractive fallback lifts the most "
        "relevant sentence from each passage instead. Either way the output is "
        "the same shape and passes through the same guards.",
    "schema validate":
        "The model output is parsed and validated against a strict Pydantic "
        "schema. Malformed output is retried once, then falls back to "
        "extractive. Free-text answers are never accepted.",
    "grounding check":
        "For every claim, a deterministic embedding similarity is computed "
        "between the claim and its cited source. The claim must cite a real "
        "source and clear the grounding threshold. When a live LLM is "
        "configured, a second, cheaper model acts as a judge and is asked, "
        "strictly, whether the source supports the claim. The deterministic "
        "result is authoritative; the judge is recorded as corroboration so the "
        "demo stays reproducible.",
    "dosage guard":
        "The showpiece check, and deterministic: every value with a clinical "
        "unit in the answer (for example '15 mg') must appear verbatim in a "
        "retrieved source, or the system refuses and names the offending value. "
        "The cheapest check catches the most dangerous mistake.",
    "decision":
        "The gate. Refuse if any input guard blocked, the retrieval gate failed, "
        "coverage failed, the model had insufficient context, any claim is "
        "ungrounded, or the dosage guard failed. Otherwise answer, assembling "
        "the grounded claims into prose with bracketed citations.",
}


def run(raw_query: str, settings: "Settings | None" = None,
        client_id: str = "global") -> AskResponse:
    # Effective settings: an explicit object, or the configured defaults.
    cfg = settings or Settings()
    timer = _Timer()
    trace: list[TraceStep] = []
    extras: dict = {"raw_query": raw_query, "settings": cfg.model_dump()}

    # --- 1. Input guards ---------------------------------------------------
    if cfg.enable_pii_redaction:
        redacted, did_redact = guards_input.redact_pii(raw_query)
        pii_status = "warn" if did_redact else "pass"
        pii_detail = "redacted sensitive tokens" if did_redact else "no PII detected"
    else:
        redacted, did_redact = raw_query, False
        pii_status, pii_detail = "skip", "guard disabled"
    extras["redacted_query"] = redacted
    trace.append(TraceStep(name="pii redaction", status=pii_status,
                           detail=pii_detail, ms=timer.lap_ms()))

    scope = guards_input.check_scope_and_injection(redacted, cfg.enable_injection_guard)
    trace.append(TraceStep(
        name="scope check",
        status="pass" if scope.ok else "fail",
        detail=scope.detail,
        ms=timer.lap_ms(),
    ))
    if not scope.ok:
        _skip_after(trace, "scope check")
        return _finish(decision="refuse",
                       answer_text=REFUSAL_PREFIX + scope.reason + " " + ROUTED,
                       refused_reason=scope.reason, claims=[], sources=[],
                       trace=trace, llm_used=False, timer=timer, extras=extras)

    rate = guards_input.check_rate_limit(client_id)
    trace.append(TraceStep(
        name="rate limit",
        status="pass" if rate.ok else "fail",
        detail=rate.detail,
        ms=timer.lap_ms(),
    ))
    if not rate.ok:
        _skip_after(trace, "rate limit")
        return _finish(decision="refuse",
                       answer_text=REFUSAL_PREFIX + rate.reason,
                       refused_reason=rate.reason, claims=[], sources=[],
                       trace=trace, llm_used=False, timer=timer, extras=extras)

    query = redacted.strip()

    # --- 2. Retrieve -------------------------------------------------------
    results = retrieval.search(query, cfg.top_k)
    # Hybrid results are in fused-rank order, so take the best cosine score
    # explicitly rather than the first result's.
    top_score = max((s for _, s in results), default=0.0)
    source_records = [r for r, _ in results]
    sources = [
        Source(
            id=r["id"], title=r["title"], snippet=r["text"], score=round(s, 4),
            topic=r.get("topic", ""), section=r.get("section", ""),
            kind=r.get("kind", "condition"), rank=i + 1,
            above_gate=s >= cfg.retrieval_min_score,
        )
        for i, (r, s) in enumerate(results)
    ]
    extras["retrieved"] = [{"id": r["id"], "score": s} for r, s in results]
    trace.append(TraceStep(
        name="retrieve",
        status="pass",
        detail=(f"top-{len(results)} from corpus "
                f"({'keyword + embedding' if config.HYBRID_RETRIEVAL else 'embedding'}), "
                f"best score {top_score:.2f}"),
        ms=timer.lap_ms(),
        data={"results": [
            {"id": r["id"], "title": r["title"], "score": round(s, 4),
             "kind": r.get("kind", ""), "section": r.get("section", "")}
            for r, s in results
        ]},
    ))

    # --- 3. Retrieval gate -------------------------------------------------
    gate_data = {"best_score": round(top_score, 4),
                 "threshold": cfg.retrieval_min_score,
                 "cleared": top_score >= cfg.retrieval_min_score}
    if top_score < cfg.retrieval_min_score:
        reason = "no sufficiently relevant source was found"
        trace.append(TraceStep(
            name="retrieval gate", status="fail",
            detail=f"best score {top_score:.2f} below threshold "
                   f"{cfg.retrieval_min_score:.2f}",
            ms=timer.lap_ms(), data=gate_data,
        ))
        _skip_after(trace, "retrieval gate")
        return _finish(decision="refuse",
                       answer_text=REFUSAL_PREFIX
                       + "No sufficiently relevant source was found. " + ROUTED,
                       refused_reason=reason, claims=[], sources=sources,
                       trace=trace, llm_used=False, timer=timer, extras=extras)
    trace.append(TraceStep(
        name="retrieval gate", status="pass",
        detail=f"best score {top_score:.2f} clears threshold "
               f"{cfg.retrieval_min_score:.2f}",
        ms=timer.lap_ms(), data=gate_data,
    ))

    # --- 4. Source coverage -----------------------------------------------
    if not cfg.enable_coverage_guard:
        covered, cover_detail = True, "guard disabled"
        trace.append(TraceStep(name="source coverage", status="skip",
                               detail="guard disabled", ms=timer.lap_ms()))
    else:
        covered, cover_detail = guards_output.coverage_check(query, source_records)
        trace.append(TraceStep(
            name="source coverage",
            status="pass" if covered else "fail",
            detail=cover_detail,
            data=guards_output.coverage_report(query, source_records),
            ms=timer.lap_ms(),
        ))
    if not covered:
        reason = cover_detail
        _skip_after(trace, "source coverage")
        return _finish(decision="refuse",
                       answer_text=REFUSAL_PREFIX
                       + "The question refers to something the trusted sources "
                       "do not cover. " + ROUTED,
                       refused_reason=reason, claims=[], sources=sources,
                       trace=trace, llm_used=False, timer=timer, extras=extras)

    # --- 5. Generate -------------------------------------------------------
    llm_answer = None if cfg.force_extractive else llm.generate_llm(
        query, source_records, cfg.temperature)
    llm_used = llm_answer is not None
    if not llm_used:
        llm_answer = llm.generate_extractive(query, results, cfg.retrieval_min_score)
    extras["llm_raw"] = llm_answer.model_dump()
    trace.append(TraceStep(
        name="generate",
        status="pass" if llm_used else "info",
        detail=("llm produced cited claims" if llm_used
                else ("naive extractive, model bypassed" if cfg.force_extractive
                      else "extractive fallback")),
        ms=timer.lap_ms(),
    ))

    # --- 6. Schema validate (already validated on parse) -------------------
    trace.append(TraceStep(
        name="schema validate", status="pass",
        detail=f"{len(llm_answer.claims)} claim(s) conform to schema",
        ms=timer.lap_ms(),
    ))

    if llm_answer.insufficient_context or not llm_answer.claims:
        reason = "the sources do not contain enough information to answer safely"
        trace.append(TraceStep(name="grounding check", status="skip",
                               detail="no claims to check", ms=0))
        trace.append(TraceStep(name="dosage guard", status="skip",
                               detail="no answer to check", ms=0))
        return _finish(decision="refuse",
                       answer_text=REFUSAL_PREFIX
                       + "The sources do not contain enough information to "
                       "answer safely. " + ROUTED,
                       refused_reason=reason, claims=[], sources=sources,
                       trace=trace, llm_used=llm_used, timer=timer, extras=extras)

    # --- 7. Grounding check ------------------------------------------------
    claims: list[Claim] = []
    ungrounded: list[str] = []
    judge_supported = 0
    judge_total = 0
    grounding_rows: list[dict] = []
    for llm_claim in llm_answer.claims:
        grounded, score = guards_output.check_claim_grounded(
            llm_claim.text, llm_claim.source_ids, cfg.grounding_min
        )
        # Optional LLM corroboration (never authoritative).
        judge_verdict = None
        for sid in llm_claim.source_ids:
            src_text = retrieval.corpus_text_for(sid)
            if src_text:
                verdict = llm.judge_claim(llm_claim.text, src_text, cfg.use_llm_judge)
                if verdict is not None:
                    judge_total += 1
                    judge_supported += 1 if verdict["supported"] else 0
                    judge_verdict = verdict
                break
        claims.append(Claim(
            text=llm_claim.text,
            source_ids=llm_claim.source_ids,
            grounded=grounded,
            grounding_score=round(score, 4),
        ))
        grounding_rows.append({
            "claim": llm_claim.text,
            "source_ids": llm_claim.source_ids,
            "deterministic_score": round(score, 4),
            "threshold": cfg.grounding_min,
            "grounded": grounded,
            "judge": judge_verdict,  # {supported, reason} or null
        })
        if not grounded:
            ungrounded.append(llm_claim.text)

    judge_note = (f", judge {judge_supported}/{judge_total} supported"
                  if judge_total else "")
    grounding_data = {
        "claims": grounding_rows,
        "method": ("deterministic embedding similarity, authoritative; "
                   "LLM judge as corroboration" if judge_total
                   else "deterministic embedding similarity"),
        "judge_model": config.JUDGE_MODEL if judge_total else None,
        "judge_summary": (f"{judge_supported}/{judge_total} supported"
                          if judge_total else "judge not run"),
    }
    if not cfg.enable_grounding_guard:
        # Guard disabled: still compute scores for display, but do not let an
        # ungrounded claim block the answer. Makes the guard's effect visible.
        ungrounded = []
        trace.append(TraceStep(name="grounding check", status="skip",
                               detail="guard disabled" + judge_note,
                               ms=timer.lap_ms(), data=grounding_data))
    elif ungrounded:
        trace.append(TraceStep(
            name="grounding check", status="fail",
            detail=f"{len(ungrounded)} claim(s) below grounding threshold "
                   f"{cfg.grounding_min:.2f}" + judge_note,
            ms=timer.lap_ms(), data=grounding_data,
        ))
    else:
        trace.append(TraceStep(
            name="grounding check", status="pass",
            detail=f"all {len(claims)} claim(s) grounded in cited sources"
                   + judge_note,
            ms=timer.lap_ms(), data=grounding_data,
        ))

    # --- 8. Dosage guard ---------------------------------------------------
    answer_body = " ".join(c.text for c in claims)
    detected_values = guards_output.extract_values(answer_body)
    if not cfg.enable_dosage_guard:
        dose_ok, dose_detail, dose_checked = True, "guard disabled", []
        trace.append(TraceStep(name="dosage guard", status="skip",
                               detail="guard disabled", ms=timer.lap_ms(),
                               data={"detected_values": detected_values,
                                     "verified": [], "enabled": False}))
    else:
        dose_ok, dose_detail, dose_checked = guards_output.dosage_guard(
            answer_body, source_records)
        trace.append(TraceStep(name="dosage guard",
                               status="pass" if dose_ok else "fail",
                               detail=dose_detail, ms=timer.lap_ms(),
                               data={"detected_values": detected_values,
                                     "verified": dose_checked,
                                     "rule": "every value-with-unit must appear "
                                             "verbatim in a retrieved source",
                                     "ok": dose_ok}))

    # --- 9. Decision gate --------------------------------------------------
    if ungrounded:
        reason = f"a claim could not be grounded: \"{ungrounded[0]}\""
        return _finish(decision="refuse",
                       answer_text=REFUSAL_PREFIX
                       + "A statement could not be grounded in the sources. "
                       + ROUTED,
                       refused_reason=reason, claims=claims, sources=sources,
                       trace=trace, llm_used=llm_used, timer=timer, extras=extras)
    if not dose_ok:
        return _finish(decision="refuse",
                       answer_text=REFUSAL_PREFIX + dose_detail.capitalize()
                       + ". " + ROUTED,
                       refused_reason=dose_detail, claims=claims, sources=sources,
                       trace=trace, llm_used=llm_used, timer=timer, extras=extras)

    # --- Answer ------------------------------------------------------------
    answer_text = _assemble_answer(claims)
    return _finish(decision="answer", answer_text=answer_text,
                   refused_reason=None, claims=claims, sources=sources,
                   trace=trace, llm_used=llm_used, timer=timer, extras=extras)


def _assemble_answer(claims: list[Claim]) -> str:
    """Join grounded claim texts into clean prose, each followed by its
    bracketed citation ids."""
    parts = []
    for claim in claims:
        text = claim.text.strip()
        if not text.endswith((".", "!", "?")):
            text += "."
        citation = "".join(f"[{sid}]" for sid in claim.source_ids)
        parts.append(f"{text} {citation}".strip())
    return " ".join(parts)

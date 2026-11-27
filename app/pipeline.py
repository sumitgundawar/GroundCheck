"""Pipeline orchestration: retrieve -> gate -> generate -> ground -> guard ->
decide -> audit. Pure, deterministic control flow. Each stage appends a
TraceStep with a status, a one-line detail, and its duration in milliseconds.

The answer-vs-refuse decision never depends on a live LLM call. If no key is set
or any call fails, the extractive fallback keeps the whole pipeline working."""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass, field

from . import (audit, config, governance, guards_input, guards_output, llm, monitoring, patient_checks, proposal, retrieval,
               tracing)
from .schemas import (AskResponse, Claim, ClaimSupport, Explanation, LLMAnswer, PatientContext, PatientFinding, Redaction, Review, Settings,
                      Source, TraceStep)

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


_NOT_SENT = re.compile(r" It was not sent for review \([^)]*\)\.$")


def _route(*, review: bool, check_only: bool, outcome: str, audit_id: str, query: str,
           reason: str, reason_code: str | None, site_id: int | None) -> Review:
    """Put a refusal in front of a reviewer, if that is possible, and say what
    happened. The answer text may claim routing only when this returns True."""
    if check_only:
        return Review(note="this was a release check")
    if not review:
        return Review(note="this was a test run")
    if outcome == "not_evaluated":
        return Review(note="the request was not evaluated")
    if not config.REVIEW_QUEUE:
        return Review(note="the review queue is switched off")
    if audit.store.backend() != "database":
        return Review(note="no review database is configured")
    case_id = governance.record_refusal(audit_id, query, reason, site_id, reason_code)
    if case_id is None:
        return Review(note="the review queue could not record it")
    return Review(routed=True, case_id=case_id, note=f"review case {case_id}")


def _routing_sentence(review: Review, outcome: str) -> str:
    if outcome == "not_evaluated":
        return ""
    return " " + ROUTED if review.routed else f" It was not sent for review ({review.note})."


def _finish(
    *,
    decision: str,
    answer_text: str,
    outcome: str | None = None,
    refused_reason: str | None,
    reason_code: str | None = None,
    claims: list[Claim],
    sources: list[Source],
    trace: list[TraceStep],
    llm_used: bool,
    timer: _Timer,
    extras: dict,
    patient_findings: list[PatientFinding] | None = None,
    explanation: Explanation | None = None,
    reason_params: dict | None = None,
) -> AskResponse:
    """answer_text, for a refusal, is the message without its routing
    sentence: whether the question was routed is only known once the review
    queue has been tried."""
    audit_id = audit.store.new_id()
    outcome = outcome or ("answered" if decision == "answer" else "refused")
    from . import sites

    site_id = extras["site_id"] if "site_id" in extras else sites.of_user(extras.get("user_id"))
    review = Review()
    if decision == "refuse":
        review = _route(review=extras.get("review", True), check_only=bool(extras.get("check_only")),
                        outcome=outcome, audit_id=audit_id, query=str(extras.get("redacted_query", "")),
                        reason=refused_reason or "", reason_code=reason_code, site_id=site_id)
        answer_text += _routing_sentence(review, outcome)
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
        outcome=outcome,
        answer_text=answer_text,
        refused_reason=refused_reason,
        claims=claims,
        sources=sources,
        trace=trace,
        audit_id=audit_id,
        total_ms=timer.total_ms(),
        llm_used=llm_used,
        patient_findings=patient_findings or [],
        reason_code=reason_code,
        review=review,
        explanation=explanation or (_explain(reason_code, refused_reason) if decision == "refuse" else Explanation()),
        redaction=extras.get("redaction") or Redaction(),
        proposal_check=extras.get("proposal_check"),
        reason_params=reason_params or {},
    )
    if extras.get("check_only"):
        return response
    saved = {k: v for k, v in extras.items()
             if k not in ("user_id", "review", "check_only", "remember", "raw_query", "redaction", "site_id", "api_key_id",
                                 "proposal_check")}
    key = extras.get("remember")
    if key and not any(step.name == "rate limit" and step.status == "fail" for step in trace):
        _remember_answer(key, response, saved)
    if not extras.get("review", True):
        saved["test_run"] = True  # kept in the audit trail, left out of usage reports
    audit.store.save(audit_id, response, saved, user_id=extras.get("user_id"), site_id=site_id,
                     api_key_id=extras.get("api_key_id"))
    if tracing.active():
        tracing.record(trace, decision=decision, outcome=outcome, reason_code=reason_code, llm_used=llm_used,
                       total_ms=response.total_ms, audit_id=audit_id, language=extras.get("language"))
    if not saved.get("test_run"):
        kind = (extras.get("provider") or {}).get("kind") if llm_used else None
        monitoring.observe_answer(decision, response.total_ms,
                                  "local" if kind == "local" else "cloud" if llm_used else "extractive")
        monitoring.observe_stages(decision, trace)
    return response


def _recover(query: str, report: dict, results: list, cfg: "Settings") -> list:
    """A question term the retrieved passages lack but the index has: one more
    search aimed at it. Only passages about the topic the question names are
    kept, so a word found in another medicine's passage can never cover it."""
    terms = [t for t in report.get("uncovered", []) if retrieval.document_frequency(t) > 0]
    named = retrieval.topics_mentioned(query)
    if not terms or not named:
        return []
    have = {r["id"] for r, _ in results}
    topics = set(named.values())
    hits = retrieval.search(" ".join(terms) + " " + " ".join(named), cfg.top_k)
    return [(r, s) for r, s in hits if r["id"] not in have and r.get("topic", "").lower() in topics]


_EXPLAIN_SUMMARY = {
    "input_blocked": "The question was stopped by an input guard before any source was searched.",
    "rate_limited": "Too many requests were sent in a short time; the question was not evaluated.",
    "no_relevant_source": "None of the sources was close enough to the question to answer from.",
    "term_not_covered": "The question names something the retrieved sources do not mention.",
    "population_not_covered": "The sources do not cover the age group the question is about.",
    "stated_value_unsupported": "The question states a dose that no source about this medicine gives.",
    "form_unsupported": "The question names a form or route the sources do not describe.",
    "limit_not_stated": "The sources mention the subject but state no such dose limit.",
    "combination_forbidden": "The sources say these must not be combined, so no combined dose exists.",
    "insufficient_context": "The retrieved sources do not contain a sentence that answers the question.",
    "claim_ungrounded": "A drafted statement could not be traced back to the sources.",
    "dose_unsupported": "The drafted answer contained a dose the sources do not support.",
    "patient_unsafe": "The patient's details make the answer unsafe to give.",
    "unsupported_language": "The question is in a language this installation cannot check against its sources.",
    "proposal_unverifiable": "The question proposes a regimen that cannot be checked against the sources.",
}
_EXPLAIN_SUGGESTION = {
    "term_not_covered": ["Check the spelling of the name.", "Ask about the medicine or condition by its listed name."],
    "population_not_covered": ["Ask about the adult guidance, or consult a specialist source."],
    "stated_value_unsupported": ["Ask without the dose to see what the sources state."],
    "form_unsupported": ["Ask about the plain form, or check the product's own label."],
    "limit_not_stated": ["Ask for the usual dose, which the sources do state."],
    "no_relevant_source": ["Name the medicine or condition and what you want to know about it."],
    "insufficient_context": ["Ask one thing at a time, naming the medicine or condition."],
    "input_blocked": ["Rephrase the question as a plain clinical question."],
    "unsupported_language": ["Ask the question in English, or in a language this installation supports."],
    "proposal_unverifiable": ["Name one medicine and give each amount with its unit and period, for example "
                              "'15 mg twice daily for 7 days'.",
                              "Ask what the source gives to see the documented regimen."],
}


def _explain(code: str | None, reason: str | None, missing: list[str] | None = None) -> Explanation:
    if code is None:
        return Explanation()
    return Explanation(summary=_EXPLAIN_SUMMARY.get(code, reason or ""), missing_terms=missing or [],
                       suggestions=_EXPLAIN_SUGGESTION.get(code, []))


REFUSAL_PREFIX = "I cannot answer this safely. "
ROUTED = "The question has been routed for review."

# Plain-English description of each stage, shown when a viewer expands a trace
# step. These make the pipeline self-documenting on the frontend.
STAGE_EXPLAIN = {
    "pii redaction":
        "Before anything is logged, embedded, or sent to a model, identifiers are "
        "replaced with placeholders such as [NAME] and [DATE]. Each detector "
        "proposes spans with a score; checksums (NHS mod-11, Luhn for cards and "
        "provider numbers, mod-97 for IBANs, social security rules) and real "
        "calendar dates confirm them, and nearby cue words ('patient', 'husband', "
        "'DOB') raise or lower the score. Names come from titles, relation and "
        "role cues, 'SURNAME, Given', and a bundled list of given names; words "
        "the documents know, medicine names and months are never names, and a "
        "dose or an age under 90 is never removed. If spaCy or Presidio is "
        "installed it can add people and places. Ages over 89 become 90. This is "
        "a safety net, not a guarantee.",
    "scope check":
        "Rejects empty or over-long queries and obvious prompt-injection "
        "patterns such as 'ignore previous instructions'. A blocked query "
        "refuses here and goes no further.",
    "rate limit":
        "A simple in-memory limiter caps requests per minute per process, so a "
        "burst of traffic cannot overwhelm the service.",
    "patient checks":
        "When the question is about a specific patient, the formulary's rules "
        "for every medicine named are applied to that patient: allergies, "
        "conditions, interactions with current medicines, age, weight, kidney and "
        "liver function, pregnancy, lab results and high-alert status. A problem "
        "that makes the answer unsafe refuses it; others are shown as warnings.",
    "retrieve":
        "The query is embedded with a sentence-transformer and compared against "
        "every document vector in the vector store, alongside a keyword search. The top-k "
        "passages are returned with cosine similarity scores.",
    "retrieval gate":
        "If even the best passage scores below the retrieval threshold, the "
        "system refuses now, before any text is generated. The cheapest refusal "
        "happens before a model is ever called.",
    "source coverage":
        "A deterministic check that the contentful terms in the question (drug "
        "and condition names, clinical terms, qualifiers like 'children') appear "
        "in the retrieved sources; everyday words like 'email' or 'meeting' are "
        "not held against them. A term the index has but the passages lack gets "
        "one more search aimed at it, kept to the topic the question names. "
        "Catches questions about things the corpus never mentions, before "
        "generation.",
    "proposal check":
        "When the question proposes a regimen ('Is 15 mg twice a day okay for X?'), its dose, frequency, route, "
        "duration, daily maximum and timing are read out and compared, part by part, with what the sources "
        "say about that medicine. Any difference gives a 'No' that names both; every part matching gives a "
        "'Yes'; a part that cannot be read or checked (no unit, an unknown or second medicine, a source that "
        "contradicts itself) refuses the question rather than guessing. No model is involved.",
    "generate":
        "The question and the retrieved sources are sent to the LLM, which must "
        "return structured JSON: a list of claims, each citing the source ids it "
        "came from. With no API key, an extractive fallback picks up to three "
        "sentences from the passages about the topic, scored on meaning, on the "
        "question's own words and on whether the passage is the kind of section "
        "that answers it (dose, side effects, symptoms...), and gives each part "
        "of a two-part question its own sentence. Either way the output is the "
        "same shape and passes through the same guards.",
    "schema validate":
        "The model output is parsed and validated against a strict Pydantic "
        "schema. Malformed output is retried once, then falls back to "
        "extractive. Free-text answers are never accepted.",
    "grounding check":
        "For every claim, a deterministic embedding similarity is computed "
        "between the claim and its cited source. The claim must cite a real "
        "source and clear the grounding threshold, and a sentence that names no "
        "medicine must belong to the passage it cites: every name and clinical "
        "term in it has to be in that passage. Each claim also records how "
        "relevant it is to the question and where in the passage it sits, for "
        "information only. When a live LLM is "
        "configured, a second, cheaper model acts as a judge and is asked, "
        "strictly, whether the source supports the claim. The deterministic "
        "result is authoritative; the judge is recorded as corroboration so the "
        "demo stays reproducible.",
    "dosage guard":
        "The showpiece check, and deterministic: every value with a clinical "
        "unit in the answer (for example '15 mg') must be stated by a retrieved "
        "source about the medicine it is attributed to. Numbers and units are "
        "compared, so '15 mg' and 'fifteen milligrams' match and '15 mcg' does "
        "not. Otherwise the system refuses and names the offending value.",
    "decision":
        "The gate. Refuse if any input guard blocked, the retrieval gate failed, "
        "coverage failed, the model had insufficient context, any claim is "
        "ungrounded, or the dosage guard failed. A refusal is put in the review "
        "queue when one is running, and the answer says whether it was; the "
        "reason is also given as a stable code. Otherwise answer, assembling the "
        "grounded claims into prose with bracketed citations.",
}


# Answers to identical questions, for a few minutes. The same question asked
# twice in a ward round shouldn't be computed twice. The key covers the
# question, the tuning settings and the index it was answered from, so a
# changed threshold or a new document never serves a stale answer. Questions
# about a specific patient are never cached: their checks depend on that
# patient. Every request is still recorded in the audit trail, and a refusal
# still opens or updates its review case.
_answers: "OrderedDict[str, tuple[float, AskResponse, dict]]" = OrderedDict()
_answer_lock = threading.Lock()


def _cache_key(raw_query: str, cfg: "Settings") -> str:
    material = json.dumps([raw_query.strip(), cfg.model_dump(), retrieval.index_version()],
                          sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _cached_answer(key: str) -> tuple[AskResponse, dict] | None:
    now = time.monotonic()
    with _answer_lock:
        found = _answers.get(key)
        if found is None:
            return None
        stored_at, response, extras = found
        if now - stored_at > config.ANSWER_CACHE_SECONDS:
            del _answers[key]
            return None
        _answers.move_to_end(key)
        return response, {**extras, "cached_seconds": round(now - stored_at, 1)}


def _remember_answer(key: str, response: AskResponse, extras: dict) -> None:
    with _answer_lock:
        _answers[key] = (time.monotonic(), response, extras)
        while len(_answers) > config.ANSWER_CACHE_SIZE:
            _answers.popitem(last=False)


def clear_answer_cache() -> None:
    with _answer_lock:
        _answers.clear()


def answer_cache_stats() -> dict:
    with _answer_lock:
        return {"answers": len(_answers), "seconds": config.ANSWER_CACHE_SECONDS}


def run(raw_query: str, settings: "Settings | None" = None,
        client_id: str = "global", user_id: int | None = None,
        review: bool = True, patient: PatientContext | None = None, check_only: bool = False,
        service: "ServiceCaller | None" = None) -> AskResponse:
    """Run a question through the pipeline. review=False keeps a refusal out of
    the review queue, for generated test questions. check_only=True is for
    checking a release before it goes live: extractive answers, no rate limit,
    and nothing recorded. `service` is an API key calling on behalf of a site:
    the record carries the key and the site, and the per-key limits in
    app/apikeys.py replace the per-client limiter."""
    if check_only:
        with llm.extractive_only():
            return _run(raw_query, settings, client_id, user_id, False, patient, True)
    # Only real questions, answered deterministically, with no patient: a
    # model's own wording varies, a patient's checks are their own, and a
    # generated test question must never be served a real user's answer, or
    # counted as use.
    cacheable = (config.ANSWER_CACHE_SECONDS > 0 and review and (patient is None or patient.is_empty())
                 and not llm.llm_available())
    if not cacheable:
        return _run(raw_query, settings, client_id, user_id, review, patient, False, service=service)
    retrieval.ensure_loaded()      # the key names the index, so load it first
    key = _cache_key(raw_query, settings or Settings())
    hit = _cached_answer(key)
    if hit is not None:
        return _reuse(*hit, user_id=user_id, review=review, service=service)
    return _run(raw_query, settings, client_id, user_id, review, patient, False, remember=key, service=service)


def _cache_age(extras: dict) -> str:
    """How long ago the remembered answer was produced, in plain words."""
    seconds = int(extras.get("cached_seconds") or 0)
    if seconds < 60:
        return f"{seconds} second{'s' if seconds != 1 else ''} ago"
    minutes = seconds // 60
    return f"{minutes} minute{'s' if minutes != 1 else ''} ago"


def _reuse(cached: AskResponse, extras: dict, user_id: int | None, review: bool,
           service: "ServiceCaller | None" = None) -> AskResponse:
    """Serve a remembered answer: the same decision, claims and trace, with its
    own audit record, so the trail still has one entry per question asked and
    a repeated refusal still adds to its review case."""
    timer = _Timer()
    from . import sites

    response = cached.model_copy(deep=True)
    response.audit_id = audit.store.new_id()
    response.total_ms = timer.total_ms()
    site_id = service.site_id if service else sites.of_user(user_id)
    if response.decision == "refuse":
        response.review = _route(review=review, check_only=False, outcome=response.outcome,
                                 audit_id=response.audit_id, query=str(extras.get("redacted_query", "")),
                                 reason=response.refused_reason or "", reason_code=response.reason_code,
                                 site_id=site_id)
        text = _NOT_SENT.sub("", response.answer_text.removesuffix(" " + ROUTED))
        response.answer_text = text + _routing_sentence(response.review, response.outcome)
    # The trace is the earlier run's. Say so, rather than leaving an audit
    # record that reads as though every check ran again for this question:
    # the stage timings below belong to the run named here.
    response.trace = [TraceStep(name="remembered answer", status="info",
                                detail=f"the same question was answered {_cache_age(extras)}, "
                                       f"and the checks below are that run's",
                                ms=0, explain="An identical question, with no patient details, asked again "
                                              "while the earlier answer was still current and the sources "
                                              "had not changed. The earlier run's checks are shown "
                                              "unchanged, and this request has its own audit record.",
                                data={"of_audit_id": cached.audit_id})] + response.trace
    audit.store.save(response.audit_id, response, {**extras, "from_cache": True}, user_id=user_id, site_id=site_id,
                     api_key_id=service.api_key_id if service else None)
    monitoring.observe_answer(response.decision, response.total_ms, "cache")
    return response


@dataclass(frozen=True)
class ServiceCaller:
    """An API key making a request: which key, and the site it speaks for."""
    api_key_id: int
    site_id: int | None


@dataclass
class Verification:
    """The outcome of checking claims against sources. `trace` holds the
    grounding, dosage and patient steps in the order they ran; `message` is the
    refusal text without its prefix."""
    decision: str                       # "answer" | "refuse"
    reason_code: str | None
    refused_reason: str | None
    message: str
    claims: list[Claim]
    findings: list[PatientFinding]
    trace: list[TraceStep]
    ungrounded: list[str] = field(default_factory=list)
    dose_ok: bool = True
    dose_detail: str = ""


def _regimen_kept(claim) -> bool:
    """Similarity cannot tell "once daily" from "twice daily". Where the claim matches a cited passage closely
    enough to be about it, the passage's frequency, duration, numbers and negations must also be the claim's."""
    from . import verify

    records = [r for r in (retrieval.record_for(i) for i in claim.source_ids) if r]
    if not records:
        return True
    found = verify.assess(claim.text, records, None)
    return found is None or bool(found.dose_values) or not (found.status == "unsupported" and found.coverage >= 0.35)


def verify_claims(question: str, claims, sources: list[dict], patient: PatientContext | None,
                  cfg: "Settings", timer: "_Timer | None" = None) -> Verification:
    """Check claims that something else drafted: each must be grounded in its
    cited source, every dose must be stated by a source about the same
    medicine, and the patient's details must not make it unsafe. `claims` are
    objects with `text` and `source_ids` (and optionally `relevance`, `support`);
    `sources` are index records. Pure apart from reading the index; the same
    stage the question pipeline runs after drafting."""
    timer = timer or _Timer()
    trace: list[TraceStep] = []
    query = question
    out_claims: list[Claim] = []
    ungrounded: list[str] = []
    judge_supported = 0
    judge_total = 0
    grounding_rows: list[dict] = []
    for llm_claim in claims:
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
        if grounded and not _regimen_kept(llm_claim):
            grounded = False
        support = llm_claim.support
        relevance = llm_claim.relevance
        if support is None and llm_claim.source_ids:
            found = retrieval.locate_sentence(llm_claim.text, llm_claim.source_ids[0])
            if found:
                support = ClaimSupport(source_id=llm_claim.source_ids[0], start=found[0], end=found[1],
                                       sentence=found[2])
        if not relevance:
            relevance = float(retrieval.embed([llm_claim.text])[0] @ retrieval.embed_query(query))
        out_claims.append(Claim(
            text=llm_claim.text,
            source_ids=llm_claim.source_ids,
            grounded=grounded,
            grounding_score=round(score, 4),
            relevance=round(relevance, 4),
            support=support,
        ))
        grounding_rows.append({
            "claim": llm_claim.text,
            "source_ids": llm_claim.source_ids,
            "deterministic_score": round(score, 4),
            "threshold": cfg.grounding_min,
            "grounded": grounded,
            "relevance": round(relevance, 4),
            "judge": judge_verdict,  # {supported, reason} or null
        })
        if not grounded:
            ungrounded.append(llm_claim.text)
    claims = out_claims

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

    # --- Dosage guard ------------------------------------------------------
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
            answer_body, sources, query)
        trace.append(TraceStep(name="dosage guard",
                               status="pass" if dose_ok else "fail",
                               detail=dose_detail, ms=timer.lap_ms(),
                               data={"detected_values": detected_values,
                                     "verified": dose_checked,
                                     "rule": "every value-with-unit must be stated by a "
                                             "retrieved source about the same medicine",
                                     "ok": dose_ok}))

    # --- Patient checks ----------------------------------------------------
    findings: list[PatientFinding] = []
    blocking = None
    if patient is None or patient.is_empty():
        trace.append(TraceStep(name="patient checks", status="skip",
                               detail="no patient details given", ms=timer.lap_ms()))
    else:
        checked = patient_checks.review(query, claims, patient)
        findings = checked.findings
        blocking = checked.blocking
        warnings = [f for f in findings if f.severity == "warn"]
        trace.append(TraceStep(
            name="patient checks",
            status="fail" if blocking else "warn" if warnings else "pass" if checked.medicines else "info",
            detail=(blocking.message if blocking
                    else f"{len(warnings)} warning(s) for {', '.join(checked.medicines)}" if warnings
                    else f"no problems for this patient with {', '.join(checked.medicines)}" if checked.medicines
                    else "no formulary medicines in the question or answer"),
            ms=timer.lap_ms(),
            data={"medicines": checked.medicines, "findings": [f.model_dump() for f in findings],
                  "derived": checked.derived},
        ))

    # --- Decision gate -----------------------------------------------------
    def refuse(code: str, reason: str, message: str) -> Verification:
        return Verification("refuse", code, reason, message, claims, findings, trace, ungrounded, dose_ok,
                            dose_detail)

    if blocking and not ungrounded and dose_ok:
        reason = blocking.message.rstrip(".")
        return refuse("patient_unsafe", reason[0].lower() + reason[1:], blocking.message)
    if ungrounded:
        return refuse("claim_ungrounded", f"a claim could not be grounded: \"{ungrounded[0]}\"",
                      "A statement could not be grounded in the sources.")
    if not dose_ok:
        return refuse("dose_unsupported", dose_detail, dose_detail[:1].upper() + dose_detail[1:] + ".")
    return Verification("answer", None, None, "", claims, findings, trace, ungrounded, dose_ok, dose_detail)


_DECIMAL_COMMA = re.compile(
    r"(?<![\d,.])(?:0,(\d+)|(\d{1,3}),(\d{1,2}))(?!\d)(?=\s*(?:mg|mcg|µg|ug|g|ml|l|milli|micro|gram|litre|liter|units?)\b)",
    re.IGNORECASE)


_ROUTE_PHRASES = ((re.compile(r"\b(?:by mouth|per os|via the oral route|by the oral route)\b", re.I), "orally"),
                  (re.compile(r"\b(?:on|to) the skin\b", re.I), "topically"),
                  (re.compile(r"\bunder the skin\b", re.I), "subcutaneously"))


def _decimal_commas(text: str) -> str:
    """"0,5 mg" is half a milligram, not "5 mg": a decimal comma is written as a point, and a route written as a
    phrase ("by mouth") as the route's own word, before any check reads it."""
    text = _DECIMAL_COMMA.sub(lambda m: f"0.{m.group(1)}" if m.group(1) else f"{m.group(2)}.{m.group(3)}", text)
    for pattern, word in _ROUTE_PHRASES:
        text = pattern.sub(word, text)
    return text


def _run(raw_query: str, settings: "Settings | None", client_id: str, user_id: int | None,
         review: bool, patient: PatientContext | None, check_only: bool, remember: str | None = None,
         service: "ServiceCaller | None" = None) -> AskResponse:
    # Effective settings: an explicit object, or the configured defaults.
    cfg = settings or Settings()
    # Turning a guard off is a demonstration, not a request option. Unless the
    # deployment has explicitly allowed it, every guard runs whatever the
    # caller sent — otherwise one JSON field from any client produces an
    # ungoverned answer that still says `decision: answer`.
    if not config.ALLOW_GUARD_OVERRIDES:
        cfg = cfg.model_copy(update={
            "enable_pii_redaction": True, "enable_injection_guard": True,
            "enable_coverage_guard": True, "enable_grounding_guard": True,
            "enable_dosage_guard": True,
            "retrieval_min_score": max(cfg.retrieval_min_score, config.RETRIEVAL_MIN_SCORE),
            "grounding_min": max(cfg.grounding_min, config.GROUNDING_MIN),
        })
    timer = _Timer()
    trace: list[TraceStep] = []
    extras: dict = {"raw_query": raw_query, "settings": cfg.model_dump(), "user_id": user_id,
                    "review": review, "check_only": check_only, "remember": remember}
    if service is not None:
        extras["site_id"], extras["api_key_id"] = service.site_id, service.api_key_id
    if patient is not None and not patient.is_empty():
        extras["patient"] = patient.model_dump(exclude_defaults=True)

    # --- 1. Input guards ---------------------------------------------------
    if cfg.enable_pii_redaction:
        cleaned = guards_input.deidentify(raw_query)
        redacted = cleaned.text
        pii_status = "warn" if cleaned.changed else "pass"
        pii_detail = cleaned.summary()
    else:
        redacted = raw_query
        pii_status, pii_detail = "skip", "guard disabled"
    extras["redacted_query"] = redacted
    extras["redaction"] = (Redaction(kinds=dict(cleaned.found), redacted_query=redacted)
                           if cfg.enable_pii_redaction else Redaction())
    trace.append(TraceStep(name="pii redaction", status=pii_status,
                           detail=pii_detail, ms=timer.lap_ms()))

    scope = guards_input.check_scope_and_injection(redacted, cfg.enable_injection_guard)
    gate = guards_input.check_language(redacted, cfg.enable_injection_guard) if scope.ok else None
    trace.append(TraceStep(
        name="scope check",
        status="pass" if scope.ok and gate.ok else "fail",
        detail=scope.detail if not (gate and gate.detail) else f"{scope.detail}; {gate.detail}",
        ms=timer.lap_ms(),
    ))
    if not scope.ok:
        _skip_after(trace, "scope check")
        return _finish(decision="refuse",
                       answer_text=REFUSAL_PREFIX + scope.reason,
                       refused_reason=scope.reason, reason_code="input_blocked", claims=[], sources=[],
                       trace=trace, llm_used=False, timer=timer, extras=extras)
    if not gate.ok:
        # Names and dates in another language are not found by the English
        # de-identifier, so the wording is not kept anywhere.
        kept = f"[question in {gate.lang} not stored]"
        extras["redacted_query"] = kept
        extras["redaction"] = Redaction(kinds=extras["redaction"].kinds, redacted_query=kept)
        _skip_after(trace, "scope check")
        return _finish(decision="refuse",
                       answer_text=REFUSAL_PREFIX + gate.reason,
                       refused_reason=gate.reason, reason_code=gate.code, claims=[], sources=[],
                       trace=trace, llm_used=False, timer=timer, extras=extras)
    if gate.english is not None:
        # Only the English reading goes on: it is de-identified again, and it,
        # not the original wording, is what the audit trail keeps.
        again = guards_input.deidentify(gate.english)
        redacted = again.text
        extras["redacted_query"] = redacted
        extras["language"] = gate.lang
        if cfg.enable_pii_redaction:
            found = Counter(cleaned.found)
            found.update(again.found)
            extras["redaction"] = Redaction(kinds=dict(found), redacted_query=redacted)
            if again.changed:
                trace[0] = TraceStep(name="pii redaction", status="warn", detail=again.summary(),
                                     ms=trace[0].ms)
        else:
            extras["redaction"] = Redaction()

    if check_only:
        rate = guards_input.GuardResult(True, detail="not applied to release checks")
    elif service is not None:
        rate = guards_input.GuardResult(True, detail="limited per API key")
    else:
        rate = guards_input.check_rate_limit(client_id)
    trace.append(TraceStep(
        name="rate limit",
        status="pass" if rate.ok else "fail",
        detail=rate.detail,
        ms=timer.lap_ms(),
    ))
    if not rate.ok:
        _skip_after(trace, "rate limit")
        return _finish(decision="refuse", outcome="not_evaluated",
                       answer_text=REFUSAL_PREFIX + rate.reason,
                       refused_reason=rate.reason, reason_code="rate_limited", claims=[], sources=[],
                       trace=trace, llm_used=False, timer=timer, extras=extras)

    query = _decimal_commas(redacted.strip())

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
                       + "No sufficiently relevant source was found.",
                       refused_reason=reason, reason_code="no_relevant_source", claims=[], sources=sources,
                       trace=trace, llm_used=False, timer=timer, extras=extras)
    trace.append(TraceStep(
        name="retrieval gate", status="pass",
        detail=f"best score {top_score:.2f} clears threshold "
               f"{cfg.retrieval_min_score:.2f}",
        ms=timer.lap_ms(), data=gate_data,
    ))

    # --- 4. Source coverage -----------------------------------------------
    if not cfg.enable_coverage_guard:
        covered, cover_detail, cover_code, cover_report = True, "guard disabled", None, {}
        trace.append(TraceStep(name="source coverage", status="skip",
                               detail="guard disabled", ms=timer.lap_ms()))
    else:
        cover_report = guards_output.coverage_report(query, source_records)
        covered, cover_detail, cover_code = guards_output.coverage_verdict(query, source_records, cover_report)
        if not covered and cover_code == "term_not_covered":
            added = _recover(query, cover_report, results, cfg)
            if added:
                results = results + added
                source_records = [r for r, _ in results]
                sources += [
                    Source(id=r["id"], title=r["title"], snippet=r["text"], score=round(s, 4),
                           topic=r.get("topic", ""), section=r.get("section", ""),
                           kind=r.get("kind", "condition"), rank=len(sources) + i + 1,
                           above_gate=s >= cfg.retrieval_min_score)
                    for i, (r, s) in enumerate(added)
                ]
                extras["retrieved"] = [{"id": r["id"], "score": s} for r, s in results]
                cover_report = guards_output.coverage_report(query, source_records)
                covered, cover_detail, cover_code = guards_output.coverage_verdict(query, source_records, cover_report)
                cover_report["targeted_retrieval"] = [r["id"] for r, _ in added]
        trace.append(TraceStep(
            name="source coverage",
            status="pass" if covered else "fail",
            detail=cover_detail,
            data=cover_report,
            ms=timer.lap_ms(),
        ))
    if not covered:
        reason = cover_detail
        if cover_code in ("stated_value_unsupported", "form_unsupported"):
            found = proposal.evaluate(query)
            if found is not None:
                extras["proposal_check"] = found.check
        _skip_after(trace, "source coverage")
        return _finish(decision="refuse",
                       answer_text=REFUSAL_PREFIX
                       + "The question refers to something the trusted sources "
                       "do not cover.",
                       refused_reason=reason, reason_code=cover_code,
                       explanation=_explain(cover_code, reason, cover_report.get("uncovered") if cover_code == "term_not_covered" else None),
                       reason_params=({"terms": cover_report.get("uncovered") or []} if cover_code == "term_not_covered"
                                      else {"detail": reason}),
                       claims=[], sources=sources,
                       trace=trace, llm_used=False, timer=timer, extras=extras)

    # --- 4b. A proposed regimen ----------------------------------------------
    proposed = proposal.evaluate(query)
    if proposed is not None:
        extras["proposal_check"] = proposed.check
        unusable = proposed.check.status == "unverifiable" or not proposed.claims
        detail = {"match": "the proposal matches the source", "mismatch": "the proposal differs from the source",
                  "unverifiable": "the proposal cannot be checked against the source"}[proposed.check.status]
        trace.append(TraceStep(
            name="proposal check", status="pass" if proposed.check.status == "match"
            else "warn" if proposed.check.status == "mismatch" else "fail",
            detail=detail, ms=timer.lap_ms(), data=proposed.check.model_dump()))
        if unusable:
            message = proposed.message or proposal.unverifiable_message(proposed.check, None)
            _skip_after(trace, "source coverage")
            return _finish(decision="refuse", answer_text=REFUSAL_PREFIX + message, refused_reason=message,
                           reason_code="proposal_unverifiable", claims=[], sources=sources, trace=trace,
                           llm_used=False, timer=timer, extras=extras,
                           reason_params={"dimensions": [{"name": d.name, "code": d.code, "reason": d.reason}
                                                         for d in proposed.check.dimensions
                                                         if d.status == "unverifiable"]})

    # --- 5. Generate -------------------------------------------------------
    provider = None if cfg.force_extractive else llm.active_provider()
    extras["provider"] = provider
    if proposed is not None:
        llm_answer = LLMAnswer(insufficient_context=False, claims=proposed.claims)
        llm_used = False
    else:
        llm_answer = None if provider is None else llm.generate_llm(
            query, source_records, cfg.temperature)
        llm_used = llm_answer is not None
        if not llm_used:
            llm_answer = llm.generate_extractive(query, results, cfg.retrieval_min_score)
    extras["llm_raw"] = llm_answer.model_dump()
    # An extractive answer may cite a passage of the topic that retrieval ranked
    # below the cut. It is a source of the answer like any other: show it, and
    # let the dosage guard read it.
    shown = {r["id"] for r in source_records}
    wanted = [sid for c in llm_answer.claims for sid in c.source_ids if sid not in shown]
    if wanted:
        scores = retrieval.passage_scores(query, wanted)
        for sid in dict.fromkeys(wanted):
            record = retrieval.record_for(sid)
            if record is None:
                continue
            score = scores.get(sid, 0.0)
            source_records.append(record)
            sources.append(Source(id=sid, title=record["title"], snippet=record["text"], score=round(score, 4),
                                  topic=record.get("topic", ""), section=record.get("section", ""),
                                  kind=record.get("kind", "condition"), rank=len(sources) + 1,
                                  above_gate=score >= cfg.retrieval_min_score))
    trace.append(TraceStep(
        name="generate",
        status="pass" if llm_used else "info",
        detail=("the cited source sentences for the proposal, no model" if proposed is not None
                else f"{'local' if provider['kind'] == 'local' else 'cloud'} model "
                f"{provider['model']} produced cited claims" if llm_used
                else ("naive extractive, model bypassed" if cfg.force_extractive
                      else ("extractive fallback: the model call failed" if provider
                            else "extractive fallback"))),
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
                       "answer safely.",
                       refused_reason=reason, reason_code="insufficient_context", claims=[], sources=sources,
                       trace=trace, llm_used=llm_used, timer=timer, extras=extras)

    # --- 7-10. Grounding, dosage, patient checks and the decision ------------
    verdict = verify_claims(query, llm_answer.claims, source_records, patient, cfg, timer=timer)
    trace.extend(verdict.trace)
    if verdict.decision == "refuse":
        return _finish(decision="refuse", answer_text=REFUSAL_PREFIX + verdict.message,
                       refused_reason=verdict.refused_reason, reason_code=verdict.reason_code,
                       claims=verdict.claims, sources=sources, trace=trace, llm_used=llm_used, timer=timer,
                       extras=extras, patient_findings=verdict.findings if verdict.reason_code == "patient_unsafe" else None,
                       reason_params=_verdict_params(verdict))
    claims, findings = verdict.claims, verdict.findings

    # --- Answer ------------------------------------------------------------
    answer_text = _assemble_answer(claims)
    if proposed is not None:
        answer_text = f"{proposed.lead} {answer_text}"
    return _finish(decision="answer", answer_text=answer_text,
                   refused_reason=None, claims=claims, sources=sources,
                   trace=trace, llm_used=llm_used, timer=timer, extras=extras,
                   patient_findings=findings)


def _verdict_params(verdict: "Verification") -> dict:
    if verdict.reason_code == "patient_unsafe":
        blocking = next((f for f in verdict.findings if f.severity == "block"), None)
        return {"rule": blocking.rule, "code": blocking.code, **blocking.params} if blocking else {}
    if verdict.reason_code == "dose_unsupported":
        return {"detail": verdict.dose_detail}
    if verdict.reason_code == "claim_ungrounded":
        return {"claims": verdict.ungrounded[:3]}
    return {}


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

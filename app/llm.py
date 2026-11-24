"""LLM integration plus the extractive fallback.

A local open-source model through Ollama (app/local_ai.py) or a cloud model
(app/providers.py: OpenAI-compatible, Anthropic, Azure OpenAI) can draft
answers; LLM_PROVIDER chooses. Left unset, a local model that is selected and
available takes precedence over the cloud one.

Every live call is wrapped so that any error, timeout, or malformed output
silently falls back to extractive generation. The user never sees an error and
the decision logic never depends on a live call succeeding."""

from __future__ import annotations

import threading
from contextlib import contextmanager

import json
import re

from . import config, local_ai, providers, retrieval
from .schemas import ClaimSupport, LLMAnswer, LLMClaim

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


def _cloud() -> dict:
    name = config.LLM_PROVIDER_EFFECTIVE or "openai_compatible"
    return {"kind": "cloud", "model": providers.model_name(name), "provider": name}


def active_provider() -> dict | None:
    """The model that will draft the next answer, as {"kind": "local" | "cloud",
    "model": name} (cloud adds "provider"), or None for extractive mode. With
    LLM_PROVIDER unset, a local model selected in the dashboard comes first;
    a provider named in LLM_PROVIDER is used alone."""
    if config.FORCE_EXTRACTIVE or getattr(_thread, "extractive", False):
        return None
    mode = config.LLM_PROVIDER
    if mode == "none":
        return None
    if mode in providers.CLOUD:
        return _cloud() if LLM_AVAILABLE else None
    local = local_ai.active_model()
    if local:
        return {"kind": "local", "model": local}
    if mode == "ollama":
        return None
    if LLM_AVAILABLE:
        return _cloud()
    return None


def llm_available() -> bool:
    return active_provider() is not None


def _complete_json(system: str, user: str, temperature: float,
                   provider: dict, judge: bool = False) -> str:
    """One JSON completion from the given provider. Raises on failure."""
    schema = JUDGE_SCHEMA if judge else GEN_SCHEMA
    if provider["kind"] == "local":
        if not judge and system == GEN_SYSTEM_PROMPT:
            system = LOCAL_GEN_SYSTEM_PROMPT
        content = local_ai.chat_json(system, user, temperature, model=provider["model"], schema=schema)
        if content is None:
            raise RuntimeError("local model call failed")
        return content
    name = provider.get("provider") or config.LLM_PROVIDER_EFFECTIVE or "openai_compatible"
    return providers.build(name).draft(system, user, temperature, schema, judge)


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


# What a question asks for: the corpus sections that answer it, and the way a
# sentence that answers it reads. A sentence from one of the sections is
# preferred; so is one that reads like an answer, wherever it sits. A question
# that matches none leaves every sentence equal.
_INTENTS = [
    ("interactions", re.compile(r"\binteract\w*|\btaken with\b|\bcombin\w*|\btogether with\b|\balongside\b|\bmixed with\b|"
                               r"\bwith other (?:medicines|drugs)\b", re.I), {"Drug Interactions"},
     re.compile(r"combin|interact|should not be taken|must not be|avoid|together|alongside", re.I)),
    ("contraindications", re.compile(r"\bcontraindicat\w*|\bnot be used\b|\bmust not\b|\bavoid\w*|\bwho should not\b|"
                                    r"\bshould not (?:take|use|be given)\b", re.I), {"Contraindications"},
     re.compile(r"contraindicat|should not be used|avoid|sensitivity|must not|not be given|allerg", re.I)),
    ("adverse effects", re.compile(r"\bside[- ]effects?\b|\badverse\b|\breactions?\b|\bunwanted\b|\bharms?\b", re.I),
     {"Adverse Reactions"}, re.compile(r"side effects?|adverse|may cause|can cause|\bcauses?\b", re.I)),
    ("dose", re.compile(r"\bdos(?:e|es|age|ing|ed)\b|\bhow much\b|\bregimen\b|\badminister\w*|\bmg\b|"
                       r"\b(?:given|take|taken)\b(?!\s+(?:with|together))", re.I), {"Dosage and Administration"},
     re.compile(r"\d+(?:\.\d+)?\s*(?:mg|mcg|µg|g|ml|units?|iu)\b|\b(?:once|twice|daily|regimen|started at)\b", re.I)),
    ("monitoring", re.compile(r"\bmonitor\w*|\bchecked\b|\bblood tests?\b|\bfollow[- ]?up\b", re.I), {"Monitoring"},
     re.compile(r"monitor|measured|checked|marker|reported on a scale|every \w+ (?:weeks?|days?|months?)", re.I)),
    ("review", re.compile(r"\breview\w*|\bprognosis\b|\boutlook\b|\brecover\w*|\bhow long\b", re.I), {"Prognosis"},
     re.compile(r"review|outlook|prognosis|recover|taper|normalis", re.I)),
    ("indication", re.compile(r"\bused for\b|\bindicat\w*|\bprescribed\b|\bwhat is \S+(?: \S+){0,2} for\b", re.I),
     {"Indications and Usage"}, re.compile(r"used (?:for|in)|indicated|management of|prescribed", re.I)),
    ("symptoms", re.compile(r"\bsymptoms?\b|\bsigns?\b|\bfeatures?\b|\bpresent(?:s|ation)?\b", re.I), {"Symptoms"},
     re.compile(r"symptom|characteri[sz]ed by|complaints|features|notice|presents? with|signs", re.I)),
    ("management", re.compile(r"\btreat\w*|\bmanag\w*|\btherapy\b|\bfirst[- ]line\b|\bnext step\b|\bescalat\w*|"
                             r"\bnot respond\w*|\bconservative\b", re.I), {"Treatment"},
     re.compile(r"managed|management|treatment|first-line|therapy|conservative|started", re.I)),
    ("cause", re.compile(r"\bcaus(?:e|es|ed)\b|\baetiology\b|\bwhy does\b", re.I), {"Causes"},
     re.compile(r"caus|arises|associated with|driver|due to", re.I)),
]
_SPLIT_PARTS = re.compile(r"\?\s+(?=\S)|;\s*|,?\s+and\s+(?=(?:can|could|is|are|does|do|should|how|what|when|which|who)\b)|"
                          r",\s*(?=(?:can|could|is|are|does|do|should|how|what|when|which|who)\b)", re.I)


def classify_intent(text: str) -> set[str]:
    """The corpus sections that answer a question, from how it is worded."""
    return _intent(text)[0]


def _intent(text: str) -> tuple[set[str], list[re.Pattern]]:
    sections: set[str] = set()
    cues: list[re.Pattern] = []
    for _name, pattern, found, cue in _INTENTS:
        if pattern.search(text):
            sections |= found
            cues.append(cue)
    if not sections and re.search(r"\bwhat is\b|\btell me about\b|\bwhat are\b", text, re.I):
        sections.add("Information")
    return sections, cues


def split_question(query: str) -> list[str]:
    """A question asking two things, as its parts. Each part must name what it
    asks for, so a stray clause is not made into a question of its own."""
    parts = [p.strip(" ?.,") for p in _SPLIT_PARTS.split(query) if p and p.strip(" ?.,")]
    parts = [p for p in parts if classify_intent(p)]
    return parts if len(parts) > 1 else [query]


def _select(part: str, pool: list[dict], skip: set[str], limit: int = 3) -> list[dict]:
    """The sentences that best answer one question: scored on meaning, on the
    question's key terms and on whether the section is the kind that answers
    it; kept above a floor and within 0.8 of the best; chosen with a penalty
    for repeating one already taken."""
    from . import guards_output

    if not pool:
        return []
    intent, cues = _intent(part)
    # The topic's own name is in every one of its sentences: it cannot tell them apart.
    from .deid import _FUNCTION

    terms = list(dict.fromkeys(w for w in retrieval.tokenize(part)
                               if len(w) >= 4 and w not in _FUNCTION and w not in skip))
    query_vec = retrieval.embed_query(part)
    scored = []
    for item in pool:
        cosine = float(item["vec"] @ query_vec)
        stems = set().union(*(guards_output._stems(w) for w in guards_output._source_vocabulary(item["sentence"])))
        overlap = sum(bool(guards_output._stems(t) & stems) for t in terms) / len(terms) if terms else 0.0
        in_section = item["section"] in intent
        reads_so = any(c.search(item["sentence"]) for c in cues)
        prior = (1.0 if in_section else 0.7 if reads_so else 0.0) if intent else 0.5
        scored.append({**item, "relevance": 0.6 * cosine + 0.25 * overlap + 0.15 * prior,
                       "in_section": in_section, "reads_so": reads_so,
                       "phrase": len(terms) >= 2 and overlap == 1.0})
    # Where the question names the kind of passage that answers it and such
    # a sentence clears the floor, the others are not candidates, unless one
    # repeats at least two of the question's own words and all of them
    # ("first-line management"). With none in such a section, a sentence that
    # reads like an answer is preferred to one that merely sounds like the
    # question.
    if intent:
        floor = config.EXTRACT_MIN_SCORE
        in_section = [x for x in scored if x["in_section"] and x["relevance"] >= floor]
        if in_section:
            scored = [x for x in scored if x["relevance"] >= floor and (x["in_section"] or x["phrase"])]
            scored = [x for x in scored if x["reads_so"]] or scored
        else:
            scored = [x for x in scored if x["reads_so"] and x["relevance"] >= floor] or scored
    top = max(scored, key=lambda x: x["relevance"])
    if top["relevance"] < config.EXTRACT_MIN_SCORE:
        return []
    floor = max(config.EXTRACT_MIN_SCORE, 0.8 * top["relevance"])
    # Sentences beside the best one in its own passage that read as answers too
    # belong with it: "monitored every three weeks" follows the marker's name.
    candidates = sorted((x for x in scored if x["relevance"] >= floor or x["phrase"]
                         or (intent and x["reads_so"] and x["record"] is top["record"]
                             and x["relevance"] >= config.EXTRACT_MIN_SCORE)),
                        key=lambda x: -x["relevance"])
    chosen: list[dict] = []
    seen_text: set[str] = set()
    while candidates and len(chosen) < limit:
        def mmr(x: dict) -> float:
            redundancy = max((float(x["vec"] @ c["vec"]) for c in chosen), default=0.0)
            return 0.7 * x["relevance"] - 0.3 * redundancy
        pick = max(candidates, key=mmr)
        candidates.remove(pick)
        key = " ".join(pick["sentence"].lower().split())
        if key in seen_text or any(float(pick["vec"] @ c["vec"]) > 0.92 for c in chosen):
            continue
        seen_text.add(key)
        chosen.append(pick)
    return chosen


def _name_the_subject(sentence: str, record: dict) -> str:
    """A dose in a sentence that does not say what is dosed ("The standard adult
    regimen is 15 mg") is attributed to nothing when the question names several
    medicines, so it leads with the subject of the passage's own title."""
    from . import guards_output

    subject = (record.get("title") or "").split(":")[0].strip()
    if ":" not in (record.get("title") or "") or not guards_output._canonical_pairs(sentence):
        return sentence
    if guards_output._named_sources(sentence, [record]):
        return sentence
    return f"{subject}: {sentence}"


def generate_extractive(query: str, scored_sources: list[tuple[dict, float]],
                        min_score: float | None = None) -> LLMAnswer:
    """Deterministic fallback: sentences lifted from the sources, chosen for the
    question rather than one per passage. When the question names a topic, the
    candidates are the sentences of that topic's passages, retrieved or not,
    since retrieval alone often ranks the wrong section of the right topic
    first. A question with several parts needs a sentence for each, or the
    sources do not answer it. Goes through the same grounding and dosage guards."""
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

    records = [record for record, score in scored_sources if score >= threshold]
    if named and records:
        have = {r["id"] for r in records}
        records += [r for r in retrieval.topic_records(set(named.values())) if r["id"] not in have]
    if not records:
        return LLMAnswer(insufficient_context=True, claims=[])

    vectors = retrieval.passage_vectors([r["text"] for r in records])
    pool = []
    for order, (record, (spans, vecs)) in enumerate(zip(records, vectors)):
        for (a, b), vec in zip(spans, vecs):
            pool.append({"record": record, "order": order, "start": a, "end": b, "vec": vec,
                         "sentence": record["text"][a:b], "section": record.get("section", "")})

    parts = split_question(query)
    # With one topic its name is in every sentence and tells none apart; with
    # several, it says which medicine a sentence is about.
    skip = ({w for topic in named.values() for w in retrieval.tokenize(topic)} | set(named)
            if len(set(named.values())) == 1 else set())
    picks: list[list[dict]] = [_select(part, pool, skip) for part in parts]
    if any(not p for p in picks):
        return LLMAnswer(insufficient_context=True, claims=[])
    cap = max(3, len(parts))
    merged: list[dict] = []
    for rank in range(3):
        for part_picks in picks:
            if rank < len(part_picks) and all(part_picks[rank] is not m for m in merged) and len(merged) < cap:
                merged.append(part_picks[rank])
    merged.sort(key=lambda x: (x["order"], x["start"]))
    several = len(set(named.values())) > 1
    claims = [LLMClaim(text=_name_the_subject(x["sentence"], x["record"]) if several else x["sentence"],
                       source_ids=[x["record"]["id"]], relevance=round(x["relevance"], 4),
                       support=ClaimSupport(source_id=x["record"]["id"], start=x["start"], end=x["end"],
                                            sentence=x["sentence"]))
              for x in merged]
    return LLMAnswer(insufficient_context=not claims, claims=claims)

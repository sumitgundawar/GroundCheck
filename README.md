---
title: GroundCheck
emoji: 🩺
colorFrom: gray
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# GroundCheck

**Grounded answers, or none at all.**

GroundCheck is a small clinical-style retrieval application that answers
questions only from a trusted set of source documents, and refuses when it
cannot ground its response. It is a demonstration of trustworthy AI
engineering: the model is the easy part, and everything around it (retrieval,
grounding, deterministic safety checks, refusal, and a full audit trail) is the
actual work.

The single idea it makes visible: **a system that refuses to answer when it is
not sure is safer than one that always answers.**

> All data is synthetic. Every condition, medication, lab marker, dosage, and
> procedure is fictional and invented for this demo. Nothing here is medical
> advice.

---

## Table of contents

1. [What it does](#what-it-does)
2. [The request lifecycle](#the-request-lifecycle)
3. [The guards in detail](#the-guards-in-detail)
4. [The synthetic corpus](#the-synthetic-corpus)
5. [The evaluation](#the-evaluation)
6. [The dashboard](#the-dashboard)
7. [Live tuning](#live-tuning)
8. [Configuration](#configuration)
9. [Local development](#local-development)
10. [Running the checks](#running-the-checks)
11. [Deployment](#deployment)
12. [Project layout](#project-layout)
13. [Honest limitations](#honest-limitations)
14. [Credits](#credits)

---

## What it does

You ask a clinical-style question. GroundCheck retrieves the most relevant
passages from its trusted corpus, asks a language model to answer using only
those passages, and then runs a series of deterministic checks before it shows
anything. If every check passes, you get an answer with citations. If any check
fails, you get a refusal with a specific reason, and the question is routed for
review.

The decision to answer or refuse is **deterministic** and never depends on a
live model call. If no API key is set, or a call fails or times out, the app
falls back to extractive generation and still works end to end. The demo cannot
break on stage.

---

## The request lifecycle

Each stage emits a trace entry (name, status, detail, milliseconds, a
plain-English explanation, and stage-specific data). The whole pipeline is in
`app/pipeline.py`.

```
question
  -> input guards     redact PII, check scope and injection, rate limit (per IP)
  -> embed query      sentence-transformers, all-MiniLM-L6-v2
  -> retrieve         FAISS top-k passages from the corpus, with cosine scores
  -> retrieval gate   if top score < threshold: REFUSE now, before any generation
  -> source coverage  if a question term appears in no source: REFUSE
  -> generate         LLM returns structured claims, each citing a source id
                      (extractive fallback if no LLM is available)
  -> schema validate  Pydantic; one retry; then extractive fallback
  -> grounding check  per claim: is it supported by its cited source?
                      (deterministic, with an optional LLM judge as corroboration)
  -> dosage guard     every value-with-unit must be supported by a source, or REFUSE
  -> decision gate    ANSWER (with citations)  |  REFUSE (route for review)
  -> audit record     full trace persisted to disk and shown in the UI
```

Status meanings: `pass` (green), `warn` (amber), `fail` (red), `skip` (grey),
`info` (blue).

---

## The guards in detail

**Input guards** (`app/guards_input.py`)

- *PII redaction.* Emails and long digit runs are replaced with `[redacted]`
  before anything is logged or sent to a model.
- *Scope and injection.* Empty or over-long queries, and obvious
  instruction-override patterns ("ignore previous instructions", "system
  prompt"), are blocked.
- *Rate limit.* A sliding-window limiter, keyed per client IP, so one caller
  cannot exhaust the limit or the LLM quota for everyone else.

**Output guards** (`app/guards_output.py`)

- *Source coverage.* A deterministic check that the contentful terms in the
  question (drug and condition names, qualifiers like "children") actually
  appear in the retrieved sources. This catches questions about entities the
  corpus never mentions, before generation. It is term-based and tolerant of
  plural and verb-form variation.
- *Grounding check.* For every claim, the stronger of two signals must clear a
  threshold: embedding cosine similarity between the claim and its cited source
  (catches faithful paraphrase), or lexical containment (catches verbatim
  extraction). The claim must also cite a real source id. When a live model is
  configured, a second, cheaper model acts as a judge and is asked, strictly,
  whether the source supports the claim. **The deterministic result is
  authoritative; the judge is corroboration only**, recorded in the trace, so a
  flaky model can never turn a refusal into an answer.
- *Dosage guard.* The showpiece, and fully deterministic. Every value with a
  clinical unit in the answer must be supported by a retrieved source. Matching
  is done on a canonical `(number, unit)` form, so `fifteen milligrams`,
  `15 mg`, and `15mg` all reduce to `15 mg`; written-out numbers, unit
  spellings, and spacing differences are all handled. If a value has no match,
  the system refuses and names it. The cheapest check catches the most
  dangerous mistake.

---

## The synthetic corpus

The corpus is **fully synthetic**, but the document *structure* mirrors two
real, public reference formats so that retrieval and grounding are exercised
realistically:

- Disease entries follow the [MedQuAD](https://github.com/abachaa/MedQuAD)
  question-type taxonomy (Information, Causes, Symptoms, Treatment, Monitoring,
  Prognosis).
- Drug entries follow the FDA Structured Product Labeling sections used by
  [DailyMed](https://dailymed.nlm.nih.gov) (Indications and Usage, Dosage and
  Administration, Contraindications, Drug Interactions, Adverse Reactions).
- Two further document kinds, lab markers and diagnostic procedures, round out
  the space.

`scripts/generate_corpus.py` produces roughly 1,880 records (plus 10 canonical
demo records kept verbatim) into `app/data/corpus.json`. It is deterministic
(seeded), generates distinct entity names so no two look alike to the
retriever, and asserts that no out-of-scope trap term leaks in, so the demo
refusals keep working at scale.

---

## The evaluation

This demonstrates "we test it like software".

`scripts/generate_golden.py` builds a large golden set
(`eval/golden.json`, ~2,000 cases) from the corpus, with a known expected
decision for each: answerable questions about entities that are in the corpus,
and must-refuse questions (unknown drugs, unknown conditions, out-of-scope
everyday topics, and missing-context qualifiers like paediatric or pregnancy).

`scripts/run_eval.py` runs every case through the pipeline in extractive mode
(deterministic, no API key) and writes `eval/eval_summary.json`. The build gate
is asymmetric and honest:

- It **fails the build** only on the unsafe direction: a must-refuse case that
  was answered.
- It **does not fail** on the safe direction: an answerable case that was
  over-refused. These are reported in the summary instead.

There is also a small set of **hand-written adversarial probes**
(`eval/adversarial.json`): leading questions asserting a wrong dose, prompt
injection, near-miss spellings, and missing-context traps. These are run as a
**diagnostic that does not gate the build**, so a genuine limitation is shown
rather than hidden.

---

## The dashboard

A calm, light, instrument-grade console (calibrated to Vercel Geist). Every
panel is built from vanilla HTML, CSS, and JavaScript, with no build step.

- **Header** with live provider state (`llm: groq` or `llm: extractive`) and the
  indexed document count.
- **Ask panel** with example chips grouped into core demo, more answers, and
  more refusals.
- **How it works** panel: the pipeline explained stage by stage, including how
  the LLM-as-a-judge layer works.
- **Tuning** panel (see below).
- **Decision** panel: a large ANSWER or REFUSED chip, the grounded answer with
  clickable citation chips, or a plain-English explanation of why it refused.
- **Retrieved sources**: rich cards showing rank, id, kind, section, topic,
  cosine score, and whether each cleared the gate. Cited sources are
  highlighted.
- **Pipeline trace**: every stage, clickable to reveal a plain-English
  description plus the stage's data, including the judge's per-claim verdict.
- **Audit record**: the full JSON for the last request, persisted to disk.
- **Corpus map**: an interactive 3D PCA projection of every document embedding.
  Drag to rotate, hover a point for its title, toggle fullscreen, click the
  topics tile to list every topic. Retrieved sources light up after a query.
- **Evaluation strip**: the headline pass numbers, a sample of cases, and the
  adversarial probe results.

---

## Live tuning

The **Tuning** panel lets you adjust the pipeline and re-ask, so you can watch
the decision change:

- **Retrieval gate**, **grounding threshold**, and **top-k** sliders.
- Toggles for the **source coverage**, **grounding**, and **dosage** guards, and
  the optional **LLM judge**.

Switching a guard off shows, visibly, what an ungoverned system would have
returned. For example, with the coverage guard off, a question about a drug that
is not in the corpus returns a confident answer about the wrong drug; with it
on, the system refuses. Overrides are sent per request; the configured defaults
are never changed.

---

## Configuration

All settings are read from the environment with safe defaults (`app/config.py`).
The only one you may want to set is `GROQ_API_KEY`.

| Variable | Default | Purpose |
| --- | --- | --- |
| `GROQ_API_KEY` | _empty_ | Enables the live LLM. Empty means extractive mode. |
| `FORCE_EXTRACTIVE` | `false` | Force extractive mode even with a key set. Use to run a public demo without spending quota. |
| `GEN_MODEL` | `llama-3.3-70b-versatile` | Generation model. |
| `JUDGE_MODEL` | `llama-3.1-8b-instant` | Optional grounding-judge model. |
| `RETRIEVAL_MIN_SCORE` | `0.30` | Cosine threshold for the retrieval gate. |
| `GROUNDING_MIN` | `0.45` | Claim-to-source threshold. |
| `TOP_K` | `4` | Passages retrieved per query. |
| `LLM_TIMEOUT_SECONDS` | `8` | Outbound call timeout. |
| `RATE_LIMIT_PER_MINUTE` | `30` | Requests per minute, per client IP. |
| `AUDIT_PERSIST` | `true` | Persist the audit trail to disk. |
| `AUDIT_LOG_PATH` | `audit/audit_log.jsonl` | Where the audit trail is written. |

No secret is ever committed. The key is read from the environment only.

---

## Local development

Requires Python 3.12 (3.11 also works locally).

macOS and Linux:

```bash
# from the repo root
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# (optional) regenerate the synthetic corpus and golden set; both are committed
python scripts/generate_corpus.py
python scripts/generate_golden.py

# build the FAISS index (downloads the embedding model once)
python scripts/build_index.py

# (optional) enable the live LLM; without this the app runs in extractive mode
export GROQ_API_KEY="your_free_key_from_console.groq.com"

# run
uvicorn app.main:app --reload --port 8000
# open http://localhost:8000
```

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
python scripts/build_index.py
$env:GROQ_API_KEY="your_free_key"
uvicorn app.main:app --reload --port 8000
```

A local `.env` file (git-ignored) is also read on startup, so you can put
`GROQ_API_KEY=...` there instead of exporting it.

---

## Running the checks

```bash
python scripts/run_eval.py     # writes eval/eval_summary.json, prints pass counts
pytest -q                      # full test suite, no network required
```

The evaluation runs in extractive mode so it is deterministic and offline. CI
(`.github/workflows/ci.yml`) builds the index, runs the evaluation, and runs the
test suite on every push and pull request.

---

## Deployment

The app is one container that listens on the port given by `app_port` (7860 on
Hugging Face Spaces). The FAISS index and evaluation are built inside the image,
so startup is instant and no model download happens at runtime.

See the deployment section of the project notes for step-by-step instructions
for Hugging Face Spaces (recommended, free) and Render. In short:

1. Push this repository to the host.
2. Add `GROQ_API_KEY` as a platform secret (never in code). Without it the app
   runs in extractive mode.
3. For a public demo where you do not want to spend quota, also set
   `FORCE_EXTRACTIVE=true`, and run the live model from localhost only.
4. The container builds the index, runs the eval, and serves the app.

**Because a public deployment exposes the endpoint, protect your key**: rely on
the per-IP rate limit, consider `FORCE_EXTRACTIVE=true` in public, and rotate
the key after a public event.

---

## Project layout

```
groundcheck/
  app/
    main.py            FastAPI app, routes, static mount, startup load
    config.py          settings and thresholds from the environment
    schemas.py         Pydantic request, response, settings, and LLM models
    pipeline.py        orchestration: retrieve -> guards -> decide -> audit
    retrieval.py       embeddings, FAISS, corpus stats and 3D projection
    guards_input.py    PII redaction, scope/injection, per-IP rate limit
    guards_output.py   coverage, grounding, dosage guards
    llm.py             Groq client, prompts, extractive fallback
    audit.py           in-memory ring buffer plus JSONL persistence
    data/              synthetic corpus and demo example queries
  scripts/
    generate_corpus.py  builds the synthetic corpus
    generate_golden.py  builds the golden evaluation set
    build_index.py      embeds the corpus, writes the FAISS index
    run_eval.py         runs the golden set and adversarial probes
  eval/
    golden.json         answerable and must-refuse cases
    adversarial.json    hand-written adversarial probes
  web/                  vanilla HTML, CSS, JS dashboard (no build step)
  tests/                pipeline, API, and guard tests (offline)
  Dockerfile, requirements.txt, .github/workflows/ci.yml
```

---

## Honest limitations

In the spirit of the demo, these are real and worth knowing:

- The corpus is procedurally generated, not real data. The structure mirrors
  real formats; the content is invented.
- The answerable half of the evaluation is somewhat self-referential by
  construction (questions are built from the corpus). The must-refuse half and
  the adversarial probes are the more meaningful safety tests.
- The evaluation runs in extractive mode, so it credits the deterministic
  guards, not the model.
- Thresholds are tuned to this synthetic corpus, not derived from first
  principles.
- The audit trail persists to a local file; a multi-instance deployment would
  use a shared store.
- The dosage guard handles digits and written-out numbers up to common ranges,
  but not every exotic format.

---

## Credits

Generation via Groq (Llama models). Embeddings via
sentence-transformers (`all-MiniLM-L6-v2`). Vector search via FAISS. Corpus
structured after MedQuAD and FDA / DailyMed labelling. Built to refuse rather
than guess.

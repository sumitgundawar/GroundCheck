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

[![CI](https://github.com/sumitgundawar/GroundCheck/actions/workflows/ci.yml/badge.svg)](https://github.com/sumitgundawar/GroundCheck/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)

[Website](https://groundcheckhealth.com) ·
[Live demo](https://huggingface.co/spaces/sumitgundawar/groundcheck) ·
[Report an issue](https://github.com/sumitgundawar/GroundCheck/issues/new/choose) ·
[Contributing](CONTRIBUTING.md)

GroundCheck is an open-source clinical-style retrieval application that answers
questions only from a trusted set of source documents, and refuses when it
cannot ground its response. It is a demonstration of trustworthy AI
engineering: the model is the easy part, and everything around it (retrieval,
grounding, deterministic safety checks, refusal, and a full audit trail) is the
actual work.

The single idea it makes visible: **a system that refuses to answer when it is
not sure is safer than one that always answers.**

> **Not a medical device and not medical advice.** GroundCheck is research and
> engineering software. It has not been clinically validated or cleared by any
> regulator. All demo data is synthetic: every condition, medication, lab
> marker, dosage, and procedure is fictional.

![The GroundCheck dashboard](site/public/screenshots/overview.webp)

---

## Quick start

Requires Python 3.12 (3.11 also works) or Docker. No API key and no GPU needed.

**With Python**

```bash
git clone https://github.com/sumitgundawar/GroundCheck.git
cd GroundCheck
python3.12 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/build_index.py    # downloads the embedding model once
uvicorn app.main:app --port 8000
```

Open http://localhost:8000.

**With Docker**

```bash
git clone https://github.com/sumitgundawar/GroundCheck.git
cd GroundCheck
docker build -t groundcheck .
docker run -p 7860:7860 groundcheck
```

Open http://localhost:7860. The image builds the index and runs the evaluation
while it builds, so it starts instantly.

Full instructions, including configuration and Windows, are in
[Local development](#local-development).

---

## Table of contents

1. [Quick start](#quick-start)
2. [What it does](#what-it-does)
3. [Screenshots](#screenshots)
4. [The request lifecycle](#the-request-lifecycle)
5. [The guards in detail](#the-guards-in-detail)
6. [The synthetic corpus](#the-synthetic-corpus)
7. [The evaluation](#the-evaluation)
8. [The dashboard](#the-dashboard)
9. [Live tuning](#live-tuning)
10. [Local AI models](#local-ai-models)
11. [Accounts and databases](#accounts-and-databases)
12. [Using the API](#using-the-api)
13. [Using your own documents](#using-your-own-documents)
14. [Configuration](#configuration)
15. [Local development](#local-development)
16. [Running the checks](#running-the-checks)
17. [Troubleshooting](#troubleshooting)
18. [Deployment](#deployment)
19. [Project layout](#project-layout)
20. [Honest limitations](#honest-limitations)
21. [Reporting issues](#reporting-issues)
22. [Contributing](#contributing)
23. [Security](#security)
24. [License](#license)
25. [Credits](#credits)

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

## Screenshots

These are unedited screenshots of the dashboard running locally in extractive
mode (no API key).

**A cited answer.** Every claim is tagged with the source it came from.

![An answer with source citations](site/public/screenshots/answer.webp)

**A refusal, with its reason.** The trace shows exactly which stage stopped the
run; later stages are marked not reached.

![A refused question and its pipeline trace](site/public/screenshots/refusal.webp)

<details>
<summary>More screenshots: evidence, tuning, corpus map, audit record, evaluation</summary>

**Retrieved sources and the pipeline trace**

![Retrieved sources with similarity scores beside the pipeline trace](site/public/screenshots/sources-trace.webp)

**Tuning panel**

![Sliders for thresholds and switches for each guard](site/public/screenshots/tuning.webp)

**Corpus map**

![A 3D map of document embeddings with retrieved sources highlighted](site/public/screenshots/corpus-map.webp)

**Audit record**

![A collapsible JSON audit record](site/public/screenshots/audit.webp)

**Evaluation**

![Golden-set results and adversarial probes](site/public/screenshots/evaluation.webp)

</details>

---

## The request lifecycle

Each stage emits a trace entry (name, status, detail, milliseconds, a
plain-English explanation, and stage-specific data). The whole pipeline is in
`app/pipeline.py`.

```
question
  -> input guards     redact PII, check scope and injection, rate limit (per IP)
  -> embed query      sentence-transformers, all-MiniLM-L6-v2
  -> retrieve         hybrid: embedding search + BM25 keywords, top-k passages
                      ranked by a blend, reported with cosine scores
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

## Local AI models

GroundCheck can draft answers with an open-source model running on your own
machine through [Ollama](https://ollama.com). No API key is needed and nothing
is sent to the cloud. The model only drafts: the same deterministic checks
decide what is shown, and if the model is unavailable or fails, answers fall
back to extractive mode.

1. Install Ollama from [ollama.com/download](https://ollama.com/download) and
   start it.
2. Open the **Local AI** panel in the dashboard. It shows your hardware, how
   much memory a model can use, and which models fit.
3. Download a model, then choose **Use this model**. The header shows the
   model in use.

| Hardware | Memory a model can use |
| --- | --- |
| Apple silicon | About two thirds of system memory, shared with the GPU |
| NVIDIA GPU | The largest GPU's memory |
| No GPU | Half of system memory, running on the CPU |

The catalogue lists ten models from 1B to 14B parameters, with download sizes
from the Ollama registry and the licence each model ships with. Models with
non-commercial licences are left out. The recommended model is the best one
that fits your machine with headroom.

A selected local model takes precedence over a cloud model. You can also set
one without the dashboard: `ollama pull llama3.2:3b`, then
`LOCAL_MODEL=llama3.2:3b`.

Downloading and switching models changes the server for everyone. With
sign-in required, only admins can do it. Without sign-in, it's allowed only
from the machine running GroundCheck; set `ADMIN_ACCESS=all` to allow it from
anywhere, or `none` to lock the choice.

Small local models are slower and less capable than large cloud models. Expect
several seconds to tens of seconds per answer on a laptop, and more refusals
when a small model drafts claims the grounding check can't verify.

---

## Accounts and databases

By default GroundCheck runs as an open demo: no sign-in, and everything is
stored in a SQLite file at `data/groundcheck.db`, created on first run.

### Requiring sign-in

For any deployment with real users, set `AUTH_REQUIRED=true`. Every API
request then needs a signed-in session, except health checks and sign-in
itself, and each user has a role:

| Role | Can |
| --- | --- |
| `clinician` | Ask questions and see their own audit records |
| `reviewer` | Also read every audit record |
| `admin` | Also manage users and local AI models |

Create the first admin from the machine running GroundCheck, either in the
dashboard (it offers this while no users exist) or on the command line:

```bash
python -m app.cli create-user --email you@example.org --role admin
```

Passwords are hashed with Argon2id and must be at least 12 characters. Users
can turn on two-factor authentication with any authenticator app. Five failed
sign-ins lock an account for 15 minutes. Sessions last 12 hours and are
stored only as hashes.

Other commands: `python -m app.cli set-password`, `list-users`,
`purge-sessions` and `migrate`. Passwords are always prompted for, never
passed as arguments.

### Using PostgreSQL or MySQL

Set `DATABASE_URL` and install the driver:

| Database | `DATABASE_URL` | Driver |
| --- | --- | --- |
| SQLite (default) | `sqlite:///data/groundcheck.db` | built in |
| PostgreSQL | `postgresql+psycopg://user:password@host:5432/groundcheck` | `pip install "psycopg[binary]"` |
| MySQL or MariaDB | `mysql+pymysql://user:password@host:3306/groundcheck` | `pip install pymysql` |

The schema is created and upgraded automatically on startup with Alembic
migrations (turn this off with `DB_AUTO_MIGRATE=false` and run
`python -m app.cli migrate` yourself). The test suite runs against SQLite and,
with `TEST_POSTGRES_URL` set, PostgreSQL. MySQL is supported through
SQLAlchemy but hasn't yet been verified against a live server.

Audit records are stored in the database with the user who asked. The
original question is never stored, only the PII-redacted version.

### Vector stores

| `VECTOR_STORE` | Where embeddings live | Use it for |
| --- | --- | --- |
| `local` (default) | `index/vectors.npy`, exact search in the app | Up to roughly a million passages on one machine |
| `qdrant` | A Qdrant server (`QDRANT_URL`) or embedded Qdrant (`QDRANT_PATH`) | Larger collections, or several app instances sharing one index |

After changing the store, rebuild the index with
`python scripts/build_index.py`. Both stores give identical results on the
full evaluation.

---

## Using the API

The dashboard is a client of a small JSON API. You can call it directly.

| Method and path | Purpose |
| --- | --- |
| `POST /api/ask` | Run a question through the pipeline |
| `GET /api/audit` | Recent audit records |
| `GET /api/audit/{id}` | One full audit record |
| `GET /api/settings` | Default settings and their allowed ranges |
| `GET /api/examples` | The example questions shown in the dashboard |
| `GET /api/eval-summary` | The latest evaluation results |
| `GET /api/corpus` | Corpus statistics and the 3D projection |
| `GET /api/health` | Status, the model drafting answers, and corpus size |
| `POST /api/sources` | Upload a document for review (multipart form: `file`, `title`, `owner`, `effective_from`, `expires_on`) |
| `GET /api/sources` | Every document version, with status and index status |
| `POST /api/sources/{id}/approve` | Approve, `/reject` or `/retire` a document version |
| `POST /api/sources/{id}/evaluate` | Generate and run a document's evaluation |
| `POST /api/index/rebuild` | Rebuild the search index |
| `GET /api/local-ai` | Hardware, Ollama status, and the model catalogue with fit |
| `POST /api/local-ai/pull` | Download a catalogue model (streams progress as NDJSON) |
| `POST /api/local-ai/select` | Use a downloaded model, or `null` to stop using one |

```bash
curl -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the recommended dose of Zalortin?"}'
```

The response contains `decision` (`answer` or `refuse`), `answer_text`,
`refused_reason`, `claims` (each with `source_ids` and a grounding score),
`sources`, the full `trace`, `audit_id`, `total_ms`, and `llm_used`. The schemas
are in `app/schemas.py`, and FastAPI serves interactive docs at `/docs`.

To override thresholds or switch guards for a single request, add a `settings`
object, for example `{"query": "...", "settings": {"top_k": 6,
"enable_dosage_guard": false}}`. Configured defaults are never changed.

---

## Using your own documents

Import your organisation's guidelines, formularies and protocols in the
dashboard's **Documents** panel, or through the API.

1. **Upload** a PDF, Word (.docx), HTML, Markdown or text file, up to 25 MB,
   with an owner and optional effective and expiry dates. GroundCheck splits it
   into sections using the document's own headings, so every citation keeps
   its heading. Uploading a new file with the same title creates the next
   version.
2. **Approve** it. Until then nothing in it can be cited. When sign-in is
   required, the person who uploaded a document can't approve it. Approving a
   version retires the previous one, and the search index rebuilds in the
   background, embedding only new text.
3. **Evaluate** it. GroundCheck generates a question for each section, which
   must be answered citing the document, and the same question about an
   invented document, which must be refused.

An approved document is cited only between its effective and expiry dates,
checked on every search. **Retire** a document to stop citing it. Set
`INCLUDE_DEMO_CORPUS=false` to answer only from your own documents.

Scanned PDFs need OCR first: GroundCheck reads the text layer only. Never put
real patient data in a document.

The thresholds were tuned on the synthetic corpus, so re-tune
`RETRIEVAL_MIN_SCORE` and `GROUNDING_MIN` for your content, and check each
document's evaluation after approving it.

You can still replace `app/data/corpus.json` (the demo corpus) with your own
records in the same format and run `python scripts/build_index.py`.

---

## Configuration

All settings are read from the environment with safe defaults (`app/config.py`).
The only one you may want to set is `GROQ_API_KEY`.

| Variable | Default | Purpose |
| --- | --- | --- |
| `GROQ_API_KEY` | _empty_ | Enables the live LLM. Empty means extractive mode. |
| `GROQ_BASE_URL` | `https://api.groq.com/openai/v1` | Any OpenAI-compatible endpoint that supports JSON output mode. |
| `DATABASE_URL` | `sqlite:///data/groundcheck.db` | Where users, sessions and audit records are stored. |
| `AUTH_REQUIRED` | `false` | Require sign-in for every API request. |
| `SESSION_HOURS` | `12` | How long a sign-in lasts. |
| `SESSION_COOKIE_SECURE` | `true` | Send the session cookie over HTTPS only. Set `false` for plain-HTTP local use with sign-in. |
| `VECTOR_STORE` | `local` | `local` or `qdrant`. |
| `QDRANT_URL` | _empty_ | A Qdrant server, for example `http://localhost:6333`. |
| `QDRANT_PATH` | `index/qdrant` | Embedded Qdrant storage, when no URL is set. |
| `OLLAMA_HOST` | `http://localhost:11434` | Where the Ollama server runs. |
| `LOCAL_MODEL` | _empty_ | A local model to use at startup, if none was chosen in the dashboard. |
| `LOCAL_AI_TIMEOUT_SECONDS` | `60` | Timeout for a local model call before falling back. |
| `ADMIN_ACCESS` | `local` | Without sign-in, where management actions (local models, documents) are allowed from: `local`, `all`, or `none`. |
| `INCLUDE_DEMO_CORPUS` | `true` | Include the synthetic demo corpus in the index alongside approved documents. |
| `FORCE_EXTRACTIVE` | `false` | Force extractive mode even with a key set. Use to run a public demo without spending quota. |
| `GEN_MODEL` | `llama-3.3-70b-versatile` | Generation model. |
| `JUDGE_MODEL` | `llama-3.1-8b-instant` | Optional grounding-judge model. |
| `RETRIEVAL_MIN_SCORE` | `0.30` | Cosine threshold for the retrieval gate. |
| `GROUNDING_MIN` | `0.45` | Claim-to-source threshold. |
| `TOP_K` | `4` | Passages retrieved per query. |
| `HYBRID_RETRIEVAL` | `true` | Combine keyword (BM25) and embedding search, so rare look-alike names aren't confused. `false` uses embeddings only. |
| `HYBRID_ALPHA` | `0.5` | Weight of the embedding score in hybrid ranking, from 0 to 1. |
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

# build the search index (downloads the embedding model once)
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

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `FileNotFoundError` for the index on startup | Run `python scripts/build_index.py` first. The index is not committed. |
| The first index build is slow or fails offline | It downloads `all-MiniLM-L6-v2` from Hugging Face once. Run it with network access, then it works offline. |
| The header shows `llm: extractive` although a key is set | Check that `FORCE_EXTRACTIVE` is not `true`, and that the key is exported in the same shell or set in `.env`. |
| Every question is refused | Check you haven't raised `RETRIEVAL_MIN_SCORE` or `GROUNDING_MIN`, or switched the corpus without rebuilding the index. |
| Refusals with a failed `rate limit` stage in the trace | You hit the per-IP rate limit. Raise `RATE_LIMIT_PER_MINUTE` for local testing or batch runs. |

---

## Deployment

The app is one container that listens on the port given by `app_port` (7860 on
Hugging Face Spaces). The search index and evaluation are built inside the image,
so startup is instant and no model download happens at runtime.

It runs on Hugging Face Spaces (Docker SDK; the front matter at the top of this
file is the Space configuration) and on Render (`render.yaml` is a Blueprint).
In short:

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
    retrieval.py       embeddings, hybrid search, corpus stats and 3D projection
    vectorstore.py     vector stores: local (NumPy) and Qdrant
    guards_input.py    PII redaction, scope/injection, per-IP rate limit
    guards_output.py   coverage, grounding, dosage guards
    llm.py             Groq client, prompts, extractive fallback
    audit.py           in-memory ring buffer plus JSONL persistence
    data/              synthetic corpus and demo example queries
  scripts/
    generate_corpus.py  builds the synthetic corpus
    generate_golden.py  builds the golden evaluation set
    build_index.py      embeds the corpus into the configured vector store
    run_eval.py         runs the golden set and adversarial probes
  eval/
    golden.json         answerable and must-refuse cases
    adversarial.json    hand-written adversarial probes
  web/                  vanilla HTML, CSS, JS dashboard (no build step)
  tests/                pipeline, API, and guard tests (offline)
  site/                 the groundcheckhealth.com website (static, Cloudflare)
  .github/              CI workflow, issue forms, pull request template
  Dockerfile, requirements.txt, render.yaml
  CONTRIBUTING.md, SECURITY.md, LICENSE
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
- The coverage guard works on words. Contractions and common conversational
  words are handled, but unusual phrasing can still cause a refusal.
- PII redaction covers email addresses and long digit runs only. It is not
  de-identification.
- "Routed for review" is recorded in the audit trail, but there is no review
  queue yet.

---

## Reporting issues

Use the issue forms at
[github.com/sumitgundawar/GroundCheck/issues/new/choose](https://github.com/sumitgundawar/GroundCheck/issues/new/choose):

- **Unsafe answer**: GroundCheck answered a question it should have refused.
  This is the most valuable report you can file. Include the exact question,
  the answer, and the audit ID.
- **Bug report**: something is broken, or a question was wrongly refused.
- **Feature request**: an improvement or a new capability.

Search existing issues first, and **never include real patient data or API
keys**.

## Contributing

Contributions are welcome: code, evaluation cases, adversarial probes, and
documentation. Read [CONTRIBUTING.md](CONTRIBUTING.md) for setup, the checks
every pull request must pass, and the extra rules for changes to the guards.
The short version:

1. Open or comment on an issue before starting anything large.
2. Branch from `main` and keep the change focused.
3. Run `python scripts/run_eval.py` (zero must-refuse cases answered) and
   `pytest -q`.
4. Open a pull request using the template.

## Security

Please don't report vulnerabilities in public issues. Follow
[SECURITY.md](SECURITY.md) to report privately.

## License

[MIT](LICENSE). You may use, modify, and distribute GroundCheck, including
commercially. Any clinical use, and the regulatory approvals and validation it
requires, is your responsibility.

---

## Credits

Generation via Groq (Llama models). Embeddings via
sentence-transformers (`all-MiniLM-L6-v2`). Vector search in NumPy or Qdrant. Corpus
structured after MedQuAD and FDA / DailyMed labelling. Built to refuse rather
than guess.

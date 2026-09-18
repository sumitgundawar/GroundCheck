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

![The GroundCheck app: a sidebar of pages and the Ask page](site/public/screenshots/ask.webp)

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
while it builds, so it starts instantly. This run keeps nothing: it is for
trying GroundCheck out. To keep data, see
[A single container, without Compose](docs/on-premises.md#a-single-container-without-compose),
or use Docker Compose, below.

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
14. [Clinical review and governance](#clinical-review-and-governance)
15. [Patient-aware checks](#patient-aware-checks)
16. [EHR integration](#ehr-integration)
17. [Training imaging models](#training-imaging-models)
18. [CT and MRI](#ct-and-mri)
19. [Knowledge releases and rollback](#knowledge-releases-and-rollback)
20. [Monitoring and alerts](#monitoring-and-alerts)
21. [Incident reporting](#incident-reporting)
22. [Configuration](#configuration)
23. [Local development](#local-development)
24. [Running the checks](#running-the-checks)
25. [Troubleshooting](#troubleshooting)
26. [Deployment](#deployment)
27. [Project layout](#project-layout)
28. [Honest limitations](#honest-limitations)
29. [Changes](#changes)
30. [Reporting issues](#reporting-issues)
31. [Contributing](#contributing)
32. [Security](#security)
33. [License](#license)
34. [Credits](#credits)

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

Unedited screenshots of the app running locally, with synthetic data. The app
has light and dark themes.

**A cited answer.** Every claim is tagged with the source it came from.

![An answer with source citations](site/public/screenshots/answer.webp)

**A refusal, with its reason.** The trace shows exactly which check stopped the
run.

![A refused question and its pipeline trace](site/public/screenshots/refusal.webp)

<details>
<summary>More screenshots: review, usage, training, model library, data protection, evaluation</summary>

**Review queue**

![Refusals and flagged answers waiting for review](site/public/screenshots/review.webp)

**Usage**

![Questions per day, refusal rate and response times](site/public/screenshots/usage.webp)

**Training a model on CT scans**

![Live loss and accuracy while training an abdominal CT organ classifier](site/public/screenshots/training.webp)

**A trained model, tried on an image**

![A model card with per-class accuracy and a CT slice classified as liver](site/public/screenshots/model.webp)

**Data protection, in dark mode**

![The audit trail verified, with encryption and retention status](site/public/screenshots/protection-dark.webp)

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
- **Documents** and **Review** panels for your own sources and the review
  queue, hazard log and reports.
- **Decision** panel: a large ANSWER or REFUSED chip, the grounded answer with
  clickable citation chips, or a plain-English explanation of why it refused.
- **Retrieved sources**: rich cards showing rank, id, kind, section, topic,
  cosine score, and whether each cleared the gate. Cited sources are
  highlighted.
- **Pipeline trace**: every stage, clickable to reveal a plain-English
  description plus the stage's data, including the judge's per-claim verdict.
- **Flag this answer**: send an answer that looks wrong to the review queue.
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
| `clinician` | Ask questions, see their own audit records, and flag answers |
| `reviewer` | Also read every audit record, work the review queue and read reports |
| `admin` | Also manage users, documents, local AI models and the hazard log |

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
`purge-sessions`, `escalate-reviews` and `migrate`. Passwords are always prompted for, never
passed as arguments.

### Single sign-on

GroundCheck signs people in through your identity provider with OpenID
Connect: Microsoft Entra ID, Okta, Google Workspace, Auth0, Keycloak, Ping or
ADFS. Register GroundCheck as a web application with the redirect address
`https://<your GroundCheck>/api/auth/sso/callback`, then set:

```bash
AUTH_REQUIRED=true
OIDC_ISSUER=https://login.microsoftonline.com/<tenant id>/v2.0
OIDC_CLIENT_ID=<application id>
OIDC_CLIENT_SECRET=<client secret>
OIDC_PROVIDER_NAME="NHS Trust account"
OIDC_ROLES_CLAIM=roles            # or groups
OIDC_ADMIN_VALUES=GroundCheck.Admin
OIDC_REVIEWER_VALUES=GroundCheck.Reviewer
OIDC_DEFAULT_ROLE=clinician       # or none, to refuse people without a mapped role
OIDC_ALLOWED_DOMAINS=nhs.net      # optional
OIDC_REQUIRE_MFA=true             # optional: refuse sign-ins without MFA
PASSWORD_SIGN_IN=false            # optional: single sign-on only
```

The sign-in page then offers "Sign in with NHS Trust account". Sign-in uses
the authorisation code flow with PKCE. The ID token's signature is checked
against the provider's published keys (RSA or EC only), with its issuer,
audience, expiry and nonce, and the sign-in must return to the browser that
started it. Roles follow the provider's claims at every sign-in, so removing
someone from a group removes their access. An existing account is linked by
email only if the provider confirms the email. Providers that only support
SAML can connect through one that bridges to OpenID Connect, such as Keycloak.

Keep one local admin account for emergencies. With `PASSWORD_SIGN_IN=false`
it can't sign in on the web, but `python -m app.cli set-password` still works
on the server.

### Several sites

One installation can serve a group of hospitals or clinics. An admin adds
**sites** on the Users page and chooses each person's site. People at a site
see only their site's questions, audit records, review cases, usage reports,
incidents and imaging series; people with no site ("Every site") see all of
them. The same refused question at two sites opens a case at each. Approved
documents, the formulary, trained models, releases and monitoring are shared
by the group.

Admins at a site manage only their site's people and can't move anyone
between sites or add sites. With single sign-on, set `OIDC_SITE_CLAIM` to a
claim holding the site's key; it sets the site at every sign-in, a missing
or unknown site is refused, and `OIDC_ALL_SITES_VALUE` names the value for
group-wide staff.

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
| `POST /api/audit/{id}/flag` | Flag an answer for review (`{"note": "..."}`) |
| `GET /api/reviews` | The review queue (`status`, `mine`, `overdue`) |
| `GET /api/reviews/{id}` | One case, with its timeline and what was shown |
| `POST /api/reviews/{id}/assign` | Assign, `/comment`, `/resolve` or `/reopen` a case |
| `GET /api/review-tests` | Tests added from resolved cases; `POST /api/review-tests/run` runs them |
| `GET /api/hazards` | The hazard log; `POST` adds and `PUT /api/hazards/{id}` updates a hazard |
| `GET /api/governance/report` | Usage, refusal and review figures for the last `days` |
| `GET /api/governance/safety-case` | A clinical safety case summary, as Markdown |

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

## Clinical review and governance

A refusal protects the patient, but it also means a clinician didn't get an
answer. The **Review** panel makes sure each one is looked at. It needs the
database (the default).

- **Review queue.** Every refusal opens a case. Asking the same question again
  while the case is open adds to it, so a reviewer sees how often it happens.
  Anyone can flag an answer that looks wrong, which opens a high-priority case.
- **Due dates and escalation.** Refusals are due within 72 hours and flagged
  answers within 24. Overdue cases are escalated to high priority when the
  queue is opened, or on a schedule with `python -m app.cli escalate-reviews`.
- **Closing the loop.** Reviewers assign cases, comment, and resolve them with
  an outcome: the refusal was correct, a document is needed, a check needs
  fixing, no action, or **add as a test**. A test keeps the question and the
  decision the reviewer says is correct, and runs from the Report tab, so the
  same mistake is caught from then on.
- **Hazard log.** Admins record what could go wrong, its cause, effect and
  controls, scored by severity and likelihood (1 to 5 each) before and after
  controls, in the shape DCB0129 and ISO 14971 expect.
- **Reports.** Questions, refusal rate and reasons, questions asked with a
  check switched off, review performance and documents approved, for a chosen
  period. **Download safety case** produces a Markdown summary of the
  evaluation, review figures and hazard log for a release.

Evaluation runs and document tests are kept in the audit trail but don't open
cases or count as use. The safety case summary supports, and doesn't replace,
sign-off by your clinical safety officer.

---

## Patient-aware checks

On the Ask page, open **For a specific patient** and enter what you know: age,
sex, weight, eGFR or serum creatinine, Child-Pugh class, pregnancy,
breastfeeding, allergies, current medicines, conditions and lab results. No
names or record numbers: the API rejects fields it doesn't know. Through the
API, send a `patient` object with `POST /api/ask`.

For every formulary medicine named in the question or the answer, GroundCheck
checks:

| Check | Refuses when | Warns when |
| --- | --- | --- |
| Allergies and conditions | The patient is allergic to the medicine, its class or a group it's ruled out for, or has a contraindicated condition | The question isn't about giving it (what it's for, say) |
| Interactions | A current medicine or class is contraindicated with it | The interaction is major, or the patient takes another medicine of the same class |
| Children | The formulary has no dosing for the patient's age, or the answer gives an adult dose (the weight-based dose is shown) | |
| Kidney and liver | The rule says avoid, or the answer's dose is above the reduced dose | The rule says caution, or a reduced dose applies |
| Pregnancy and breastfeeding | The medicine is to be avoided, or has no pregnancy safety information | There's no breastfeeding information |
| Lab results | A value crosses an avoid threshold | A value crosses a caution threshold |
| Dose | The answer's dose is above the patient's maximum | The medicine is high alert |
| Missing data | A dose question lacks what the rules need: age, weight, kidney or liver function | |

Creatinine clearance is calculated with Cockcroft-Gault when eGFR isn't given.
Every finding names its rule and source, and the trace records the checks.

The rules come from a formulary in a validated JSON format (`app/formulary.py`).
The demo formulary is generated from the synthetic corpus by
`scripts/build_formulary.py`, with illustrative kidney, liver, weight, child,
pregnancy, lab and high-alert rules. **For clinical use, replace it** with rules
from a licensed drug database and point `FORMULARY_PATH` at the file; it's
validated when the app starts. The **Medicines** page shows every rule.

`eval/patient_cases.json` holds 383 patient scenarios (regenerate with
`scripts/generate_patient_cases.py`), and `scripts/run_eval.py` fails the build
if any scenario that must be refused is answered.

---

## EHR integration

GroundCheck works with electronic health records through HL7 FHIR R4, SMART
on FHIR and CDS Hooks. Admins see the addresses to register and a CDS console
on the **EHR integration** page.

### Reading a patient

From a patient's FHIR record GroundCheck reads age and sex, the latest eGFR,
serum creatinine, weight, potassium, sodium, INR, platelets, ALT, QTc and
pregnancy status (by LOINC, with units converted), active allergies,
medicines and conditions. Names, identifiers and contact details are never
copied. Results that are old (`FHIR_LAB_MAX_AGE_DAYS`), in units it can't
convert, or implausible are flagged or left out. The patient is kept,
encrypted, for `EHR_CONTEXT_MINUTES` for that browser.

- **SMART on FHIR launch.** Register GroundCheck with your EHR using the launch
  URL `https://<your GroundCheck>/api/ehr/launch` and redirect URL
  `https://<your GroundCheck>/api/ehr/callback`, then set `SMART_CLIENT_ID`
  and `SMART_ALLOWED_ISSUERS` (your EHR's FHIR base URL). EHR launch and
  standalone launch both use PKCE, and only allowed issuers can launch.
- **Open FHIR servers.** For development, list servers in `FHIR_OPEN_SERVERS`
  (for example `https://hapi.fhir.org/baseR4`) to load a patient by ID from the
  patient panel.

### Saving to the record

With `SMART_WRITE_NOTES=true`, a clinician can save an answer asked for the
launched patient to their record as a preliminary `DocumentReference`: the
question, the answer or reason, cited sources, the patient checks, their
comment and the audit record ID. The EHR's access token is kept encrypted
server-side and never sent to the browser.

### CDS Hooks

The discovery endpoint is `https://<your GroundCheck>/cds-services`. The
`order-select` and `order-sign` services check each draft `MedicationRequest`
against the prefetched patient and return a card per finding: **critical** for
unsafe orders (with override reasons), **warning** and **info** otherwise, each
naming its formulary rule. Calls must carry a JWT from a trusted EHR:
`CDS_HOOKS_TRUSTED=https://ehr.example.org=https://ehr.example.org/jwks.json`.
`CDS_HOOKS_ALLOW_UNSIGNED=true` is for sandbox testing only.

### Testing against public sandboxes

`RUN_LIVE_EHR=1 pytest tests/test_ehr.py` loads patients from the public HAPI
FHIR server, and runs a full SMART launch, patient load and note write
against the [SMART Health IT launcher](https://launch.smarthealthit.org).

---

## Training imaging models

The **Training** page trains an image classifier on a folder of your own
images, on this machine's hardware, and saves it to the **Model library**.

1. **Choose a folder.** One subfolder per class (`normal/`, `abnormal/`), or
   `train/`, `val/` and `test/` folders each holding class subfolders. PNG,
   JPEG, BMP, TIFF and WebP are read; 16-bit images are windowed to 8 bits.
   Without split folders, images are split 70/15/15 by class. Folders are only
   read from `TRAINING_DATA_DIRS`.
2. **Choose where to train.** NVIDIA and AMD GPUs, the Apple GPU and the CPU
   are detected. With several GPUs you pick one; the recommended one is
   selected.
3. **Name the model and start.** Choose a small CNN (fast, from scratch) or
   ResNet-18 (optionally from ImageNet weights; check their terms for
   commercial use). Loss and accuracy update live, training stops early when
   validation stops improving, and it can be cancelled. Training runs in its
   own process, so the app stays responsive.

Every saved model refuses to guess, like the rest of GroundCheck:

- **Confidence threshold.** Chosen on validation images as the lowest
  confidence at which answered images are at least `MODEL_TARGET_ACCURACY`
  (95%) correct, by the lower end of a 95% confidence interval. Below it, the
  model abstains, and it never answers below `MODEL_MIN_CONFIDENCE` (50%). The
  library shows whether the target also held on the test
  images, and marks a model experimental if not.
- **Unfamiliar images.** At the end of every block of the network, the
  model records the mean and spread of each feature map for its training
  images, per class. An image whose statistics are far from every class at
  any block (Mahalanobis distance, calibrated to flag about 2% of validation
  images) is refused rather than forced into a class. Early blocks see
  texture, contrast and noise, so they catch another modality or scanner;
  late blocks catch the wrong anatomy. Checking the last block alone missed
  every MRI patch shown to a CT model; checking every block flagged them.
  Run `python scripts/refit_novelty.py <model id>` to refit a model trained
  before this check.
- **Model card.** Accuracy, balanced accuracy, AUC, calibration error,
  sensitivity, specificity and precision for every class, a confusion matrix,
  training history, the dataset's fingerprint and the hardware used.

A model is a self-contained folder (`models/library/<id>/`: weights in
safetensors format and `model.json`). Download it as a zip from the library
and copy it into another installation's library to use it there.

### Try it on public CT scans

```bash
python scripts/fetch_scan_dataset.py organamnist        # abdominal CT slices, 11 organs, 200 MB
python scripts/fetch_scan_dataset.py pneumoniamnist     # paediatric chest X-rays, 2 classes, 20 MB
```

Both come from [MedMNIST v2](https://medmnist.com) under CC BY 4.0 and are
written to `data/datasets/`, with a `DATASET.md` giving the source and
citations. Then choose the folder on the Training page.

Models trained here are for research and evaluation. They are not medical
devices and haven't been validated for clinical use. To run a model over CT
or MRI series, see [CT and MRI](#ct-and-mri).

---

## CT and MRI

The **CT and MRI** page imports whole series, runs a model from the library
over them, and lets a clinician read the images and sign a report that a PACS
can store.

**Importing.** Upload `.dcm` files, a folder or a zip (up to 3,000 files), or
search a PACS and retrieve a series over DICOMweb (QIDO-RS and WADO-RS) when
`DICOMWEB_URL` is set. CT and MR image storage is accepted. Every instance is
de-identified before anything is stored, following the DICOM PS3.15 Basic
Application Level Confidentiality Profile with the Retain Patient
Characteristics and Clean Descriptors options:

- names, IDs, dates and times, institutions, physicians, device serial
  numbers, free-text comments and private tags are removed or emptied
- UIDs are replaced consistently, so a series stays a series
- the patient's own name and IDs are removed from descriptions, which then
  pass through text de-identification
- age (over 89 becomes 90), sex, size and weight are kept
- images marked as having patient details burned into the pixels are refused

Slices are ordered head to feet (axial), front to back (coronal) or right to
left (sagittal) and turned to standard radiological display from
`ImageOrientationPatient`, whatever order and orientation they were stored in.
Pixels are stored in `IMAGING_DIR`, encrypted with `DATA_ENCRYPTION_KEYS` when
set. For a series from a PACS, the original identifiers are kept, encrypted,
only so a signed report can be filed with the right study.

**Viewing.** Scroll, drag or use the arrow keys through the slices. CT has
abdomen, soft tissue, lung, bone and brain windows. Orientation letters, the
field of view and slice spacing are shown.

**Running a model.** A model from the library needs imaging settings, set by
an administrator in the Model library under **CT and MRI use**: the modality
it was trained on, its training images' window and orientation, and the width
of anatomy each training image showed. The series is searched slice by slice
with overlapping regions of that physical size, so scanners with different
pixel spacing are treated alike. Each region is answered, or abstained on
when the model isn't confident or the region is unlike its training images.
The model refuses a series of another modality, and abstains on the whole
series when at least half of its regions are unfamiliar. Regions it answers
are drawn on the images and shown as a map of slices per label.

**Reporting.** The clinician records whether they agree with the model, or
didn't use it, and writes findings and an impression. A signed report can't
be edited: an amendment supersedes it and both are kept. A signed report
downloads as a DICOM Comprehensive SR that references every image in the
series, names the model and its output, and records the clinician's
agreement. For a series retrieved from a PACS, **Send to the PACS** stores
the report there with STOW-RS, under the original study.

### Try it on public scans

```bash
python scripts/fetch_dicom_samples.py
```

This downloads an abdominal CT (Pancreas-CT, 181 slices) and a prostate MRI
(Prostate-Diagnosis, 20 slices) from The Cancer Imaging Archive, under CC BY
3.0, into `data/imaging-samples/`, with the citations. Upload a folder on the
CT and MRI page.

To try a PACS, `docker compose -f deploy/docker-compose.yml --profile pacs up`
starts Orthanc with DICOMweb. The retrieval code is also tested against the
public Orthanc demo server (`RUN_LIVE_PACS=1 pytest tests/test_imaging.py`).

The demonstration CT model was trained on MedMNIST crops centred on single
organs, not on whole CT slices. On a real abdominal CT it abstains, because
78% of the regions are unlike its training images, and it refuses MRI. That
is the check doing its job. A model for real use needs training images cut
from whole series the way they'll be analysed, and local validation.

---

## Speed, caching and hardware

GroundCheck sizes itself to the machine it runs on, and does no work twice.

**Where work runs.** A question's embedding runs on the CPU: it takes a few
milliseconds, and it's safe to run from many requests at once. Batch work —
building the index, analysing a scan — runs on an NVIDIA or Apple GPU when
there is one (`BATCH_DEVICE=auto`), one job at a time. On an M1 laptop that
is 2,000 passages embedded in 2.2 seconds instead of 38.5, and a 181-slice CT
analysed in 3.4 seconds instead of 31.

> Don't set `EMBED_DEVICE=mps`. Apple's GPU aborts the process when several
> requests embed at the same time, which is why per-request work stays on the
> CPU.

**What is cached.**

| Cache | Holds | Dropped when |
| --- | --- | --- |
| Question embeddings | The vector for a question | The index changes |
| Sentence embeddings | Each passage's sentence vectors | The index changes |
| Answers | The decision, claims, sources and trace for an identical question with identical settings | `ANSWER_CACHE_SECONDS` (300) passes, or the index changes |

A cached answer is never shared across a change of documents or settings, and
questions about a specific patient, questions answered by a language model,
and generated test questions are never cached at all. Every request still
writes its own audit record, and a repeated refusal still adds to its review
case. `ANSWER_CACHE_SECONDS=0` turns the answer cache off.

**Sizing.** `app/resources.py` reads the cores, memory and accelerator this
process really has — including a container's cgroup limits, so a 2-core, 4 GB
pod sizes itself as 2 cores and 4 GB — and derives the embedding batch size,
the imaging batch size, how many series stay in memory, how large the
sentence cache is, how much memory a training run may use, and how many web
workers the machine can run. Override any of them with `EMBED_BATCH_SIZE`,
`IMAGING_BATCH_SIZE`, `IMAGING_CACHE_SERIES`, `SENTENCE_CACHE_SIZE`,
`TRAINING_CACHE_MB` or `WEB_WORKERS`.

**Measured on an M1 laptop, extractive mode:** 61 ms to answer a question,
9 ms to refuse one, 24 ms for a repeated question.

---

## Knowledge releases and rollback

Every rebuild of the search index, whether after a document is approved or
retired or by hand, makes a **release**: a snapshot of the passages and their
vectors, with the documents it holds and fingerprints of its content and the
formulary. Before a release goes live it's checked while live questions keep
using the current index:

- every must-refuse question in the golden set and every must-refuse patient
  scenario, when the demo corpus is included
- every test question added from a review

If any question that must be refused is answered, the release is **blocked**:
it never goes live, and the Monitoring page raises a critical alert.
Answerable questions that are now refused are listed, but don't block it.
Checks run in extractive mode, whatever model is configured, and record
nothing in the audit trail; with the demo corpus they take about 35 seconds.

A release that passes goes live straight away, or waits for an admin with
`RELEASE_AUTO_PROMOTE=false`. The **Releases** page shows the history, what
each release adds, updates or removes compared with the live one, and its
check. **Roll back** makes the previous release live again from its
snapshot, and any kept release can be made live again. The last
`RELEASES_KEEP` (10) snapshots are kept, and always the live one and the one
before it. The index in use when releases begin is recorded as the first.

---

## Monitoring and alerts

The **Monitoring** page (reviewers and admins) shows what's firing, the last
hour's questions, refusal rate and response time, and every check with its
threshold. Checks run every `ALERT_INTERVAL_SECONDS` (60) from the database,
so every instance agrees, and on demand with **Check now**:

| Check | Alerts when |
| --- | --- |
| Refusal rate | The last hour's refusal rate is `ALERT_REFUSAL_RISE` (15 points) or three standard errors above the last week's, with at least `ALERT_MIN_QUESTIONS` questions. Critical at twice that. |
| Response time | The last hour's 95th-percentile response time is above `ALERT_LATENCY_P95_MS` (15 s). |
| Question drift | The best search scores of the last day have shifted from the last month's: population stability index at least `ALERT_DRIFT_PSI` (0.25). People may be asking about things the documents don't cover. |
| Overdue reviews | Any review case is past due. Critical at `ALERT_OVERDUE_CRITICAL` (10). |
| Document expiry | An approved document expires within `ALERT_DOCUMENT_EXPIRY_DAYS` (14). Critical once expired. |
| Audit trail | The tamper-evident audit chain doesn't verify, checked every `ALERT_CHAIN_CHECK_MINUTES` (360). |
| Server errors | 5% of this instance's requests in 15 minutes failed. |
| Imaging failures | An imaging analysis failed in the last day. |
| Knowledge releases | The newest release failed its safety check. |

Test questions are left out. An alert fires once, stays firing while the
condition holds, can be acknowledged, and resolves by itself. With
`ALERT_WEBHOOK_URL` set, firing, escalation and resolution are posted as JSON
with a `text` field, which Slack and Microsoft Teams incoming webhooks accept.
Turn checks off with `ALERT_DISABLED_RULES=latency,imaging_failures`.

**Prometheus.** `/metrics` serves request counts and latency by route,
questions by decision and what drafted them, imaging analyses by outcome, open
and overdue review cases, and firing alerts. Set `METRICS_TOKEN` and scrape
with it as a bearer token; without a token, only requests from the same
machine are answered.

```yaml
scrape_configs:
  - job_name: groundcheck
    authorization: { credentials: "<METRICS_TOKEN>" }
    static_configs: [{ targets: ["groundcheck:8000"] }]
```

**Health probes.** `/healthz/live` answers while the process runs.
`/healthz/ready` returns 503 until the database answers and the search index
is loaded, for load balancers and Kubernetes readiness probes.

---

## Incident reporting

Anyone signed in can **report an incident**: from the Incidents page, from
an answer (with its audit ID filled in), or from a firing alert. An incident
records what it concerns (an answer or refusal, a patient safety check,
imaging, EHR integration, data protection, an outage, or something else),
the harm to a patient graded as in NHS patient safety reporting (none or near
miss, low, moderate, severe, death), when it happened, and a description.

Reviewers and admins see every incident; clinicians see the ones they
reported and can add comments. Reviewers take ownership, regrade the harm,
record the root cause, the actions taken and any external report, link a
hazard log entry, and close the incident. Closing needs a root cause and
actions. Every change and comment goes into the incident's timeline.

Two kinds of incident show a deadline and can't be closed until an external
report reference, or the reason it isn't reportable, is recorded:

- **Personal data breaches**: a notifiable breach must be reported to the
  data protection authority within 72 hours of becoming aware of it (UK and
  EU GDPR article 33). The deadline counts from when the incident was
  reported, and can be corrected.
- **Severe harm or death**: the organisation decides whether to report to
  the medical device regulator, such as the MHRA or the FDA's MedWatch.

Titles, descriptions, root causes, actions and comments are encrypted with
`DATA_ENCRYPTION_KEYS`. **Download CSV** exports the log for your safety
committee or quality system.

### Surveillance reports

The Review page's report tab also downloads a **post-market surveillance
report** for a period (90 days by default): what people asked, why answers
were refused, what reviewers found and how quickly, every incident with its
harm and whether a regulator decision is needed, the operational alerts that
fired, and which knowledge releases went live. It ends with the actions to
work through and a line for a quality or clinical safety lead to sign.
Regulated use expects this at regular intervals; GroundCheck gathers the
evidence, and a qualified person reviews and signs it.

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
| `FHIR_OPEN_SERVERS` | _empty_ | FHIR servers a patient can be loaded from without SMART, comma-separated. For development. |
| `SMART_CLIENT_ID` | _empty_ | GroundCheck's client ID registered with the EHR. |
| `SMART_CLIENT_SECRET` | _empty_ | The client secret, for confidential clients. |
| `SMART_ALLOWED_ISSUERS` | _empty_ | FHIR base URLs of EHRs allowed to launch GroundCheck, comma-separated. |
| `SMART_WRITE_NOTES` | `false` | Ask for write permission and let clinicians save reviewed answers as notes. |
| `EHR_CONTEXT_MINUTES` | `60` | How long a patient loaded from an EHR is kept. |
| `FHIR_LAB_MAX_AGE_DAYS` | `90` | Lab results older than this are flagged as possibly out of date. |
| `CDS_HOOKS_TRUSTED` | _empty_ | EHRs allowed to call the CDS services, as `issuer=JWKS URL` pairs. |
| `CDS_HOOKS_ALLOW_UNSIGNED` | `false` | Accept unsigned CDS calls. Testing only. |
| `FORMULARY_PATH` | `app/data/formulary.json` | Medicine rules for patient-aware checks. Replace the synthetic demo formulary before clinical use. |
| `TRAINING_DATA_DIRS` | `data/datasets` and your home folder | Folders the training studio may read images from, comma-separated. On a shared server, list only dataset folders. |
| `MODEL_LIBRARY_DIR` | `models/library` | Where trained models are saved. |
| `TRAINING_RUNS_DIR` | `models/runs` | Training run settings, progress and logs. |
| `MODEL_TARGET_ACCURACY` | `0.95` | Accuracy a model must show on answered validation images when setting its confidence threshold. |
| `MODEL_MIN_CONFIDENCE` | `0.5` | A trained model never answers below this confidence. |
| `BATCH_DEVICE` | `auto` | Where batch work runs: `auto`, `cuda`, `mps` or `cpu`. |
| `EMBED_DEVICE` | `cpu` | Where a question's embedding runs. Leave it on the CPU. |
| `ANSWER_CACHE_SECONDS` | `300` | How long an identical question keeps its answer. 0 turns it off. |
| `ANSWER_CACHE_SIZE` | `500` | How many answers to keep. |
| `RELEASE_CHECKS` | `true` | Check each rebuilt index against the safety tests before it goes live. |
| `RELEASE_AUTO_PROMOTE` | `true` | Put a release that passes live straight away. `false` waits for an admin. |
| `RELEASES_KEEP` | `10` | How many release snapshots to keep. |
| `RELEASES_DIR` | `INDEX_DIR/releases` | Where release snapshots are kept. |
| `OIDC_SITE_CLAIM` | _empty_ | A claim with the key of the person's site. Missing or unknown sites are refused. |
| `OIDC_ALL_SITES_VALUE` | _empty_ | The site claim value meaning every site. |
| `METRICS_TOKEN` | _empty_ | Bearer token for `/metrics`. Without it, only local requests are answered. |
| `ALERT_WEBHOOK_URL` | _empty_ | Where alerts are posted when they fire and resolve. |
| `ALERT_INTERVAL_SECONDS` | `60` | How often alert checks run. 0 turns background checks off. |
| `ALERT_DISABLED_RULES` | _empty_ | Checks to turn off, comma-separated. |
| `INSTANCE_NAME` | the host name | Named in alert notifications. |
| `IMAGING_DIR` | `data/imaging` | Where imported CT and MRI series are stored, encrypted when `DATA_ENCRYPTION_KEYS` is set. |
| `IMAGING_MAX_UPLOAD_MB` | `1024` | Largest imaging upload. |
| `DICOMWEB_URL` | _empty_ | A PACS or VNA's DICOMweb root, to search, retrieve series and send signed reports. |
| `DICOMWEB_AUTHORIZATION` | _empty_ | The `Authorization` header sent to the PACS, such as `Bearer <token>`. |
| `ORGANISATION_NAME` | _empty_ | Recorded as the verifying organisation in signed imaging reports. |
| `DATA_ENCRYPTION_KEYS` | _empty_ | Base64 keys, comma-separated, to encrypt stored questions, answers and review notes. The first encrypts. |
| `DATA_ENCRYPTION_RETIRED_KEYS` | _empty_ | Keys that only decrypt, for rotation. |
| `AUDIT_SIGNING_KEYS` | _empty_ | Base64 keys to sign the audit chain. The first signs. |
| `AUDIT_RETENTION_DAYS` | `0` | Delete audit records older than this when retention runs. 0 keeps them. |
| `REVIEW_RETENTION_DAYS` | `0` | Delete resolved review cases older than this when retention runs. 0 keeps them. |
| `API_DOCS` | `true` | Serve interactive API docs at `/docs`. Set `false` in production. |
| `REVIEW_QUEUE` | `true` | Open a review case for every refusal. |
| `REVIEW_SLA_HOURS` | `72` | When a refusal case is due. |
| `FLAGGED_SLA_HOURS` | `24` | When a flagged answer case is due. |
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
python scripts/run_eval.py                  # 2,008 golden cases, 16 probes, 383 patient scenarios
pytest -q                                   # 360 tests, including property-based fuzzing
python scripts/stress_eval.py --sample 5000 # a sample of the large evaluation
python scripts/stress_eval.py               # all 252,825 cases (about an hour on 7 cores)
```

Everything runs in extractive mode, so it's deterministic and offline. CI
(`.github/workflows/ci.yml`) builds the index, runs the evaluation, the test
suite and a stress sample on every push and pull request.

### The large evaluation

`scripts/stress_eval.py` generates 252,825 questions and patient scenarios,
with the expected decision taken from the corpus and the formulary, never from
GroundCheck's own code. Any answer to a question that must be refused fails
the run.

| Family | Cases | Must |
| --- | --- | --- |
| `unknown_entity` | 91,840 | refuse: medicines and conditions that don't exist |
| `population` | 60,175 | refuse: real topics for children, infants, ages 0 to 11, pregnancy |
| `near_miss` | 41,300 | refuse: one-letter misspellings of real names |
| `answerable` | 27,768 | answer: corpus questions in many phrasings, casings and spacings |
| `injection` | 12,056 | refuse: answerable questions wrapped in instructions to ignore the sources |
| `patient_*` | 12,765 | the formulary's own rules: allergy, interaction, child, pregnancy, missing data |
| `wrong_dose` | 3,970 | refuse: a real medicine asserted at a dose the sources don't give |
| `mixed_unknown` | 2,400 | refuse: a real medicine together with one that doesn't exist |
| `absent_fact`, `out_of_scope` | 680 | refuse: alcohol, grams, cures, other species, everyday questions |

A 5,000-case sample of the first run found two safety gaps that the 2,008-case
evaluation missed, both now fixed with regression tests: misspelled medicine
names were answered, and a dose stated in a question was accepted because an
unrelated medicine's passage happened to contain that number.

`tests/test_fuzz.py` checks the guards' invariants over thousands of generated
inputs with Hypothesis, and `scripts/stress_eval.py --families ...` runs one
family at a time.

### Security testing

| Check | Tool | Result |
| --- | --- | --- |
| Dependency vulnerabilities | `pip-audit -r requirements.txt` | 0 known advisories |
| Static analysis | `bandit -r app` | no medium or high findings |
| Web vulnerabilities | OWASP ZAP baseline and active API scan | 0 failures over 305 endpoint variants |

Re-run them before a release, and after upgrading dependencies. CI also
produces a CycloneDX bill of materials and a licence list for every build; see
[docs/third-party.md](docs/third-party.md).

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

**In a hospital or clinic network**, use the hardened stack in `deploy/`
(GroundCheck, PostgreSQL and HTTPS, with no internet access at runtime) and
follow [docs/on-premises.md](docs/on-premises.md): installation, single
sign-on, keys, scheduled jobs, backups, upgrades and a hardening checklist.

**As a public demo**, the app is one container that listens on the port given by `app_port` (7860 on
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
    training/          training studio: datasets, worker process, model library
    imaging/           DICOM import and de-identification, analysis, reports, DICOMweb
    releases.py        checked knowledge releases, promotion and rollback
    sites.py           several hospitals or clinics in one installation
    resources.py       the machine's cores, memory and accelerator, and the sizes they imply
    monitoring.py      Prometheus metrics, alert rules and notifications
    incidents.py       incident reporting, investigation and regulator deadlines
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
- The audit trail is stored in the database. Several instances need a shared
  database such as PostgreSQL.
- The dosage guard handles digits and written-out numbers up to common ranges,
  but not every exotic format.
- The coverage guard works on words. Contractions and common conversational
  words are handled, but unusual phrasing can still cause a refusal.
- De-identification of questions and DICOM text is rule-based. It catches
  common names, identifiers and dates, and a DICOM series' own patient name
  and IDs, but it can miss unusual ones. Pixel data isn't scanned for
  burned-in text; images that declare it are refused.
- The demonstration imaging models were trained on small public datasets and
  abstain on most real scans. They show the workflow, not diagnostic
  performance.
- The safety case summary and hazard log are tools for your own clinical
  safety process. GroundCheck is not a certified medical device.

---

## Changes

[CHANGELOG.md](CHANGELOG.md) lists what each release changed, and which
problems it fixed.

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

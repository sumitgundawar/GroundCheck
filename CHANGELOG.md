# Changelog

Dates are when the work landed on `main`. Versions follow the app's
`config.VERSION`.

## 3.0.0 — unreleased (branch v3)

### Service API

- **API keys.** `gck_` keys with scopes (`ask`, `verify`, `usage`, `audit:read`), site binding, expiry,
  revocation, per-key rate limit and monthly quota. Administered at `/api/keys`; creation and revocation
  are chained into the audit trail without the secret.
- **`/api/v1`** (`ask`, `verify`, `usage`, `audit/{id}`, `health`): one error envelope, `X-Request-Id`,
  `X-RateLimit-*` and `Retry-After`, exact metering of successes and failures, usage by key, day, site,
  endpoint or decision with CSV, and a committed contract (`openapi/v1.json`) with a compatibility test.
- **Verify an answer from any model.** `POST /api/v1/verify` checks claims, doses, negations, frequencies,
  routes, populations, citations and the patient against the approved sources, and says whether the
  answer addresses the question. Doses in another unit are compared as quantities; millilitres convert
  only when a source states the strength. `scripts/verify_eval.py` gates CI on zero wrong answers passed;
  `scripts/build_verify_holdout.py` adds mutation families that are never tuned on.
- A knowledge pack's evaluation cases now gate a release: any case decided differently blocks it.
- Administrative events in the audit chain are skipped by every report, monitor, list and lookup.

### Proposed regimens

- **A question that proposes a regimen** ("Is 15 mg twice a day okay for X?") is read into dose, frequency,
  route, duration, daily maximum and timing and compared, part by part, with the cited source (`app/proposal.py`).
  The answer leads with `No — the source gives 15 mg once daily, not 15 mg twice daily.` or `Yes — this matches
  the source: ...`, then the cited sentences; units are compared as quantities (0.015 g is 15 mg), and
  `bd`, `tds`, `q8h` and "every 8 hours" are doses a day. Anything that cannot be read or checked (no unit,
  tablets, mg/kg, "1,000 mg", a second medicine, a source that contradicts itself, a question about whether an
  amount is "too much") is refused with `proposal_unverifiable` rather than guessed. Same in extractive and model
  modes (the model is not asked), and over `/api/v1/ask`, which returns `proposal_check` (`match`, `mismatch` or
  `unverifiable`, with each part's proposed and documented value and source id).
- A proposed dose or route that the source documents but the coverage guards used to refuse (`stated_value_unsupported`,
  `form_unsupported`, `term_not_covered` on "subcut", `limit_not_stated`) now gets the same verdict: `No — the source
  gives 15 mg once daily, not 50 mg once daily.` The proposed value appears only in the "not" clause. It applies only
  to a question (a "?" or "correct", "okay" and the like) about one named medicine, with route synonyms (subcut/SC,
  IV, PO/by mouth, IM) mapped and no patient age or weight in the text; any other uncovered term, a form the
  question names that is not a route (XR, tablets), or a source that does not state that part still refuses.
  Matching proposals that were over-refused ("subcut", "25000 mcg", "no more than 40 mcg per day") now get "Yes".
- A maximum daily dose is judged only against a maximum the source states; the usual regimen is not a maximum, so
  with none stated the question is refused. A negatively phrased question ("not correct?", "isn't", "never",
  "false", "contraindicated", "or not") and an open-ended duration ("at least", "or longer", "up to") are refused
  as `proposal_unverifiable` rather than given a yes or no. "What is the dose of X 15 mg twice daily?" gets the
  verdict on the stated regimen.
- Model claims are cleaned before the guards: bidirectional and zero-width controls are removed and a citation that
  names no source is dropped (a claim left with no real source is ungrounded). Two approved documents that give
  different doses of the same medicine are refused with the new reason code `source_conflict` (flagged for review)
  instead of one being served. Both SDKs redact the API key from exception messages if a server echoes it.
- Decimal commas ("0,5 mg") and "by mouth" are rewritten before the guards read them. `scripts/proposal_eval.py` gates on zero wrong answers over
  `eval/proposal_cases.json`.

### Tenants and sites

- **Tenant isolation.** A site's admins and staff no longer see other sites through `GET /api/sites`, the
  sites list in `GET /api/users`, `GET /api/sources` and `/api/sources/{id}`, `GET /api/releases`
  (other sites' documents, who promoted a release, reasons that name a document) or the index status.
  Documents carry a site (migration 0014; documents that already existed are shared by the group). Retrieval,
  the vocabulary the coverage guard uses, the answer cache key, document versions, duplicate checks and the
  replacement of older versions are all per site; shared documents stay visible to every site. A person with no
  site is answered from shared documents only. Approving, rejecting, retiring or evaluating a document is limited
  to its own site (shared documents to the group), and only an admin with no site promotes or rolls back a release,
  runs retention, edits the hazard log or manages local models and training. Saving an answer to the record checks
  the site. `tests/test_tenant_isolation.py` (every GET route classified, every role) and
  `tests/test_tenant_documents.py` cover it.
- **`tenant create`** (`python -m app.cli tenant create`): a site, its admin, an API key and a quota in one step, undone
  entirely if any part fails, audited without the secret.
- **Fixed:** a person's site was remembered by user id alone, so after the database changed a stale site id could
  fail the audit record's foreign key and lose the record. It is now remembered per database.

### Knowledge

- **Signed knowledge packs** (`app/packs.py`): install, verify and uninstall a publisher's documents with their
  evaluation cases; every case must pass before the pack can go live, and overrides are audited.
- **DailyMed SPL importer** and **formulary adapters** for loading drug labels and medicine rules from published
  sources, with fixtures and tests for malformed and hostile input.
- **Model providers** (`LLM_PROVIDER`): OpenAI-compatible, Anthropic, Azure OpenAI, Ollama or none, with the
  `GROQ_*` variables kept as aliases. 429, 5xx, timeouts, malformed and cut-off replies all fall back safely.

### Clinical integration

- **EHR write-back.** Optional FHIR `Provenance` (`SMART_WRITE_PROVENANCE`) and `AuditEvent`
  (`SMART_WRITE_AUDITEVENT`) next to a saved note; both off by default, validated as FHIR R4, and a failed write
  never costs the note. `POST /api/ehr/notes` reports `provenance` and `audit_event` status.
- **CDS Hooks conformance.** hookInstance UUIDs, JWT checks including `jti` replay, `fhirServer` fallback, service id
  matched to the hook, and malformed, deeply nested or odd feedback bodies answered as 400 or 200, never 500.
- **FHIR profile validation** of what is read and written (`app/fhir_profiles.py`).

### Imaging

- **Burned-in text.** Every image's pixels are searched for text whatever its tags say; text is blacked out
  (`IMAGING_BURNED_IN_ACTION=redact`, default) or the image refused (`block`), then searched again. OCR is used when
  installed (`IMAGING_OCR=auto|off`). On a synthetic set recall is 0.991 and precision 0.994; "no text found" is never
  called clean.
- **NIfTI volumes** can be imported, viewed, analysed and used for training, with volume-level splits.

### Evidence for buyers

- **Feedback** on answers (`/api/audit/{id}/feedback`): one rating per person, a "wrong" opens a review case, identifiers
  removed from comments, the signed audit record unchanged.
- **Adoption metrics** and a **pilot report** (`/api/metrics/adoption`, `/api/governance/pilot-report`), and a signed
  **evidence pack** (`/api/governance/evidence-pack`) with per-file hashes, a secret scan and the SBOM when present.
- **Compliance registers:** 104 controls across HIPAA, GDPR, SOC 2 and ISO 27001 (`scripts/controls_report.py --check`), 49
  requirements and 12 hazards traced to tests (`scripts/trace_matrix.py --strict`, pytest marker `req`), and the list
  of third-party software and models (`scripts/soup_list.py`).
- **Install telemetry** (`TELEMETRY`, off by default): weekly bucketed counts only, fields documented and tested against
  the README.
- **Held-out evaluation split** (`scripts/eval_split.py`) by entity with a secret seed kept outside the repository, and a
  `heldout` block in `run_eval.py`. The committed golden set is not split until the owner applies it.
- **Model benchmark** (`scripts/benchmark.py run|report|verify-report`): suites with guards on and off, unsafe-answer and
  over-refusal rates, cost, and a signed HTML report whose caveats cannot be switched off.

### Languages, tracing and recovery

- **Languages.** Questions are language-checked first; a language with no pack is refused as `unsupported_language`
  (`SUPPORTED_LANGUAGES`, default `en`). Spanish is supported by closed-vocabulary translation (`app/langpack.py`): an
  unknown word, unit or ambiguous number refuses. Answers stay English source text. On the Spanish test set (406 cases)
  nothing unsafe was answered; on wording the pack was not built from about 12% is answered and the rest refused.
  `EMBEDDING_MODEL` names the embedding model (`EMBED_MODEL` still works).
- **Interface translation.** Every visible string in the dashboard goes through `t()` with `web/i18n/en.json` and a Spanish
  `es.json`, a locale selector and a pseudo-locale for layout tests; tests fail on any visible literal that is not keyed.
- **Tracing.** Optional OpenTelemetry export (`OTEL_EXPORTER_OTLP_ENDPOINT`, `OTEL_EXPORTER_OTLP_PROTOCOL`,
  `OTEL_SERVICE_NAME`): one span per pipeline stage, fixed attributes, never the question or who asked.
- **Restore drill.** `scripts/db_restore_drill.py [--tamper-check]` backs up, wipes and restores a PostgreSQL or SQLite
  database and fails unless row counts, the audit chain head and audit verification match; CI runs it against PostgreSQL.

### Quality assurance fixes

- `/api/v1/verify` no longer passes an answer whose overdose is a sentence of only a quantity ("Also 15 g."), and reads
  `b.i.d.`, `t.i.d.`, `q.i.d.`, `bd`, `tds` and `qds` as frequencies.
- Answers written by a language model are also checked for frequency, duration, numbers and negation against the source,
  not only for word overlap.
- A path id beyond the database's integer range is 404, not 500; NaN or Infinity in a body is 422 without echoing the input.
- Free text is de-identified before it is stored or exported: CDS feedback override reasons, patient allergies, medicines and
  conditions in the audit record, incident and review-case text, and uploaded file names.
- De-identification now finds lower-case and initial-style names, e-mail written `[at]`/`[dot]`, digits spoken in words,
  and accented names.
- A child the formulary does not cover is refused as `paediatric_not_covered` even when other patient data is also
  missing, instead of being asked for that data first.

### SDKs

- `sdk/python` and `sdk/typescript` (ESM and CommonJS), each tested against a mock server, with the Python client also
  run against the real server.

## 2.0.0 — unreleased

### Refusals say what happened

- **Reason codes.** Every refusal carries a stable `reason_code`
  (`term_not_covered`, `population_not_covered`, `dose_unsupported`,
  `patient_unsafe`...) next to the unchanged `refused_reason` text. Review
  reports group by the code, and by phrase for older records.
- **Routing is real.** The answer says "routed for review" only when a review
  case was opened, and says why not otherwise ("not sent for review (this was a
  test run)"). `review` on the response gives `routed` and the `case_id`; a
  remembered answer re-checks this for each request.
- **Explanations and redaction on the response.** `explanation` names the
  missing terms and what to try instead; `redaction` gives the placeholder
  counts and the question with placeholders, never the removed values. Claims
  carry `relevance` and the `support` sentence and offsets they were taken from.

### De-identification

- Candidate spans with scores, checksums (NHS mod-11, Luhn for cards and
  provider numbers, mod-97 IBANs, social security rules), real calendar dates,
  and cue words within four words. New `[CARD]` placeholder.
- Names from titles in any case, relation and role cues ("husband Raj", "seen
  by nurse Kelly Brown"), "SURNAME, Given", and a bundled list of given names.
  Words the documents know, medicine names and months are never names, and a
  dose or an age under 90 is never removed.
- Optional named-entity recognition (`DEID_NER=auto|off|on`) when Presidio or
  spaCy is installed; nothing changes when it is not.
- `scripts/pii_eval.py` measures it on `eval/pii_cases.json` (389 synthetic
  sentences, 340 gold spans, including "John Smith has a fever" and "Sarah, 34,
  pregnant"): precision 0.90 and recall 0.67 before, 1.00 and 1.00 now. The
  cases and the detectors were written by the same people, so read the number
  as a regression guard, not a forecast.

### Coverage guard

- A question word has to be found in the sources only if it is a qualifier, a
  form or species, a clinical term (`app/data/clinical_terms.txt`) or not
  ordinary English (`app/data/common_words.txt`): an invented or misspelled
  medicine is still caught, "email", "meeting" and "yearly" no longer refuse a
  question. Over-refusal on 320 everyday phrasings (`eval/phrasing.json`) fell
  from 87.5% to 0.3%.
- A term the index has but the retrieved passages lack gets one more search,
  limited to the topic the question names. Its refusal reads "any retrieved
  source" rather than "any trusted source".

### Extractive answers

- Sentences are pooled from the passages about the topic, retrieved or not, and
  scored 0.6 meaning, 0.25 the question's own words, 0.15 whether the section
  is the kind that answers (dose, side effects, symptoms...); at most three
  are kept, and each part of a two-part question needs its own.
  `scripts/answer_quality_eval.py`: claim precision 0.20 to 1.00, section
  recall 0.79 to 1.00, claims per answer 3.99 to 1.25.

### Guards

- A generic sentence that names no medicine ("Once X settles, the dose is
  tapered") must belong to the passage it cites: every name and clinical term
  in it has to be in that passage. Wrong claims accepted by the guard
  evaluation fell from 6 to 0; wrong doses stay at 0.

### Evaluation

- `run_eval.py` also writes `pii`, `answer_quality` and `phrasing` blocks.
- Two patient scenarios changed expectation, because the answer now states the
  Mendel solution regimen where it used to quote another sentence: an 8-year-old
  asked about the dose, or the standard course, is refused with the
  weight-based dose the formulary gives.

## 1.0.0 — 2026-09-18

The release that turns the reference pipeline into a platform a clinic or
hospital can run. It has not been used on real patients, and it is not a
cleared medical device: see "Honest limitations" in the README for what still
needs people rather than code.

### Answers and safety

- **Patient-aware checks.** Answers are checked against the patient in front of
  the clinician: allergies, contraindicated conditions, interactions with
  current medicines, kidney function (eGFR or creatinine clearance), liver
  function, weight, age, pregnancy, breastfeeding, lab results and high-alert
  medicines, from a structured formulary. A problem that makes the answer
  unsafe refuses it.
- **Hybrid retrieval** (keyword and embeddings), so a rare look-alike name
  isn't blurred into a similar one.
- **Guards fixed by testing at scale** (see Testing): misspelled medicine
  names, doses supported by the wrong medicine's passage, other species, and
  prompt injections written with unusual spacing.

### Knowledge

- **Your own documents:** import PDF, Word, HTML and Markdown, split by their
  own headings, with owners, versions, effective and expiry dates, and
  approval by someone other than the uploader.
- **Knowledge releases:** every index rebuild is checked against the safety
  tests before it goes live, blocked if it answers anything that must be
  refused, and reversible in one step.

### Imaging

- **Training studio** for imaging models on your own hardware, with a model
  library, abstention thresholds and an unfamiliar-image check.
- **CT and MRI:** DICOM upload and PACS retrieval over DICOMweb,
  de-identification to the PS3.15 profile, a viewer with windows and
  orientation, model analysis with regions, and signed reports exported as
  DICOM Structured Reports and sent back to the PACS.

### Clinical governance

- Review queue, hazard log, safety case summary, and a post-market
  surveillance report.
- Incident reporting graded by harm, with the GDPR 72-hour and medical device
  regulator decisions blocking closure until they're recorded.

### Integration

- **EHR:** FHIR R4 patient loading, SMART on FHIR launch, CDS Hooks cards on
  medication orders, and reviewed answers written back as preliminary notes.
- **Single sign-on** with OpenID Connect, roles and sites from claims.

### Running it

- Accounts, roles, two-factor, PostgreSQL and MySQL support, encryption at
  rest, a signed tamper-evident audit trail, retention and rekeying.
- **Several sites** in one installation, each seeing only its own records.
- **Monitoring:** Prometheus metrics, health probes, and alerts on refusal
  rates, response times, drift from the documents, overdue reviews, expiring
  documents, the audit trail and failed releases.
- **Deployment:** hardened Docker Compose with PostgreSQL and Caddy, and a
  Helm chart for Kubernetes.
- **One container on its own,** for a small site behind an existing reverse
  proxy: `docs/on-premises.md` lists the settings that keep the database,
  index and audit trail on the mounted volume.

### Speed

- Extractive answers take 61 ms instead of 665: the sentences of every
  retrieved passage are embedded in one call, not one call each.
- Building the index and analysing a scan run on an NVIDIA or Apple GPU when
  there is one: 2,000 passages embed in 2.2 s instead of 38.5, and a
  181-slice CT is analysed in 3.4 s instead of 31. A question's own embedding
  stays on the CPU, where many requests can run at once.
- Question and sentence embeddings are kept, and an identical question keeps
  its answer for `ANSWER_CACHE_SECONDS`. Never across a change of documents,
  settings or patient, and every request is still audited.
- Batch sizes, caches and the recommended worker count come from the real
  cores, memory and accelerator, including a container's cgroup limits.
- Measured: 100,000 requests through PostgreSQL at 30 a second with 40
  concurrent users, no errors, and every audit record present and verified.

### Testing

- 2,008-case evaluation, 383 patient scenarios and 16 adversarial probes gate
  the build, as before.
- **265,778 generated questions and patient scenarios** (`scripts/stress_eval.py`),
  with expectations from the corpus and the formulary.
- **Property-based fuzzing** of the guards' invariants.
- **Security:** `pip-audit`, `bandit`, and OWASP ZAP baseline and active scans
  (`scripts/security_scan.sh`); a CycloneDX bill of materials and licence list
  from CI. The scans' one medium alert is the cross-origin policy the CDS Hooks
  specification requires, and the README says why it stands.
- **Soak test** (`scripts/soak.py`): 100,000 requests on PostgreSQL with 40
  users, no errors, no unsafe answers, and every audit record present and
  chained.
- **A browser journey** through the app (`tests/ui/`), run in CI: questions and
  their citations, the trace and audit record, a patient loaded, every page,
  themes, the keyboard and a phone-sized screen.
- **The release image, verified as shipped:** signing in, refusing, answering,
  the cache, metrics, the hardware panel and the audit chain all checked
  inside the container, and the data checked again after a restart.

### Fixed

- Embeddings run on the CPU by default: on Apple silicon the GPU aborted the
  process when several requests embedded at once.
- Database migrations take a lock, so several workers or replicas can start
  together.
- Release snapshots are safe when instances share a volume.
- The container image now has the PostgreSQL and MySQL drivers, owns its data
  volume, builds its index before recording a release, and stops cleanly on
  signals.
- Training checkpoints load tensors and plain data only, never code.
- **From an independent audit of the repository,** five ways an unsafe answer
  could get through: a dose stated above a weight-based maximum was recorded as
  a note rather than blocked; a current medicine written as "Tessorin10mg"
  matched nothing, so every rule about it was skipped; an interaction written on
  one medicine's entry was invisible from the other side; an unrecorded sex
  silently meant male in the creatinine calculation, moving a patient across a
  dose rule; and a modified-release or intravenous form borrowed the plain
  product's dose.
- Two-factor codes could be guessed without limit; they now count against the
  same lockout as passwords.
- The audit chain did not cover which site a record belonged to, so a record
  could be moved between sites without breaking it.
- A release that failed part-way could leave the search index and the database
  disagreeing about what was live; startup now restores from the release's own
  snapshot.
- With a child's details loaded, an answer about a medicine's place in
  treatment now says the formulary covers adults only. The dose questions were
  already refused; the informational ones passed with nothing said.
- The Medicines page no longer runs off the side of a phone screen.
- 31 dependency advisories closed by upgrading FastAPI, Starlette,
  cryptography, transformers, sentence-transformers and pytest.

## 0.1.0 — earlier

The grounded retrieval core: the eleven-stage pipeline, deterministic guards,
citations, the audit trail, the tuning dashboard, the JSON API, local AI models
through Ollama, and the website.

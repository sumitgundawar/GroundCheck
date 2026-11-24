# Changelog

Dates are when the work landed on `main`. Versions follow the app's
`config.VERSION`.

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

# Changelog

Dates are when the work landed on this branch. Versions follow the app's
`config.VERSION`.

## 1.0.0 — unreleased

The release that turns the reference pipeline into a platform a clinic or
hospital can run. Everything below is in this branch and not yet merged.

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
- **252,825 generated questions and patient scenarios** (`scripts/stress_eval.py`),
  with expectations from the corpus and the formulary.
- **Property-based fuzzing** of the guards' invariants.
- **Security:** `pip-audit`, `bandit` and OWASP ZAP baseline and active scans;
  a CycloneDX bill of materials and licence list from CI.

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
- 31 dependency advisories closed by upgrading FastAPI, Starlette,
  cryptography, transformers, sentence-transformers and pytest.

## 0.1.0 — earlier

The grounded retrieval core: the eleven-stage pipeline, deterministic guards,
citations, the audit trail, the tuning dashboard, the JSON API, local AI models
through Ollama, and the website.

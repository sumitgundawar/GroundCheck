# Contributing to GroundCheck

Thanks for helping. GroundCheck exists to show that a clinical AI system can
refuse when it isn't sure, so the most valuable contributions are the ones that
make its refusals more correct: a question it answered but shouldn't have, a
check that can be fooled, or an evaluation case that exposes either.

This guide covers how to report problems, how to set up a development
environment, and what a pull request needs before it can be merged.

## Contents

1. [Ground rules](#ground-rules)
2. [Reporting issues](#reporting-issues)
3. [Development setup](#development-setup)
4. [Making a change](#making-a-change)
5. [Pull request checklist](#pull-request-checklist)
6. [Changes to the guards](#changes-to-the-guards)
7. [The website](#the-website)
8. [Licensing](#licensing)

## Ground rules

- **Never commit real patient data.** No real names, identifiers, records,
  scans or free text from a clinical system, not even in a test or an issue.
  The corpus is synthetic on purpose. If you need an example, invent one.
- **Never commit secrets.** API keys are read from the environment. `.env` is
  git-ignored; keep it that way.
- **Safety beats coverage.** A change that answers more questions but lets a
  single must-refuse case through will not be merged.
- Be respectful and assume good intent in issues, reviews and discussions.

## Reporting issues

Open an issue with the form that fits:
[github.com/sumitgundawar/GroundCheck/issues/new/choose](https://github.com/sumitgundawar/GroundCheck/issues/new/choose)

| You found | Use | What helps most |
| --- | --- | --- |
| An answer that should have been a refusal | **Unsafe answer** | The exact question, the decision, the refusal reason or answer text, and the audit ID |
| A refusal that should have been an answer | **Bug report** | The question, the trace stage that refused, and why the sources do cover it |
| Something broken: a crash, a setup failure, a UI bug | **Bug report** | Steps to reproduce, expected and actual behaviour, OS and Python version |
| An idea or a missing capability | **Feature request** | The problem you're trying to solve, before the solution |

Before opening an issue, search the existing ones. If you find a match, add
your details there instead of opening a duplicate.

**Security vulnerabilities are not issues.** Don't open a public issue for a
way to bypass a guard in a deployed system, leak data, or exhaust a provider
quota. Follow [SECURITY.md](SECURITY.md) instead.

## Development setup

Requirements: Python 3.12 (3.11 also works), git, and about 1 GB of disk for
the embedding model and dependencies. Docker is optional.

```bash
git clone https://github.com/sumitgundawar/GroundCheck.git
cd GroundCheck

python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt

python scripts/build_index.py      # downloads the embedding model once
uvicorn app.main:app --reload --port 8000
```

Open http://localhost:8000. Without `GROQ_API_KEY` the app runs in extractive
mode, which is all you need for most work. The decision logic is identical in
both modes.

## Making a change

1. **Start from an issue.** For anything larger than a small fix, comment on an
   issue (or open one) first, so we can agree on the approach before you spend
   time on it.
2. **Branch from `main`** with a short, descriptive name, for example
   `fix/coverage-guard-age-terms`.
3. **Keep the change focused.** One concern per pull request. Refactors go in
   their own pull request.
4. **Match the surrounding code.** Type hints on public functions, docstrings
   that explain *why*, and no new dependencies without discussion.
5. **Run the checks** before pushing:

   ```bash
   python scripts/run_eval.py   # must report every must-refuse case as refused
   pytest -q                    # the full suite, offline
   ```

   CI runs the same commands on every push and pull request.

6. **Write the commit message** in the imperative mood ("Fix coverage guard
   for age qualifiers"), with a body that explains why the change is needed.

## Pull request checklist

A pull request is ready for review when:

- [ ] It links the issue it resolves.
- [ ] `python scripts/run_eval.py` passes, with **zero** must-refuse cases
      answered.
- [ ] `pytest -q` passes.
- [ ] New behaviour has tests, and a bug fix has a test that failed before it.
- [ ] Any change to answer or refuse behaviour includes the before and after
      evaluation numbers in the description.
- [ ] Documentation (README, configuration table, website) is updated if
      behaviour or configuration changed.
- [ ] No real patient data, secrets or generated files (`index/`, `audit/`,
      `eval/eval_summary.json`) are committed.

## Changes to the guards

The guards in `app/guards_input.py` and `app/guards_output.py` decide what
users see, so they get extra scrutiny:

- **Deterministic only.** A guard's decision must not depend on a model call,
  randomness or the network. A model may *corroborate* (as the LLM judge
  does), but must never be able to turn a refusal into an answer.
- **Add evaluation cases.** A guard change should come with cases in
  `eval/golden.json` (generated by `scripts/generate_golden.py`) or hand-written
  probes in `eval/adversarial.json` that show what it fixes.
- **Report both directions.** State how many must-refuse and answerable cases
  change, even if the change is an improvement.
- **Explain threshold changes.** If you change a default in `app/config.py`,
  include the evaluation results that justify the new value.

## The website

The project website (groundcheckhealth.com) lives in [`site/`](site/). It is
static HTML, CSS and JavaScript with no build step. See
[site/README.md](site/README.md) for how to run and check it locally.

## Licensing

GroundCheck is released under the [MIT License](LICENSE). By submitting a
contribution, you agree that it is licensed under the same terms.

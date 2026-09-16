## What and why

<!-- What does this change, and why is it needed? Link the issue: "Closes #123". -->

## Evaluation

<!-- Required if this can change an answer or refuse decision. Paste the
     summary line from `python scripts/run_eval.py` before and after. -->

| | Before | After |
| --- | --- | --- |
| Must-refuse correct | | |
| Answerable correct | | |
| Adversarial probes | | |

## Checklist

- [ ] `python scripts/run_eval.py` reports zero must-refuse cases answered
- [ ] `pytest -q` passes
- [ ] New behaviour has tests; a bug fix has a test that failed before it
- [ ] README, configuration table or website updated if behaviour changed
- [ ] No real patient data, secrets or generated files committed

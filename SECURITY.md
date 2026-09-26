# Security policy

## Supported versions

GroundCheckHealth is developed on the `main` branch. Security fixes are made on
`main`; there are no separately maintained release branches.

## Reporting a vulnerability

**Please don't report security vulnerabilities in public issues, pull requests
or discussions.**

Report privately through GitHub:
[Report a vulnerability](https://github.com/sumitgundawar/GroundCheck/security/advisories/new)

Include:

- a description of the problem and its impact
- the steps, or the exact input, needed to reproduce it
- the commit or version you tested
- any suggested fix, if you have one

We aim to acknowledge reports within a week and will keep you updated as the
fix progresses. You'll be credited in the advisory unless you'd rather stay
anonymous.

## What counts as a vulnerability

In scope:

- A way to make a deployed instance **answer a question its guards should
  refuse**, reliably, by crafting the input (for example, bypassing the
  injection or dosage checks)
- **Personal data** reaching logs, the audit trail or a model provider despite
  the redaction guard
- Exposure of **API keys** or other secrets
- Bypassing the **rate limit** to exhaust a provider quota
- Path traversal, injection or remote code execution in the API

Out of scope, and better reported as a regular issue:

- A single question that is answered but should be refused, with no general
  technique behind it. This is a correctness bug. Please use the **Unsafe
  answer** issue form, which is exactly where we want it.
- Findings that need an attacker who already controls the host, the corpus
  file or the environment variables

## Safe harbour

We won't pursue or support legal action against good-faith research that
follows this policy, avoids privacy violations and service disruption, and
gives us reasonable time to fix a problem before it is disclosed.

## Not a medical device

GroundCheckHealth is research and engineering software. It is not a medical device,
has not been clinically validated, and must not be used to make clinical
decisions.

# groundcheck-client

Python client for the GroundCheck verification API. Standard library only, Python 3.9 or later.

```python
from groundcheck_client import Client, Patient

client = Client("gck_your_key", base_url="https://groundcheck.example.org")

result = client.verify(
    question="What is the usual adult dose of Zentrilex?",
    answer="Take 10 mg once daily.",
    sources=["DOC1-001"],
    patient=Patient(age_years=50, egfr=80),
)
if result.passed:
    print("verified", result.audit_id)
else:
    print(result.decision, result.reason_code, result.explanation.summary)

reply = client.ask("What is the usual adult dose of Zentrilex?")
print(reply.answered, reply.answer_text)
```

The key defaults to `$GROUNDCHECK_API_KEY` and the base URL to `$GROUNDCHECK_BASE_URL`.

## Behaviour

- **Retries.** A 429 or 5xx response is retried (default 2 retries) with exponential backoff, honouring `Retry-After`.
  A `Retry-After` longer than `max_retry_wait` (30 s) raises `RateLimitError` at once. A dropped connection or timeout is
  retried for GET only, never for POST, so a request is not processed twice. Other 4xx responses are never retried.
- **Errors.** `BadRequestError` (400), `AuthenticationError` (401), `PermissionDeniedError` (403), `NotFoundError` (404),
  `PayloadTooLargeError` (413), `UnprocessableError` (422), `RateLimitError` (429, with `retry_after`), `ServerError` (5xx),
  `APIConnectionError`, `APITimeoutError`, `InvalidResponseError` (a reply that isn't valid JSON or lacks a field). All derive
  from `GroundCheckError` and carry `status`, `code` and `request_id`.
- **Safety.** `VerifyResponse.passed` is true only for an exact `"pass"`; `AskResponse.answered` only for a checked answer.
  A response with a missing or mistyped field raises rather than being guessed.
- **Secrets.** The key is never shown in `repr`, logs or exceptions, and a `Client` can't be pickled. Plain `http` is refused
  for non-local hosts unless `allow_insecure_http=True`. Redirects are never followed.
- **Timeout.** Applies to connecting, to each read, and as a limit on reading a body.

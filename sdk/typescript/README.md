# @groundcheck/client

TypeScript client for the GroundCheck verification API. Uses the platform `fetch` (Node 18 or later, edge runtimes, browsers behind a proxy), has no runtime dependencies, and ships ESM, CommonJS and types.

```ts
import { GroundCheck } from "@groundcheck/client";

const client = new GroundCheck({ apiKey: "gck_your_key", baseUrl: "https://groundcheck.example.org" });

const result = await client.verify({
  question: "What is the usual adult dose of Zentrilex?",
  answer: "Take 10 mg once daily.",
  sources: ["DOC1-001"],
  patient: { ageYears: 50, egfr: 80 },
});
if (result.passed) console.log("verified", result.auditId);
else console.log(result.decision, result.reasonCode, result.explanation.summary);

const reply = await client.ask({ question: "What is the usual adult dose of Zentrilex?" });
console.log(reply.answered, reply.answerText);
```

The key defaults to `GROUNDCHECK_API_KEY` and the base URL to `GROUNDCHECK_BASE_URL` where an environment exists.

## Behaviour

- **Retries.** A 429 or 5xx response is retried (default 2 retries) with exponential backoff, honouring `Retry-After`. A
  `Retry-After` longer than `maxRetryWaitMs` (30 s) throws `RateLimitError` at once. A network failure or timeout is retried
  for GET only, never for POST. Other 4xx responses are never retried.
- **Errors.** `BadRequestError` (400), `AuthenticationError` (401), `PermissionDeniedError` (403), `NotFoundError` (404),
  `PayloadTooLargeError` (413), `UnprocessableError` (422), `RateLimitError` (429, with `retryAfter` in seconds), `ServerError`
  (5xx), `APIConnectionError`, `APITimeoutError`, `InvalidResponseError`, `InvalidArgumentError` (rejected before sending). All
  extend `GroundCheckError` and carry `status`, `code` and `requestId`.
- **Safety.** `VerifyResponse.passed` is true only for an exact `"pass"`; `AskResponse.answered` only for a checked answer. A
  response with a missing or mistyped field throws rather than being guessed. `NaN`, `Infinity`, unpaired surrogates and
  unknown patient fields are rejected instead of being sent.
- **Secrets.** The key is kept in a private field: it does not appear in `JSON.stringify`, `console.log`, `String()` or errors.
  Plain `http` is refused for non-local hosts unless `allowInsecureHttp` is set. Redirects are never followed.
- **Timeout.** `timeoutMs` covers the whole call, body included. Pass `{ signal }` as the second argument to cancel a call.

## Build and test

    npm install
    npm run build      # dist/esm and dist/cjs
    npm test           # vitest, against a mock HTTP server using responses recorded from the service

import { readFileSync } from "node:fs";
import { inspect } from "node:util";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import {
  APIConnectionError,
  APITimeoutError,
  AuthenticationError,
  BadRequestError,
  GroundCheck,
  GroundCheckError,
  InvalidArgumentError,
  InvalidResponseError,
  NotFoundError,
  PayloadTooLargeError,
  PermissionDeniedError,
  RateLimitError,
  ServerError,
  UnexpectedStatusError,
  UnprocessableError,
  parseRetryAfter,
} from "../src/index.js";
import type { ClientOptions } from "../src/index.js";
import { MockServer } from "./mockServer.js";

const KEY = "gck_s3cretvalueABCDEFGHIJKLMNOP";
const fixture = (name: string) => JSON.parse(readFileSync(new URL(`./fixtures/${name}`, import.meta.url), "utf8"));
const ASK = fixture("ask_answer.json");
const REFUSAL = fixture("ask_refusal.json");
const BLOCKED = fixture("ask_patient_blocked.json");
const PASS = fixture("verify_pass.json");
const FAIL = fixture("verify_fail.json");
const USAGE = fixture("usage.json");
const HEALTH = fixture("health.json");
const err = (code = "bad", message = "Nope.") => ({ error: { code, message } });

let server: MockServer;
beforeEach(async () => {
  server = await new MockServer().start();
});
afterEach(async () => {
  await server.stop();
});

function make(opts: ClientOptions = {}) {
  const sleeps: number[] = [];
  const client = new GroundCheck({
    apiKey: KEY,
    baseUrl: server.url,
    timeoutMs: 5000,
    jitter: () => 1,
    sleep: async (ms) => void sleeps.push(ms),
    ...opts,
  });
  return { client, sleeps };
}

describe("construction", () => {
  it("requires a key, from the option or the environment", () => {
    const saved = process.env["GROUNDCHECK_API_KEY"];
    delete process.env["GROUNDCHECK_API_KEY"];
    expect(() => new GroundCheck()).toThrow(InvalidArgumentError);
    expect(() => new GroundCheck({ apiKey: "" })).toThrow(InvalidArgumentError);
    process.env["GROUNDCHECK_API_KEY"] = KEY;
    expect(new GroundCheck({ baseUrl: "https://x.example" })).toBeTruthy();
    if (saved === undefined) delete process.env["GROUNDCHECK_API_KEY"];
    else process.env["GROUNDCHECK_API_KEY"] = saved;
  });

  it("reads the base URL from the environment and strips trailing slashes", async () => {
    process.env["GROUNDCHECK_BASE_URL"] = server.url + "//";
    server.fallback = { body: HEALTH };
    try {
      const c = new GroundCheck({ apiKey: KEY });
      expect((await c.health()).ok).toBe(true);
      expect(server.requests[0]?.path).toBe("/api/v1/health");
    } finally {
      delete process.env["GROUNDCHECK_BASE_URL"];
    }
  });

  it.each(["has space", "tab\tkey", "line\nbreak", "a\r\nX-Evil: 1", "ünïcode", "key\u0000"])(
    "rejects a key that could inject headers: %j",
    (key) => {
      expect(() => new GroundCheck({ apiKey: key, baseUrl: "https://x.example" })).toThrow(InvalidArgumentError);
    },
  );

  it.each([
    "ftp://x.example",
    "x.example",
    "",
    "http://",
    "https://u:p@x.example",
    "https://x.example?a=1",
    "https://x.example#f",
    "file:///etc/passwd",
    "javascript:alert(1)",
  ])("rejects the base URL %j", (baseUrl) => {
    expect(() => new GroundCheck({ apiKey: KEY, baseUrl })).toThrow(InvalidArgumentError);
  });

  it("allows plain http only for loopback unless opted in", () => {
    for (const ok of ["http://localhost:8000", "http://127.0.0.1:1", "http://[::1]:8000", "http://api.localhost"]) {
      expect(new GroundCheck({ apiKey: KEY, baseUrl: ok })).toBeTruthy();
    }
    expect(() => new GroundCheck({ apiKey: KEY, baseUrl: "http://api.example.com" })).toThrow(/plain http/);
    expect(() => new GroundCheck({ apiKey: KEY, baseUrl: "http://127.0.0.1.evil.com" })).toThrow(/plain http/);
    expect(new GroundCheck({ apiKey: KEY, baseUrl: "http://api.example.com", allowInsecureHttp: true })).toBeTruthy();
  });

  it("keeps a path prefix", async () => {
    server.fallback = { body: HEALTH };
    await new GroundCheck({ apiKey: KEY, baseUrl: server.url + "/gateway/" }).health();
    expect(server.requests[0]?.path).toBe("/gateway/api/v1/health");
  });

  it.each([
    { timeoutMs: 0 },
    { timeoutMs: -1 },
    { timeoutMs: NaN },
    { timeoutMs: Infinity },
    { maxRetries: -1 },
    { maxRetries: 11 },
    { maxRetries: 1.5 },
  ])("rejects bad options %j", (opts) => {
    expect(() => new GroundCheck({ apiKey: KEY, baseUrl: "https://x.example", ...opts })).toThrow(InvalidArgumentError);
  });

  it("never shows the key through JSON, String, inspect or Object.keys", () => {
    const { client } = make();
    for (const text of [
      JSON.stringify(client),
      String(client),
      inspect(client, { showHidden: true, depth: 5 }),
      JSON.stringify(Object.entries(client)),
      Object.getOwnPropertyNames(client).join(),
    ]) {
      expect(text).not.toContain(KEY);
      expect(text).not.toContain("s3cret");
    }
  });

  it("never puts the key in an error", async () => {
    server.enqueue({ status: 401, body: err("auth", "Bad key.") });
    const { client } = make();
    const e = await client.ask({ question: "q" }).catch((x: unknown) => x as GroundCheckError);
    expect(e).toBeInstanceOf(AuthenticationError);
    const text = String(e) + JSON.stringify(e) + (e as Error).stack + inspect(e, { depth: 6 });
    expect(text).not.toContain(KEY);
    expect(text).not.toContain("Bearer");
  });

  it("sends the key as a bearer token with JSON headers", async () => {
    server.fallback = { body: ASK };
    await make().client.ask({ question: "What is the dose?" });
    const seen = server.requests[0]!;
    expect(seen.headers["authorization"]).toBe(`Bearer ${KEY}`);
    expect(seen.headers["content-type"]).toBe("application/json");
    expect(seen.headers["accept"]).toBe("application/json");
  });
});

describe("endpoints, with responses recorded from the real service", () => {
  it("ask returns a typed answer", async () => {
    server.fallback = {
      body: ASK,
      headers: { "X-Request-Id": "req-1", "X-RateLimit-Limit": "60", "X-RateLimit-Remaining": "59", "X-RateLimit-Reset": "1700000000" },
    };
    const r = await make().client.ask({ question: "What is the standard dose of Caloradine?", site: "north", patient: { ageYears: 50, egfr: 80 } });
    const seen = server.requests[0]!;
    expect(seen.method).toBe("POST");
    expect(seen.path).toBe("/api/v1/ask");
    expect(JSON.parse(seen.body)).toEqual({
      question: "What is the standard dose of Caloradine?",
      site: "north",
      patient: { age_years: 50, egfr: 80 },
    });
    expect(r.decision).toBe("answer");
    expect(r.answered).toBe(true);
    expect(r.auditId).toBe("rec-ask-1");
    expect(r.claims.length).toBeGreaterThan(0);
    expect(r.claims[0]!.grounded).toBe(true);
    expect(r.claims[0]!.support?.sentence).toContain("15 mg");
    expect(r.sources[0]!.id).toBeTruthy();
    expect(r.requestId).toBe("req-1");
    expect(r.rateLimit).toEqual({ limit: 60, remaining: 59, reset: 1700000000 });
    expect(r.trace.length).toBeGreaterThan(0);
  });

  it("a refusal is not answered, and keeps its reason", async () => {
    server.fallback = { body: REFUSAL };
    const r = await make().client.ask({ question: "What is the standard dose of Quelvarin?" });
    expect(r.answered).toBe(false);
    expect(r.decision).toBe("refuse");
    expect(r.reasonCode).toBe("term_not_covered");
    expect(r.refusedReason).toBeTruthy();
    server.fallback = { body: { ...ASK, outcome: "not_evaluated" } };
    expect((await make().client.ask({ question: "q" })).answered).toBe(false);
  });

  it("ask surfaces patient findings", async () => {
    server.fallback = { body: BLOCKED };
    const r = await make().client.ask({ question: "q", patient: { ageYears: 60, weightKg: 70, egfr: 45, allergies: ["penicillin"], labs: { inr: 2.1 } } });
    expect(JSON.parse(server.requests[0]!.body).patient).toEqual({
      age_years: 60, weight_kg: 70, egfr: 45, allergies: ["penicillin"], labs: { inr: 2.1 },
    });
    expect(r.patientFindings[0]!.code).toBe("renal_dose");
    expect(r.answered).toBe(false);
  });

  it("verify sends the answer and source ids and parses a pass", async () => {
    server.fallback = { body: PASS, headers: { "X-Request-Id": "v-1" } };
    const r = await make().client.verify({ question: "What is the standard dose of Caloradine?", answer: "x", sources: ["CALO-001"], site: "s1", patient: { ageYears: 30 } });
    expect(server.requests[0]!.path).toBe("/api/v1/verify");
    expect(JSON.parse(server.requests[0]!.body)).toEqual({
      question: "What is the standard dose of Caloradine?", answer: "x", sources: ["CALO-001"], site: "s1", patient: { age_years: 30 },
    });
    expect(r.passed).toBe(true);
    expect(r.reasonCode).toBe("verified");
    expect(r.requestId).toBe("v-1");
    expect(r.claims[0]!.support).not.toBeNull();
  });

  it("verify failures carry dose issues and are never passed", async () => {
    server.fallback = { body: FAIL };
    const r = await make().client.verify({ question: "q", answer: "The standard dose of Caloradine is 500 mg twice daily." });
    expect("sources" in JSON.parse(server.requests[0]!.body)).toBe(false);
    expect(r.passed).toBe(false);
    expect(r.decision).toBe("fail");
    expect(r.unsupportedClaims).toEqual([0]);
    expect(r.doseIssues[0]!.value).toBe("500 mg");
    expect(r.claims[0]!.issues.length).toBeGreaterThan(0);
  });

  it.each(["review", "PASS", "passed", "", "unknown", "Pass "])("only an exact pass counts: %j", async (decision) => {
    server.fallback = { body: { ...PASS, decision } };
    expect((await make().client.verify({ question: "q", answer: "a" })).passed).toBe(false);
  });

  it("usage builds the query and parses rows", async () => {
    server.fallback = { body: USAGE };
    const { client } = make();
    const r = await client.usage({ groupBy: "day", from: new Date("2026-11-01T10:00:00Z"), to: "2026-11-24" });
    expect(server.requests[0]!.method).toBe("GET");
    expect(server.requests[0]!.path).toBe("/api/v1/usage?group_by=day&from=2026-11-01&to=2026-11-24");
    expect(server.requests[0]!.body).toBe("");
    expect(server.requests[0]!.headers["content-type"]).toBeUndefined();
    expect([r.groupBy, r.from, r.to, r.total, r.billable]).toEqual(["day", "2026-11-01", "2026-11-24", 17, 14]);
    expect(r.rows).toHaveLength(2);
    await client.usage();
    expect(server.requests[1]!.path).toBe("/api/v1/usage?group_by=key");
    await client.usage({ from: "2026-01-02" });
    expect(server.requests[2]!.path).toBe("/api/v1/usage?group_by=key&from=2026-01-02");
  });

  it("health", async () => {
    server.fallback = { body: HEALTH, headers: { "X-Request-Id": "h" } };
    const h = await make().client.health();
    expect(h.ok).toBe(true);
    expect(h.details["version"]).toBe("3.0.0");
    expect(h.requestId).toBe("h");
    server.fallback = { body: { status: "degraded" } };
    expect((await make().client.health()).ok).toBe(false);
  });
});

describe("argument validation happens before anything is sent", () => {
  const cases: [string, (c: GroundCheck) => Promise<unknown>][] = [
    ["empty question", (c) => c.ask({ question: "" })],
    ["blank question", (c) => c.ask({ question: "   " })],
    ["missing question", (c) => c.ask({} as never)],
    ["number question", (c) => c.ask({ question: 5 as never })],
    ["empty site", (c) => c.ask({ question: "q", site: "" })],
    ["number site", (c) => c.ask({ question: "q", site: 5 as never })],
    ["array patient", (c) => c.ask({ question: "q", patient: [] as never })],
    ["null patient", (c) => c.ask({ question: "q", patient: null as never })],
    ["unknown patient field", (c) => c.ask({ question: "q", patient: { name: "Jane" } as never })],
    ["NaN in patient", (c) => c.ask({ question: "q", patient: { egfr: NaN } })],
    ["Infinity in patient", (c) => c.ask({ question: "q", patient: { egfr: Infinity } })],
    ["BigInt in patient", (c) => c.ask({ question: "q", patient: { egfr: 1n as never } })],
    ["lone surrogate", (c) => c.ask({ question: "bad \ud800 surrogate" })],
    ["empty answer", (c) => c.verify({ question: "q", answer: "" })],
    ["empty verify question", (c) => c.verify({ question: "", answer: "a" })],
    ["string sources", (c) => c.verify({ question: "q", answer: "a", sources: "DOC1" as never })],
    ["number sources", (c) => c.verify({ question: "q", answer: "a", sources: [1, 2] as never })],
    ["bad groupBy", (c) => c.usage({ groupBy: "week" as never })],
    ["bad from", (c) => c.usage({ from: "not a date" })],
    ["impossible date", (c) => c.usage({ from: "2026-02-30" })],
    ["invalid Date", (c) => c.usage({ to: new Date("nope") })],
    ["number date", (c) => c.usage({ from: 20260101 as never })],
  ];
  it.each(cases)("%s", async (_name, call) => {
    await expect(call(make().client)).rejects.toBeInstanceOf(InvalidArgumentError);
    expect(server.requests).toHaveLength(0);
  });
});

describe("error statuses", () => {
  const table: [number, new (...a: never[]) => Error][] = [
    [400, BadRequestError], [401, AuthenticationError], [403, PermissionDeniedError], [404, NotFoundError],
    [413, PayloadTooLargeError], [422, UnprocessableError], [409, UnexpectedStatusError], [418, UnexpectedStatusError],
  ];
  it.each(table)("%i maps to its class and is never retried", async (status, cls) => {
    server.fallback = { status, body: err("some_code", "Specific message."), headers: { "X-Request-Id": "rid-9", "Retry-After": "1" } };
    const { client, sleeps } = make({ maxRetries: 5 });
    const e = (await client.ask({ question: "q" }).catch((x: unknown) => x)) as GroundCheckError;
    expect(e.constructor).toBe(cls);
    expect([e.status, e.code, e.message, e.requestId]).toEqual([status, "some_code", "Specific message.", "rid-9"]);
    expect(String(e)).toContain("rid-9");
    expect(server.requests).toHaveLength(1);
    expect(sleeps).toEqual([]);
    expect(e).toBeInstanceOf(GroundCheckError);
  });

  it("429 is retried, then raised with retryAfter", async () => {
    server.fallback = { status: 429, body: err("rate_limited", "Slow down."), headers: { "Retry-After": "2" } };
    const { client, sleeps } = make({ maxRetries: 2 });
    const e = (await client.verify({ question: "q", answer: "a" }).catch((x: unknown) => x)) as RateLimitError;
    expect(e).toBeInstanceOf(RateLimitError);
    expect([e.retryAfter, e.status, e.code]).toEqual([2, 429, "rate_limited"]);
    expect(server.requests).toHaveLength(3);
    expect(sleeps).toEqual([2000, 2000]);
  });

  it("honours Retry-After and then succeeds with the same body", async () => {
    server.enqueue({ status: 429, body: err("r", "x"), headers: { "Retry-After": "3" } });
    server.fallback = { body: PASS };
    const { client, sleeps } = make();
    expect((await client.verify({ question: "q", answer: "a" })).passed).toBe(true);
    expect(sleeps).toEqual([3000]);
    expect(server.requests[0]!.body).toBe(server.requests[1]!.body);
  });

  it("accepts Retry-After as a fraction or an HTTP date", async () => {
    const date = new Date(Date.now() + 10_000).toUTCString();
    server.enqueue({ status: 429, body: err("r", "x"), headers: { "Retry-After": date } }, { status: 429, body: err("r", "x"), headers: { "Retry-After": "0.25" } });
    server.fallback = { body: PASS };
    const { client, sleeps } = make({ maxRetries: 3 });
    await client.verify({ question: "q", answer: "a" });
    expect(sleeps[0]).toBeGreaterThan(8000);
    expect(sleeps[0]).toBeLessThanOrEqual(10_000);
    expect(sleeps[1]).toBe(250);
  });

  it.each(["soon", "-5", "NaN", "Infinity", "", "Wed, 99 Foo 2020", "1e999"])("unusable Retry-After %j falls back to backoff", async (header) => {
    server.enqueue({ status: 429, body: err("r", "x"), headers: { "Retry-After": header } });
    server.fallback = { body: PASS };
    const { client, sleeps } = make({ backoffBaseMs: 500 });
    await client.verify({ question: "q", answer: "a" });
    expect(sleeps).toHaveLength(1);
    expect([500, 0]).toContain(sleeps[0]);
  });

  it("raises at once when Retry-After exceeds the cap", async () => {
    server.fallback = { status: 429, body: err("rate_limited", "Come back later."), headers: { "Retry-After": "3600" } };
    const { client, sleeps } = make({ maxRetryWaitMs: 30_000 });
    const e = (await client.ask({ question: "q" }).catch((x: unknown) => x)) as RateLimitError;
    expect(e.retryAfter).toBe(3600);
    expect(server.requests).toHaveLength(1);
    expect(sleeps).toEqual([]);
  });

  it.each([500, 502, 503, 504, 599])("%i is retried with exponential backoff, then raised", async (status) => {
    server.fallback = { status, body: err("server", "Down.") };
    const { client, sleeps } = make({ maxRetries: 3, backoffBaseMs: 500, backoffMaxMs: 1500 });
    const e = (await client.ask({ question: "q" }).catch((x: unknown) => x)) as ServerError;
    expect(e).toBeInstanceOf(ServerError);
    expect(e.status).toBe(status);
    expect(server.requests).toHaveLength(4);
    expect(sleeps).toEqual([500, 1000, 1500]);
  });

  it("recovers after 5xx responses, including an HTML one", async () => {
    server.enqueue({ status: 503, body: err("x", "y") }, { status: 502, body: "<html>Bad gateway</html>" });
    server.fallback = { body: ASK };
    const { client } = make({ maxRetries: 2 });
    expect((await client.ask({ question: "q" })).answered).toBe(true);
    expect(server.requests).toHaveLength(3);
  });

  it("scales backoff by jitter, and zero retries means one attempt", async () => {
    server.fallback = { status: 500, body: err("x", "y") };
    const a = make({ maxRetries: 2, jitter: () => 0.5 });
    await expect(a.client.ask({ question: "q" })).rejects.toBeInstanceOf(ServerError);
    expect(a.sleeps).toEqual([250, 500]);
    server.requests.length = 0;
    await expect(make({ maxRetries: 0 }).client.ask({ question: "q" })).rejects.toBeInstanceOf(ServerError);
    expect(server.requests).toHaveLength(1);
  });

  it("ignores Retry-After on a 500 for the wait", async () => {
    server.enqueue({ status: 500, body: err("x", "y"), headers: { "Retry-After": "9" } });
    server.fallback = { body: ASK };
    const { client, sleeps } = make();
    await client.ask({ question: "q" });
    expect(sleeps).toEqual([500]);
  });

  it("gives a generic message for odd error bodies", async () => {
    const bodies: unknown[] = ["<html><h1>502</h1></html>", "", [1, 2], { error: "just a string" }, { error: { code: 5, message: "" } }, { detail: "x" }, null];
    for (const body of bodies) {
      server.fallback = { status: 400, body: body === null ? "null" : body };
      const e = (await make().client.ask({ question: "q" }).catch((x: unknown) => x)) as BadRequestError;
      expect([e.message, e.code]).toEqual(["HTTP 400", undefined]);
    }
  });

  it("truncates very long messages", async () => {
    server.fallback = { status: 400, body: err("c".repeat(500), "m".repeat(5000)) };
    const e = (await make().client.ask({ question: "q" }).catch((x: unknown) => x)) as BadRequestError;
    expect([e.message.length, e.code!.length]).toEqual([1000, 100]);
  });

  it("does not follow redirects", async () => {
    server.fallback = { status: 302, body: "", headers: { Location: "/elsewhere" } };
    const e = (await make().client.ask({ question: "q" }).catch((x: unknown) => x)) as UnexpectedStatusError;
    expect(e).toBeInstanceOf(UnexpectedStatusError);
    expect(e.status).toBe(302);
    expect(server.requests).toHaveLength(1);
  });

  it("parses Retry-After headers", () => {
    expect(parseRetryAfter("5")).toBe(5);
    expect(parseRetryAfter("0")).toBe(0);
    expect(parseRetryAfter("-3")).toBe(0);
    expect(parseRetryAfter(" 2.5 ")).toBe(2.5);
    expect(parseRetryAfter(undefined)).toBeUndefined();
    expect(parseRetryAfter(null)).toBeUndefined();
    expect(parseRetryAfter("abc")).toBeUndefined();
    expect(parseRetryAfter(new Date(1_000_000 + 7000).toUTCString(), 1_000_000)).toBe(7);
    expect(parseRetryAfter(new Date(0).toUTCString(), 1_000_000)).toBe(0);
  });
});

describe("network failures", () => {
  it("a refused connection is a connection error", async () => {
    const dead = await new MockServer().start();
    const url = dead.url;
    await dead.stop();
    const c = new GroundCheck({ apiKey: KEY, baseUrl: url, sleep: async () => undefined });
    await expect(c.ask({ question: "q" })).rejects.toBeInstanceOf(APIConnectionError);
  });

  it("does not retry a POST after the connection drops", async () => {
    server.enqueue({ drop: true });
    server.fallback = { body: ASK };
    await expect(make().client.ask({ question: "q" })).rejects.toBeInstanceOf(APIConnectionError);
    expect(server.requests).toHaveLength(1);
  });

  it("retries a GET after the connection drops", async () => {
    server.enqueue({ drop: true }, { drop: true });
    server.fallback = { body: HEALTH };
    const { client, sleeps } = make({ maxRetries: 2 });
    expect((await client.health()).ok).toBe(true);
    expect(server.requests).toHaveLength(3);
    expect(sleeps).toEqual([500, 1000]);
  });

  it("gives up on a GET after the retries", async () => {
    server.fallback = { drop: true };
    await expect(make({ maxRetries: 1 }).client.health()).rejects.toBeInstanceOf(APIConnectionError);
    expect(server.requests).toHaveLength(2);
  });

  it("times out on a slow server", async () => {
    server.fallback = { body: ASK, delayMs: 1500 };
    const start = Date.now();
    const e = await make({ timeoutMs: 300 }).client.ask({ question: "q" }).catch((x: unknown) => x);
    expect(e).toBeInstanceOf(APITimeoutError);
    expect(e).toBeInstanceOf(APIConnectionError);
    expect(Date.now() - start).toBeLessThan(1200);
    expect(server.requests).toHaveLength(1);
  });

  it("cuts off a server that trickles its body", async () => {
    server.fallback = { trickle: true };
    const start = Date.now();
    await expect(make({ timeoutMs: 400 }).client.ask({ question: "q" })).rejects.toBeInstanceOf(APITimeoutError);
    expect(Date.now() - start).toBeLessThan(2000);
  });

  it("retries a GET after a timeout", async () => {
    server.enqueue({ body: HEALTH, delayMs: 1000 });
    server.fallback = { body: HEALTH };
    expect((await make({ timeoutMs: 300 }).client.health()).ok).toBe(true);
    expect(server.requests).toHaveLength(2);
  });

  it("honours the caller's abort signal and does not retry it", async () => {
    server.fallback = { body: HEALTH, delayMs: 800 };
    const ctrl = new AbortController();
    setTimeout(() => ctrl.abort(), 100);
    const e = (await make({ maxRetries: 3 }).client.health({ signal: ctrl.signal }).catch((x: unknown) => x)) as APIConnectionError;
    expect(e).toBeInstanceOf(APIConnectionError);
    expect(e).not.toBeInstanceOf(APITimeoutError);
    expect(e.code).toBe("aborted");
    expect(server.requests).toHaveLength(1);
    const already = new AbortController();
    already.abort();
    await expect(make().client.health({ signal: already.signal })).rejects.toBeInstanceOf(APIConnectionError);
  });
});

describe("malformed responses", () => {
  it.each([
    "", "{", "not json", "<html></html>", "null", "[]", "42", '"text"', '{"decision": "answer"}',
    "[".repeat(5000) + "]".repeat(5000),
  ])("raises InvalidResponseError for %j", async (body) => {
    server.fallback = { body, headers: { "X-Request-Id": "r" } };
    const e = (await make().client.ask({ question: "q" }).catch((x: unknown) => x)) as InvalidResponseError;
    expect(e).toBeInstanceOf(InvalidResponseError);
  });

  it("rejects invalid UTF-8", async () => {
    server.fallback = { body: Buffer.from([0x7b, 0xff, 0xfe, 0x7d]) };
    await expect(make().client.ask({ question: "q" })).rejects.toBeInstanceOf(InvalidResponseError);
  });

  it.each<[string, Record<string, unknown>]>([
    ["decision", { decision: 5 }], ["audit_id", { audit_id: null }], ["total_ms string", { total_ms: "41" }],
    ["total_ms float", { total_ms: 4.5 }], ["claims string", { claims: "none" }], ["claims item", { claims: [1] }],
    ["claim text", { claims: [{ text: 5 }] }], ["source missing fields", { sources: [{ id: "x" }] }],
    ["llm_used", { llm_used: "no" }], ["trace", { trace: [1] }], ["finding", { patient_findings: [{ severity: "warn" }] }],
    ["redaction kinds", { redaction: { kinds: { a: "b" } } }], ["redaction array", { redaction: [] }],
    ["review routed", { review: { routed: "yes" } }], ["review case", { review: { case_id: "1" } }],
    ["explanation", { explanation: { missing_terms: [1] } }], ["source_ids", { claims: [{ text: "t", source_ids: [1] }] }],
    ["support", { claims: [{ text: "t", support: { start: 0 } }] }],
  ])("rejects a wrongly typed field: %s", async (_n, change) => {
    server.fallback = { body: { ...ASK, ...change } };
    await expect(make().client.ask({ question: "q" })).rejects.toBeInstanceOf(InvalidResponseError);
  });

  it("rejects missing required fields", async () => {
    for (const field of ["decision", "answer_text", "audit_id", "total_ms"]) {
      const body = { ...ASK };
      delete body[field];
      server.fallback = { body };
      await expect(make().client.ask({ question: "q" })).rejects.toThrow(new RegExp(field));
    }
    for (const field of ["decision", "reason_code", "audit_id", "latency_ms"]) {
      const body = { ...PASS };
      delete body[field];
      server.fallback = { body };
      await expect(make().client.verify({ question: "q", answer: "a" })).rejects.toThrow(new RegExp(field));
    }
  });

  it("rejects bad verify lists", async () => {
    for (const change of [{ unsupported_claims: ["0"] }, { dose_issues: [{ claim: "x" }] }, { claims: [{ grounded: true }] }]) {
      server.fallback = { body: { ...PASS, ...change } };
      await expect(make().client.verify({ question: "q", answer: "a" })).rejects.toBeInstanceOf(InvalidResponseError);
    }
  });

  it("rejects bad usage and health bodies", async () => {
    server.fallback = { body: { ...USAGE, rows: [1] } };
    await expect(make().client.usage()).rejects.toBeInstanceOf(InvalidResponseError);
    server.fallback = { body: { ...USAGE, total: "17" } };
    await expect(make().client.usage()).rejects.toBeInstanceOf(InvalidResponseError);
    server.fallback = { body: {} };
    await expect(make().client.health()).rejects.toBeInstanceOf(InvalidResponseError);
  });

  it("ignores fields a newer server adds", async () => {
    server.fallback = { body: { ...ASK, new_field: { a: 1 }, claims: [{ ...ASK.claims[0], extra: 1 }] } };
    expect((await make().client.ask({ question: "q" })).answered).toBe(true);
    server.fallback = { body: { ...PASS, new_field: [1] } };
    expect((await make().client.verify({ question: "q", answer: "a" })).passed).toBe(true);
  });

  it("tolerates a UTF-8 byte order mark", async () => {
    server.fallback = { body: Buffer.concat([Buffer.from([0xef, 0xbb, 0xbf]), Buffer.from(JSON.stringify(PASS))]) };
    expect((await make().client.verify({ question: "q", answer: "a" })).passed).toBe(true);
  });

  it("fills defaults for missing optional fields", async () => {
    server.fallback = { body: { decision: "refuse", answer_text: "", audit_id: "a", total_ms: 1 } };
    const r = await make().client.ask({ question: "q" });
    expect([r.outcome, r.claims.length, r.review.routed, r.rateLimit]).toEqual(["refused", 0, false, undefined]);
  });

  it("refuses a response over the size limit", async () => {
    server.fallback = { body: { ...ASK, answer_text: "x".repeat(5000) } };
    await expect(make({ maxResponseBytes: 1000 }).client.ask({ question: "q" })).rejects.toThrow(/larger/);
  });

  it("ignores garbage rate limit headers", async () => {
    server.fallback = { body: ASK, headers: { "X-RateLimit-Limit": "abc", "X-RateLimit-Remaining": "5" } };
    const r = await make().client.ask({ question: "q" });
    expect(r.rateLimit).toEqual({ limit: undefined, remaining: 5, reset: undefined });
  });
});

describe("unicode and size", () => {
  const texts = [
    "Dosis für Müller: 5 mg ✓", "日本語の質問 — 用量は？", "שאלה בעברית مرحبا", "emoji 💊🧪 and combining é",
    "zero​width and ‮RTL override", "null \u0000 byte escape", 'quote " backslash \\ tab \t newline \n', "𝒳 astral plane",
  ];
  it.each(texts)("survives the round trip exactly: %j", async (text) => {
    server.fallback = { body: { ...ASK, answer_text: text } };
    const r = await make().client.ask({ question: text });
    expect(JSON.parse(server.requests[0]!.body).question).toBe(text);
    expect(r.answerText).toBe(text);
  });

  it("sends a 5 MB request intact", async () => {
    const text = "long question ".repeat(400_000).slice(0, 5_000_000);
    server.fallback = { body: ASK };
    await make({ timeoutMs: 30_000 }).client.ask({ question: text });
    expect(JSON.parse(server.requests[0]!.body).question).toBe(text);
  });

  it("parses a 5 MB response", async () => {
    const big = "word ".repeat(1_000_000);
    server.fallback = { body: { ...ASK, answer_text: big } };
    expect((await make({ timeoutMs: 30_000 }).client.ask({ question: "q" })).answerText.length).toBe(big.length);
  });

  it("maps 413 for a payload the server rejects, without retrying", async () => {
    server.fallback = { status: 413, body: err("payload_too_large", "Too large.") };
    await expect(make().client.verify({ question: "q", answer: "a".repeat(100_000) })).rejects.toBeInstanceOf(PayloadTooLargeError);
    expect(server.requests).toHaveLength(1);
  });

  it("serves many concurrent calls from one client", async () => {
    server.fallback = { body: ASK };
    const { client } = make();
    const results = await Promise.all(Array.from({ length: 30 }, (_, i) => client.ask({ question: `question ${i}` })));
    expect(results).toHaveLength(30);
    expect(server.requests.map((r) => JSON.parse(r.body).question).sort()).toEqual(
      Array.from({ length: 30 }, (_, i) => `question ${i}`).sort(),
    );
  });

  it("uses an injected fetch, so it can run on any runtime", async () => {
    let seenUrl = "";
    const client = new GroundCheck({
      apiKey: KEY,
      baseUrl: "https://api.example.com",
      fetch: async (url) => {
        seenUrl = url;
        return new Response(JSON.stringify(HEALTH), { status: 200, headers: { "content-type": "application/json" } });
      },
    });
    expect((await client.health()).ok).toBe(true);
    expect(seenUrl).toBe("https://api.example.com/api/v1/health");
  });
});

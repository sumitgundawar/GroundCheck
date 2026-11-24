import {
  APIConnectionError,
  APITimeoutError,
  InvalidArgumentError,
  InvalidResponseError,
  RateLimitError,
  ServerError,
  STATUS_ERRORS,
  UnexpectedStatusError,
} from "./errors.js";
import type { APIStatusError } from "./errors.js";
import {
  parseAsk,
  parseHealth,
  parseUsage,
  parseVerify,
  patientToWire,
} from "./models.js";
import type {
  AskResponse,
  Health,
  Patient,
  RateLimit,
  UsageReport,
  VerifyResponse,
} from "./models.js";

export const DEFAULT_BASE_URL = "http://localhost:8000";
export const VERSION = "0.1.0";

export type FetchLike = (input: string, init: RequestInit) => Promise<Response>;

export interface ClientOptions {
  /** Defaults to GROUNDCHECK_API_KEY where an environment exists. */
  apiKey?: string;
  /** Defaults to GROUNDCHECK_BASE_URL, else http://localhost:8000. */
  baseUrl?: string;
  /** Milliseconds for the whole call, body included. Default 30000. */
  timeoutMs?: number;
  /** Extra attempts after a 429 or 5xx (and, for GET, a network failure). Default 2. */
  maxRetries?: number;
  backoffBaseMs?: number;
  backoffMaxMs?: number;
  /** The longest the client sleeps for one retry; a longer Retry-After raises RateLimitError at once. */
  maxRetryWaitMs?: number;
  /** Plain http is refused for non-loopback hosts unless this is true. */
  allowInsecureHttp?: boolean;
  /** For tests and unusual runtimes. Defaults to the global fetch. */
  fetch?: FetchLike;
  sleep?: (ms: number) => Promise<void>;
  /** Returns a number in [0.5, 1] that scales each backoff. */
  jitter?: () => number;
  maxResponseBytes?: number;
}

export interface CallOptions {
  signal?: AbortSignal;
}

export interface AskParams {
  question: string;
  site?: string;
  patient?: Patient;
}

export interface VerifyParams extends AskParams {
  answer: string;
  /** Source ids the answer must rest on. */
  sources?: string[];
}

export interface UsageParams {
  groupBy?: "key" | "day" | "site" | "endpoint" | "decision";
  from?: string | Date;
  to?: string | Date;
}

const LOOPBACK = /^(localhost|.+\.localhost|127(\.\d{1,3}){3}|\[::1\]|::1)$/i;
const DATE = /^\d{4}-\d{2}-\d{2}$/;

function envVar(name: string): string | undefined {
  const proc = (globalThis as { process?: { env?: Record<string, string | undefined> } }).process;
  return proc?.env?.[name];
}

function day(value: string | Date | undefined, name: string): string | undefined {
  if (value === undefined) return undefined;
  if (value instanceof Date) {
    if (Number.isNaN(value.getTime())) throw new InvalidArgumentError(`${name} is an invalid date`);
    return value.toISOString().slice(0, 10);
  }
  if (typeof value === "string" && DATE.test(value) && !Number.isNaN(Date.parse(value + "T00:00:00Z"))) {
    // Date.parse accepts 2026-02-30 in some engines; compare the round trip.
    if (new Date(value + "T00:00:00Z").toISOString().slice(0, 10) === value) return value;
  }
  throw new InvalidArgumentError(`${name} must be a date like 2026-11-24`);
}

/** Seconds from a Retry-After header (seconds or an HTTP date); undefined if unusable. */
export function parseRetryAfter(value: string | null | undefined, now: number = Date.now()): number | undefined {
  if (!value || !value.trim()) return undefined;
  const text = value.trim();
  let seconds: number;
  if (/^[+-]?(\d+\.?\d*|\.\d+)(e[+-]?\d+)?$/i.test(text)) {
    seconds = Number(text);
  } else {
    const when = Date.parse(text);
    if (Number.isNaN(when)) return undefined;
    seconds = (when - now) / 1000;
  }
  if (!Number.isFinite(seconds)) return undefined;
  return Math.max(0, seconds);
}

function intHeader(headers: Headers, name: string): number | undefined {
  const v = headers.get(name);
  if (v === null || !/^\d+$/.test(v.trim())) return undefined;
  return Number(v);
}

function text(value: unknown, name: string): string {
  if (typeof value !== "string") throw new InvalidArgumentError(`${name} must be a string.`);
  if (!value.trim()) throw new InvalidArgumentError(`${name} is empty.`);
  return value;
}

const defaultSleep = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));

export class GroundCheck {
  readonly baseUrl: string;
  readonly timeoutMs: number;
  readonly maxRetries: number;
  readonly #apiKey: string;
  readonly #fetch: FetchLike;
  readonly #sleep: (ms: number) => Promise<void>;
  readonly #jitter: () => number;
  readonly #backoffBase: number;
  readonly #backoffMax: number;
  readonly #maxRetryWait: number;
  readonly #maxResponse: number;

  constructor(options: ClientOptions = {}) {
    const key = options.apiKey ?? envVar("GROUNDCHECK_API_KEY");
    if (typeof key !== "string" || !key) {
      throw new InvalidArgumentError("An API key is required: pass apiKey or set GROUNDCHECK_API_KEY.");
    }
    if (/[^\x21-\x7e]/.test(key)) {
      throw new InvalidArgumentError("The API key contains spaces, control or non-ASCII characters.");
    }
    const url = options.baseUrl ?? envVar("GROUNDCHECK_BASE_URL") ?? DEFAULT_BASE_URL;
    let parsed: URL;
    try {
      parsed = new URL(url);
    } catch {
      throw new InvalidArgumentError("baseUrl must be an http or https URL.");
    }
    if ((parsed.protocol !== "http:" && parsed.protocol !== "https:") || !parsed.hostname) {
      throw new InvalidArgumentError("baseUrl must be an http or https URL.");
    }
    if (parsed.username || parsed.password || parsed.search || parsed.hash) {
      throw new InvalidArgumentError("baseUrl can't contain credentials, a query or a fragment.");
    }
    if (parsed.protocol === "http:" && !options.allowInsecureHttp && !LOOPBACK.test(parsed.hostname)) {
      throw new InvalidArgumentError(
        "Refusing to send an API key over plain http to a non-local host. Use https, or set allowInsecureHttp.",
      );
    }
    const timeoutMs = options.timeoutMs ?? 30_000;
    if (typeof timeoutMs !== "number" || !Number.isFinite(timeoutMs) || timeoutMs <= 0) {
      throw new InvalidArgumentError("timeoutMs must be a positive number.");
    }
    const maxRetries = options.maxRetries ?? 2;
    if (!Number.isInteger(maxRetries) || maxRetries < 0 || maxRetries > 10) {
      throw new InvalidArgumentError("maxRetries must be an integer from 0 to 10.");
    }
    const fetchImpl = options.fetch ?? (globalThis as { fetch?: FetchLike }).fetch;
    if (typeof fetchImpl !== "function") throw new InvalidArgumentError("No fetch is available; pass options.fetch.");
    this.#apiKey = key;
    this.baseUrl = url.replace(/\/+$/, "");
    this.timeoutMs = timeoutMs;
    this.maxRetries = maxRetries;
    this.#fetch = fetchImpl.bind(globalThis) as FetchLike;
    this.#sleep = options.sleep ?? defaultSleep;
    this.#jitter = options.jitter ?? (() => 0.5 + Math.random() / 2);
    this.#backoffBase = options.backoffBaseMs ?? 500;
    this.#backoffMax = options.backoffMaxMs ?? 8000;
    this.#maxRetryWait = options.maxRetryWaitMs ?? 30_000;
    this.#maxResponse = options.maxResponseBytes ?? 32 * 1024 * 1024;
  }

  toJSON(): Record<string, unknown> {
    return { baseUrl: this.baseUrl, apiKey: "***" };
  }

  toString(): string {
    return `GroundCheck(baseUrl=${this.baseUrl}, apiKey=***)`;
  }

  [Symbol.for("nodejs.util.inspect.custom")](): string {
    return this.toString();
  }

  async ask(params: AskParams, options: CallOptions = {}): Promise<AskResponse> {
    const body: Record<string, unknown> = { question: text(params?.question, "question") };
    this.#optional(body, params);
    const { data, meta } = await this.#request("POST", "/api/v1/ask", body, options);
    return { ...parseAsk(data), ...meta };
  }

  async verify(params: VerifyParams, options: CallOptions = {}): Promise<VerifyResponse> {
    const body: Record<string, unknown> = {
      question: text(params?.question, "question"),
      answer: text(params?.answer, "answer"),
    };
    if (params.sources !== undefined) {
      if (!Array.isArray(params.sources) || !params.sources.every((s) => typeof s === "string")) {
        throw new InvalidArgumentError("sources must be an array of source id strings.");
      }
      body["sources"] = [...params.sources];
    }
    this.#optional(body, params);
    const { data, meta } = await this.#request("POST", "/api/v1/verify", body, options);
    return { ...parseVerify(data), ...meta };
  }

  async usage(params: UsageParams = {}, options: CallOptions = {}): Promise<UsageReport> {
    const groupBy = params.groupBy ?? "key";
    if (!["key", "day", "site", "endpoint", "decision"].includes(groupBy)) {
      throw new InvalidArgumentError("groupBy must be key, day, site, endpoint or decision.");
    }
    const query = new URLSearchParams({ group_by: groupBy });
    const from = day(params.from, "from");
    const to = day(params.to, "to");
    if (from) query.set("from", from);
    if (to) query.set("to", to);
    const { data, meta } = await this.#request("GET", `/api/v1/usage?${query.toString()}`, undefined, options);
    return { ...parseUsage(data), ...meta };
  }

  async health(options: CallOptions = {}): Promise<Health> {
    const { data, meta } = await this.#request("GET", "/api/v1/health", undefined, options);
    return { ...parseHealth(data), requestId: meta.requestId };
  }

  #optional(body: Record<string, unknown>, params: AskParams): void {
    if (params.site !== undefined) {
      if (typeof params.site !== "string" || !params.site) {
        throw new InvalidArgumentError("site must be a non-empty string.");
      }
      body["site"] = params.site;
    }
    if (params.patient !== undefined) {
      if (params.patient === null || typeof params.patient !== "object" || Array.isArray(params.patient)) {
        throw new InvalidArgumentError("patient must be an object.");
      }
      const known = new Set([
        "ageYears", "sex", "weightKg", "pregnant", "breastfeeding", "egfr", "creatinineUmolL", "childPugh",
        "allergies", "medicines", "conditions", "labs",
      ]);
      const unknown = Object.keys(params.patient).filter((k) => !known.has(k));
      if (unknown.length) throw new InvalidArgumentError(`Unknown patient field: ${unknown[0]}.`);
      body["patient"] = patientToWire(params.patient);
    }
  }

  #delay(attempt: number, retryAfter: number | undefined): number {
    if (retryAfter !== undefined) return retryAfter * 1000;
    return Math.min(this.#backoffMax, this.#backoffBase * 2 ** attempt) * this.#jitter();
  }

  async #request(
    method: "GET" | "POST",
    path: string,
    body: Record<string, unknown> | undefined,
    options: CallOptions,
  ): Promise<{ data: unknown; meta: { requestId?: string; rateLimit?: RateLimit } }> {
    let payload: string | undefined;
    if (body !== undefined) {
      try {
        payload = JSON.stringify(body, (_k, v) => {
          if (typeof v === "number" && !Number.isFinite(v)) throw new TypeError("NaN and Infinity can't be sent");
          if (typeof v === "string" && typeof (v as { isWellFormed?: () => boolean }).isWellFormed === "function" &&
              !(v as unknown as { isWellFormed: () => boolean }).isWellFormed()) {
            throw new TypeError("a string contains an unpaired surrogate");
          }
          return v;
        });
      } catch (err) {
        throw new InvalidArgumentError(`The request can't be encoded as JSON: ${(err as Error).message}`);
      }
    }
    const url = this.baseUrl + path;
    for (let attempt = 0; ; attempt++) {
      const last = attempt >= this.maxRetries;
      let result: { status: number; headers: Headers; raw: Uint8Array };
      try {
        result = await this.#send(method, url, payload, options.signal);
      } catch (err) {
        if (err instanceof APIConnectionError && method === "GET" && !last && !options.signal?.aborted) {
          await this.#sleep(this.#delay(attempt, undefined));
          continue;
        }
        throw err;
      }
      const { status, headers, raw } = result;
      const requestId = headers.get("x-request-id") ?? undefined;
      if (status >= 200 && status < 300) {
        const hasLimit = [...headers.keys()].some((k) => k.startsWith("x-ratelimit-"));
        return {
          data: this.#decode(raw, status, requestId),
          meta: {
            requestId,
            ...(hasLimit
              ? {
                  rateLimit: {
                    limit: intHeader(headers, "x-ratelimit-limit"),
                    remaining: intHeader(headers, "x-ratelimit-remaining"),
                    reset: intHeader(headers, "x-ratelimit-reset"),
                  },
                }
              : {}),
          },
        };
      }
      const retryable = status === 429 || status >= 500;
      const wait = status === 429 || status === 503 ? parseRetryAfter(headers.get("retry-after")) : undefined;
      if (retryable && !last && (wait === undefined || wait * 1000 <= this.#maxRetryWait)) {
        await this.#sleep(this.#delay(attempt, wait));
        continue;
      }
      throw this.#error(status, headers, raw, wait);
    }
  }

  async #send(
    method: string,
    url: string,
    payload: string | undefined,
    outer: AbortSignal | undefined,
  ): Promise<{ status: number; headers: Headers; raw: Uint8Array }> {
    const controller = new AbortController();
    let timedOut = false;
    const timer = setTimeout(() => {
      timedOut = true;
      controller.abort();
    }, this.timeoutMs);
    const onAbort = () => controller.abort();
    if (outer) {
      if (outer.aborted) controller.abort();
      else outer.addEventListener("abort", onAbort, { once: true });
    }
    try {
      const res = await this.#fetch(url, {
        method,
        body: payload,
        redirect: "manual",
        signal: controller.signal,
        headers: {
          Authorization: `Bearer ${this.#apiKey}`,
          Accept: "application/json",
          ...(payload !== undefined ? { "Content-Type": "application/json" } : {}),
        },
      });
      const raw = await this.#readBody(res);
      return { status: res.status, headers: res.headers, raw };
    } catch (err) {
      if (err instanceof InvalidResponseError) throw err;
      if (timedOut) throw new APITimeoutError(`The request timed out after ${this.timeoutMs} ms.`);
      if (outer?.aborted) throw new APIConnectionError("The request was aborted.", { code: "aborted" });
      if (err instanceof Error && err.name === "AbortError") {
        throw new APITimeoutError(`The request timed out after ${this.timeoutMs} ms.`);
      }
      throw new APIConnectionError(`Couldn't reach the server: ${err instanceof Error ? err.name : "error"}`);
    } finally {
      clearTimeout(timer);
      outer?.removeEventListener("abort", onAbort);
    }
  }

  async #readBody(res: Response): Promise<Uint8Array> {
    if (!res.body) return new Uint8Array(0);
    const reader = res.body.getReader();
    const chunks: Uint8Array[] = [];
    let size = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      size += value.byteLength;
      if (size > this.#maxResponse) {
        void reader.cancel().catch(() => undefined);
        throw new InvalidResponseError("The response is larger than the client accepts.", { status: res.status });
      }
      chunks.push(value);
    }
    const out = new Uint8Array(size);
    let offset = 0;
    for (const c of chunks) {
      out.set(c, offset);
      offset += c.byteLength;
    }
    return out;
  }

  #decode(raw: Uint8Array, status: number, requestId: string | undefined): unknown {
    try {
      return JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(raw));
    } catch {
      throw new InvalidResponseError("The server's response wasn't valid JSON.", { status, requestId });
    }
  }

  #error(status: number, headers: Headers, raw: Uint8Array, retryAfter: number | undefined): APIStatusError {
    const requestId = headers.get("x-request-id") ?? undefined;
    let code: string | undefined;
    let message = `HTTP ${status}`;
    try {
      const parsed: unknown = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(raw));
      const err = parsed !== null && typeof parsed === "object" ? (parsed as Record<string, unknown>)["error"] : undefined;
      if (err !== null && typeof err === "object") {
        const e = err as Record<string, unknown>;
        if (typeof e["message"] === "string" && e["message"].trim()) message = e["message"].trim().slice(0, 1000);
        if (typeof e["code"] === "string") code = e["code"].slice(0, 100);
      }
    } catch {
      // not JSON: keep the generic message
    }
    const init = { status, code, requestId };
    if (status === 429) return new RateLimitError(message, { ...init, retryAfter });
    if (status >= 500) return new ServerError(message, init);
    const Cls = STATUS_ERRORS[status] ?? UnexpectedStatusError;
    return new Cls(message, init);
  }
}

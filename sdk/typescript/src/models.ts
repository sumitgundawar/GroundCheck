import { InvalidResponseError } from "./errors.js";

export interface ClaimSupport {
  sourceId: string;
  start: number;
  end: number;
  sentence: string;
}
export interface Explanation {
  summary: string;
  missingTerms: string[];
  suggestions: string[];
}
export interface Redaction {
  kinds: Record<string, number>;
  redactedQuery: string;
  redactedAnswer: string;
}
export interface Review {
  routed: boolean;
  caseId: number | null;
}
export interface PatientFinding {
  severity: string;
  code: string;
  medicine: string;
  message: string;
  ruleId: string;
  source: string;
}
export interface RateLimit {
  limit?: number;
  remaining?: number;
  reset?: number;
}
export interface Patient {
  ageYears?: number;
  sex?: "female" | "male" | "other";
  weightKg?: number;
  pregnant?: boolean;
  breastfeeding?: boolean;
  egfr?: number;
  creatinineUmolL?: number;
  childPugh?: "A" | "B" | "C";
  allergies?: string[];
  medicines?: string[];
  conditions?: string[];
  labs?: Partial<Record<"potassium" | "sodium" | "inr" | "platelets" | "alt" | "qtc", number>>;
}

export interface VerifiedClaim {
  text: string;
  grounded: boolean;
  sourceId: string | null;
  support: ClaimSupport | null;
  relevance: number;
  issues: string[];
}
export interface DoseIssue {
  claim: number;
  value: string;
  message: string;
}
export interface SourceRef {
  id: string;
  title: string;
  topic: string;
}
export interface VerifyResponse {
  /** "pass", "fail" or "review". Treat anything but "pass" as not verified; see `passed`. */
  decision: string;
  reasonCode: string;
  claims: VerifiedClaim[];
  unsupportedClaims: number[];
  doseIssues: DoseIssue[];
  explanation: Explanation;
  redaction: Redaction;
  auditId: string;
  latencyMs: number;
  sources: SourceRef[];
  patientFindings: PatientFinding[];
  review: Review;
  /** True only for an exact "pass". */
  passed: boolean;
  requestId?: string;
  rateLimit?: RateLimit;
}

export interface AskClaim {
  text: string;
  sourceIds: string[];
  grounded: boolean;
  groundingScore: number;
  relevance: number;
  support: ClaimSupport | null;
}
export interface AskSource {
  id: string;
  title: string;
  snippet: string;
  score: number;
  topic: string;
  section: string;
  kind: string;
  rank: number;
  aboveGate: boolean;
}
export interface AskResponse {
  /** "answer" or "refuse". */
  decision: string;
  /** "answered", "refused" or "not_evaluated". */
  outcome: string;
  answerText: string;
  auditId: string;
  totalMs: number;
  llmUsed: boolean;
  refusedReason: string | null;
  reasonCode: string | null;
  claims: AskClaim[];
  sources: AskSource[];
  patientFindings: PatientFinding[];
  review: Review;
  explanation: Explanation;
  redaction: Redaction;
  trace: Record<string, unknown>[];
  /** True only when the service answered and checked the answer. */
  answered: boolean;
  requestId?: string;
  rateLimit?: RateLimit;
}
export interface UsageReport {
  groupBy: string;
  from: string;
  to: string;
  rows: Record<string, unknown>[];
  total: number;
  billable: number;
  requestId?: string;
  rateLimit?: RateLimit;
}
export interface Health {
  status: string;
  ok: boolean;
  details: Record<string, unknown>;
  requestId?: string;
}

// --- Parsing: strict about types, ignoring fields a newer server adds -----------

type Obj = Record<string, unknown>;

function fail(where: string, what: string): never {
  throw new InvalidResponseError(`Unexpected response: ${where} ${what}`);
}

function obj(value: unknown, where: string): Obj {
  if (value === null || typeof value !== "object" || Array.isArray(value)) fail(where, "should be an object");
  return value as Obj;
}

function str(d: Obj, key: string, where: string, dflt?: string): string {
  if (!(key in d)) {
    if (dflt !== undefined) return dflt;
    fail(`${where}.${key}`, "is missing");
  }
  if (typeof d[key] !== "string") fail(`${where}.${key}`, "should be a string");
  return d[key] as string;
}

function optStr(d: Obj, key: string, where: string): string | null {
  const v = d[key];
  if (v === undefined || v === null) return null;
  if (typeof v !== "string") fail(`${where}.${key}`, "should be a string or null");
  return v as string;
}

function bool(d: Obj, key: string, where: string, dflt?: boolean): boolean {
  if (!(key in d) && dflt !== undefined) return dflt;
  if (typeof d[key] !== "boolean") fail(`${where}.${key}`, "should be true or false");
  return d[key] as boolean;
}

function int(d: Obj, key: string, where: string, dflt?: number): number {
  if (!(key in d) && dflt !== undefined) return dflt;
  const v = d[key];
  if (typeof v !== "number" || !Number.isInteger(v)) fail(`${where}.${key}`, "should be an integer");
  return v as number;
}

function num(d: Obj, key: string, where: string, dflt?: number): number {
  if (!(key in d) && dflt !== undefined) return dflt;
  const v = d[key];
  if (typeof v !== "number" || !Number.isFinite(v)) fail(`${where}.${key}`, "should be a number");
  return v as number;
}

function list(d: Obj, key: string, where: string): unknown[] {
  const v = d[key] ?? [];
  if (!Array.isArray(v)) fail(`${where}.${key}`, "should be a list");
  return v as unknown[];
}

function strs(d: Obj, key: string, where: string): string[] {
  const items = list(d, key, where);
  if (!items.every((i) => typeof i === "string")) fail(`${where}.${key}`, "should be a list of strings");
  return items as string[];
}

function support(value: unknown, where: string): ClaimSupport | null {
  if (value === undefined || value === null) return null;
  const d = obj(value, where);
  return {
    sourceId: str(d, "source_id", where),
    start: int(d, "start", where),
    end: int(d, "end", where),
    sentence: str(d, "sentence", where),
  };
}

function explanation(value: unknown): Explanation {
  const d = obj(value ?? {}, "explanation");
  return {
    summary: str(d, "summary", "explanation", ""),
    missingTerms: strs(d, "missing_terms", "explanation"),
    suggestions: strs(d, "suggestions", "explanation"),
  };
}

function redaction(value: unknown): Redaction {
  const d = obj(value ?? {}, "redaction");
  const kinds = d["kinds"] ?? {};
  const k = obj(kinds, "redaction.kinds");
  for (const v of Object.values(k)) {
    if (typeof v !== "number" || !Number.isInteger(v)) fail("redaction.kinds", "should map names to counts");
  }
  return {
    kinds: { ...(k as Record<string, number>) },
    redactedQuery: str(d, "redacted_query", "redaction", ""),
    redactedAnswer: str(d, "redacted_answer", "redaction", ""),
  };
}

function review(value: unknown): Review {
  const d = obj(value ?? {}, "review");
  const c = d["case_id"];
  if (c !== undefined && c !== null && (typeof c !== "number" || !Number.isInteger(c))) {
    fail("review.case_id", "should be an integer or null");
  }
  return { routed: bool(d, "routed", "review", false), caseId: (c as number | null | undefined) ?? null };
}

function findings(d: Obj, where: string): PatientFinding[] {
  return list(d, "patient_findings", where).map((f, i) => {
    const w = `patient_findings[${i}]`;
    const o = obj(f, w);
    return {
      severity: str(o, "severity", w),
      code: str(o, "code", w),
      medicine: str(o, "medicine", w),
      message: str(o, "message", w),
      ruleId: str(o, "rule_id", w, ""),
      source: str(o, "source", w, ""),
    };
  });
}

export function parseVerify(data: unknown): VerifyResponse {
  const w = "verify response";
  const d = obj(data, w);
  const unsupported = list(d, "unsupported_claims", w);
  if (!unsupported.every((i) => typeof i === "number" && Number.isInteger(i))) {
    fail(`${w}.unsupported_claims`, "should be a list of integers");
  }
  const decision = str(d, "decision", w);
  return {
    decision,
    reasonCode: str(d, "reason_code", w),
    claims: list(d, "claims", w).map((c, i) => {
      const cw = `claims[${i}]`;
      const o = obj(c, cw);
      return {
        text: str(o, "text", cw),
        grounded: bool(o, "grounded", cw),
        sourceId: optStr(o, "source_id", cw),
        support: support(o["support"], `${cw}.support`),
        relevance: num(o, "relevance", cw, 0),
        issues: strs(o, "issues", cw),
      };
    }),
    unsupportedClaims: unsupported as number[],
    doseIssues: list(d, "dose_issues", w).map((x, i) => {
      const dw = `dose_issues[${i}]`;
      const o = obj(x, dw);
      return { claim: int(o, "claim", dw), value: str(o, "value", dw), message: str(o, "message", dw) };
    }),
    explanation: explanation(d["explanation"]),
    redaction: redaction(d["redaction"]),
    auditId: str(d, "audit_id", w),
    latencyMs: int(d, "latency_ms", w),
    sources: list(d, "sources", w).map((s, i) => {
      const sw = `sources[${i}]`;
      const o = obj(s, sw);
      return { id: str(o, "id", sw), title: str(o, "title", sw), topic: str(o, "topic", sw, "") };
    }),
    patientFindings: findings(d, w),
    review: review(d["review"]),
    passed: decision === "pass",
  };
}

export function parseAsk(data: unknown): AskResponse {
  const w = "ask response";
  const d = obj(data, w);
  const trace = list(d, "trace", w);
  if (!trace.every((t) => t !== null && typeof t === "object" && !Array.isArray(t))) {
    fail(`${w}.trace`, "should be a list of objects");
  }
  const decision = str(d, "decision", w);
  const outcome = str(d, "outcome", w, decision === "answer" ? "answered" : "refused");
  return {
    decision,
    outcome,
    answerText: str(d, "answer_text", w),
    auditId: str(d, "audit_id", w),
    totalMs: int(d, "total_ms", w),
    llmUsed: bool(d, "llm_used", w, false),
    refusedReason: optStr(d, "refused_reason", w),
    reasonCode: optStr(d, "reason_code", w),
    claims: list(d, "claims", w).map((c, i) => {
      const cw = `claims[${i}]`;
      const o = obj(c, cw);
      return {
        text: str(o, "text", cw),
        sourceIds: strs(o, "source_ids", cw),
        grounded: bool(o, "grounded", cw, false),
        groundingScore: num(o, "grounding_score", cw, 0),
        relevance: num(o, "relevance", cw, 0),
        support: support(o["support"], `${cw}.support`),
      };
    }),
    sources: list(d, "sources", w).map((s, i) => {
      const sw = `sources[${i}]`;
      const o = obj(s, sw);
      return {
        id: str(o, "id", sw),
        title: str(o, "title", sw),
        snippet: str(o, "snippet", sw),
        score: num(o, "score", sw),
        topic: str(o, "topic", sw, ""),
        section: str(o, "section", sw, ""),
        kind: str(o, "kind", sw, ""),
        rank: int(o, "rank", sw, 0),
        aboveGate: bool(o, "above_gate", sw, true),
      };
    }),
    patientFindings: findings(d, w),
    review: review(d["review"]),
    explanation: explanation(d["explanation"]),
    redaction: redaction(d["redaction"]),
    trace: trace as Record<string, unknown>[],
    answered: decision === "answer" && outcome === "answered",
  };
}

export function parseUsage(data: unknown): UsageReport {
  const w = "usage response";
  const d = obj(data, w);
  const rows = list(d, "rows", w);
  if (!rows.every((r) => r !== null && typeof r === "object" && !Array.isArray(r))) {
    fail(`${w}.rows`, "should be a list of objects");
  }
  return {
    groupBy: str(d, "group_by", w),
    from: str(d, "from", w),
    to: str(d, "to", w),
    rows: rows as Record<string, unknown>[],
    total: int(d, "total", w),
    billable: int(d, "billable", w),
  };
}

export function parseHealth(data: unknown): Health {
  const d = obj(data, "health response");
  const status = str(d, "status", "health response");
  return { status, ok: status === "ok" || status === "healthy", details: { ...d } };
}

export function patientToWire(p: Patient): Record<string, unknown> {
  const map: [keyof Patient, string][] = [
    ["ageYears", "age_years"],
    ["sex", "sex"],
    ["weightKg", "weight_kg"],
    ["pregnant", "pregnant"],
    ["breastfeeding", "breastfeeding"],
    ["egfr", "egfr"],
    ["creatinineUmolL", "creatinine_umol_l"],
    ["childPugh", "child_pugh"],
    ["allergies", "allergies"],
    ["medicines", "medicines"],
    ["conditions", "conditions"],
    ["labs", "labs"],
  ];
  const out: Record<string, unknown> = {};
  for (const [from, to] of map) {
    const v = p[from];
    if (v !== undefined && v !== null) out[to] = v;
  }
  return out;
}

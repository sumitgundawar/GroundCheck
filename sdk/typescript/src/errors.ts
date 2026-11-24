export interface ErrorInit {
  status?: number;
  code?: string;
  requestId?: string;
}

/** Base class. `status` is the HTTP status; it is undefined when no response arrived. */
export class GroundCheckError extends Error {
  readonly status?: number;
  readonly code?: string;
  readonly requestId?: string;

  constructor(message: string, init: ErrorInit = {}) {
    super(message);
    this.name = new.target.name;
    this.status = init.status;
    this.code = init.code;
    this.requestId = init.requestId;
  }

  override toString(): string {
    const extra = [
      ["status", this.status],
      ["code", this.code],
      ["request_id", this.requestId],
    ]
      .filter(([, v]) => v !== undefined && v !== "")
      .map(([k, v]) => `${k}=${v}`);
    return `${this.name}: ${this.message}${extra.length ? ` (${extra.join(", ")})` : ""}`;
  }
}

/** The call was rejected before anything was sent. */
export class InvalidArgumentError extends GroundCheckError {}
/** The server could not be reached, or the connection broke. */
export class APIConnectionError extends GroundCheckError {}
/** The request took longer than the timeout. */
export class APITimeoutError extends APIConnectionError {}
/** The server answered, but not with something this client understands. */
export class InvalidResponseError extends GroundCheckError {}
/** The server answered with an error status. */
export class APIStatusError extends GroundCheckError {}
export class BadRequestError extends APIStatusError {}
export class AuthenticationError extends APIStatusError {}
export class PermissionDeniedError extends APIStatusError {}
export class NotFoundError extends APIStatusError {}
export class PayloadTooLargeError extends APIStatusError {}
export class UnprocessableError extends APIStatusError {}
export class ServerError extends APIStatusError {}
/** A redirect, or a 4xx status the API does not use. */
export class UnexpectedStatusError extends APIStatusError {}

export class RateLimitError extends APIStatusError {
  /** Seconds the server asked the caller to wait, if it said. */
  readonly retryAfter?: number;

  constructor(message: string, init: ErrorInit & { retryAfter?: number } = {}) {
    super(message, init);
    this.retryAfter = init.retryAfter;
  }
}

export const STATUS_ERRORS: Record<number, new (message: string, init?: ErrorInit) => APIStatusError> = {
  400: BadRequestError,
  401: AuthenticationError,
  403: PermissionDeniedError,
  404: NotFoundError,
  413: PayloadTooLargeError,
  422: UnprocessableError,
};

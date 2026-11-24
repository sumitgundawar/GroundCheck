"""Exceptions. One class per HTTP status the API uses, all deriving from GroundCheckError."""

from __future__ import annotations

from typing import Optional


class GroundCheckError(Exception):
    """Base class. `status` is the HTTP status (None when no response arrived)."""

    def __init__(self, message: str, *, status: Optional[int] = None, code: Optional[str] = None,
                 request_id: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.status = status
        self.code = code
        self.request_id = request_id

    def __str__(self) -> str:
        extra = [f"{k}={v}" for k, v in (("status", self.status), ("code", self.code),
                                          ("request_id", self.request_id)) if v is not None and v != ""]
        return f"{self.message} ({', '.join(extra)})" if extra else self.message


class InvalidArgumentError(GroundCheckError, ValueError):
    """The call was rejected before anything was sent."""


class APIConnectionError(GroundCheckError):
    """The server couldn't be reached, or the connection broke."""


class APITimeoutError(APIConnectionError):
    """The request took longer than the timeout."""


class InvalidResponseError(GroundCheckError):
    """The server answered, but not with something this client understands."""


class APIStatusError(GroundCheckError):
    """The server answered with an error status."""


class BadRequestError(APIStatusError): ...          # 400
class AuthenticationError(APIStatusError): ...      # 401
class PermissionDeniedError(APIStatusError): ...    # 403
class NotFoundError(APIStatusError): ...            # 404
class PayloadTooLargeError(APIStatusError): ...     # 413
class UnprocessableError(APIStatusError): ...       # 422


class RateLimitError(APIStatusError):               # 429
    def __init__(self, message: str, *, retry_after: Optional[float] = None, **kw):
        super().__init__(message, **kw)
        self.retry_after = retry_after


class ServerError(APIStatusError): ...              # 5xx


class UnexpectedStatusError(APIStatusError): ...    # anything else (redirects, other 4xx)


STATUS_ERRORS = {400: BadRequestError, 401: AuthenticationError, 403: PermissionDeniedError, 404: NotFoundError,
                 413: PayloadTooLargeError, 422: UnprocessableError}

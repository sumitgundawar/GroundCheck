"""The client. Standard library only."""

from __future__ import annotations

import dataclasses
import email.utils
import http.client
import ipaddress
import json
import logging
import math
import os
import random
import re
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from . import errors
from .errors import (APIConnectionError, APITimeoutError, InvalidArgumentError, InvalidResponseError, RateLimitError,
                     ServerError, UnexpectedStatusError)
from .models import (AskResponse, Health, Patient, RateLimit, UsageReport, VerifyResponse)

log = logging.getLogger("groundcheck_client")

DEFAULT_BASE_URL = "http://localhost:8000"
MAX_RESPONSE_BYTES = 32 * 1024 * 1024
RETRY_STATUSES_FROM = 500
_LOOPBACK_NAMES = {"localhost"}


class _Secret:
    """Holds the API key so it can't leak through repr, str, logging or a traceback's locals dump."""

    __slots__ = ("_value",)

    def __init__(self, value: str):
        self._value = value

    def reveal(self) -> str:
        return self._value

    def __repr__(self) -> str:
        return "<secret>"

    __str__ = __repr__

    def __reduce__(self):
        raise TypeError("A secret can't be pickled.")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Never follow a redirect: it would send the API key wherever the server points."""

    def redirect_request(self, *args, **kwargs):
        return None


def _is_loopback(host: str) -> bool:
    if host in _LOOPBACK_NAMES or host.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _day(value: Union[str, date, datetime, None], name: str) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str):
        try:
            return date.fromisoformat(value).isoformat()
        except ValueError:
            pass
    raise InvalidArgumentError(f"{name} must be a date like 2026-11-24")


def _retry_after(value: Optional[str]) -> Optional[float]:
    """Seconds from a Retry-After header (a number of seconds or an HTTP date); None if unusable."""
    if not value:
        return None
    value = value.strip()
    try:
        seconds = float(value)
    except ValueError:
        try:
            when = email.utils.parsedate_to_datetime(value)
        except (TypeError, ValueError, IndexError):
            return None
        if when is None:
            return None
        now = datetime.now(when.tzinfo) if when.tzinfo else datetime.utcnow()
        seconds = (when - now).total_seconds()
    if math.isnan(seconds) or math.isinf(seconds):
        return None
    return max(0.0, seconds)


def _int_header(headers: Mapping[str, str], name: str) -> Optional[int]:
    try:
        return int(headers.get(name, ""))
    except (TypeError, ValueError):
        return None


class Client:
    """Client for the GroundCheck API.

    api_key        defaults to $GROUNDCHECK_API_KEY. Sent as "Authorization: Bearer <key>".
    base_url       defaults to $GROUNDCHECK_BASE_URL, else http://localhost:8000. Plain http is only
                   allowed for loopback addresses unless allow_insecure_http=True, so a key can't go
                   over the network unencrypted by accident.
    timeout        seconds, applied to connecting and to each read, and as a limit on reading a body.
    max_retries    extra attempts after a 429 or 5xx response (and, for GET, a dropped connection).
                   4xx responses other than 429 are never retried.
    max_retry_wait the longest the client will sleep for one retry; a longer Retry-After raises
                   RateLimitError straight away instead.
    """

    def __init__(self, api_key: Optional[str] = None, *, base_url: Optional[str] = None, timeout: float = 30.0,
                 max_retries: int = 2, backoff_base: float = 0.5, backoff_max: float = 8.0,
                 max_retry_wait: float = 30.0, allow_insecure_http: bool = False,
                 user_agent: Optional[str] = None,
                 sleep: Callable[[float], None] = time.sleep, jitter: Callable[[], float] = lambda: random.uniform(0.5, 1.0)):
        key = api_key if api_key is not None else os.environ.get("GROUNDCHECK_API_KEY")
        if not isinstance(key, str) or not key:
            raise InvalidArgumentError("An API key is required: pass api_key or set GROUNDCHECK_API_KEY.")
        if any(ord(c) <= 32 or ord(c) >= 127 for c in key):
            raise InvalidArgumentError("The API key contains spaces, control or non-ASCII characters.")
        url = base_url if base_url is not None else os.environ.get("GROUNDCHECK_BASE_URL", DEFAULT_BASE_URL)
        parts = urllib.parse.urlsplit(url if isinstance(url, str) else "")
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise InvalidArgumentError("base_url must be an http or https URL.")
        if parts.username or parts.password or parts.query or parts.fragment:
            raise InvalidArgumentError("base_url can't contain credentials, a query or a fragment.")
        if parts.scheme == "http" and not allow_insecure_http and not _is_loopback(parts.hostname):
            raise InvalidArgumentError("Refusing to send an API key over plain http to a non-local host. "
                                       "Use https, or pass allow_insecure_http=True.")
        if not (isinstance(timeout, (int, float)) and not isinstance(timeout, bool) and timeout > 0
                and math.isfinite(timeout)):
            raise InvalidArgumentError("timeout must be a positive number of seconds.")
        if not (isinstance(max_retries, int) and not isinstance(max_retries, bool) and 0 <= max_retries <= 10):
            raise InvalidArgumentError("max_retries must be an integer from 0 to 10.")
        self._key = _Secret(key)
        self.base_url = url.rstrip("/")
        self.timeout = float(timeout)
        self.max_retries = max_retries
        self.backoff_base, self.backoff_max, self.max_retry_wait = backoff_base, backoff_max, max_retry_wait
        self._sleep, self._jitter = sleep, jitter
        self.user_agent = user_agent or "groundcheck-client-python/0.1.0"
        self._opener = urllib.request.build_opener(_NoRedirect)

    def __repr__(self) -> str:
        return f"Client(base_url={self.base_url!r}, api_key='***', timeout={self.timeout}, max_retries={self.max_retries})"

    def __reduce__(self):
        raise TypeError("A Client holds an API key and can't be pickled.")

    def __enter__(self) -> "Client":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        """Nothing is held open; present so the client can be used in a `with` block."""

    # --- Endpoints -----------------------------------------------------------

    def ask(self, question: str, *, site: Optional[str] = None,
            patient: Union[Patient, Mapping[str, Any], None] = None) -> AskResponse:
        body: Dict[str, Any] = {"question": _text(question, "question")}
        self._optional(body, site, patient)
        data, meta = self._request("POST", "/api/v1/ask", body)
        return dataclasses.replace(AskResponse.from_dict(data), **meta)

    def verify(self, question: str, answer: str, *, site: Optional[str] = None,
               sources: Optional[Sequence[str]] = None,
               patient: Union[Patient, Mapping[str, Any], None] = None) -> VerifyResponse:
        body: Dict[str, Any] = {"question": _text(question, "question"), "answer": _text(answer, "answer")}
        if sources is not None:
            if isinstance(sources, (str, bytes)) or not all(isinstance(s, str) for s in sources):
                raise InvalidArgumentError("sources must be a list of source id strings.")
            body["sources"] = list(sources)
        self._optional(body, site, patient)
        data, meta = self._request("POST", "/api/v1/verify", body)
        return dataclasses.replace(VerifyResponse.from_dict(data), **meta)

    def usage(self, group_by: str = "key", *, start: Union[str, date, datetime, None] = None,
              end: Union[str, date, datetime, None] = None) -> UsageReport:
        if group_by not in ("key", "day", "site", "endpoint", "decision"):
            raise InvalidArgumentError("group_by must be key, day, site, endpoint or decision.")
        query = {"group_by": group_by}
        for name, value in (("from", _day(start, "start")), ("to", _day(end, "end"))):
            if value:
                query[name] = value
        data, meta = self._request("GET", "/api/v1/usage?" + urllib.parse.urlencode(query), None)
        return dataclasses.replace(UsageReport.from_dict(data), **meta)

    def health(self) -> Health:
        data, meta = self._request("GET", "/api/v1/health", None)
        return dataclasses.replace(Health.from_dict(data), request_id=meta["request_id"])

    # --- Internals -----------------------------------------------------------

    @staticmethod
    def _optional(body: Dict[str, Any], site: Optional[str], patient: Any) -> None:
        if site is not None:
            if not isinstance(site, str) or not site:
                raise InvalidArgumentError("site must be a non-empty string.")
            body["site"] = site
        if patient is not None:
            if isinstance(patient, Patient):
                patient = patient.to_dict()
            if not isinstance(patient, Mapping):
                raise InvalidArgumentError("patient must be a Patient or a mapping.")
            body["patient"] = dict(patient)

    def _delay(self, attempt: int, retry_after: Optional[float]) -> float:
        if retry_after is not None:
            return retry_after
        return min(self.backoff_max, self.backoff_base * (2 ** attempt)) * self._jitter()

    def _request(self, method: str, path: str, body: Optional[Dict[str, Any]]) -> Tuple[Any, Dict[str, Any]]:
        payload = None
        if body is not None:
            try:
                payload = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
            except (TypeError, ValueError, UnicodeEncodeError) as exc:
                raise InvalidArgumentError(f"The request can't be encoded as JSON: {exc}") from None
        url = self.base_url + path
        for attempt in range(self.max_retries + 1):
            last = attempt == self.max_retries
            try:
                status, headers, raw = self._send(method, url, payload)
            except (APITimeoutError, APIConnectionError):
                if method == "GET" and not last:
                    log.warning("%s %s: connection problem, retrying (%d/%d)", method, path, attempt + 1,
                                self.max_retries)
                    self._sleep(self._delay(attempt, None))
                    continue
                raise
            log.debug("%s %s -> %d (attempt %d)", method, path, status, attempt + 1)
            request_id = headers.get("x-request-id")
            if 200 <= status < 300:
                return self._decode(raw, status, request_id), {
                    "request_id": request_id,
                    **({"rate_limit": self._rate_limit(headers)} if self._has_rate_limit(headers) else {})}
            retryable = status == 429 or status >= RETRY_STATUSES_FROM
            wait = _retry_after(headers.get("retry-after")) if status == 429 or status == 503 else None
            if retryable and not last and (wait is None or wait <= self.max_retry_wait):
                delay = self._delay(attempt, wait)
                log.warning("%s %s -> %d, retrying in %.2fs (%d/%d)", method, path, status, delay, attempt + 1,
                            self.max_retries)
                self._sleep(delay)
                continue
            raise self._error(status, headers, raw, wait, self._key.reveal())
        raise AssertionError("unreachable")  # pragma: no cover

    def _send(self, method: str, url: str, payload: Optional[bytes]) -> Tuple[int, Mapping[str, str], bytes]:
        req = urllib.request.Request(url, data=payload, method=method, headers={
            "Authorization": "Bearer " + self._key.reveal(), "Accept": "application/json",
            "User-Agent": self.user_agent, **({"Content-Type": "application/json"} if payload is not None else {})})
        deadline = time.monotonic() + self.timeout
        try:
            try:
                resp = self._opener.open(req, timeout=self.timeout)
            except urllib.error.HTTPError as err:
                resp = err
            with resp:
                status = resp.status if hasattr(resp, "status") else resp.code
                headers = {k.lower(): v for k, v in resp.headers.items()}
                chunks: List[bytes] = []
                size = 0
                while True:
                    if time.monotonic() > deadline:
                        raise APITimeoutError(f"The response took longer than {self.timeout:g}s.")
                    chunk = (getattr(resp, "read1", None) or resp.read)(65536)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > MAX_RESPONSE_BYTES:
                        raise InvalidResponseError("The response is larger than the client accepts.", status=status)
                    chunks.append(chunk)
                return status, headers, b"".join(chunks)
        except (socket.timeout, TimeoutError):
            raise APITimeoutError(f"The request timed out after {self.timeout:g}s.") from None
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, (socket.timeout, TimeoutError)):
                raise APITimeoutError(f"The request timed out after {self.timeout:g}s.") from None
            raise APIConnectionError(f"Couldn't reach the server: {exc.reason}") from None
        except (http.client.HTTPException, ConnectionError, OSError) as exc:
            if isinstance(exc, APITimeoutError):
                raise
            raise APIConnectionError(f"The connection failed: {type(exc).__name__}") from None

    @staticmethod
    def _decode(raw: bytes, status: int, request_id: Optional[str]) -> Any:
        try:
            return json.loads(raw.decode("utf-8-sig"))
        except (UnicodeDecodeError, json.JSONDecodeError, RecursionError):
            raise InvalidResponseError("The server's response wasn't valid JSON.", status=status,
                                       request_id=request_id) from None

    @staticmethod
    def _has_rate_limit(headers: Mapping[str, str]) -> bool:
        return any(k.startswith("x-ratelimit-") for k in headers)

    @staticmethod
    def _rate_limit(headers: Mapping[str, str]) -> RateLimit:
        return RateLimit(_int_header(headers, "x-ratelimit-limit"), _int_header(headers, "x-ratelimit-remaining"),
                         _int_header(headers, "x-ratelimit-reset"))

    @staticmethod
    def _scrub(text: str, key: str) -> str:
        """An error message must never carry the API key, even when the server echoes it back."""
        if key:
            text = text.replace(key, "***")
        text = re.sub(r"gck_[A-Za-z0-9_-]{4,}", "***", text)
        return re.sub(r"(?i)\b(bearer|basic)\s+[^\s\"',;]+", r"\1 ***", text)

    @staticmethod
    def _error(status: int, headers: Mapping[str, str], raw: bytes, retry_after: Optional[float], key: str = ""):
        scrub = lambda v: Client._scrub(v, key)  # noqa: E731
        request_id = headers.get("x-request-id")
        code = None
        message = f"HTTP {status}"
        try:
            parsed = json.loads(raw.decode("utf-8"))
            err = parsed.get("error") if isinstance(parsed, dict) else None
            if isinstance(err, dict):
                if isinstance(err.get("message"), str) and err["message"].strip():
                    message = scrub(err["message"].strip())[:1000]
                if isinstance(err.get("code"), str):
                    code = scrub(err["code"])[:100]
        except (UnicodeDecodeError, json.JSONDecodeError, RecursionError):
            pass
        kw = {"status": status, "code": code, "request_id": scrub(request_id) if request_id else request_id}
        if status == 429:
            return RateLimitError(message, retry_after=retry_after, **kw)
        if status >= 500:
            return ServerError(message, **kw)
        return errors.STATUS_ERRORS.get(status, UnexpectedStatusError)(message, **kw)


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise InvalidArgumentError(f"{name} must be a string.")
    if not value.strip():
        raise InvalidArgumentError(f"{name} is empty.")
    return value

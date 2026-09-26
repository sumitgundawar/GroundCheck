"""How this process logs.

Without this, Python falls back to its handler of last resort: WARNING and
above, no timestamp, no level, no logger name, straight to stderr. That is how
"Database unavailable, continuing without persistence" — an answer served with
no audit trail behind it — reached an operator as one bare line with nothing to
tell them when it happened or which instance said it.

Format is plain text by default and JSON when something is collecting logs,
since a log shipper can read fields but not prose.
"""

from __future__ import annotations

import json
import logging
import logging.config
import time

from . import config

_configured = False

# Fields on a LogRecord that every record carries; anything else an emitter
# attached with `extra=` is context worth keeping.
_STANDARD = frozenset(vars(logging.LogRecord("", 0, "", 0, "", (), None)))


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        out = {
            "time": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
                    + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _STANDARD and not key.startswith("_"):
                out[key] = value
        if record.exc_info:
            out["exception"] = self.formatException(record.exc_info)
        return json.dumps(out, ensure_ascii=False, default=str)


def configure(force: bool = False) -> None:
    """Set up logging once. Safe to call from anywhere."""
    global _configured
    if _configured and not force:
        return
    level = (config.LOG_LEVEL or "INFO").upper()
    if level not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        level = "INFO"
    handler = logging.StreamHandler()
    if (config.LOG_FORMAT or "plain").lower() == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)-8s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S"))
    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level)
    # Access logs are uvicorn's to emit; this only decides how they look.
    for noisy in ("uvicorn.access", "httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(max(getattr(logging, level), logging.WARNING))
    _configured = True

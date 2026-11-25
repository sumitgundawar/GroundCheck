"""OpenTelemetry tracing of the pipeline, optional and off by default.

When OTEL_EXPORTER_OTLP_ENDPOINT is set (and the OpenTelemetry packages are
installed) every answered or refused question exports one root span and one
child span per stage that ran, with the stage's status and duration. Nothing
about the question goes into a span: not its text, not the stage's detail line
(which can quote a term from it), not who asked. Attributes come from the fixed
list below, and each value is a stage name, a status or a reason code.

Everything is imported lazily and every failure is swallowed: tracing never
changes an answer, and with it off the cost is one boolean test."""

from __future__ import annotations

import logging
import threading
import time

from . import config

log = logging.getLogger("groundcheck.tracing")

# The only attributes a span may carry.
ALLOWED_ATTRIBUTES = frozenset({
    "groundcheck.stage", "groundcheck.status", "groundcheck.duration_ms", "groundcheck.decision",
    "groundcheck.outcome", "groundcheck.reason_code", "groundcheck.llm_used", "groundcheck.stages_run",
    "groundcheck.language", "groundcheck.audit_id",
})

_on = False
_tracer = None
_provider = None
_lock = threading.Lock()
_tried = False
errors = 0


def active() -> bool:
    """Whether spans are being exported. Cheap enough for every request."""
    if _on:
        return True
    if _tried or not config.OTEL_EXPORTER_OTLP_ENDPOINT:
        return False
    return _start()


def _exporter():
    protocol = config.OTEL_EXPORTER_OTLP_PROTOCOL
    endpoint = config.OTEL_EXPORTER_OTLP_ENDPOINT
    if protocol.startswith("grpc"):
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

        return OTLPSpanExporter(endpoint=endpoint)
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    return OTLPSpanExporter(endpoint=endpoint.rstrip("/") + "/v1/traces" if not endpoint.rstrip("/").endswith("/v1/traces") else endpoint)


def _build(exporter, simple: bool):
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor

    provider = TracerProvider(resource=Resource.create({"service.name": config.OTEL_SERVICE_NAME}))
    provider.add_span_processor((SimpleSpanProcessor if simple else BatchSpanProcessor)(exporter))
    return provider


def _start() -> bool:
    global _on, _tracer, _provider, _tried
    with _lock:
        if _on:
            return True
        _tried = True
        try:
            _provider = _build(_exporter(), simple=False)
            _tracer = _provider.get_tracer("groundcheck.pipeline")
            _on = True
        except Exception as exc:  # noqa: BLE001 - not installed, or a bad endpoint: run without tracing
            log.warning("OpenTelemetry tracing is off: %s", exc)
            _on = False
        return _on


def use_exporter(exporter) -> None:
    """Send spans to this exporter, synchronously (tests)."""
    global _on, _tracer, _provider, _tried
    with _lock:
        _provider = _build(exporter, simple=True)
        _tracer = _provider.get_tracer("groundcheck.pipeline")
        _on, _tried = True, True


def reset() -> None:
    """Back to off (tests)."""
    global _on, _tracer, _provider, _tried, errors
    with _lock:
        if _provider is not None:
            try:
                _provider.shutdown()
            except Exception:  # noqa: BLE001
                pass
        _on, _tracer, _provider, _tried, errors = False, None, None, False, 0


def shutdown() -> None:
    """Flush what is queued; called when the app stops."""
    if _provider is not None:
        try:
            _provider.force_flush(5000)
        except Exception:  # noqa: BLE001
            pass


def _clean(attributes: dict) -> dict:
    return {k: v for k, v in attributes.items() if k in ALLOWED_ATTRIBUTES and v is not None}


def record(trace, *, decision: str, outcome: str | None, reason_code: str | None, llm_used: bool,
           total_ms: int, audit_id: str, language: str | None = None) -> None:
    """Export one question's stages. Never raises."""
    global errors
    if not _on or _tracer is None:
        return
    try:
        from opentelemetry.trace import Status, StatusCode

        end = time.time_ns()
        ran = [s for s in trace if s.status != "skip" and s.name != "decision"]
        start = end - int(total_ms) * 1_000_000
        root = _tracer.start_span("groundcheck.ask", start_time=start, attributes=_clean({
            "groundcheck.decision": decision, "groundcheck.outcome": outcome, "groundcheck.reason_code": reason_code,
            "groundcheck.llm_used": bool(llm_used), "groundcheck.stages_run": len(ran),
            "groundcheck.duration_ms": int(total_ms), "groundcheck.language": language, "groundcheck.audit_id": audit_id,
        }))
        from opentelemetry import trace as otel_trace

        context = otel_trace.set_span_in_context(root)
        cursor = start
        for step in ran:
            ms = int(step.ms or 0)
            child = _tracer.start_span(f"groundcheck.{step.name.replace(' ', '_')}", context=context, start_time=cursor,
                                       attributes=_clean({"groundcheck.stage": step.name, "groundcheck.status": step.status,
                                                          "groundcheck.duration_ms": ms}))
            cursor += ms * 1_000_000
            if step.status == "fail":
                child.set_status(Status(StatusCode.ERROR, "refused at this stage"))
            else:
                child.set_status(Status(StatusCode.OK))
            child.end(end_time=cursor)
        root.set_status(Status(StatusCode.OK))
        root.end(end_time=max(end, cursor))
    except Exception as exc:  # noqa: BLE001
        errors += 1
        log.debug("span export failed: %s", exc)

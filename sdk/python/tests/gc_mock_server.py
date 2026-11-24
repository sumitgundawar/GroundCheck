"""A scriptable HTTP server for testing the SDK. Not the real app."""

from __future__ import annotations

import json
import socket
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


@dataclass
class Step:
    status: int = 200
    body: object = b""
    headers: dict = field(default_factory=dict)
    delay: float = 0.0
    drop: bool = False            # close the connection without answering
    trickle: bool = False         # send headers, then the body one byte at a time
    raw: bool = False             # body is already bytes; don't JSON-encode


@dataclass
class Seen:
    method: str
    path: str
    headers: dict
    body: bytes

    def json(self):
        return json.loads(self.body.decode("utf-8"))


class MockServer:
    def __init__(self):
        self.requests: list[Seen] = []
        self.steps: list[Step] = []
        self.default = Step(200, {})
        self.lock = threading.Lock()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def _handle(self):
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length) if length else b""
                with outer.lock:
                    outer.requests.append(Seen(self.command, self.path,
                                               {k.lower(): v for k, v in self.headers.items()}, body))
                    step = outer.steps.pop(0) if outer.steps else outer.default
                if step.delay:
                    time.sleep(step.delay)
                if step.drop:
                    self.close_connection = True
                    try:
                        self.connection.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
                    return
                payload = step.body if step.raw or isinstance(step.body, bytes) else json.dumps(step.body).encode()
                self.send_response(step.status)
                headers = {"Content-Type": "application/json", **step.headers}
                for k, v in headers.items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(payload) if not step.trickle else 1000))
                self.send_header("Connection", "close")
                self.end_headers()
                if step.trickle:
                    try:
                        for _ in range(1000):
                            self.wfile.write(b" ")
                            self.wfile.flush()
                            time.sleep(0.05)
                    except OSError:
                        pass
                    return
                self.wfile.write(payload)

            do_GET = do_POST = do_PUT = do_DELETE = _handle

        ThreadingHTTPServer.request_queue_size = 128
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.thread = threading.Thread(target=lambda: self.httpd.serve_forever(poll_interval=0.01), daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()

    def enqueue(self, *steps: Step) -> None:
        self.steps.extend(steps)


ASK_OK = {
    "decision": "answer", "outcome": "answered", "answer_text": "Take 10 mg once daily. [DOC1-001]",
    "refused_reason": None, "claims": [{"text": "Take 10 mg once daily.", "source_ids": ["DOC1-001"], "grounded": True,
                                          "grounding_score": 0.93, "relevance": 0.8,
                                          "support": {"source_id": "DOC1-001", "start": 0, "end": 22,
                                                      "sentence": "Take 10 mg once daily."}}],
    "sources": [{"id": "DOC1-001", "title": "Label: Dosage", "snippet": "Take 10 mg once daily.", "score": 0.71,
                 "topic": "label", "section": "Dosage", "kind": "organisation", "rank": 1, "above_gate": True}],
    "trace": [{"name": "retrieval", "status": "pass", "detail": "ok", "ms": 3}], "audit_id": "abc123", "total_ms": 41,
    "llm_used": False, "patient_findings": [{"severity": "warn", "code": "renal_dose", "medicine": "X",
                                              "message": "Check kidney function.", "rule_id": "R-1", "source": "S"}],
    "reason_code": None, "review": {"routed": False, "case_id": None},
    "explanation": {"summary": "", "missing_terms": [], "suggestions": []},
    "redaction": {"kinds": {"phone": 1}, "redacted_query": "Call [PHONE]"},
}

VERIFY_OK = {
    "decision": "pass", "reason_code": "verified",
    "claims": [{"text": "Take 10 mg once daily.", "grounded": True, "source_id": "DOC1-001",
                "support": {"source_id": "DOC1-001", "start": 0, "end": 22, "sentence": "Take 10 mg once daily."},
                "relevance": 0.9, "issues": []}],
    "unsupported_claims": [], "dose_issues": [], "explanation": {"summary": "All claims supported."},
    "redaction": {"kinds": {}, "redacted_query": "q", "redacted_answer": "a"}, "audit_id": "v123", "latency_ms": 12,
    "sources": [{"id": "DOC1-001", "title": "Label"}], "patient_findings": [], "review": {"routed": False},
}

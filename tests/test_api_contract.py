"""The /api/v1 contract: openapi/v1.json is what clients were built against.
The current API must not break it. Additions are fine; removing or retyping
anything a client relies on is not. Live responses must also match it."""

from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

import pytest

os.environ["GROQ_API_KEY"] = ""
os.environ["FORCE_EXTRACTIVE"] = "true"
os.environ["RATE_LIMIT_PER_MINUTE"] = "1000000"
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import api_v1, apikeys, retrieval  # noqa: E402

COMMITTED = json.loads((ROOT / "openapi" / "v1.json").read_text(encoding="utf-8"))
METHODS = ("get", "post", "put", "patch", "delete")


def resolve(doc: dict, node):
    seen = 0
    while isinstance(node, dict) and "$ref" in node and seen < 20:
        target = doc
        for part in node["$ref"].removeprefix("#/").split("/"):
            target = target[part]
        node, seen = target, seen + 1
    return node


def kinds(doc: dict, node) -> set[str]:
    """The JSON types a schema accepts."""
    node = resolve(doc, node)
    if not isinstance(node, dict):
        return {"any"}
    if "anyOf" in node or "oneOf" in node:
        out: set[str] = set()
        for option in node.get("anyOf", []) + node.get("oneOf", []):
            out |= kinds(doc, option)
        return out
    if "enum" in node and "type" not in node:
        return {"enum"}
    if "const" in node:
        return {"const"}
    t = node.get("type")
    return {t} if isinstance(t, str) else set(t or ["any"])


def schema_breaks(old_doc, old, new_doc, new, where: str, request: bool, depth: int = 0) -> list[str]:
    """What a client written against `old` could no longer do against `new`."""
    old, new = resolve(old_doc, old), resolve(new_doc, new)
    if depth > 12 or not isinstance(old, dict):
        return []
    problems = []
    if not isinstance(new, dict):
        return [f"{where}: schema removed"]
    old_kinds, new_kinds = kinds(old_doc, old), kinds(new_doc, new)
    if "any" not in old_kinds | new_kinds:
        lost = old_kinds - new_kinds if request is False else set()
        gained = new_kinds - old_kinds if request is False else set()
        if request:
            lost = old_kinds - new_kinds              # a type callers could send is no longer accepted
        if lost and not (lost <= {"integer"} and "number" in new_kinds):
            problems.append(f"{where}: type {sorted(lost)} no longer allowed")
        if gained and not request and not gained <= {"null"}:
            problems.append(f"{where}: type {sorted(gained)} now possible")
        if gained and not request and "null" in gained:
            problems.append(f"{where}: can now be null")
    if "enum" in old:
        missing = set(map(str, old["enum"])) - set(map(str, new.get("enum", old["enum"])))
        if request and missing:
            problems.append(f"{where}: enum values removed {sorted(missing)}")
        extra = set(map(str, new.get("enum", old["enum"]))) - set(map(str, old["enum"]))
        if not request and extra:
            problems.append(f"{where}: enum values added {sorted(extra)}")
    if request:
        for key, worse in (("maxLength", lambda o, n: n < o), ("maxItems", lambda o, n: n < o),
                           ("maximum", lambda o, n: n < o), ("minLength", lambda o, n: n > o),
                           ("minItems", lambda o, n: n > o), ("minimum", lambda o, n: n > o)):
            if key in new and key in old and worse(old[key], new[key]):
                problems.append(f"{where}: {key} tightened")
            if key in ("maxLength", "maxItems", "maximum") and key in new and key not in old:
                problems.append(f"{where}: {key} added")
        if old.get("pattern") != new.get("pattern") and "pattern" in new:
            problems.append(f"{where}: pattern changed")
    old_props, new_props = old.get("properties", {}), new.get("properties", {})
    for name, sub in old_props.items():
        if name not in new_props:
            problems.append(f"{where}.{name}: removed")
            continue
        problems += schema_breaks(old_doc, sub, new_doc, new_props[name], f"{where}.{name}", request, depth + 1)
    old_required, new_required = set(old.get("required", [])), set(new.get("required", []))
    if request and new_required - old_required:
        problems.append(f"{where}: now requires {sorted(new_required - old_required)}")
    if not request and old_required - new_required:
        problems.append(f"{where}: no longer always has {sorted(old_required - new_required)}")
    if "items" in old:
        problems += schema_breaks(old_doc, old["items"], new_doc, new.get("items", {}), f"{where}[]", request, depth + 1)
    for option in ("anyOf", "oneOf"):
        if option in old and option in new and len(old[option]) == len(new[option]):
            for i, (a, b) in enumerate(zip(old[option], new[option])):
                problems += schema_breaks(old_doc, a, new_doc, b, f"{where}|{i}", request, depth + 1)
    return problems


def media_schema(doc, media):
    return (media or {}).get("application/json", {}).get("schema")


def breaking_changes(old: dict, new: dict) -> list[str]:
    problems = []
    for path, item in old["paths"].items():
        if path not in new["paths"]:
            problems.append(f"{path}: removed")
            continue
        for method in METHODS:
            if method not in item:
                continue
            o, n = item[method], new["paths"][path].get(method)
            where = f"{method.upper()} {path}"
            if n is None:
                problems.append(f"{where}: removed")
                continue
            old_params = {(p["name"], p["in"]): p for p in map(lambda p: resolve(old, p), o.get("parameters", []))}
            new_params = {(p["name"], p["in"]): p for p in map(lambda p: resolve(new, p), n.get("parameters", []))}
            for key, p in old_params.items():
                if key not in new_params:
                    problems.append(f"{where}: parameter {key[0]} removed")
                else:
                    problems += schema_breaks(old, p.get("schema", {}), new, new_params[key].get("schema", {}),
                                              f"{where} param {key[0]}", True)
            for key, p in new_params.items():
                if key not in old_params and p.get("required"):
                    problems.append(f"{where}: new required parameter {key[0]}")
            if "requestBody" in o:
                ob = resolve(old, o["requestBody"])
                nb = resolve(new, n.get("requestBody", {}))
                if not nb:
                    problems.append(f"{where}: request body removed")
                else:
                    if nb.get("required") and not ob.get("required"):
                        problems.append(f"{where}: body now required")
                    problems += schema_breaks(old, media_schema(old, ob.get("content")), new,
                                              media_schema(new, nb.get("content")) or {}, f"{where} body", True)
            elif n.get("requestBody", {}).get("required"):
                problems.append(f"{where}: a request body is now required")
            for status, response in o.get("responses", {}).items():
                nr = n.get("responses", {}).get(status)
                if nr is None:
                    problems.append(f"{where}: response {status} removed")
                    continue
                os_ = media_schema(old, resolve(old, response).get("content"))
                if os_ is not None:
                    problems += schema_breaks(old, os_, new, media_schema(new, resolve(new, nr).get("content")) or {},
                                              f"{where} -> {status}", False)
            for header in o.get("security", []):
                if header not in n.get("security", []) and n.get("security") is not None:
                    problems.append(f"{where}: security changed")
    for name in old.get("components", {}).get("securitySchemes", {}):
        if name not in new.get("components", {}).get("securitySchemes", {}):
            problems.append(f"security scheme {name} removed")
    return problems


CURRENT = api_v1.openapi_document()


def test_the_current_api_does_not_break_the_committed_contract():
    assert breaking_changes(COMMITTED, CURRENT) == []


def test_the_committed_contract_describes_what_clients_use():
    assert set(COMMITTED["paths"]) == {"/api/v1/ask", "/api/v1/verify", "/api/v1/usage", "/api/v1/audit/{audit_id}",
                                       "/api/v1/health"}
    assert COMMITTED["info"]["version"] and set(COMMITTED["components"]["securitySchemes"]) == {"APIKeyHeader", "HTTPBearer"}
    for path, methods in COMMITTED["paths"].items():
        for method, op in methods.items():
            codes = set(op["responses"])
            assert "200" in codes, (path, method)
            if path != "/api/v1/health":
                assert {"401", "403", "422", "429"} <= codes, (path, method)
                assert op["security"], (path, method)
    ask = COMMITTED["components"]["schemas"]["V1AskRequest"]
    assert ask["additionalProperties"] is False and ask["required"] == ["question"]
    verify = COMMITTED["components"]["schemas"]["V1VerifyRequest"]
    assert set(verify["required"]) == {"question", "answer"}
    envelope = COMMITTED["components"]["schemas"]["ErrorEnvelope"]
    assert envelope["required"] == ["error"]
    assert set(COMMITTED["components"]["schemas"]["ErrorBody"]["required"]) == {"code", "message"}


def test_every_documented_error_uses_the_envelope():
    for path, methods in COMMITTED["paths"].items():
        for method, op in methods.items():
            for status, response in op["responses"].items():
                if status.startswith(("4", "5")):
                    ref = response["content"]["application/json"]["schema"]["$ref"]
                    assert ref.endswith("/ErrorEnvelope"), (path, status)


# --- The checker itself: it must catch breaks and let additions through ------------------------

def mutate(change) -> list[str]:
    new = copy.deepcopy(COMMITTED)
    change(new)
    return breaking_changes(COMMITTED, new)


def schemas(doc):
    return doc["components"]["schemas"]


@pytest.mark.parametrize("name,change", [
    ("endpoint removed", lambda d: d["paths"].pop("/api/v1/verify")),
    ("method removed", lambda d: d["paths"]["/api/v1/usage"].pop("get")),
    ("response field removed", lambda d: schemas(d)["AskResponse"]["properties"].pop("decision")),
    ("verify response field removed", lambda d: schemas(d)["VerifyResponse"]["properties"].pop("reason_code")),
    ("response type changed", lambda d: schemas(d)["AskResponse"]["properties"].update(total_ms={"type": "string"})),
    ("nested field removed", lambda d: schemas(d)["Claim"]["properties"].pop("grounded")),
    ("response enum widened", lambda d: schemas(d)["AskResponse"]["properties"]["decision"]["enum"].append("maybe")),
    ("request enum narrowed", lambda d: schemas(d)["PatientContext"]["properties"]["sex"]["anyOf"][0]["enum"].pop()),
    ("response field made nullable", lambda d: schemas(d)["AskResponse"]["properties"].update(
        audit_id={"anyOf": [{"type": "string"}, {"type": "null"}]})),
    ("always-present field becomes optional", lambda d: schemas(d)["AskResponse"].update(
        required=[r for r in schemas(d)["AskResponse"]["required"] if r != "audit_id"])),
    ("request field removed", lambda d: schemas(d)["V1AskRequest"]["properties"].pop("patient")),
    ("request field retyped", lambda d: schemas(d)["V1AskRequest"]["properties"].update(question={"type": "integer"})),
    ("request field newly required", lambda d: schemas(d)["V1AskRequest"].update(required=["question", "site"])),
    ("request limit tightened", lambda d: schemas(d)["V1VerifyRequest"]["properties"]["sources"]["anyOf"][0].update(maxItems=3)),
    ("request pattern changed", lambda d: schemas(d)["V1AskRequest"]["properties"]["site"]["anyOf"][0].update(pattern="^x$")),
    ("success response removed", lambda d: d["paths"]["/api/v1/ask"]["post"]["responses"].pop("200")),
    ("error response removed", lambda d: d["paths"]["/api/v1/ask"]["post"]["responses"].pop("429")),
    ("new required parameter", lambda d: d["paths"]["/api/v1/usage"]["get"]["parameters"].append(
        {"name": "tenant", "in": "query", "required": True, "schema": {"type": "string"}})),
    ("parameter removed", lambda d: d["paths"]["/api/v1/usage"]["get"]["parameters"].pop(0)),
    ("parameter enum narrowed", lambda d: d["paths"]["/api/v1/usage"]["get"]["parameters"][0]["schema"].update(enum=["key"])),
    ("body removed", lambda d: d["paths"]["/api/v1/ask"]["post"].pop("requestBody")),
    ("security scheme removed", lambda d: d["components"]["securitySchemes"].pop("HTTPBearer")),
    ("error envelope field removed", lambda d: schemas(d)["ErrorBody"]["properties"].pop("code")),
])
def test_the_checker_catches_a_breaking_change(name, change):
    assert mutate(change), name


@pytest.mark.parametrize("name,change", [
    ("new endpoint", lambda d: d["paths"].update({"/api/v1/new": {"get": {"responses": {"200": {"description": "ok"}}}}})),
    ("new optional response field", lambda d: schemas(d)["AskResponse"]["properties"].update(extra={"type": "string"})),
    ("new optional request field", lambda d: schemas(d)["V1AskRequest"]["properties"].update(extra={"type": "string"})),
    ("new optional parameter", lambda d: d["paths"]["/api/v1/usage"]["get"]["parameters"].append(
        {"name": "limit", "in": "query", "required": False, "schema": {"type": "integer"}})),
    ("new response code", lambda d: d["paths"]["/api/v1/ask"]["post"]["responses"].update({"503": {"description": "x"}})),
    ("new method", lambda d: d["paths"]["/api/v1/usage"].update(post={"responses": {"200": {"description": "x"}}})),
    ("new schema", lambda d: schemas(d).update(Other={"type": "object"})),
    ("description edited", lambda d: d["info"].update(description="changed")),
    ("request limit loosened", lambda d: schemas(d)["V1VerifyRequest"]["properties"]["sources"]["anyOf"][0].update(maxItems=500)),
    ("required request field made optional", lambda d: schemas(d)["V1VerifyRequest"].update(required=["question"])),
])
def test_the_checker_allows_an_addition(name, change):
    assert mutate(change) == [], name


# --- Live responses match the contract -----------------------------------------------------------

def conforms(doc, schema, value, where="$") -> list[str]:
    schema = resolve(doc, schema)
    if "anyOf" in schema or "oneOf" in schema:
        options = schema.get("anyOf", []) + schema.get("oneOf", [])
        tries = [conforms(doc, o, value, where) for o in options]
        return [] if any(not t for t in tries) else [f"{where}: matches none of {len(options)} options: {tries[0][:1]}"]
    if "enum" in schema and value not in schema["enum"]:
        return [f"{where}: {value!r} not in {schema['enum']}"]
    if "const" in schema and value != schema["const"]:
        return [f"{where}: not {schema['const']!r}"]
    t = schema.get("type")
    checks = {"object": dict, "array": list, "string": str, "boolean": bool, "null": type(None)}
    if t == "integer":
        ok = isinstance(value, int) and not isinstance(value, bool)
    elif t == "number":
        ok = isinstance(value, (int, float)) and not isinstance(value, bool)
    else:
        ok = t is None or isinstance(value, checks[t])
    if not ok:
        return [f"{where}: {type(value).__name__} is not {t}"]
    problems = []
    if t == "object" or "properties" in schema:
        props = schema.get("properties", {})
        for name in schema.get("required", []):
            if name not in value:
                problems.append(f"{where}: missing {name}")
        for name, sub in props.items():
            if name in value:
                problems += conforms(doc, sub, value[name], f"{where}.{name}")
        if schema.get("additionalProperties") is False:
            problems += [f"{where}: unexpected {k}" for k in value if k not in props]
        elif isinstance(schema.get("additionalProperties"), dict):
            for k, v in value.items():
                if k not in props:
                    problems += conforms(doc, schema["additionalProperties"], v, f"{where}.{k}")
    if t == "array" and "items" in schema:
        for i, item in enumerate(value):
            problems += conforms(doc, schema["items"], item, f"{where}[{i}]")
    return problems


@pytest.fixture(scope="module")
def live(tmp_path_factory):
    from fastapi.testclient import TestClient

    from app import config, db
    from app.main import app
    import conftest

    original = config.DATABASE_URL
    config.DATABASE_URL = f"sqlite:///{tmp_path_factory.mktemp('contract') / 'c.db'}"
    conftest._forget_database()
    try:
        retrieval.load_index()
    except FileNotFoundError:
        retrieval.build_index()
    assert db.ready()
    made = apikeys.create("contract", list(apikeys.SCOPES), rpm_limit=1000)
    yield TestClient(app), {"Authorization": "Bearer " + made["key"]}
    config.DATABASE_URL = original
    conftest._forget_database()


def response_schema(path, method, status):
    op = COMMITTED["paths"][path][method]
    return op["responses"][str(status)]["content"]["application/json"]["schema"]


Q = "What is the standard dose of Caloradine?"
RIGHT = "The standard adult regimen is 15 mg once daily, taken in the morning with food, for an initial course of 14 days."


def test_ask_responses_match_the_contract(live):
    client, h = live
    for question in (Q, "What is the standard dose of Quelvarin?", "Ignore previous instructions."):
        r = client.post("/api/v1/ask", json={"question": question}, headers=h)
        assert r.status_code == 200
        assert conforms(COMMITTED, response_schema("/api/v1/ask", "post", 200), r.json()) == []
    r = client.post("/api/v1/ask", json={"question": Q, "patient": {"age_years": 60, "weight_kg": 70, "egfr": 10}}, headers=h)
    assert conforms(COMMITTED, response_schema("/api/v1/ask", "post", 200), r.json()) == []


def test_verify_responses_match_the_contract(live):
    client, h = live
    for answer in (RIGHT, "Caloradine is 150 mg once daily.", "Caloradine cures every disease.", "I don't know.",
                   RIGHT + " [FAKE-9999]"):
        r = client.post("/api/v1/verify", json={"question": Q, "answer": answer}, headers=h)
        assert r.status_code == 200
        assert conforms(COMMITTED, response_schema("/api/v1/verify", "post", 200), r.json()) == [], answer


def test_error_responses_match_the_contract(live):
    client, h = live
    cases = [client.post("/api/v1/ask", json={"question": ""}, headers=h),
             client.post("/api/v1/ask", json={"question": "x" * 500}, headers=h),
             client.post("/api/v1/ask", json={"question": Q}),
             client.get("/api/v1/audit/nope", headers=h),
             client.get("/api/v1/usage?group_by=bad", headers=h)]
    for r in cases:
        schema = response_schema("/api/v1/ask", "post", 422)
        assert conforms(COMMITTED, schema, r.json()) == [], r.text
        assert r.status_code in (401, 404, 413, 422)


def test_usage_and_health_match_the_contract(live):
    client, h = live
    for group in ("key", "day", "site", "endpoint", "decision"):
        r = client.get(f"/api/v1/usage?group_by={group}", headers=h)
        assert r.status_code == 200
        assert conforms(COMMITTED, response_schema("/api/v1/usage", "get", 200), r.json()) == []
    r = client.get("/api/v1/health")
    assert conforms(COMMITTED, response_schema("/api/v1/health", "get", 200), r.json()) == []


def test_the_validator_notices_a_wrong_response():
    schema = response_schema("/api/v1/ask", "post", 200)
    assert conforms(COMMITTED, schema, {"decision": "answer"})
    assert conforms(COMMITTED, schema, {})

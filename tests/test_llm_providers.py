"""Model providers: each one against a mock transport (no network), the failure
modes every one must survive, fallback to extractive, and the rule that the
guards judge a provider's draft exactly as they judge any other."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import httpx
import pytest

os.environ["GROQ_API_KEY"] = ""
os.environ["RATE_LIMIT_PER_MINUTE"] = "1000000"
os.environ["AUDIT_PERSIST"] = "false"
os.environ["REVIEW_QUEUE"] = "false"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config, llm, pipeline, providers, retrieval  # noqa: E402

QUESTION = "What is the standard dose of Caloradine?"
GOOD = {"insufficient_context": False, "claims": [{
    "text": "The standard adult regimen is 15 mg once daily, taken in the morning with food, for an initial course of 14 days.",
    "source_ids": ["CALO-001"]}]}
WRONG_DOSE = {"insufficient_context": False, "claims": [{
    "text": "The standard adult regimen is 150 mg once daily, taken in the morning with food, for an initial course of 14 days.",
    "source_ids": ["CALO-001"]}]}
UNGROUNDED = {"insufficient_context": False, "claims": [{
    "text": "Caloradine is also recommended for migraine prevention in adults.", "source_ids": ["CALO-001"]}]}
SECRET = "sk-test-SECRET-123456"


@pytest.fixture(autouse=True, scope="module")
def _index():
    try:
        retrieval.load_index()
    except FileNotFoundError:
        retrieval.build_index()


@pytest.fixture()
def calls():
    return []


def _install(monkeypatch, handler, calls):
    def wrapped(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return handler(request)

    monkeypatch.setattr(providers, "_client", lambda: httpx.Client(transport=httpx.MockTransport(wrapped), timeout=5))


def _select(monkeypatch, name):
    monkeypatch.setattr(config, "FORCE_EXTRACTIVE", False)
    monkeypatch.setattr(config, "LLM_PROVIDER", name)
    monkeypatch.setattr(config, "LLM_PROVIDER_EFFECTIVE", name)
    monkeypatch.setattr(config, "LLM_API_KEY", SECRET)
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", SECRET)
    monkeypatch.setattr(config, "AZURE_OPENAI_API_KEY", SECRET)
    monkeypatch.setattr(config, "AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
    monkeypatch.setattr(config, "AZURE_OPENAI_DEPLOYMENT", "gc-gpt")
    monkeypatch.setattr(llm, "LLM_AVAILABLE", True)
    monkeypatch.setattr(llm, "_thread", type("T", (), {})())


def _reply(name, content, **extra):
    """A well-formed provider reply carrying `content` (a dict, or raw text)."""
    import json

    text = content if isinstance(content, str) else json.dumps(content)
    if name == "anthropic":
        block = ({"type": "text", "text": text} if isinstance(content, str)
                 else {"type": "tool_use", "name": "reply", "id": "t1", "input": content})
        return {"content": [block], "stop_reason": extra.get("stop", "tool_use")}
    return {"choices": [{"message": {"content": text}, "finish_reason": extra.get("stop", "stop")}]}


ALL = ["openai_compatible", "anthropic", "azure_openai"]


# --- Request shape, per provider -------------------------------------------------

def test_openai_compatible_request(monkeypatch, calls):
    _select(monkeypatch, "openai_compatible")
    _install(monkeypatch, lambda r: httpx.Response(200, json=_reply("openai_compatible", GOOD)), calls)
    out = providers.build("openai_compatible").draft("sys", "usr", 0.0, llm.GEN_SCHEMA)
    assert llm._parse_llm_answer(out).claims[0].source_ids == ["CALO-001"]
    request = calls[0]
    assert str(request.url).endswith("/chat/completions")
    assert request.headers["authorization"] == f"Bearer {SECRET}"
    import json

    body = json.loads(request.content)
    assert body["response_format"] == {"type": "json_object"} and body["temperature"] == 0.0
    assert [m["role"] for m in body["messages"]] == ["system", "user"]


def test_azure_request_uses_deployment_and_api_key_header(monkeypatch, calls):
    _select(monkeypatch, "azure_openai")
    _install(monkeypatch, lambda r: httpx.Response(200, json=_reply("azure_openai", GOOD)), calls)
    providers.build("azure_openai").draft("s", "u", 0.0, llm.GEN_SCHEMA)
    request = calls[0]
    assert request.url.path == "/openai/deployments/gc-gpt/chat/completions"
    assert request.url.params["api-version"] == config.AZURE_OPENAI_API_VERSION
    assert request.headers["api-key"] == SECRET and "authorization" not in request.headers


def test_anthropic_request_forces_the_schema_tool(monkeypatch, calls):
    import json

    _select(monkeypatch, "anthropic")
    _install(monkeypatch, lambda r: httpx.Response(200, json=_reply("anthropic", GOOD)), calls)
    out = providers.build("anthropic").draft("sys", "usr", 0.0, llm.GEN_SCHEMA)
    assert json.loads(out) == GOOD
    request = calls[0]
    assert request.url.path == "/v1/messages"
    assert request.headers["x-api-key"] == SECRET and request.headers["anthropic-version"]
    body = json.loads(request.content)
    assert body["tool_choice"] == {"type": "tool", "name": "reply"}
    assert body["tools"][0]["input_schema"] == llm.GEN_SCHEMA
    assert body["system"] == "sys" and body["messages"] == [{"role": "user", "content": "usr"}]


def test_judge_uses_the_judge_model(monkeypatch, calls):
    import json

    _select(monkeypatch, "openai_compatible")
    monkeypatch.setattr(config, "GEN_MODEL", "gen-m")
    monkeypatch.setattr(config, "JUDGE_MODEL", "judge-m")
    _install(monkeypatch, lambda r: httpx.Response(200, json=_reply("openai_compatible", {"supported": True, "reason": "x"})), calls)
    verdict = llm.judge_claim("c", "s", use_judge=True)
    assert verdict == {"supported": True, "reason": "x"}
    assert json.loads(calls[0].content)["model"] == "judge-m"


# --- Failure modes, per provider -------------------------------------------------

def _raising(exc):
    def handler(request):
        raise exc
    return handler


FAILURES = {
    "timeout": _raising(httpx.ReadTimeout("slow")),
    "connect_error": _raising(httpx.ConnectError("down")),
    "dropped_mid_stream": _raising(httpx.RemoteProtocolError("peer closed connection without response")),
    "http_429": lambda r: httpx.Response(429, json={"error": "slow down"}),
    "http_500": lambda r: httpx.Response(500, text="boom"),
    "http_503": lambda r: httpx.Response(503, text="busy"),
    "http_401": lambda r: httpx.Response(401, json={"error": "bad key"}),
    "not_json_body": lambda r: httpx.Response(200, text="<html>gateway</html>"),
    "json_list_body": lambda r: httpx.Response(200, json=[1, 2]),
}


@pytest.mark.parametrize("name", ALL)
@pytest.mark.parametrize("failure", sorted(FAILURES))
def test_a_failing_provider_raises_provider_error_without_leaking_the_key(monkeypatch, calls, name, failure):
    _select(monkeypatch, name)
    _install(monkeypatch, FAILURES[failure], calls)
    with pytest.raises(providers.ProviderError) as info:
        providers.build(name).draft("s", "u", 0.0, llm.GEN_SCHEMA)
    assert SECRET not in str(info.value) and "Bearer" not in str(info.value)


@pytest.mark.parametrize("name", ALL)
@pytest.mark.parametrize("failure", sorted(FAILURES))
def test_a_failing_provider_falls_back_to_the_extractive_answer(monkeypatch, calls, name, failure):
    _select(monkeypatch, name)
    _install(monkeypatch, FAILURES[failure], calls)
    assert llm.generate_llm(QUESTION, retrieval.topic_records({"caloradine"}) or [], 0.0) is None
    response = pipeline.run(QUESTION)
    assert response.decision == "answer" and not response.llm_used
    assert "15 mg" in response.answer_text
    assert len(calls) >= 4  # two draft attempts here, two more in the pipeline, then the judge
    trace = {step.name: step for step in response.trace}
    assert "model call failed" in trace["generate"].detail


BAD_CONTENT = {
    "malformed_json": "{not json",
    "truncated_json": '{"insufficient_context": false, "claims": [{"text": "The standard',
    "prose_not_json": "Sure! The dose is 15 mg.",
    "empty_string": "",
    "whitespace_only": "   \n",
    "wrong_shape": '{"answer": "15 mg"}',
    "claims_not_a_list": '{"insufficient_context": false, "claims": "15 mg"}',
    "claim_without_sources": '{"insufficient_context": false, "claims": [{"text": "15 mg once daily"}]}',
}


@pytest.mark.parametrize("name", ALL)
@pytest.mark.parametrize("kind", sorted(BAD_CONTENT))
def test_unusable_model_output_falls_back_to_extractive(monkeypatch, calls, name, kind):
    _select(monkeypatch, name)
    _install(monkeypatch, lambda r: httpx.Response(200, json=_reply(name, BAD_CONTENT[kind])), calls)
    response = pipeline.run(QUESTION)
    assert response.decision == "answer" and not response.llm_used and "15 mg" in response.answer_text


@pytest.mark.parametrize("name", ALL)
def test_empty_content_variants_raise(monkeypatch, calls, name):
    _select(monkeypatch, name)
    empties = ([{"content": []}, {"content": None}, {"content": [{"type": "text", "text": ""}]}, {}]
               if name == "anthropic" else
               [{"choices": []}, {"choices": [{"message": {"content": None}}]},
                {"choices": [{"message": {"content": ""}}]}, {}, {"choices": [{}]}])
    for body in empties:
        _install(monkeypatch, lambda r, b=body: httpx.Response(200, json=b), calls)
        with pytest.raises(providers.ProviderError):
            providers.build(name).draft("s", "u", 0.0, llm.GEN_SCHEMA)


@pytest.mark.parametrize("name", ALL)
def test_a_reply_cut_off_by_the_token_limit_is_refused_even_if_it_parses(monkeypatch, calls, name):
    _select(monkeypatch, name)
    stop = "max_tokens" if name == "anthropic" else "length"
    _install(monkeypatch, lambda r: httpx.Response(200, json=_reply(name, GOOD, stop=stop)), calls)
    with pytest.raises(providers.ProviderError, match="cut off"):
        providers.build(name).draft("s", "u", 0.0, llm.GEN_SCHEMA)
    assert pipeline.run(QUESTION).llm_used is False


def test_one_transient_failure_is_retried(monkeypatch, calls):
    _select(monkeypatch, "openai_compatible")
    state = {"n": 0}

    def handler(request):
        state["n"] += 1
        return (httpx.Response(503) if state["n"] == 1
                else httpx.Response(200, json=_reply("openai_compatible", GOOD)))

    _install(monkeypatch, handler, calls)
    response = pipeline.run(QUESTION)
    assert response.llm_used and response.decision == "answer" and len(calls) >= 2


def test_anthropic_text_block_is_accepted_when_no_tool_use(monkeypatch, calls):
    _select(monkeypatch, "anthropic")
    import json

    _install(monkeypatch, lambda r: httpx.Response(200, json=_reply("anthropic", json.dumps(GOOD))), calls)
    assert pipeline.run(QUESTION).llm_used


# --- The guards judge every provider's output -----------------------------------

@pytest.mark.parametrize("name", ALL)
def test_a_good_draft_is_used_and_cited(monkeypatch, calls, name):
    _select(monkeypatch, name)
    _install(monkeypatch, lambda r: httpx.Response(200, json=_reply(name, GOOD)), calls)
    response = pipeline.run(QUESTION)
    assert response.decision == "answer" and response.llm_used
    assert response.claims[0].source_ids == ["CALO-001"]


@pytest.mark.parametrize("name", ALL)
def test_a_provider_returning_a_wrong_dose_is_refused(monkeypatch, calls, name):
    _select(monkeypatch, name)
    _install(monkeypatch, lambda r: httpx.Response(200, json=_reply(name, WRONG_DOSE)), calls)
    response = pipeline.run(QUESTION)
    assert response.decision == "refuse" and "150 mg once daily" not in response.answer_text
    assert response.reason_code in ("dose_unsupported", "claim_ungrounded")


@pytest.mark.parametrize("name", ALL)
def test_a_provider_returning_an_ungrounded_claim_is_refused(monkeypatch, calls, name):
    _select(monkeypatch, name)
    _install(monkeypatch, lambda r: httpx.Response(200, json=_reply(name, UNGROUNDED)), calls)
    response = pipeline.run(QUESTION)
    assert response.decision == "refuse" and "migraine" not in response.answer_text


@pytest.mark.parametrize("name", ALL)
def test_a_provider_citing_a_source_it_was_not_given_is_refused(monkeypatch, calls, name):
    _select(monkeypatch, name)
    forged = {"insufficient_context": False, "claims": [{"text": GOOD["claims"][0]["text"], "source_ids": ["NOPE-999"]}]}
    _install(monkeypatch, lambda r: httpx.Response(200, json=_reply(name, forged)), calls)
    assert pipeline.run(QUESTION).decision == "refuse"


@pytest.mark.parametrize("name", ALL)
def test_a_provider_saying_insufficient_context_is_a_refusal_not_a_fallback(monkeypatch, calls, name):
    _select(monkeypatch, name)
    _install(monkeypatch, lambda r: httpx.Response(200, json=_reply(name, {"insufficient_context": True, "claims": []})), calls)
    response = pipeline.run(QUESTION)
    assert response.decision == "refuse" and response.reason_code == "insufficient_context"


@pytest.mark.parametrize("name", ALL)
def test_a_provider_cannot_answer_a_question_the_sources_do_not_cover(monkeypatch, calls, name):
    _select(monkeypatch, name)
    claim = {"insufficient_context": False, "claims": [{"text": "The standard adult dose is 20 mg daily.", "source_ids": ["CALO-001"]}]}
    _install(monkeypatch, lambda r: httpx.Response(200, json=_reply(name, claim)), calls)
    response = pipeline.run("What is the standard dose of Zyntrafen?")
    assert response.decision == "refuse" and not calls  # stopped before any model call


# --- Selection ------------------------------------------------------------------

def test_none_is_always_extractive(monkeypatch, calls):
    _select(monkeypatch, "none")
    _install(monkeypatch, lambda r: httpx.Response(200, json=_reply("openai_compatible", GOOD)), calls)
    assert llm.active_provider() is None and not llm.llm_available()
    assert pipeline.run(QUESTION).llm_used is False and not calls


def test_ollama_means_local_only(monkeypatch):
    from app import local_ai

    _select(monkeypatch, "ollama")
    monkeypatch.setattr(local_ai, "active_model", lambda: None)
    assert llm.active_provider() is None
    monkeypatch.setattr(local_ai, "active_model", lambda: "llama3.2:3b")
    assert llm.active_provider() == {"kind": "local", "model": "llama3.2:3b"}


@pytest.mark.parametrize("name,model", [("openai_compatible", config.GEN_MODEL), ("azure_openai", "gc-gpt"),
                                        ("anthropic", "claude-haiku-4-5")])
def test_a_named_provider_is_used_alone_and_reported(monkeypatch, name, model):
    from app import local_ai

    _select(monkeypatch, name)
    monkeypatch.setattr(config, "GEN_MODEL_EXPLICIT", "")
    monkeypatch.setattr(local_ai, "active_model", lambda: "llama3.2:3b")  # ignored when a provider is named
    assert llm.active_provider() == {"kind": "cloud", "model": model, "provider": name}


def test_force_extractive_beats_every_provider(monkeypatch, calls):
    for name in ALL:
        _select(monkeypatch, name)
        monkeypatch.setattr(config, "FORCE_EXTRACTIVE", True)
        assert llm.active_provider() is None
    with llm.extractive_only():
        monkeypatch.setattr(config, "FORCE_EXTRACTIVE", False)
        assert llm.active_provider() is None


def test_extractive_only_context_wins_over_a_provider(monkeypatch, calls):
    _select(monkeypatch, "anthropic")
    monkeypatch.setattr(llm, "_thread", __import__("threading").local())
    with llm.extractive_only():
        assert llm.active_provider() is None
    assert llm.active_provider()["provider"] == "anthropic"


@pytest.mark.parametrize("name,keys", [
    ("openai_compatible", {"LLM_API_KEY": ""}),
    ("anthropic", {"ANTHROPIC_API_KEY": ""}),
    ("azure_openai", {"AZURE_OPENAI_API_KEY": ""}),
    ("azure_openai", {"AZURE_OPENAI_ENDPOINT": ""}),
    ("azure_openai", {"AZURE_OPENAI_DEPLOYMENT": ""}),
])
def test_a_provider_without_credentials_is_not_configured(monkeypatch, name, keys):
    _select(monkeypatch, name)
    for key, value in keys.items():
        monkeypatch.setattr(config, key, value)
    assert providers.configured(name) is False
    assert config.llm_configured() is False


def test_configured_when_credentials_present(monkeypatch):
    for name in ALL:
        _select(monkeypatch, name)
        assert providers.configured(name) is True


def test_unset_provider_keeps_the_groq_alias_behaviour(monkeypatch):
    import importlib

    for var in ("LLM_PROVIDER", "LLM_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "gsk_alias")
    monkeypatch.setenv("GROQ_BASE_URL", "https://groq.example/v1")
    try:
        importlib.reload(config)
        assert config.LLM_API_KEY == "gsk_alias" and config.GROQ_API_KEY == "gsk_alias"
        assert config.LLM_BASE_URL == "https://groq.example/v1" == config.GROQ_BASE_URL
        assert config.LLM_PROVIDER_EFFECTIVE == "openai_compatible" and config.llm_configured()
        monkeypatch.setenv("GROQ_API_KEY", "")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "ant")
        importlib.reload(config)
        assert config.LLM_PROVIDER_EFFECTIVE == "" and not config.llm_configured()  # unset never guesses another provider
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        importlib.reload(config)
        assert config.LLM_PROVIDER_EFFECTIVE == "anthropic" and config.llm_configured()
        monkeypatch.setenv("LLM_PROVIDER", "Azure-OpenAI")
        importlib.reload(config)
        assert config.LLM_PROVIDER == "azure_openai"
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_an_unknown_provider_name_is_treated_as_none(monkeypatch):
    import importlib

    monkeypatch.setenv("LLM_PROVIDER", "mystery")
    monkeypatch.setenv("LLM_API_KEY", "k")
    try:
        importlib.reload(config)
        assert config.LLM_PROVIDER_EFFECTIVE == "" and not config.llm_configured()
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_building_a_non_cloud_provider_fails():
    with pytest.raises(providers.ProviderError):
        providers.build("ollama")


def test_the_decision_is_the_same_whichever_provider_drafted_it(monkeypatch, calls):
    results = {}
    for name in ALL:
        _select(monkeypatch, name)
        _install(monkeypatch, lambda r, n=name: httpx.Response(200, json=_reply(n, GOOD)), calls)
        r = pipeline.run(QUESTION)
        results[name] = (r.decision, r.answer_text, tuple(c.text for c in r.claims))
    assert len(set(results.values())) == 1
    _select(monkeypatch, "none")
    r = pipeline.run(QUESTION)
    assert (r.decision, r.answer_text) == next(iter(results.values()))[:2]

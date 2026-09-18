"""Local AI tests. Offline: Ollama is never contacted. Hardware parsing, model
fit and recommendation, provider precedence, fallback, and the API's
permission and validation rules."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

os.environ["GROQ_API_KEY"] = ""
os.environ["FORCE_EXTRACTIVE"] = "true"
os.environ["RATE_LIMIT_PER_MINUTE"] = "1000000"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

from app import config, llm, local_ai, retrieval  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated_state(tmp_path, monkeypatch):
    """Never read or write the real selection file, and never reach Ollama."""
    monkeypatch.setattr(config, "LOCAL_AI_STATE_PATH", tmp_path / "local_ai.json")
    monkeypatch.setattr(config, "LOCAL_MODEL", "")
    monkeypatch.setattr(local_ai, "_selected", None)
    monkeypatch.setattr(local_ai, "_selected_loaded", False)
    local_ai.forget_availability()
    monkeypatch.setattr(local_ai, "ollama_version", lambda: None)
    monkeypatch.setattr(local_ai, "installed_models", lambda: [])
    monkeypatch.setattr(local_ai, "loaded_models", lambda: [])
    yield
    local_ai.forget_availability()


# --- Hardware ----------------------------------------------------------------

def test_parse_nvidia_smi_reads_every_gpu():
    gpus = local_ai.parse_nvidia_smi("NVIDIA GeForce RTX 4090, 24564\nNVIDIA GeForce RTX 3060, 12288\n")
    assert [g.name for g in gpus] == ["NVIDIA GeForce RTX 4090", "NVIDIA GeForce RTX 3060"]
    assert [g.memory_gb for g in gpus] == [24.0, 12.0]


def test_parse_nvidia_smi_skips_malformed_lines():
    assert local_ai.parse_nvidia_smi("garbage\nRTX, not-a-number\n") == ()


def test_memory_budget_by_accelerator():
    two_gpus = (local_ai.Gpu("A", 24.0), local_ai.Gpu("B", 12.0))
    assert local_ai.model_memory_budget(16.0, True, ()) == ("apple-silicon", 10.4)
    assert local_ai.model_memory_budget(64.0, False, two_gpus) == ("nvidia", 24.0)
    assert local_ai.model_memory_budget(32.0, False, ()) == ("cpu", 16.0)


def test_detect_hardware_returns_a_usable_budget():
    hw = local_ai.detect_hardware()
    assert hw.memory_gb > 0
    assert hw.accelerator in {"apple-silicon", "nvidia", "cpu"}
    assert 0 < hw.model_memory_gb <= max([hw.memory_gb, *(g.memory_gb for g in hw.gpus)])


# --- Fit and recommendation --------------------------------------------------

def test_fit_levels():
    model = local_ai._BY_NAME["llama3.2:3b"]  # needs about 3.4 GB
    assert local_ai.fit(model, 10.0) == "good"
    assert local_ai.fit(model, 3.5) == "tight"
    assert local_ai.fit(model, 2.0) == "too-large"


def test_recommendation_is_the_best_model_that_fits_with_headroom():
    assert local_ai.recommend(10.4).name == "qwen2.5:7b"
    assert local_ai.recommend(4.0).name == "llama3.2:1b"
    assert local_ai.recommend(1.0) is None


def test_catalogue_excludes_non_commercial_licences():
    assert all("research" not in m.licence.lower() for m in local_ai.CATALOGUE)


def test_model_names_match_with_or_without_latest_tag():
    installed = [{"name": "phi4-mini:latest", "size_gb": 2.5}]
    assert local_ai.is_installed("phi4-mini", installed)
    assert not local_ai.is_installed("phi4-mini:14b", installed)


# --- Selection and provider precedence ----------------------------------------

def test_selection_persists(tmp_path):
    local_ai.select_model("llama3.2:3b")
    assert (tmp_path / "local_ai.json").read_text().count("llama3.2:3b") == 1
    local_ai._selected_loaded = False  # simulate a restart
    assert local_ai.selected_model() == "llama3.2:3b"


def test_local_model_is_used_only_when_ollama_has_it(monkeypatch):
    monkeypatch.setattr(config, "FORCE_EXTRACTIVE", False)
    monkeypatch.setattr(llm, "LLM_AVAILABLE", False)
    local_ai.select_model("llama3.2:3b")
    assert llm.active_provider() is None  # Ollama not running

    monkeypatch.setattr(local_ai, "ollama_version", lambda: "0.32.15")
    monkeypatch.setattr(local_ai, "installed_models", lambda: [{"name": "llama3.2:3b", "size_gb": 2.0}])
    local_ai.forget_availability()
    assert llm.active_provider() == {"kind": "local", "model": "llama3.2:3b"}


def test_local_model_takes_precedence_over_cloud(monkeypatch):
    monkeypatch.setattr(config, "FORCE_EXTRACTIVE", False)
    monkeypatch.setattr(llm, "LLM_AVAILABLE", True)
    assert llm.active_provider() == {"kind": "cloud", "model": config.GEN_MODEL}
    monkeypatch.setattr(local_ai, "ollama_version", lambda: "0.32.15")
    monkeypatch.setattr(local_ai, "installed_models", lambda: [{"name": "gemma3:4b", "size_gb": 3.3}])
    local_ai.select_model("gemma3:4b")
    assert llm.active_provider()["kind"] == "local"


def test_force_extractive_ignores_local_model(monkeypatch):
    monkeypatch.setattr(config, "FORCE_EXTRACTIVE", True)
    monkeypatch.setattr(local_ai, "ollama_version", lambda: "0.32.15")
    monkeypatch.setattr(local_ai, "installed_models", lambda: [{"name": "gemma3:4b", "size_gb": 3.3}])
    local_ai.select_model("gemma3:4b")
    assert llm.active_provider() is None


def test_failed_local_call_falls_back_to_extractive(monkeypatch):
    monkeypatch.setattr(llm, "active_provider", lambda: {"kind": "local", "model": "llama3.2:3b"})
    monkeypatch.setattr(local_ai, "chat_json", lambda *a, **k: None)
    assert llm.generate_llm("What is the standard dose of Caloradine?", []) is None


def test_local_generation_uses_schema_and_local_prompt(monkeypatch):
    seen = {}

    def fake_chat(system, user, temperature=0.0, model=None, schema=None):
        seen.update(system=system, schema=schema, model=model)
        return '{"insufficient_context": false, "claims": [{"text": "x", "source_ids": ["CALO-001"]}]}'

    monkeypatch.setattr(llm, "active_provider", lambda: {"kind": "local", "model": "llama3.2:3b"})
    monkeypatch.setattr(local_ai, "chat_json", fake_chat)
    answer = llm.generate_llm("What is the standard dose of Caloradine?", [])
    assert answer is not None and answer.claims[0].source_ids == ["CALO-001"]
    assert seen["system"] == llm.LOCAL_GEN_SYSTEM_PROMPT
    assert seen["schema"] == llm.GEN_SCHEMA
    assert seen["model"] == "llama3.2:3b"


# --- API ---------------------------------------------------------------------

@pytest.fixture()
def client():
    try:
        retrieval.load_index()
    except FileNotFoundError:
        retrieval.build_index()
    from app.main import app
    with TestClient(app) as c:
        yield c


def test_status_endpoint_reports_hardware_and_catalogue(client):
    body = client.get("/api/local-ai").json()
    assert body["ollama"]["running"] is False
    assert body["can_manage"] is True
    assert len(body["models"]) == len(local_ai.CATALOGUE)
    assert {"fit", "recommended", "installed", "memory_needed_gb"} <= set(body["models"][0])


def test_managing_models_is_refused_from_another_machine(client):
    headers = {"X-Forwarded-For": "203.0.113.9"}
    assert client.post("/api/local-ai/select", json={"model": None}, headers=headers).status_code == 403
    assert client.post("/api/local-ai/pull", json={"model": "gemma3:1b"}, headers=headers).status_code == 403
    assert client.get("/api/local-ai", headers=headers).json()["can_manage"] is False


def test_management_can_be_disabled(client, monkeypatch):
    monkeypatch.setattr(config, "ADMIN_ACCESS", "none")
    assert client.post("/api/local-ai/select", json={"model": None}).status_code == 403


def test_only_catalogue_models_can_be_pulled_or_selected(client):
    assert client.post("/api/local-ai/pull", json={"model": "evil/model"}).status_code == 400
    assert client.post("/api/local-ai/select", json={"model": "evil/model"}).status_code == 400


def test_selecting_a_model_that_is_not_downloaded_is_a_conflict(client):
    r = client.post("/api/local-ai/select", json={"model": "gemma3:1b"})
    assert r.status_code == 409
    assert "Download" in r.json()["detail"]


def test_switching_back_to_no_local_model(client):
    local_ai.select_model("gemma3:1b")
    r = client.post("/api/local-ai/select", json={"model": None})
    assert r.status_code == 200
    assert r.json()["selected"] is None
    assert local_ai.selected_model() is None


def test_a_local_model_that_returns_nonsense_is_treated_as_no_answer(monkeypatch):
    """Small local models are the ones most likely to return something that
    isn't the schema. Every shape of that has to end as no answer, not as a
    half-parsed one, and never as an exception reaching the person asking."""
    monkeypatch.setattr(llm, "active_provider", lambda: {"kind": "local", "model": "llama3.2:3b"})
    for reply in [None,                                    # the model gave nothing
                  "I'm afraid I can't help with that.",    # prose instead of JSON
                  "{",                                     # truncated
                  '{"claims": "not a list"}',              # right key, wrong type
                  '{"insufficient_context": false}',       # no claims at all
                  '{"claims": [{"text": "x"}]}']:          # a claim citing nothing
        monkeypatch.setattr(local_ai, "chat_json",
                            lambda *a, reply=reply, **k: reply)
        answer = llm.generate_llm("What is the standard dose of Caloradine?", [])
        assert answer is None or not answer.claims or all(not c.source_ids for c in answer.claims), reply

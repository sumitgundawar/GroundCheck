"""Model providers that can draft an answer.

A provider turns a system prompt, a user message and a JSON schema into the
text of one JSON object. It never decides anything: whatever it returns goes
through the same parsing, grounding and dosage checks as any other draft, and
any failure here makes the pipeline fall back to extractive generation.

Providers speak plain HTTP with httpx, so a new one needs no new dependency and
every one can be tested against a mock transport. Secrets are sent only in
request headers and never appear in an error message."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass

import httpx

from . import config

NAMES = ("openai_compatible", "anthropic", "azure_openai", "ollama", "none")
CLOUD = ("openai_compatible", "anthropic", "azure_openai")
MAX_OUTPUT_TOKENS = 1024


_usage = threading.local()


def reset_usage() -> None:
    _usage.tokens = [0, 0, 0]


def usage_total() -> dict:
    """Tokens used by provider calls in this thread since reset_usage():
    prompt, completion, and how many calls reported nothing."""
    tokens = getattr(_usage, "tokens", [0, 0, 0])
    return {"prompt_tokens": tokens[0], "completion_tokens": tokens[1], "calls_without_usage": tokens[2]}


def _note_usage(prompt, completion) -> None:
    tokens = getattr(_usage, "tokens", None)
    if tokens is None:
        return
    if isinstance(prompt, int) and isinstance(completion, int) and prompt >= 0 and completion >= 0:
        tokens[0] += prompt
        tokens[1] += completion
    else:
        tokens[2] += 1


class ProviderError(RuntimeError):
    """A provider call failed. The message is safe to log: it holds no secret
    and nothing the model or the caller wrote."""


def _client() -> httpx.Client:
    return httpx.Client(timeout=config.LLM_TIMEOUT_SECONDS)


def _post(url: str, headers: dict, payload: dict) -> dict:
    try:
        with _client() as client:
            response = client.post(url, headers=headers, json=payload)
    except httpx.TimeoutException as exc:
        raise ProviderError("the model call timed out") from exc
    except httpx.HTTPError as exc:
        raise ProviderError(f"the model call failed ({type(exc).__name__})") from exc
    if response.status_code == 429:
        raise ProviderError("the model provider is rate limiting (429)")
    if response.status_code >= 500:
        raise ProviderError(f"the model provider failed ({response.status_code})")
    if response.status_code >= 400:
        raise ProviderError(f"the model provider refused the request ({response.status_code})")
    try:
        body = response.json()
    except ValueError as exc:
        raise ProviderError("the model provider returned something that is not JSON") from exc
    if not isinstance(body, dict):
        raise ProviderError("the model provider returned an unexpected body")
    return body


def _need(text) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ProviderError("the model returned no content")
    return text


@dataclass
class OpenAICompatible:
    """Any server that speaks the OpenAI chat completions protocol: OpenAI,
    Groq, vLLM, LM Studio, Together and so on."""

    name = "openai_compatible"
    base_url: str
    api_key: str
    model: str
    judge_model: str

    def url(self) -> str:
        return f"{self.base_url.rstrip('/')}/chat/completions"

    def headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}"}

    def payload(self, system: str, user: str, temperature: float, judge: bool) -> dict:
        return {"model": self.judge_model if judge else self.model, "temperature": temperature,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "response_format": {"type": "json_object"}, "max_tokens": MAX_OUTPUT_TOKENS}

    def draft(self, system: str, user: str, temperature: float, schema: dict, judge: bool = False) -> str:
        body = _post(self.url(), self.headers(), self.payload(system, user, temperature, judge))
        try:
            choice = body["choices"][0]
            text, finish = choice["message"]["content"], choice.get("finish_reason")
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError("the model reply had an unexpected shape") from exc
        usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
        _note_usage(usage.get("prompt_tokens"), usage.get("completion_tokens"))
        if finish == "length":
            raise ProviderError("the model reply was cut off")
        return _need(text)


@dataclass
class AzureOpenAI(OpenAICompatible):
    """An Azure OpenAI deployment: the deployment name picks the model."""

    name = "azure_openai"
    api_version: str = "2024-10-21"

    def url(self) -> str:
        return (f"{self.base_url.rstrip('/')}/openai/deployments/{self.model}/chat/completions"
                f"?api-version={self.api_version}")

    def headers(self) -> dict:
        return {"api-key": self.api_key}


@dataclass
class Anthropic:
    """The Anthropic Messages API. The schema is passed as a forced tool, so
    the reply is a structured tool input rather than text to be parsed."""

    name = "anthropic"
    base_url: str
    api_key: str
    model: str
    judge_model: str
    version: str = "2023-06-01"

    def draft(self, system: str, user: str, temperature: float, schema: dict, judge: bool = False) -> str:
        payload = {"model": self.judge_model if judge else self.model, "max_tokens": MAX_OUTPUT_TOKENS,
                   "temperature": temperature, "system": system,
                   "messages": [{"role": "user", "content": user}],
                   "tools": [{"name": "reply", "description": "Return the reply as JSON.", "input_schema": schema}],
                   "tool_choice": {"type": "tool", "name": "reply"}}
        headers = {"x-api-key": self.api_key, "anthropic-version": self.version}
        body = _post(f"{self.base_url.rstrip('/')}/v1/messages", headers, payload)
        usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
        _note_usage(usage.get("input_tokens"), usage.get("output_tokens"))
        if body.get("stop_reason") == "max_tokens":
            raise ProviderError("the model reply was cut off")
        blocks = body.get("content")
        if not isinstance(blocks, list):
            raise ProviderError("the model reply had an unexpected shape")
        for block in blocks:
            if isinstance(block, dict) and block.get("type") == "tool_use" and isinstance(block.get("input"), dict):
                return json.dumps(block["input"])
        for block in blocks:
            if isinstance(block, dict) and block.get("type") == "text":
                return _need(block.get("text"))
        raise ProviderError("the model returned no content")


def configured(name: str | None = None) -> bool:
    """Whether the named cloud provider (default: the selected one) has what it needs."""
    name = name or config.LLM_PROVIDER_EFFECTIVE
    if name == "openai_compatible":
        return bool(config.LLM_API_KEY)
    if name == "anthropic":
        return bool(config.ANTHROPIC_API_KEY)
    if name == "azure_openai":
        return bool(config.AZURE_OPENAI_API_KEY and config.AZURE_OPENAI_ENDPOINT and config.AZURE_OPENAI_DEPLOYMENT)
    return False


def build(name: str):
    """The provider object for a cloud provider name."""
    if name == "openai_compatible":
        return OpenAICompatible(config.LLM_BASE_URL, config.LLM_API_KEY, config.GEN_MODEL, config.JUDGE_MODEL)
    if name == "anthropic":
        model = config.GEN_MODEL_EXPLICIT or "claude-haiku-4-5"
        return Anthropic(config.ANTHROPIC_BASE_URL, config.ANTHROPIC_API_KEY, model,
                         config.JUDGE_MODEL_EXPLICIT or model)
    if name == "azure_openai":
        return AzureOpenAI(config.AZURE_OPENAI_ENDPOINT, config.AZURE_OPENAI_API_KEY,
                           config.AZURE_OPENAI_DEPLOYMENT, config.AZURE_OPENAI_DEPLOYMENT,
                           api_version=config.AZURE_OPENAI_API_VERSION)
    raise ProviderError(f"{name} is not a cloud provider")


def model_name(name: str) -> str:
    return build(name).model

"""Provider-neutral LLM configuration."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Optional
from urllib.parse import urlparse


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
UPSTAGE_BASE_URL = "https://api.upstage.ai/v1/solar"


def _clean_env_list(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(part.strip() for part in value.split(",") if part.strip())


class _RoundRobin:
    """Rotation counter shared by every copy of a config.

    ``copy_with`` goes through ``dataclasses.replace``, which re-initialises
    ``init=False`` fields from their defaults. If the counter lived directly on
    the dataclass, every copy would restart at the first entry -- and the
    benchmark copies the config once per agent run (see
    ``_normalize_agent_max_tokens`` in run_benchmark.py), so nothing would ever
    rotate. Holding it in a shared object that ``copy_with`` passes by reference
    keeps one rotation across all copies.
    """

    __slots__ = ("_index", "_lock")

    def __init__(self) -> None:
        self._index = 0
        self._lock = threading.Lock()

    def pick(self, items: list):
        with self._lock:
            item = items[self._index % len(items)]
            self._index += 1
            return item


def _first_non_empty(values: Iterable[Optional[str]]) -> Optional[str]:
    for value in values:
        if value:
            return value
    return None


@dataclass
class LLMConfig:
    """Configuration for any chat model backend.

    `api_spec` is the API contract, not a vendor name. Most hosted and
    local backends fit `openai_compatible` by changing `base_url`.
    """

    model: str
    api_spec: str = "openai_compatible"
    provider: str = "custom"
    base_url: Optional[str] = None
    base_urls: tuple[str, ...] = ()
    endpoint_strategy: str = "first"
    api_key: Optional[str] = None
    api_key_env: Optional[str] = None
    api_key_envs: tuple[str, ...] = ()
    credential_strategy: str = "first"
    temperature: float = 0.7
    max_tokens: Optional[int] = 4096
    reasoning_budget: Optional[int] = None
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    allow_dummy_api_key: bool = False
    _credential_rr: _RoundRobin = field(default_factory=_RoundRobin, init=False, repr=False)
    _endpoint_rr: _RoundRobin = field(default_factory=_RoundRobin, init=False, repr=False)

    def copy_with(self, **updates: Any) -> "LLMConfig":
        clone = replace(self, **{k: v for k, v in updates.items() if v is not None})
        # Share the rotation state: replace() would have handed the clone fresh
        # counters, so every copy would start over at the first entry.
        clone._credential_rr = self._credential_rr
        clone._endpoint_rr = self._endpoint_rr
        return clone

    @property
    def all_api_key_envs(self) -> tuple[str, ...]:
        envs: list[str] = []
        if self.api_key_env:
            envs.append(self.api_key_env)
        envs.extend(self.api_key_envs)
        return tuple(dict.fromkeys(envs))

    @property
    def all_base_urls(self) -> tuple[str, ...]:
        """Every endpoint this config may talk to, in order, deduplicated.

        Both fields accept a comma-separated string so a single
        ``<SLOT>_AGENT_MODEL_BASE_URL`` listing several pods keeps working
        without the callers having to split it.
        """
        urls: list[str] = []
        for raw in (self.base_url, *self.base_urls):
            urls.extend(_clean_env_list(raw) if raw else ())
        return tuple(dict.fromkeys(urls))

    def resolve_base_url(self) -> Optional[str]:
        """Pick the endpoint for the next model instance.

        With several pods serving the same model, round-robin spreads the load
        the way ``resolve_api_key`` spreads credentials. Endpoints must serve
        the SAME model — the caller is responsible for that, exactly as with
        the embedding encoder.
        """
        urls = self.all_base_urls
        if not urls:
            return None
        if len(urls) == 1 or self.endpoint_strategy != "round_robin":
            return urls[0]
        return self._endpoint_rr.pick(list(urls))

    def is_local_endpoint(self) -> bool:
        urls = self.all_base_urls
        if not urls:
            return False
        # Homogeneous pod sets in practice; judge by the first.
        host = urlparse(urls[0]).hostname or ""
        return host in {"localhost", "127.0.0.1", "::1", "0.0.0.0"}

    def resolve_api_key(self) -> Optional[str]:
        if self.api_key:
            return self.api_key

        keys = [os.getenv(env) for env in self.all_api_key_envs]
        keys = [key for key in keys if key]

        if keys:
            if self.credential_strategy == "round_robin":
                return self._credential_rr.pick(keys)
            return keys[0]

        if self.allow_dummy_api_key or self.is_local_endpoint():
            return "not-needed"

        return None

    def metadata(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "api_spec": self.api_spec,
            "model": self.model,
            "base_url": self.base_url,
            "base_urls": self.all_base_urls,
            "endpoint_strategy": self.endpoint_strategy,
            "api_key_envs": self.all_api_key_envs,
            "credential_strategy": self.credential_strategy,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "reasoning_budget": self.reasoning_budget,
        }


def llm_config_from_legacy(
    *,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = 4096,
    reasoning_budget: Optional[int] = None,
) -> LLMConfig:
    """Build an LLMConfig from the old provider/model fields."""

    provider = (provider or "openrouter").lower()

    local_profiles = {
        "local-ollama": "http://localhost:11434/v1",
        "ollama": "http://localhost:11434/v1",
        "local-lmstudio": "http://localhost:1234/v1",
        "lmstudio": "http://localhost:1234/v1",
        "local-vllm": "http://localhost:8000/v1",
        "vllm": "http://localhost:8000/v1",
    }
    if provider in local_profiles:
        return LLMConfig(
            provider=provider,
            api_spec="openai_compatible",
            model=model or "local-model",
            base_url=base_url or local_profiles[provider],
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_budget=reasoning_budget,
            allow_dummy_api_key=True,
        )

    if provider == "anthropic":
        return LLMConfig(
            provider="anthropic",
            api_spec="anthropic",
            model=model or "claude-3-5-sonnet-latest",
            api_key=api_key,
            api_key_env="ANTHROPIC_API_KEY",
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_budget=reasoning_budget,
        )

    if provider == "upstage":
        return LLMConfig(
            provider="upstage",
            api_spec="openai_compatible",
            model=model or "solar-pro3",
            base_url=base_url or UPSTAGE_BASE_URL,
            api_key=api_key,
            api_key_envs=("UPSTAGE_API_KEY", "UPSTAGE_API_KEY2", "UPSTAGE_API_KEY3"),
            credential_strategy="round_robin",
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_budget=reasoning_budget,
        )

    if provider == "openai":
        return LLMConfig(
            provider="openai",
            api_spec="openai_compatible",
            model=model or "gpt-4o-mini",
            base_url=base_url,
            api_key=api_key,
            api_key_env="OPENAI_API_KEY",
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_budget=reasoning_budget,
        )

    return LLMConfig(
        provider=provider,
        api_spec="openai_compatible",
        model=model or "anthropic/claude-opus-4.5",
        base_url=base_url or OPENROUTER_BASE_URL,
        api_key=api_key,
        api_key_env="OPENROUTER_API_KEY",
        temperature=temperature,
        max_tokens=max_tokens,
        reasoning_budget=reasoning_budget,
    )


def llm_config_from_env(
    *,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    api_spec: Optional[str] = None,
    base_url: Optional[str] = None,
    base_urls: Optional[tuple[str, ...]] = None,
    api_key: Optional[str] = None,
    api_key_env: Optional[str] = None,
    api_key_envs: Optional[tuple[str, ...]] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    reasoning_budget: Optional[int] = None,
) -> LLMConfig:
    """Resolve LLM config from CLI overrides and environment variables."""

    provider = provider or os.getenv("AGENT_MODEL_PROVIDER") or "openrouter"
    model = model or os.getenv("AGENT_MODEL_NAME")
    api_spec = api_spec or os.getenv("AGENT_MODEL_API_SPEC")
    base_url = base_url or os.getenv("AGENT_MODEL_BASE_URL")
    # Plural wins when both are set; the singular still accepts a comma list.
    base_urls = base_urls or _clean_env_list(os.getenv("AGENT_MODEL_BASE_URLS"))
    api_key = api_key or os.getenv("AGENT_MODEL_API_KEY")
    api_key_env = api_key_env or os.getenv("AGENT_MODEL_API_KEY_ENV")
    api_key_envs = api_key_envs or _clean_env_list(os.getenv("AGENT_MODEL_API_KEY_ENVS"))
    credential_strategy = os.getenv("AGENT_MODEL_CREDENTIAL_STRATEGY")
    reasoning_budget = reasoning_budget or int(os.getenv("REASONING_BUDGET", "0") or "0") or None

    env_temperature = os.getenv("AGENT_MODEL_TEMPERATURE")
    env_max_tokens = os.getenv("AGENT_MODEL_MAX_TOKENS")
    if temperature is None and env_temperature:
        temperature = float(env_temperature)
    if max_tokens is None and env_max_tokens:
        max_tokens = int(env_max_tokens)

    if api_spec or base_url or base_urls or api_key or api_key_env or api_key_envs:
        cfg = LLMConfig(
            provider=provider,
            api_spec=api_spec or "openai_compatible",
            model=model or "local-model",
            base_url=base_url,
            base_urls=base_urls,
            endpoint_strategy=os.getenv("AGENT_MODEL_ENDPOINT_STRATEGY")
            or ("round_robin"
                if len(_clean_env_list(base_url)) + len(base_urls) > 1 else "first"),
            api_key=api_key,
            api_key_env=api_key_env,
            api_key_envs=api_key_envs,
            credential_strategy=credential_strategy or ("round_robin" if len(api_key_envs) > 1 else "first"),
            temperature=temperature if temperature is not None else 0.7,
            max_tokens=max_tokens if max_tokens is not None else 4096,
            reasoning_budget=reasoning_budget,
        )
    else:
        cfg = llm_config_from_legacy(
            provider=provider,
            model=model,
            temperature=temperature if temperature is not None else 0.7,
            max_tokens=max_tokens if max_tokens is not None else 4096,
            reasoning_budget=reasoning_budget,
        )

    return cfg

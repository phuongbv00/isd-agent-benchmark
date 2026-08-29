"""Factories for chat-model clients."""

from __future__ import annotations

import os
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from openai import OpenAI

from shared.llm.config import LLMConfig


def _chat_kwargs(config: LLMConfig) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": config.model,
        "temperature": config.temperature,
        # Construction-time callback -> counts tokens for EVERY agent uniformly
        # (persists across .bind(), so it survives the JSON-mode binding agents
        # apply). config._token_counter, NOT a module-level singleton — see
        # shared/llm/token_accounting.py for why.
        "callbacks": [config._token_counter],
    }
    if config.max_tokens is not None:
        kwargs["max_tokens"] = config.max_tokens
    # For Small-context backends
    cap_env = os.getenv("AGENT_MODEL_MAX_TOKENS_CAP")
    if cap_env:
        cap = int(cap_env)
        current = kwargs.get("max_tokens")
        kwargs["max_tokens"] = cap if current is None else min(current, cap)
    if config.model_kwargs:
        kwargs["model_kwargs"] = config.model_kwargs
    return kwargs


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def create_chat_model(config: LLMConfig):
    """Create a LangChain chat model from provider-neutral config."""

    api_spec = config.api_spec.lower()
    api_key = config.resolve_api_key()

    if api_spec == "anthropic":
        kwargs = _chat_kwargs(config)
        if api_key:
            kwargs["api_key"] = api_key
        return ChatAnthropic(**kwargs)

    if api_spec in {"openai", "openai_compatible"}:
        kwargs = _chat_kwargs(config)
        if api_key:
            kwargs["api_key"] = api_key
        # One endpoint per model instance; with several pods behind the same
        # model this round-robins them (see LLMConfig.resolve_base_url).
        endpoint = config.resolve_base_url()
        if endpoint:
            kwargs["base_url"] = endpoint
        if _env_flag("AGENT_MODEL_STREAMING"):
            kwargs["streaming"] = True
            kwargs["stream_usage"] = True
        if _env_flag("AGENT_MODEL_DISABLE_THINKING"):
            kwargs["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": False}
            }
        return ChatOpenAI(**kwargs)

    raise ValueError(f"Unsupported LLM API spec: {config.api_spec}")


def create_openai_client(config: LLMConfig) -> OpenAI:
    """Create an OpenAI SDK client for OpenAI-compatible backends."""

    if config.api_spec.lower() not in {"openai", "openai_compatible"}:
        raise ValueError(
            "OpenAI SDK client requires api_spec='openai_compatible' or 'openai'"
        )

    kwargs: dict[str, Any] = {"api_key": config.resolve_api_key()}
    endpoint = config.resolve_base_url()
    if endpoint:
        kwargs["base_url"] = endpoint
    return OpenAI(**kwargs)

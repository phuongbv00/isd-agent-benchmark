"""Factories for chat-model clients."""

from __future__ import annotations

from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from openai import OpenAI

from shared.llm.config import LLMConfig


def _chat_kwargs(config: LLMConfig) -> dict[str, Any]:
    from shared.llm.token_accounting import HANDLER

    kwargs: dict[str, Any] = {
        "model": config.model,
        "temperature": config.temperature,
        # Construction-time callback -> counts tokens for EVERY agent uniformly
        # (persists across .bind(), so it survives the JSON-mode binding agents
        # apply). See shared/llm/token_accounting.py.
        "callbacks": [HANDLER],
    }
    if config.max_tokens is not None:
        kwargs["max_tokens"] = config.max_tokens
    if config.model_kwargs:
        kwargs["model_kwargs"] = config.model_kwargs
    return kwargs


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
        if config.base_url:
            kwargs["base_url"] = config.base_url
        return ChatOpenAI(**kwargs)

    raise ValueError(f"Unsupported LLM API spec: {config.api_spec}")


def create_openai_client(config: LLMConfig) -> OpenAI:
    """Create an OpenAI SDK client for OpenAI-compatible backends."""

    if config.api_spec.lower() not in {"openai", "openai_compatible"}:
        raise ValueError(
            "OpenAI SDK client requires api_spec='openai_compatible' or 'openai'"
        )

    kwargs: dict[str, Any] = {"api_key": config.resolve_api_key()}
    if config.base_url:
        kwargs["base_url"] = config.base_url
    return OpenAI(**kwargs)

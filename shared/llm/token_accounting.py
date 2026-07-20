"""Uniform, agent-agnostic token accounting for the benchmark.

Every agent gets its LLM from ``shared.llm.create_chat_model`` (the single
source of LLM configuration), so attaching one LangChain callback handler
there counts tokens for ALL agents without touching any agent code. Counts
accumulate into a **thread-local** so the benchmark's parallel per-agent
workers never mix each other's usage.

The benchmark runner brackets each agent run with ``reset()`` / ``snapshot()``
and writes the snapshot into that run's metadata, giving one comparable
``token_usage`` field across every agent (baselines included).
"""
from __future__ import annotations

import threading
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

_local = threading.local()


def reset() -> None:
    _local.prompt = 0
    _local.completion = 0
    _local.calls = 0


def _add(prompt: int, completion: int) -> None:
    _local.prompt = getattr(_local, "prompt", 0) + int(prompt or 0)
    _local.completion = getattr(_local, "completion", 0) + int(completion or 0)
    _local.calls = getattr(_local, "calls", 0) + 1


def snapshot() -> dict[str, int]:
    prompt = getattr(_local, "prompt", 0)
    completion = getattr(_local, "completion", 0)
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "llm_calls": getattr(_local, "calls", 0),
    }


def _usage_from_llm_result(response: Any) -> tuple[int, int]:
    """Extract (prompt, completion) from a LangChain ``LLMResult``, robust to
    the two shapes providers use: ``llm_output['token_usage']`` (OpenAI-style)
    and per-generation ``message.usage_metadata`` (LangChain-normalized)."""
    llm_output = getattr(response, "llm_output", None) or {}
    usage = (llm_output.get("token_usage") or llm_output.get("usage") or {}) if isinstance(llm_output, dict) else {}
    if usage:
        prompt = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
        completion = usage.get("completion_tokens") or usage.get("output_tokens") or 0
        if prompt or completion:
            return int(prompt), int(completion)

    prompt = completion = 0
    for generation_list in getattr(response, "generations", None) or []:
        for generation in generation_list:
            message = getattr(generation, "message", None)
            meta = getattr(message, "usage_metadata", None) if message is not None else None
            if isinstance(meta, dict):
                prompt += int(meta.get("input_tokens", 0) or 0)
                completion += int(meta.get("output_tokens", 0) or 0)
    return prompt, completion


class TokenCountingHandler(BaseCallbackHandler):
    """Stateless (module-level counters) — one shared instance is fine."""

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:  # noqa: D401
        prompt, completion = _usage_from_llm_result(response)
        _add(prompt, completion)


HANDLER = TokenCountingHandler()

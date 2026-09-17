"""Uniform, agent-agnostic token accounting for the benchmark.

Every agent gets its LLM from ``shared.llm.create_chat_model`` (the single
source of LLM configuration), which attaches a ``TokenCountingHandler`` as a
construction-time callback -- so counting tokens for ALL agents needs no
change to any agent code. The handler comes from the ``LLMConfig`` object
itself (``config._token_counter``, see ``shared/llm/config.py``), NOT a
module-level singleton: ``run_benchmark.py``'s ``_get_agent_runner`` opens one
scope per agent run with ``new_token_scope()``, and from there the counter
FOLLOWS THE CONFIG LINEAGE -- ``copy_with()`` re-shares it, exactly as it
re-shares the credential/endpoint round-robin state -- so every
``create_chat_model()`` call made during that run reports into the same
counter, whether it is made directly, from a thread the agent spawned, or
off a copy the agent made to pin its own model or temperature.

That last clause is load-bearing. When ``copy_with()`` handed the clone a
fresh counter and ``_get_agent_runner`` leaned on that side effect, the
contract silently required agents never to copy the config they were given
-- and all six baselines copy it in their constructor, so their
``token_usage`` was 0 on every run from 2026-08-29 until the lineage rule
replaced it. Only ``alignmentgraph-isd`` kept reporting real numbers, and
only because its own copy sits behind a guard the benchmark never trips.

That object-reference approach, not a thread-local or a contextvar, is
deliberate: an agent whose own execution model fans work out across a
thread pool (e.g. a LangGraph ``Send()``-based parallel step) makes some of
its LLM calls from worker threads the benchmark's own outer per-agent
thread never touches. ``threading.local()`` (the previous design here) and
``contextvars.ContextVar`` BOTH fail for that case: neither
``concurrent.futures.ThreadPoolExecutor`` nor LangGraph's own executor
built on it propagates the submitting thread's state into the worker
thread -- confirmed against CPython's ``concurrent.futures.thread._WorkItem
.run()``, which calls ``self.fn(*args, **kwargs)`` directly, no
``contextvars.copy_context()`` involved. A plain object reference captured
by every call site (here: the shared ``LLMConfig``) survives that just
fine, because closures and dataclass fields don't care which thread reads
them -- only ambient, thread-affine state does.
"""
from __future__ import annotations

import threading
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler


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
    """One run's token/call tally, attached as a LangChain callback on every
    chat-model instance ``create_chat_model`` builds for that run.

    Lock-guarded, not thread-local: see the module docstring for why a fan-out
    agent needs every thread's report to land in the SAME counter, and why a
    plain object reference (this instance, held by the run's ``LLMConfig``)
    is what makes that work regardless of which thread calls ``on_llm_end``.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._prompt = 0
        self._completion = 0
        self._calls = 0

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:  # noqa: D401
        prompt, completion = _usage_from_llm_result(response)
        with self._lock:
            self._prompt += prompt
            self._completion += completion
            self._calls += 1

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "prompt_tokens": self._prompt,
                "completion_tokens": self._completion,
                "total_tokens": self._prompt + self._completion,
                "llm_calls": self._calls,
            }

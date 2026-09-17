"""Runtime LLM configuration used by tool-style agents."""

from __future__ import annotations

import threading
from typing import Optional

from shared.llm.config import LLMConfig, llm_config_from_env


_default_config: Optional[LLMConfig] = None
_config_lock = threading.Lock()


def configure_default_llm(config: LLMConfig) -> None:
    global _default_config
    with _config_lock:
        _default_config = config


def get_default_llm_config() -> LLMConfig:
    with _config_lock:
        if _default_config is not None:
            return _default_config
    return llm_config_from_env()


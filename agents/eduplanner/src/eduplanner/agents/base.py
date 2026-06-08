"""에이전트 기본 클래스"""

from abc import ABC, abstractmethod
from typing import Any, Optional

from pydantic import BaseModel

from shared.llm import LLMConfig, create_chat_model, llm_config_from_legacy


class AgentConfig(BaseModel):
    """에이전트 설정"""
    model: str = "solar-mini"
    temperature: float = 0.7
    max_tokens: int = 4096
    provider: str = "upstage"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    llm_config: Any = None

    def to_llm_config(self) -> LLMConfig:
        if self.llm_config is not None:
            return self.llm_config.copy_with(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        return llm_config_from_legacy(
            provider=self.provider,
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )


class BaseAgent(ABC):
    """에이전트 기본 클래스"""

    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        self._llm = None

    @property
    def llm(self):
        """LLM 인스턴스 반환 (지연 초기화)"""
        if self._llm is None:
            self._llm = self._create_llm()
        return self._llm

    def _create_llm(self):
        """LLM 인스턴스 생성"""
        return create_chat_model(self.config.to_llm_config())

    @abstractmethod
    def run(self, *args, **kwargs) -> Any:
        """에이전트 실행"""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """에이전트 이름"""
        pass

    @property
    @abstractmethod
    def role(self) -> str:
        """에이전트 역할 설명"""
        pass

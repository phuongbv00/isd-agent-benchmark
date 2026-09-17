from eduplanner.agents.base import AgentConfig
from eduplanner.agents.main import EduPlannerAgent
from shared.llm import LLMConfig


def test_subagents_inherit_parent_llm_config():
    llm_config = LLMConfig(
        model="openai/gpt-4o-mini",
        api_spec="openai_compatible",
        provider="openrouter",
        api_key="test-key",
        base_url="https://example.test/v1",
        temperature=0.2,
        max_tokens=1234,
    )
    config = AgentConfig(
        model=llm_config.model,
        provider=llm_config.provider,
        llm_config=llm_config,
        temperature=0.2,
        max_tokens=1234,
    )

    agent = EduPlannerAgent(config=config)

    assert agent.evaluator.config.model == config.model
    assert agent.evaluator.config.provider == config.provider
    assert agent.evaluator.config.llm_config is llm_config
    assert agent.evaluator.config.temperature == 0.7
    assert agent.evaluator.config.max_tokens == 4096
    assert agent.evaluator.config.to_llm_config().api_key == llm_config.api_key
    assert agent.evaluator.config.to_llm_config().base_url == llm_config.base_url

    assert agent.analyst.config.model == config.model
    assert agent.analyst.config.provider == config.provider
    assert agent.analyst.config.llm_config is llm_config
    assert agent.analyst.config.temperature == 0.7
    assert agent.analyst.config.max_tokens == 4096

    assert agent.optimizer.config.model == config.model
    assert agent.optimizer.config.provider == config.provider
    assert agent.optimizer.config.llm_config is llm_config
    assert agent.optimizer.config.temperature == 0.3
    assert agent.optimizer.config.max_tokens == 8192

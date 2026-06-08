from shared.llm import llm_config_from_env, llm_config_from_legacy


def test_local_provider_uses_dummy_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    config = llm_config_from_legacy(provider="local-lmstudio", model="local-model")

    assert config.api_spec == "openai_compatible"
    assert config.base_url == "http://localhost:1234/v1"
    assert config.resolve_api_key() == "not-needed"


def test_upstage_provider_rotates_key_envs(monkeypatch):
    monkeypatch.setenv("UPSTAGE_API_KEY", "key-1")
    monkeypatch.setenv("UPSTAGE_API_KEY2", "key-2")
    monkeypatch.delenv("UPSTAGE_API_KEY3", raising=False)

    config = llm_config_from_legacy(provider="upstage", model="solar-pro3")

    assert config.resolve_api_key() == "key-1"
    assert config.resolve_api_key() == "key-2"
    assert config.resolve_api_key() == "key-1"


def test_direct_openai_compatible_config_from_env(monkeypatch):
    monkeypatch.setenv("AGENT_MODEL_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.setenv("AGENT_MODEL_NAME", "qwen-local")
    monkeypatch.delenv("AGENT_MODEL_API_KEY", raising=False)

    config = llm_config_from_env()

    assert config.api_spec == "openai_compatible"
    assert config.model == "qwen-local"
    assert config.resolve_api_key() == "not-needed"


def test_agent_model_env_aliases_are_role_first(monkeypatch):
    monkeypatch.setenv("AGENT_MODEL_PROVIDER", "upstage")
    monkeypatch.setenv("AGENT_MODEL_API_SPEC", "openai_compatible")
    monkeypatch.setenv("AGENT_MODEL_NAME", "solar-pro3")
    monkeypatch.setenv("AGENT_MODEL_BASE_URL", "https://api.upstage.ai/v1/solar")
    monkeypatch.setenv("AGENT_MODEL_API_KEY_ENVS", "UPSTAGE_API_KEY,UPSTAGE_API_KEY2")
    monkeypatch.setenv("AGENT_MODEL_CREDENTIAL_STRATEGY", "round_robin")
    monkeypatch.setenv("UPSTAGE_API_KEY", "key-1")
    monkeypatch.setenv("UPSTAGE_API_KEY2", "key-2")

    config = llm_config_from_env()

    assert config.provider == "upstage"
    assert config.api_spec == "openai_compatible"
    assert config.all_api_key_envs == ("UPSTAGE_API_KEY", "UPSTAGE_API_KEY2")
    assert config.resolve_api_key() == "key-1"
    assert config.resolve_api_key() == "key-2"

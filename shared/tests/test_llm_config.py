from shared.llm import LLMConfig, llm_config_from_env, llm_config_from_legacy


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


def test_single_base_url_is_returned_unchanged():
    config = LLMConfig(model="m", base_url="https://pod-a/v1")
    assert config.all_base_urls == ("https://pod-a/v1",)
    assert [config.resolve_base_url() for _ in range(3)] == ["https://pod-a/v1"] * 3


def test_multiple_endpoints_round_robin():
    config = LLMConfig(
        model="m",
        base_urls=("https://a/v1", "https://b/v1", "https://c/v1"),
        endpoint_strategy="round_robin",
    )
    picked = [config.resolve_base_url() for _ in range(4)]
    assert picked == ["https://a/v1", "https://b/v1", "https://c/v1", "https://a/v1"]


def test_singular_base_url_accepts_a_comma_list():
    # <SLOT>_AGENT_MODEL_BASE_URL may carry several pods; callers must not
    # have to split it themselves.
    config = LLMConfig(
        model="m",
        base_url="https://a/v1, https://b/v1",
        endpoint_strategy="round_robin",
    )
    assert config.all_base_urls == ("https://a/v1", "https://b/v1")
    assert config.resolve_base_url() == "https://a/v1"
    assert config.resolve_base_url() == "https://b/v1"


def test_duplicate_endpoints_collapse():
    config = LLMConfig(model="m", base_url="https://a/v1", base_urls=("https://a/v1",))
    assert config.all_base_urls == ("https://a/v1",)


def test_base_urls_env_enables_round_robin(monkeypatch):
    monkeypatch.setenv("AGENT_MODEL_NAME", "Qwen/Qwen3.5-9B")
    monkeypatch.setenv("AGENT_MODEL_BASE_URLS", "https://p1/v1,https://p2/v1")
    monkeypatch.delenv("AGENT_MODEL_BASE_URL", raising=False)
    monkeypatch.delenv("AGENT_MODEL_ENDPOINT_STRATEGY", raising=False)

    config = llm_config_from_env()

    assert config.endpoint_strategy == "round_robin"
    assert config.all_base_urls == ("https://p1/v1", "https://p2/v1")
    assert config.resolve_base_url() == "https://p1/v1"
    assert config.resolve_base_url() == "https://p2/v1"


def test_endpoint_round_robin_is_thread_safe():
    import collections
    import threading

    config = LLMConfig(
        model="m",
        base_urls=tuple(f"https://p{i}/v1" for i in range(4)),
        endpoint_strategy="round_robin",
    )
    counts: collections.Counter = collections.Counter()
    lock = threading.Lock()

    def hammer():
        for _ in range(100):
            url = config.resolve_base_url()
            with lock:
                counts[url] += 1

    threads = [threading.Thread(target=hammer) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # every endpoint used equally: no index lost to a race
    assert sum(counts.values()) == 800
    assert set(counts.values()) == {200}


def test_rotation_survives_copy_with(monkeypatch):
    """Copies must share the rotation, not restart it.

    run_benchmark.py copies the config once per agent run (to pin max_tokens),
    so if `copy_with` handed out fresh counters every request would go to the
    first endpoint and the first key — round-robin would be silently dead.
    """
    monkeypatch.setenv("K1", "key-1")
    monkeypatch.setenv("K2", "key-2")
    base = LLMConfig(
        model="m",
        base_urls=("https://a/v1", "https://b/v1"),
        endpoint_strategy="round_robin",
        api_key_envs=("K1", "K2"),
        credential_strategy="round_robin",
    )
    urls = [base.copy_with(max_tokens=16384).resolve_base_url() for _ in range(4)]
    keys = [base.copy_with(max_tokens=16384).resolve_api_key() for _ in range(4)]
    assert urls == ["https://a/v1", "https://b/v1"] * 2
    assert keys == ["key-1", "key-2"] * 2


# ── token accounting follows the config lineage ─────────────────────────────
#
# run_benchmark reads `resolved_llm_config._token_counter` AFTER the agent
# returns, so an agent whose LLM calls report into a different counter object
# ships token_usage 0. Copies used to get a fresh counter, which made "never
# copy the config you were handed" a silent precondition that every baseline
# broke in its constructor.

def _usage(config) -> dict[str, int]:
    return config._token_counter.snapshot()


class _Result:
    """The OpenAI-shaped LLMResult the handler reads usage off."""

    def __init__(self, prompt: int, completion: int) -> None:
        self.llm_output = {
            "token_usage": {"prompt_tokens": prompt, "completion_tokens": completion}
        }


def _record(config, prompt: int, completion: int) -> None:
    """One LLM call's worth of usage, exactly as the callback would report it."""
    config._token_counter.on_llm_end(_Result(prompt, completion))


def test_the_token_counter_survives_copy_with():
    base = LLMConfig(model="m")
    _record(base.copy_with(temperature=0.3), 10, 5)
    _record(base.copy_with(model="other"), 1, 2)
    assert _usage(base) == {
        "prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18, "llm_calls": 2,
    }


def test_a_copy_of_a_copy_still_reports_home():
    """Agents chain copies (benchmark pins max_tokens, agent then pins model)."""
    base = LLMConfig(model="m")
    _record(base.copy_with(max_tokens=16384).copy_with(model="x", temperature=0.1), 3, 4)
    assert _usage(base)["total_tokens"] == 7


def test_new_token_scope_is_the_only_thing_that_resets_the_tally():
    base = LLMConfig(model="m")
    _record(base, 10, 5)

    scope = base.new_token_scope()
    assert _usage(scope) == {
        "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "llm_calls": 0,
    }
    _record(scope.copy_with(temperature=0.2), 7, 3)
    assert _usage(scope)["total_tokens"] == 10
    assert _usage(base)["total_tokens"] == 15      # the earlier scope is untouched


def test_new_token_scope_keeps_the_rotation(monkeypatch):
    """A new accounting scope must not restart endpoint/credential round-robin:
    that rotates across the WHOLE benchmark session, not per agent run."""
    monkeypatch.setenv("K1", "key-1")
    monkeypatch.setenv("K2", "key-2")
    base = LLMConfig(
        model="m",
        base_urls=("https://a/v1", "https://b/v1"),
        endpoint_strategy="round_robin",
        api_key_envs=("K1", "K2"),
        credential_strategy="round_robin",
    )
    assert base.new_token_scope().resolve_base_url() == "https://a/v1"
    assert base.new_token_scope().resolve_base_url() == "https://b/v1"
    assert base.new_token_scope().resolve_api_key() == "key-1"
    assert base.new_token_scope().resolve_api_key() == "key-2"

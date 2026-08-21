from types import SimpleNamespace

from agent_for_business.tau_provider import Tau2RetailProvider


def test_provider_builds_retail_run_and_selects_one_task():
    config_calls = []
    load_calls = []
    run_calls = []

    def config_factory(**kwargs):
        config_calls.append(kwargs)
        return SimpleNamespace(**kwargs)

    def task_loader(domain, task_split_name, task_ids):
        load_calls.append((domain, task_split_name, task_ids))
        return [SimpleNamespace(id=task_ids[0])]

    expected_result = object()

    def single_task_runner(config, task, seed):
        run_calls.append((config, task, seed))
        return expected_result

    provider = Tau2RetailProvider(
        agent_llm="local/qwen3.5-2b",
        user_llm="deepseek-flash",
        config_factory=config_factory,
        task_loader=task_loader,
        single_task_runner=single_task_runner,
    )

    result = provider.run(task_id="retail-011", seed=47)

    assert result is expected_result
    assert load_calls == [("retail", "base", ["retail-011"])]
    assert run_calls[0][1].id == "retail-011"
    assert run_calls[0][2] == 47
    assert config_calls[0]["domain"] == "retail"
    assert config_calls[0]["llm_agent"] == "local/qwen3.5-2b"
    assert config_calls[0]["llm_user"] == "deepseek-flash"


def test_default_runner_passes_agent_llm_config_to_evaluator(monkeypatch):
    captured = {}

    import tau2.runner

    def fake_run_single_task(config, task, *, seed, **kwargs):
        captured.update(kwargs)
        return "simulation"

    monkeypatch.setattr(tau2.runner, "run_single_task", fake_run_single_task)
    config = SimpleNamespace(
        llm_agent="gpt-5.6-luna",
        llm_args_agent={
            "api_key": "secret",
            "api_base": "https://api.example.com/v1",
        },
    )

    result = Tau2RetailProvider._default_single_task_runner(
        config,
        SimpleNamespace(id="retail-011"),
        seed=47,
    )

    assert result == "simulation"
    assert captured == {
        "evaluation_llm": "gpt-5.6-luna",
        "evaluation_llm_args": {
            "api_key": "secret",
            "api_base": "https://api.example.com/v1",
        },
    }

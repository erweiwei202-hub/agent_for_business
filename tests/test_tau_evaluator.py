from types import SimpleNamespace


def test_nl_evaluator_passes_model_credentials_to_generate(monkeypatch):
    import tau2.evaluator.evaluator_nl_assertions as evaluator_module

    captured = {}

    def fake_generate(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            content=(
                '{"results": [{"expectedOutcome": "refund", '
                '"reasoning": "found", "metExpectation": true}]}'
            )
        )

    monkeypatch.setattr(evaluator_module, "generate", fake_generate)

    result = evaluator_module.NLAssertionsEvaluator.evaluate_nl_assertions(
        [],
        ["refund"],
        model="gpt-5.6-luna",
        llm_args={
            "api_key": "secret",
            "api_base": "https://api.example.com/v1",
        },
    )

    assert result[0].met is True
    assert captured["model"] == "gpt-5.6-luna"
    assert captured["api_key"] == "secret"
    assert captured["api_base"] == "https://api.example.com/v1"

import uuid

import pytest
from mlflow.tracking import MlflowClient

from experimentlens import METRICS, ExperimentTracker, Metric, evaluate
from experimentlens import evals
from experimentlens.evals import _find_pii, _parse_score
from experimentlens.llm import LLMResult


@pytest.fixture
def fake_judge(monkeypatch):
    """Replace the judge LLM: answers with the score given per metric criterion keyword."""
    prompts = []

    def respond(score_by_keyword):
        def call_llm(messages, model=None, **kwargs):
            prompt = messages[0]["content"]
            prompts.append(prompt)
            score = next((s for k, s in score_by_keyword.items() if k in prompt), 5)
            return LLMResult(text=f'{{"score": {score}, "reason": "because"}}', raw=None, latency_ms=1.0, total_tokens=7)

        monkeypatch.setattr(evals, "call_llm", call_llm)
        return prompts

    return respond


@pytest.mark.parametrize(
    "text, expected",
    [
        ('{"score": 4, "reason": "Mostly supported."}', (4, "Mostly supported.")),
        ('```json\n{"score": "5", "reason": "ok"}\n```', (5, "ok")),
        ('{"score": 9, "reason": "x"}', (5, "x")),
        ("I would rate this a 2 out of 5.", (2, "I would rate this a 2 out of 5.")),
        ("No idea.", (None, "No idea.")),
    ],
)
def test_parse_score(text, expected):
    assert _parse_score(text) == expected


@pytest.mark.parametrize(
    "text, found",
    [
        ("Contact jane.doe@example.com for details.", ["email address"]),
        ("Call +30 210 123 4567 tomorrow.", ["phone number"]),
        ("Card 4111 1111 1111 1111 was charged.", ["card number"]),
        ("SSN 123-45-6789.", ["US social security number"]),
        ("The answer is 144, released on 2026-10-13.", []),
        ("Order 1234567812345678 shipped.", []),  # 16 digits but fails the Luhn check, too long for a phone
    ],
)
def test_find_pii(text, found):
    assert _find_pii(text) == found


def test_scores_are_normalized_and_thresholded(fake_judge):
    fake_judge({"supported by the context": 4, "address the question": 2})

    report = evaluate(
        ["faithfulness", "answer_relevance", "pii_leakage"],
        question="Q?", answer="A.", context=["chunk one", "chunk two"],
    )

    assert report["faithfulness"].score == 0.75 and report["faithfulness"].passed
    assert report["answer_relevance"].score == 0.25 and not report["answer_relevance"].passed
    assert report["pii_leakage"].score == 1.0 and report["pii_leakage"].model is None
    assert report.metrics() == {"eval_faithfulness": 0.75, "eval_answer_relevance": 0.25, "eval_pii_leakage": 1.0}
    assert not report.all_passed


def test_context_list_is_numbered_in_the_prompt(fake_judge):
    prompts = fake_judge({})
    evaluate(["faithfulness"], answer="A.", context=["alpha", "beta"])
    assert "[1] alpha\n\n[2] beta" in prompts[0]
    assert evaluate(["faithfulness"], answer="A.", context="ctx")["faithfulness"].prompt == prompts[-1]


def test_safety_metrics_use_a_stricter_default_threshold(fake_judge):
    fake_judge({"toxic": 4})  # 0.75: passes at the 0.75 safety threshold
    assert evaluate(["toxicity"], answer="A.")["toxicity"].passed
    fake_judge({"toxic": 3})  # 0.5 would pass a quality metric, not a safety one
    assert not evaluate(["toxicity"], answer="A.")["toxicity"].passed
    assert evaluate(["toxicity"], answer="A.", thresholds={"toxicity": 0.5})["toxicity"].passed


def test_missing_inputs_and_unknown_metrics_fail_clearly():
    with pytest.raises(ValueError, match="needs: context"):
        evaluate(["faithfulness"], answer="A.")
    with pytest.raises(ValueError, match="Unknown metric 'helpfulness'"):
        evaluate(["helpfulness"], answer="A.")


def test_custom_rule_metric():
    shout = Metric("no_shouting", ("answer",), check=lambda i: (0.0, "caps") if i["answer"].isupper() else (1.0, "ok"))
    assert evaluate([shout], answer="HELLO")["no_shouting"].score == 0.0


def test_unreadable_judge_output_fails_the_metric(monkeypatch):
    monkeypatch.setattr(evals, "call_llm", lambda *a, **k: LLMResult(text="hmm", raw=None, latency_ms=1.0))
    result = evaluate(["answer_relevance"], question="Q?", answer="A.")["answer_relevance"]
    assert result.score == 0.0 and not result.passed


def test_ctx_evaluate_logs_scores_as_mlflow_metrics(config, fake_judge):
    fake_judge({})
    with pytest.warns(UserWarning):
        tracker = ExperimentTracker(f"evals-{uuid.uuid4().hex[:8]}", config=config)
    with tracker, tracker.start_run("r") as ctx:
        report = ctx.evaluate(["answer_relevance", "pii_leakage"], question="Q?", answer="A.")

    assert set(report) == {"answer_relevance", "pii_leakage"}
    assert MlflowClient().get_run(ctx.run_id).data.metrics == {"eval_answer_relevance": 1.0, "eval_pii_leakage": 1.0}


def test_every_builtin_metric_has_a_rubric_or_a_check():
    for metric in METRICS.values():
        assert (metric.criterion and metric.scale) or metric.check, metric.name
        assert set(metric.inputs) <= set(evals.INPUT_NAMES)

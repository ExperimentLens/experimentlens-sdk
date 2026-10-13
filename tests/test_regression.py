import json
import uuid

import pytest

from experimentlens import (
    ExperimentTracker,
    RegressionError,
    assert_no_regression,
    compare,
    find_regressions,
    load_results,
    run_experiment,
)
from experimentlens import regression


def rows(variant, metric, values):
    return [{"variant": variant, "status": "ok", metric: v} for v in values]


def test_clear_drop_in_a_higher_is_better_metric_regresses():
    results = rows("base", "eval_faithfulness", [0.9, 0.95, 0.9, 1.0]) + rows("new", "eval_faithfulness", [0.5, 0.55, 0.5, 0.6])

    [c] = compare(results, "base", "new", ["eval_faithfulness"])

    assert c.regressed and c.significant and not c.lower_is_better
    assert c.delta == pytest.approx(-0.4)


def test_latency_increase_regresses_and_decrease_does_not():
    slower = rows("base", "latency_ms", [100, 110, 105]) + rows("new", "latency_ms", [300, 310, 305])
    faster = rows("base", "latency_ms", [300, 310, 305]) + rows("new", "latency_ms", [100, 110, 105])

    assert find_regressions(slower, "base", "new", ["latency_ms"])
    assert not find_regressions(faster, "base", "new", ["latency_ms"])


def test_difference_within_noise_is_not_a_regression():
    results = rows("base", "score", [0.2, 0.9, 0.4, 0.8]) + rows("new", "score", [0.1, 0.9, 0.3, 0.8])
    [c] = compare(results, "base", "new", ["score"])
    assert c.significant is False and not c.regressed


def test_single_runs_fall_back_to_the_tolerance():
    results = rows("base", "score", [0.80]) + rows("new", "score", [0.78])
    assert find_regressions(results, "base", "new", ["score"])
    assert not find_regressions(results, "base", "new", ["score"], tolerance=0.05)


def test_assert_no_regression_raises_and_notifies(monkeypatch):
    sent = []
    monkeypatch.setattr(regression, "notify", lambda url, regs, **kw: sent.append((url, regs, kw)))
    results = rows("base", "score", [0.9]) + rows("new", "score", [0.1])

    with pytest.raises(RegressionError, match="new regressed against base on 1 metric"):
        assert_no_regression(results, "base", "new", ["score"], webhook_url="https://hooks.example/x")
    assert sent[0][0] == "https://hooks.example/x"


def test_notify_posts_a_slack_style_payload(monkeypatch):
    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def urlopen(request, timeout):
        captured["body"] = json.loads(request.data)
        return Response()

    monkeypatch.setattr(regression.urllib.request, "urlopen", urlopen)
    [c] = compare(rows("base", "score", [0.9]) + rows("new", "score", [0.1]), "base", "new", ["score"])
    regression.notify("https://hooks.example/x", [c], baseline="base", candidate="new")

    assert "*new* regressed against *base*" in captured["body"]["text"]
    assert "score: 0.900 → 0.100" in captured["body"]["text"]


def test_unknown_group_is_reported():
    with pytest.raises(ValueError, match="No runs with variant='missing'"):
        compare(rows("base", "score", [1.0]), "base", "missing", ["score"])


def test_load_results_reads_runs_back_from_mlflow(config):
    name = f"reg-{uuid.uuid4().hex[:8]}"
    with ExperimentTracker(name, config=config, experiment_type="ml") as tracker:
        run_experiment(
            tracker,
            {"base": {"k": 1}, "new": {"k": 2}},
            [{"id": "q1"}, {"id": "q2"}],
            lambda variant, cfg, case, ctx: {"score": 1.0 if variant == "base" else 0.0},
        )

    results = load_results(name, tracking_uri=config.mlflow_tracking_uri)

    assert sorted(r["variant"] for r in results) == ["base", "base", "new", "new"]
    assert find_regressions(results, "base", "new", ["score"])

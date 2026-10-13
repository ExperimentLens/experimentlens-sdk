import math
import threading

import pytest
from mlflow.tracking import MlflowClient

from experimentlens import config_hash, print_summary, run_experiment, summarize
from experimentlens.runner import _flatten_params, _t_critical_95

VARIANTS = {
    "concise": {"system_prompt": "Answer in one short sentence.", "temperature": 0.0},
    "detailed": {"system_prompt": "Explain your reasoning.", "temperature": 0.7},
}
DATASET = [{"id": "fr", "question": "Capital of France?"}, {"id": "jp", "question": "Capital of Japan?"}]


def _runs(tracker):
    return MlflowClient().search_runs([tracker.experiment_id], max_results=1000)


def test_runs_every_variant_case_and_repeat_in_order(ml_tracker):
    def run_case(variant_name, variant, case, ctx):
        return {"score": 1.0 if variant_name == "concise" else 0.5, "latency_ms": 100}

    results = run_experiment(ml_tracker, VARIANTS, DATASET, run_case, repeats=3, max_workers=4)

    assert [(r["variant"], r["case_id"], r["repeat"]) for r in results] == [
        (v, c, rep) for v in VARIANTS for c in ("fr", "jp") for rep in range(3)
    ]
    assert all(r["status"] == "ok" for r in results)

    runs = _runs(ml_tracker)
    assert len(runs) == 12
    run = next(r for r in runs if r.info.run_name == "detailed__jp__r2")
    assert run.data.params == {
        "variant": "detailed",
        "case_id": "jp",
        "system_prompt": "Explain your reasoning.",
        "temperature": "0.7",
    }
    assert run.data.metrics == {"score": 0.5, "latency_ms": 100.0}
    assert run.data.tags["experimentlens.repeat"] == "2"
    assert run.data.tags["experimentlens.config_hash"] == config_hash(VARIANTS["detailed"])
    assert run.data.tags["experiment_type"] == "ml"
    assert "experimentlens.sdk_version" in run.data.tags


def test_parallel_workers_actually_overlap(ml_tracker):
    barrier = threading.Barrier(2, timeout=10)

    def run_case(variant_name, variant, case, ctx):
        barrier.wait()  # deadlocks (and times out) unless two cases run at once
        return {"ok": 1}

    results = run_experiment(ml_tracker, {"a": {}}, DATASET, run_case, max_workers=2)

    assert [r["status"] for r in results] == ["ok", "ok"]


def test_a_failing_case_is_recorded_without_stopping_the_rest(ml_tracker):
    def run_case(variant_name, variant, case, ctx):
        if case["id"] == "jp":
            raise RuntimeError("model timed out")
        return {"score": 1}

    results = run_experiment(ml_tracker, {"a": {}}, DATASET, run_case)

    assert [r["status"] for r in results] == ["ok", "error"]
    assert results[1]["error"] == "model timed out"
    failed = next(r for r in _runs(ml_tracker) if r.info.run_name == "a__jp")
    assert failed.info.status == "FAILED"
    assert failed.data.tags["error"] == "RuntimeError: model timed out"


def test_rejects_invalid_repeats_and_workers(ml_tracker):
    with pytest.raises(ValueError):
        run_experiment(ml_tracker, VARIANTS, DATASET, lambda *a: {}, repeats=0)
    with pytest.raises(ValueError):
        run_experiment(ml_tracker, VARIANTS, DATASET, lambda *a: {}, max_workers=0)


def test_nested_configs_become_dotted_params():
    assert _flatten_params({"retriever": {"k": 5, "type": "mmr"}, "stop": ["\n"], "model": "x"}) == {
        "retriever.k": "5",
        "retriever.type": "mmr",
        "stop": '["\\n"]',
        "model": "x",
    }


def test_config_hash_ignores_key_order():
    assert config_hash({"a": 1, "b": {"c": 2}}) == config_hash({"b": {"c": 2}, "a": 1})
    assert config_hash({"a": 1}) != config_hash({"a": 2})


def test_summary_gives_mean_and_95_percent_interval():
    results = [{"variant": "a", "status": "ok", "score": v, "passed": v > 2} for v in [1, 2, 3, 4, 5]]
    results.append({"variant": "a", "status": "error", "error": "boom"})

    [row] = summarize(results, ["score", "passed"])

    assert (row["n"], row["errors"]) == (5, 1)
    assert row["score"]["mean"] == 3
    assert row["score"]["ci_low"] == pytest.approx(3 - 2.776 * math.sqrt(2.5) / math.sqrt(5), abs=1e-3)
    assert row["passed"]["mean"] == pytest.approx(0.6)


def test_summary_of_a_single_run_has_no_interval():
    [row] = summarize([{"variant": "a", "score": 0.9}], ["score", "missing"])

    assert row["score"] == {"mean": 0.9, "std": 0.0, "ci_low": None, "ci_high": None, "n": 1}
    assert row["missing"]["mean"] is None


def test_t_critical_values():
    assert _t_critical_95(1) == pytest.approx(12.706)
    assert _t_critical_95(30) == pytest.approx(2.042)
    assert _t_critical_95(60) == pytest.approx(2.000, abs=2e-3)
    assert _t_critical_95(10_000) == pytest.approx(1.960, abs=1e-3)


def test_print_summary(capsys):
    print_summary([{"variant": "a", "score": 1.0}, {"variant": "a", "score": 3.0}], ["score"])

    out = capsys.readouterr().out
    assert "2.000 ± 12.706" in out
    assert "95% confidence interval" in out

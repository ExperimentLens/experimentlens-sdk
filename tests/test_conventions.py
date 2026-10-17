import json

import mlflow

from experimentlens import run_experiment
from experimentlens.conventions import infer_direction


def _tags(tracker):
    return mlflow.tracking.MlflowClient().get_experiment(tracker.experiment_id).tags


def test_evaluate_and_cost_metrics_declare_themselves(ml_tracker):
    with ml_tracker.start_run("r1") as ctx:
        report = ctx.evaluate(["pii_leakage"], answer="Mail me at a@b.com", thresholds={"pii_leakage": 1.0})
        ctx.log_metrics({"latency_ms": 12.0, "accuracy": 0.9})

    tags = _tags(ml_tracker)
    assert report["pii_leakage"].threshold == 1.0
    assert json.loads(tags["experimentlens.metric.eval_pii_leakage"]) == {
        "pass": 1.0, "direction": "maximize", "kind": "safety",
    }
    assert json.loads(tags["experimentlens.metric.latency_ms"]) == {"direction": "minimize", "kind": "cost"}
    assert "experimentlens.metric.accuracy" not in tags


def test_explicit_description_wins_over_the_name_guess(ml_tracker):
    ml_tracker.describe_metric("wait_time_s", direction="minimize", kind="other")
    with ml_tracker.start_run("r1") as ctx:
        ctx.log_metrics({"wait_time_s": 3.0})

    assert json.loads(_tags(ml_tracker)["experimentlens.metric.wait_time_s"]) == {"direction": "minimize", "kind": "other"}


def test_run_experiment_records_params_and_baseline(ml_tracker):
    run_experiment(
        ml_tracker, {"v1": {}, "v2": {}}, [{"id": "c1"}], lambda *_: {"score": 1.0}, baseline="v1",
    )
    tags = _tags(ml_tracker)
    assert tags["experimentlens.group_param"] == "variant"
    assert tags["experimentlens.case_param"] == "case_id"
    assert tags["experimentlens.baseline"] == "v1"


def test_direction_guess_matches_the_ui():
    assert infer_direction("eval_toxicity") == "maximize"
    assert infer_direction("hallucinations") == "minimize"
    assert infer_direction("wait_time") == "minimize"
    assert infer_direction("token_f1") == "maximize"

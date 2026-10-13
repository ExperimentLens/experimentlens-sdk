import json
import pickle
import uuid

import pytest
from mlflow.tracking import MlflowClient

from experimentlens import ExperimentTracker
from experimentlens.tracking import _compute_roc_data


def test_ml_experiment_is_tagged_for_the_ml_views_and_has_no_traces(ml_tracker):
    experiment = MlflowClient().get_experiment(ml_tracker.experiment_id)

    assert experiment.tags["experiment_type"] == "ml"
    assert "langfuse.user_id" not in experiment.tags

    with ml_tracker.start_run("r", params={"depth": 3}) as ctx:
        ctx.log_metrics({"acc": 0.9, "skipped": None})
    run = MlflowClient().get_run(ctx.run_id)

    assert run.data.metrics == {"acc": 0.9}
    assert not any(k.startswith("langfuse.") for k in run.data.tags)


def test_llm_experiment_without_keys_warns_and_logs_mlflow_only(config):
    with pytest.warns(UserWarning, match="Langfuse disabled"):
        tracker = ExperimentTracker(f"llm-{uuid.uuid4().hex[:8]}", config=config)

    with tracker:
        with tracker.start_run("r") as ctx:
            ctx.score("judge", True, data_type="BOOLEAN")  # no-op without Langfuse
            with ctx.observe("llm_call", as_type="generation") as obs:
                obs.update(output="ok")
        run = MlflowClient().get_run(ctx.run_id)

    # Only claim a linked trace when one is really sent, or the UI shows an empty Traces tab.
    assert "langfuse.session_id" not in run.data.tags
    assert MlflowClient().get_experiment(tracker.experiment_id).tags["experiment_type"] == "llm"


def test_reuses_an_existing_experiment(config):
    name = f"reuse-{uuid.uuid4().hex[:8]}"
    first = ExperimentTracker(name, config=config, experiment_type="ml")

    assert ExperimentTracker(name, config=config, experiment_type="ml").experiment_id == first.experiment_id


def test_explainability_files_land_where_the_ui_reads_them(ml_tracker, tmp_path):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("sklearn")
    from sklearn.linear_model import LogisticRegression

    X = pd.DataFrame({"a": [0, 1, 0, 1, 0, 1], "b": [1, 1, 0, 0, 1, 0]})
    y = pd.Series([0, 1, 0, 1, 0, 1], name="label")
    model = LogisticRegression().fit(X, y)

    with ml_tracker.start_run("explain") as ctx:
        ctx.log_explainability_data(
            X_train=X, X_test=X, y_train=y, y_test=y, y_pred=model.predict(X),
            model=model, y_score=model.predict_proba(X),
        )

    client = MlflowClient()
    files = sorted(f.path for f in client.list_artifacts(ctx.run_id, "explainability"))
    assert files == [
        f"explainability/{name}"
        for name in ["X_test.csv", "X_train.csv", "Y_pred.csv", "Y_test.csv", "Y_train.csv", "model.pkl", "roc_data.json"]
    ]

    local = client.download_artifacts(ctx.run_id, "explainability", str(tmp_path))
    with open(f"{local}/roc_data.json") as f:
        roc = json.load(f)
    assert set(roc) == {"fpr", "tpr", "thresholds", "roc_auc"}
    with open(f"{local}/model.pkl", "rb") as f:
        assert pickle.load(f).predict(X).tolist() == model.predict(X).tolist()
    with open(f"{local}/Y_test.csv") as f:
        assert f.readline().strip() == "label"


def test_roc_data_is_valid_json_for_binary_and_multiclass():
    pytest.importorskip("sklearn")

    binary = _compute_roc_data([0, 1, 1, 0], [[0.9, 0.1], [0.2, 0.8], [0.4, 0.6], [0.7, 0.3]])
    assert binary["roc_auc"] == pytest.approx(1.0)
    json.dumps(binary, allow_nan=False)  # +inf threshold must have been replaced

    multi = _compute_roc_data([0, 1, 2], [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
    assert multi["roc_auc"] == pytest.approx(1.0)


def test_failed_run_records_a_bounded_error_tag(ml_tracker):
    with pytest.raises(ValueError):
        with ml_tracker.start_run("boom") as ctx:
            raise ValueError("x" * 10_000)

    tag = MlflowClient().get_run(ctx.run_id).data.tags["error"]
    assert tag.startswith("ValueError: xxx")
    assert len(tag) == 5000

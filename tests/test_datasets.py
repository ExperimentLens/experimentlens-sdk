import uuid
from datetime import datetime, timezone

import pytest
from mlflow.tracking import MlflowClient

from experimentlens import Dataset, ExperimentTracker, load_dataset, run_experiment, upload_dataset
from experimentlens import datasets as ds
from experimentlens.datasets import fingerprint


class FakeLangfuse:
    """In-memory stand-in for the Langfuse dataset REST endpoints."""

    def __init__(self):
        self.datasets, self.items, self.queries = {}, {}, []

    def request(self, method, path, body=None, query=None):
        if method == "POST" and path == "/api/public/v2/datasets":
            if body["name"] in self.datasets:
                raise RuntimeError("exists")
            self.datasets[body["name"]] = body
            return body
        if method == "POST" and path == "/api/public/dataset-items":
            self.items[body["id"]] = {**body, "status": "ACTIVE"}
            return body
        if method == "GET" and path.startswith("/api/public/v2/datasets/"):
            return self.datasets.get(path.rsplit("/", 1)[1])
        if method == "GET" and path == "/api/public/dataset-items":
            self.queries.append(query)
            data = [i for i in self.items.values() if i["datasetName"] == query["datasetName"]]
            return {"data": data, "meta": {"totalPages": 1}}
        raise AssertionError(f"unexpected {method} {path}")


@pytest.fixture
def fake(monkeypatch):
    client = FakeLangfuse()
    monkeypatch.setattr(ds, "_client", lambda config: client)
    return client


CASES = [
    {"id": "fr", "question": "Capital of France?", "expected": "Paris"},
    {"id": "jp", "question": "Capital of Japan?", "expected": "Tokyo"},
]


def test_fingerprint_ignores_order_and_changes_with_content():
    assert fingerprint(CASES) == fingerprint(list(reversed(CASES)))
    assert fingerprint(CASES) != fingerprint([{**CASES[0], "expected": "Lyon"}, CASES[1]])


def test_upload_then_load_round_trips_cases(fake):
    dataset = upload_dataset("capitals", CASES, description="demo")

    assert len(dataset) == 2 and dataset.description == "demo"
    by_id = {c["id"]: c for c in dataset}
    assert by_id["fr"]["question"] == "Capital of France?" and by_id["fr"]["expected"] == "Paris"
    assert dataset.version == fingerprint(dataset.cases) and len(dataset.version) == 12


def test_reuploading_a_case_updates_its_item(fake):
    upload_dataset("capitals", CASES)
    dataset = upload_dataset("capitals", [{**CASES[0], "expected": "Paris, France"}])

    assert len(dataset) == 2
    assert {c["id"]: c["expected"] for c in dataset}["fr"] == "Paris, France"


def test_load_requires_a_utc_version(fake):
    with pytest.raises(ValueError, match="timezone-aware"):
        load_dataset("capitals", version=datetime(2026, 1, 1))
    load_dataset("capitals", version=datetime(2026, 1, 1, tzinfo=timezone.utc))
    assert fake.queries[-1]["version"] == "2026-01-01T00:00:00Z"


def test_case_ids_are_required(fake):
    with pytest.raises(ValueError, match="needs an 'id' key"):
        upload_dataset("capitals", [{"question": "?"}])


def test_run_experiment_tags_runs_with_the_dataset_version(config):
    dataset = Dataset(name="capitals", cases=[dict(c) for c in CASES], version=fingerprint(CASES), snapshot=datetime.now(timezone.utc))
    name = f"ds-{uuid.uuid4().hex[:8]}"
    with ExperimentTracker(name, config=config, experiment_type="ml") as tracker:
        results = run_experiment(tracker, {"a": {"k": 1}}, dataset, lambda v, c, case, ctx: {"score": 1.0})

    run_id = MlflowClient().search_runs([tracker.experiment_id])[0].info.run_id
    tags = MlflowClient().get_run(run_id).data.tags
    assert tags["experimentlens.dataset"] == "capitals"
    assert tags["experimentlens.dataset_version"] == dataset.version
    assert MlflowClient().get_experiment(tracker.experiment_id).tags["experimentlens.datasets"] == "capitals"
    assert [r["case_id"] for r in results] == ["fr", "jp"]

import os
import uuid

import pytest

from experimentlens import ExperimentTracker, SDKConfig

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")


@pytest.fixture(scope="session")
def tracking_uri(tmp_path_factory):
    """A throwaway local MLflow store, so tests never touch a real tracking server."""
    return f"sqlite:///{tmp_path_factory.mktemp('mlflow') / 'mlflow.db'}"


@pytest.fixture(autouse=True)
def isolated_cwd(tmp_path, monkeypatch):
    # MLflow puts artifacts under ./mlruns, and the SDK looks for a .env from the
    # working directory upwards; both must stay inside the test's temp folder.
    monkeypatch.chdir(tmp_path)
    for name in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST", "LANGFUSE_BASE_URL"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def config(tracking_uri):
    return SDKConfig(mlflow_tracking_uri=tracking_uri)


@pytest.fixture
def ml_tracker(config):
    with ExperimentTracker(f"test-{uuid.uuid4().hex[:8]}", config=config, experiment_type="ml") as tracker:
        yield tracker

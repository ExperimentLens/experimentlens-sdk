from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - python-dotenv is an optional convenience
    load_dotenv = None

# Local Ollama needs no API key, so examples run out of the box. Override with
# EXPERIMENTLENS_MODEL (or LLM_MODEL, the name the ExperimentLens docs use) /
# EXPERIMENTLENS_JUDGE_MODEL, or per call via `model=`.
FALLBACK_MODEL = "ollama/llama3.2"


def default_model() -> str:
    return os.getenv("EXPERIMENTLENS_MODEL") or os.getenv("LLM_MODEL") or FALLBACK_MODEL


def default_judge_model() -> str:
    return os.getenv("EXPERIMENTLENS_JUDGE_MODEL") or default_model()


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    if load_dotenv is not None:
        load_dotenv(path, override=False)
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _find_env_file() -> Optional[Path]:
    """Nearest .env walking up from the working directory, then from the running
    script's directory -- so `python examples/foo.py` or an IDE run button launched
    from another folder still finds the project's .env."""
    import __main__

    starts = [Path.cwd()]
    main_file = getattr(__main__, "__file__", None)
    if main_file:
        starts.append(Path(main_file).resolve().parent)
    for start in starts:
        for directory in (start, *start.parents):
            candidate = directory / ".env"
            if candidate.is_file():
                return candidate
    return None


@dataclass
class SDKConfig:
    """Connection settings for the MLflow/Langfuse instances backing ExperimentLens.

    Build it with `SDKConfig.from_env()` to read environment variables (loading a
    .env file first if present), optionally overriding any field:

        SDKConfig.from_env(mlflow_tracking_uri="http://mlflow.internal:5000")

    or construct it directly with `SDKConfig(...)` to skip the environment entirely.

    `langfuse_project_id` may be left unset: the tracker then looks it up from the
    Langfuse API keys.
    """

    mlflow_tracking_uri: str = "http://127.0.0.1:5000"
    langfuse_public_key: Optional[str] = None
    langfuse_secret_key: Optional[str] = None
    langfuse_host: str = "http://localhost:3000"
    langfuse_project_id: Optional[str] = None
    # Optional second OTLP/HTTP destination (Phoenix, Logfire, Honeycomb, ...). Spans
    # still go to Langfuse; this sends a copy of each one there too.
    otlp_endpoint: Optional[str] = None

    def __post_init__(self) -> None:
        # Langfuse appends "/api/public/otel/v1/traces" verbatim; a trailing slash
        # yields "//api/...", which self-hosted Langfuse answers with a 308 the OTLP
        # exporter won't follow -- spans are silently dropped while scores still land.
        self.langfuse_host = self.langfuse_host.rstrip("/")

    @classmethod
    def from_env(cls, env_file: Optional[str] = None, **overrides) -> "SDKConfig":
        path = Path(env_file) if env_file else _find_env_file()
        if path is not None:
            _load_env_file(path)

        values = dict(
            mlflow_tracking_uri=(
                os.getenv("MLFLOW_TRACKING_URI")
                or os.getenv("MLFLOW_TRACKING_URL")
                or cls.mlflow_tracking_uri
            ),
            langfuse_public_key=os.getenv("LANGFUSE_PUBLIC_KEY") or None,
            langfuse_secret_key=os.getenv("LANGFUSE_SECRET_KEY") or None,
            langfuse_host=(
                os.getenv("LANGFUSE_HOST")
                or os.getenv("LANGFUSE_BASE_URL")
                or cls.langfuse_host
            ),
            langfuse_project_id=os.getenv("LANGFUSE_PROJECT_ID") or None,
            otlp_endpoint=os.getenv("EXPERIMENTLENS_OTLP_ENDPOINT") or None,
        )
        values.update(overrides)
        return cls(**values)

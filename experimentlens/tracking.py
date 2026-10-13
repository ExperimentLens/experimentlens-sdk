from __future__ import annotations

import time
import warnings
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

import mlflow
from mlflow.tracking import MlflowClient

from . import __version__
from .config import SDKConfig

_MAX_TAG_LENGTH = 5000


class _NoopObservation:
    """Stand-in returned by RunContext.observe() when Langfuse is disabled or fails,
    so experiment code doesn't need to branch on whether tracing is configured."""

    def update(self, *args, **kwargs) -> "_NoopObservation":
        return self

    def score(self, *args, **kwargs) -> None:
        return None

    def end(self, *args, **kwargs) -> "_NoopObservation":
        return self


class ExperimentTracker:
    """MLflow logging, with optional dual Langfuse tracing for LLM/agent experiments.

    `experiment_type` is the one thing vis-frontend uses to pick which view an
    experiment gets (see `experiment_type` tag checks in monitoring-page.tsx):

      - "llm" (default): every run also gets a linked Langfuse trace, using the
        convention the UI looks for --
            Langfuse trace `user_id`    = MLflow experiment_id
            Langfuse trace `session_id` = MLflow run_id
        That's the only mechanism the UI uses to find a run's traces, so
        `start_run()` sets this up automatically for every run.

      - "ml": plain MLflow only. No Langfuse client, no trace/session tags on
        runs, so the Traces tab correctly stays empty/hidden for a tabular ML
        experiment (see the `traces` capability in experimentCapabilities.ts,
        which is detected from `langfuse.*` tags on a run) instead of showing
        an irrelevant empty tab. Use `RunContext.log_metrics/log_params/
        log_dataset` and log your model directly via `mlflow.sklearn.log_model`
        (or the equivalent for your framework) -- see examples/ml_experiment.py.
    """

    def __init__(
        self,
        experiment_name: str,
        config: Optional[SDKConfig] = None,
        experiment_type: str = "llm",
    ) -> None:
        self.config = config or SDKConfig.from_env()
        self.experiment_type = experiment_type
        self._use_langfuse = experiment_type == "llm"

        mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
        self._client = MlflowClient()
        self.experiment_id = self._get_or_create_experiment(experiment_name)
        mlflow.set_experiment(experiment_name)

        # Tags vis-api / vis-frontend rely on to classify and link this experiment.
        self._client.set_experiment_tag(self.experiment_id, "experiment_type", experiment_type)

        self._langfuse = None
        self.langfuse_project_id: Optional[str] = None
        if not self._use_langfuse:
            return

        self._client.set_experiment_tag(
            self.experiment_id, "langfuse.user_id", str(self.experiment_id)
        )

        if self.config.langfuse_public_key and self.config.langfuse_secret_key:
            try:
                from langfuse import Langfuse

                self._langfuse = Langfuse(
                    public_key=self.config.langfuse_public_key,
                    secret_key=self.config.langfuse_secret_key,
                    # base_url, not host: Langfuse ranks the LANGFUSE_BASE_URL env var
                    # above `host`, which would bypass our normalized value.
                    base_url=self.config.langfuse_host,
                )
            except Exception as exc:  # pragma: no cover - defensive
                warnings.warn(f"Langfuse disabled: could not initialize client ({exc})")
        else:
            warnings.warn(
                "Langfuse disabled: LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY not set. "
                "Runs will still log to MLflow, but no traces will appear in the "
                "Observability / trajectory views."
            )

        project_id = self.config.langfuse_project_id or self._detect_langfuse_project_id()
        self.langfuse_project_id = project_id
        if project_id:
            self._client.set_experiment_tag(self.experiment_id, "langfuse.project_id", project_id)
        elif self._langfuse is not None:
            warnings.warn(
                "Could not determine the Langfuse project id; set LANGFUSE_PROJECT_ID "
                "so the UI can link this experiment to its traces."
            )

    def _detect_langfuse_project_id(self) -> Optional[str]:
        """Look up the project the Langfuse API keys belong to."""
        if self._langfuse is None:
            return None
        try:
            projects = self._langfuse.api.projects.get()
            return projects.data[0].id if projects.data else None
        except Exception:
            return None

    def _get_or_create_experiment(self, name: str) -> str:
        exp = mlflow.get_experiment_by_name(name)
        if exp is not None:
            return exp.experiment_id
        return self._client.create_experiment(name)

    @contextmanager
    def start_run(
        self,
        run_name: str,
        params: Optional[dict] = None,
        tags: Optional[dict] = None,
        variant: Optional[str] = None,
    ) -> Iterator["RunContext"]:
        """Start one MLflow run + matching Langfuse trace. Use as a context manager
        around a single experiment case (e.g. one variant x one dataset item)."""
        with mlflow.start_run(run_name=run_name) as run:
            run_id = run.info.run_id

            run_tags = {"experiment_type": self.experiment_type, "experimentlens.sdk_version": __version__}
            # Only claim a linked trace when one is actually being sent; otherwise the
            # UI shows a Traces tab that can never fill.
            if self._langfuse is not None:
                run_tags["langfuse.user_id"] = str(self.experiment_id)
                run_tags["langfuse.session_id"] = run_id
            if tags:
                run_tags.update(tags)
            mlflow.set_tags(run_tags)
            if params:
                mlflow.log_params(params)

            with ExitStack() as stack:
                root_span = None
                if self._langfuse is not None:
                    try:
                        root_span = stack.enter_context(
                            self._langfuse.start_as_current_observation(
                                name=run_name, as_type="span", input=params or {}
                            )
                        )
                        from langfuse import propagate_attributes

                        langfuse_tags = [t for t in [variant, *(tags or {}).values()] if t]
                        stack.enter_context(
                            propagate_attributes(
                                user_id=str(self.experiment_id),
                                session_id=run_id,
                                metadata={
                                    "mlflow_experiment_id": self.experiment_id,
                                    "mlflow_run_id": run_id,
                                    **{k: str(v) for k, v in (params or {}).items()},
                                },
                                tags=langfuse_tags,
                            )
                        )
                    except Exception as exc:  # pragma: no cover - defensive
                        warnings.warn(f"Langfuse trace setup failed, continuing MLflow-only: {exc}")
                        root_span = None

                ctx = RunContext(tracker=self, run_id=run_id, root_span=root_span)
                try:
                    yield ctx
                except Exception as exc:
                    # MLflow rejects tag values over 5000 characters, which would mask the real error.
                    mlflow.set_tag("error", f"{type(exc).__name__}: {exc}"[:_MAX_TAG_LENGTH])
                    raise
                finally:
                    if root_span is not None:
                        self._tag_trace_link(root_span)
                    if self._langfuse is not None:
                        try:
                            self._langfuse.flush()
                        except Exception:
                            pass

    def _tag_trace_link(self, root_span: Any) -> None:
        """Write the Langfuse trace id/url back onto the MLflow run, so ExperimentLens
        can open the matching trace straight from the run."""
        try:
            trace_id = root_span.trace_id
            tags = {"langfuse.trace_id": trace_id}
            if self.langfuse_project_id:
                tags["langfuse.trace_url"] = (
                    f"{self.config.langfuse_host}/project/{self.langfuse_project_id}/traces/{trace_id}"
                )
            mlflow.set_tags(tags)
        except Exception:
            pass

    def __enter__(self) -> "ExperimentTracker":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def close(self) -> None:
        """Flush pending traces and shut down the Langfuse client. Called automatically
        when the tracker is used as a context manager (`with ExperimentTracker(...) as t:`)."""
        if self._langfuse is not None:
            try:
                self._langfuse.flush()
                self._langfuse.shutdown()
            except Exception:
                pass


@dataclass
class RunContext:
    """Handed to experiment code inside a `with tracker.start_run(...) as ctx:` block."""

    tracker: ExperimentTracker
    run_id: str
    root_span: Any = None

    def set_trace_io(self, input: Any = None, output: Any = None) -> None:
        """Set the trace-level input/output shown at the top of the Langfuse trace
        view (distinct from the per-step input/output inside `observe()`/
        `log_generation()`). The vis-frontend's trace UI looks specifically for a
        `question` key in the input and an `answer` key in the output, e.g.:

            ctx.set_trace_io(
                input={"question": case["question"]},
                output={"answer": result.text, "judge_pass_rate": float(passed)},
            )

        Without this, the trace root span's INPUT/OUTPUT panels show empty even
        though nested generations (e.g. the LLM call itself) have their own
        input/output populated."""
        if self.root_span is None:
            return
        try:
            self.root_span.update(input=input, output=output)
        except Exception:
            pass

    def log_params(self, params: dict) -> None:
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict) -> None:
        clean = {k: v for k, v in metrics.items() if v is not None}
        if clean:
            mlflow.log_metrics(clean)

    def set_tags(self, tags: dict) -> None:
        mlflow.set_tags(tags)

    def log_artifact_dict(self, filename: str, data: dict) -> None:
        """Log a JSON-serializable record (full I/O, intermediate steps, etc.) as an
        MLflow artifact under this run, e.g. ctx.log_artifact_dict("trace.json", {...})."""
        mlflow.log_dict(data, filename)

    def log_dataset(
        self,
        data,
        name: str = "dataset",
        context: str = "evaluation",
        targets: Optional[str] = None,
        file_format: str = "csv",
    ) -> None:
        """Log an input dataset for this run: registers it as an MLflow input
        dataset (shows up as an INPUT DataAsset in the ExperimentLens UI's Data
        tab, via `dataset_inputs`) AND logs the actual rows as a downloadable
        `<name>.csv`/`.parquet` artifact under the run, since `dataset_inputs`
        alone only stores a schema/profile/digest summary — not the row data
        itself, so there'd be nothing to open otherwise.

        `data` can be a list of dicts (e.g. a test case, or the whole dataset) or
        a pandas DataFrame; `targets` names the ground-truth column, if any;
        `file_format` is "csv" (default, human-readable) or "parquet".
        `run_experiment()` already calls this once per run with that run's single
        case — call it yourself (e.g. with the full dataset) for anything beyond
        that default.

        The registered input dataset's `source` points at the actual logged
        artifact (an HTTPDatasetSource built from this run's tracking URL) — not
        the default "code" source mlflow.data.from_pandas falls
        back to (a JSON blob describing the calling script, not a usable link)."""
        import os
        import tempfile

        import pandas as pd
        import mlflow.data
        from mlflow.data.http_dataset_source import HTTPDatasetSource

        if isinstance(data, pd.DataFrame):
            df = data
        else:
            df = pd.DataFrame(data if isinstance(data, list) else [data])

        suffix = ".parquet" if file_format == "parquet" else ".csv"
        filename = f"{name}{suffix}"
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, filename)
            if file_format == "parquet":
                df.to_parquet(path, index=False)
            else:
                df.to_csv(path, index=False)
            mlflow.log_artifact(path)

        source_uri = (
            f"{self.tracker.config.mlflow_tracking_uri.rstrip('/')}"
            f"/get-artifact?run_id={self.run_id}&path={filename}"
        )
        dataset = mlflow.data.from_pandas(
            df, source=HTTPDatasetSource(source_uri), name=name, targets=targets
        )
        mlflow.log_input(dataset, context=context)

    def log_explainability_data(
        self,
        X_train=None,
        X_test=None,
        y_train=None,
        y_test=None,
        y_pred=None,
        model=None,
        y_score=None,
        roc_data: Optional[dict] = None,
        extra: Optional[dict] = None,
    ) -> None:
        """Log the files ExperimentLens' explainability views read, into this run's
        `explainability/` artifact folder (see "MLflow experiment with
        explainability" in the ExperimentLens docs): X_train.csv, X_test.csv,
        Y_train.csv, Y_test.csv, Y_pred.csv, model.pkl and roc_data.json.

        Every argument is optional — pass whatever you have. Data arguments accept a
        DataFrame, a Series, a numpy array, or a plain list; label columns without a
        name are called `target`. `model` is any object picklable with `pickle` (e.g.
        a fitted sklearn estimator).

        For the ROC curve, either pass `roc_data` ({fpr, tpr, thresholds, roc_auc})
        yourself, or pass `y_score` (e.g. `model.predict_proba(X_test)`) together
        with `y_test` and it is computed for you (needs scikit-learn; multi-class
        labels are micro-averaged). `extra` logs additional named CSVs the same way,
        e.g. `extra={"feature_importances": importances_df}`.
        """
        import json
        import os
        import pickle
        import tempfile

        import pandas as pd

        def _to_df(data, col_name):
            if data is None:
                return None
            if isinstance(data, pd.DataFrame):
                return data
            if isinstance(data, pd.Series):
                return data.to_frame(data.name if data.name is not None else col_name)
            df = pd.DataFrame(data)
            if df.shape[1] == 1 and col_name == "target":
                df.columns = ["target"]
            return df

        if roc_data is None and y_score is not None and y_test is not None:
            roc_data = _compute_roc_data(y_test, y_score)

        files = {
            "X_train": _to_df(X_train, "X_train"),
            "X_test": _to_df(X_test, "X_test"),
            "Y_train": _to_df(y_train, "target"),
            "Y_test": _to_df(y_test, "target"),
            "Y_pred": _to_df(y_pred, "target"),
        }
        for name, data in (extra or {}).items():
            files[name] = _to_df(data, name)

        with tempfile.TemporaryDirectory() as tmp_dir:
            for name, df in files.items():
                if df is None:
                    continue
                path = os.path.join(tmp_dir, f"{name}.csv")
                df.to_csv(path, index=False)
                mlflow.log_artifact(path, artifact_path="explainability")

            if model is not None:
                model_path = os.path.join(tmp_dir, "model.pkl")
                with open(model_path, "wb") as f:
                    pickle.dump(model, f)
                mlflow.log_artifact(model_path, artifact_path="explainability")

            if roc_data is not None:
                roc_path = os.path.join(tmp_dir, "roc_data.json")
                with open(roc_path, "w", encoding="utf-8") as f:
                    json.dump(roc_data, f, indent=2)
                mlflow.log_artifact(roc_path, artifact_path="explainability")

    @contextmanager
    def observe(self, name: str, as_type: str = "span", **kwargs) -> Iterator[Any]:
        """Wrap a real unit of work (an LLM call, a tool call, a pipeline stage) as a
        nested Langfuse observation with accurate start/end timestamps, e.g.:

            with ctx.observe("llm_call", as_type="generation", model="ollama/llama3.2") as obs:
                result = call_llm(messages, model="ollama/llama3.2")
                obs.update(output=result.text, usage_details={...})

        `as_type` is one of Langfuse's observation types: "span", "generation", "tool",
        "chain", "retriever", "agent", "evaluator", "guardrail".
        No-ops safely if Langfuse isn't configured.

        For `as_type="generation"`, the vis-frontend's dedicated PROMPT panel reads
        `input.prompt` as a plain string (see GenInput in agentic-conventions.ts) —
        passing a raw chat-messages list as `input` will show in the raw JSON view
        but leave that panel empty. Use `experimentlens.messages_to_prompt(messages)`
        to build it, e.g. `input={"prompt": messages_to_prompt(messages), "model": ...}`.
        """
        if self.tracker._langfuse is None:
            yield _NoopObservation()
            return
        try:
            with self.tracker._langfuse.start_as_current_observation(
                name=name, as_type=as_type, **kwargs
            ) as obs:
                yield obs
        except Exception as exc:  # pragma: no cover - defensive
            warnings.warn(f"Langfuse observe('{name}') failed: {exc}")
            yield _NoopObservation()

    def log_generation(
        self,
        name: str,
        *,
        input: Any = None,
        output: Any = None,
        model: Optional[str] = None,
        latency_ms: float = 0.0,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        cost_usd: Optional[float] = None,
    ) -> None:
        """Convenience for logging an LLM call you've already finished (e.g. the
        result of `llm.call_llm(...)`), when you don't need `observe()`'s real-time
        timing. Note: since the call already completed, the logged observation's
        duration will be near-zero even though `latency_ms` is still recorded as an
        MLflow metric via RunContext.log_metrics — use `observe()` instead if you need
        accurate per-step timing in the Langfuse trace view."""
        usage_details = {
            "input": prompt_tokens,
            "output": completion_tokens,
            "total": total_tokens,
        }
        self._log_finished_observation(
            "generation",
            name,
            input=input,
            output=output,
            model=model,
            usage_details={k: v for k, v in usage_details.items() if v},
            cost_details={"total": cost_usd} if cost_usd is not None else None,
        )

    def log_tool_call(self, name: str, *, input: Any = None, output: Any = None) -> None:
        self._log_finished_observation("tool", name, input=input, output=output)

    def log_span(self, name: str, *, input: Any = None, output: Any = None) -> None:
        self._log_finished_observation("span", name, input=input, output=output)

    def log_event(self, name: str, *, output: Any = None, level: str = "WARNING") -> None:
        """Log a one-off point-in-time event (e.g. a replanning trigger, a guardrail
        block) rather than a span with duration."""
        if self.tracker._langfuse is None:
            return
        try:
            self.tracker._langfuse.create_event(name=name, output=output, level=level)
        except Exception:
            pass

    def log_judge(
        self,
        name: str,
        question: str,
        answer: str,
        criteria: str,
        model: Optional[str] = None,
    ):
        """Run an LLM-as-judge check and log it so the vis-frontend recognizes it as
        a judge: name gets prefixed `judge_` (isJudge() in agentic-conventions.ts
        matches on that prefix, or on a boolean `output.passed`), and the verdict is
        also attached to the trace as a BOOLEAN score. Returns a JudgeResult(passed,
        rationale, llm_result) so you can also fold `passed` into your own metrics.

        Prefer a stronger/different model than the one under test to reduce judge
        bias, e.g. model="openai/gpt-4o-mini". If omitted, EXPERIMENTLENS_JUDGE_MODEL
        is used (see llm_judge).
        """
        from .config import default_judge_model
        from .judge import llm_judge

        model = model or default_judge_model()

        judge_name = name if name.startswith("judge_") else f"judge_{name}"
        verdict = llm_judge(question, answer, criteria, model=model)

        self._log_finished_observation(
            "generation",
            judge_name,
            input={
                "prompt": f"Question: {question}\nAnswer: {answer}\nCriteria: {criteria}",
                "model": model,
            },
            output={"answer": verdict.rationale, "passed": verdict.passed, "rationale": verdict.rationale},
            model=model,
            usage_details={"total": verdict.llm_result.total_tokens} if verdict.llm_result.total_tokens else None,
        )
        self.score(judge_name, verdict.passed, data_type="BOOLEAN", comment=verdict.rationale)
        return verdict

    def evaluate(
        self,
        metrics,
        *,
        question: Optional[str] = None,
        answer: Optional[str] = None,
        context=None,
        expected: Optional[str] = None,
        model: Optional[str] = None,
        thresholds: Optional[dict] = None,
    ):
        """Score this run's output on built-in or custom metrics (see
        `experimentlens.evals.METRICS`) and log the results, e.g.

            report = ctx.evaluate(["faithfulness", "answer_relevance", "toxicity", "pii_leakage"],
                                  question=q, answer=a, context=chunks)
            return {"latency_ms": ..., **report.metrics()}

        Each score (0..1, higher is better) is logged as an MLflow metric
        `eval_<name>`, so runs can be compared and checked for regressions. On the
        trace, LLM-judged metrics appear as `judge_<name>` verdicts (the Judges
        views pick them up) and rule-based ones as BOOLEAN checker scores. Returns
        an EvalReport of {name: EvalResult(score, passed, reason, ...)}.

        Return `report.metrics()` from `run_case` as well, so `summarize()` and
        `find_regressions()` see the scores in run_experiment's results.
        """
        from .evals import evaluate

        report = evaluate(
            metrics, question=question, answer=answer, context=context,
            expected=expected, model=model, thresholds=thresholds,
        )
        for name, result in report.items():
            if result.model is None:
                self.score(name, result.passed, data_type="BOOLEAN", comment=result.reason)
                continue
            judge_name = f"judge_{name}"
            tokens = result.llm_result.total_tokens if result.llm_result else 0
            self._log_finished_observation(
                "generation",
                judge_name,
                input={"prompt": result.prompt, "model": result.model},
                output={"answer": result.reason, "passed": result.passed, "score": result.score, "rationale": result.reason},
                model=result.model,
                usage_details={"total": tokens} if tokens else None,
            )
            self.score(judge_name, result.passed, data_type="BOOLEAN", comment=result.reason)
        self.log_metrics(report.metrics())
        return report

    def score(self, name: str, value: Any, data_type: str = "NUMERIC", comment: Optional[str] = None) -> None:
        """Attach a score to this run's trace, e.g. ctx.score("task_success", True, "BOOLEAN")."""
        if self.root_span is None:
            return
        try:
            self.root_span.score(name=name, value=value, data_type=data_type, comment=comment)
        except Exception:
            pass

    def _log_finished_observation(
        self,
        as_type: str,
        name: str,
        *,
        input: Any = None,
        output: Any = None,
        model: Optional[str] = None,
        usage_details: Optional[dict] = None,
        cost_details: Optional[dict] = None,
    ) -> None:
        if self.tracker._langfuse is None:
            return
        try:
            obs = self.tracker._langfuse.start_observation(
                name=name,
                as_type=as_type,
                input=input,
                output=output,
                model=model,
                usage_details=usage_details or None,
                cost_details=cost_details,
            )
            obs.end(end_time=time.time_ns())
        except Exception:
            pass


def _compute_roc_data(y_test, y_score) -> dict:
    """ROC in the shape ExperimentLens expects: {fpr, tpr, thresholds, roc_auc}.
    Multi-class scores are micro-averaged, as in the ExperimentLens docs example."""
    import numpy as np
    from sklearn.metrics import auc, roc_curve
    from sklearn.preprocessing import label_binarize

    y_true = np.asarray(y_test).ravel()
    scores = np.asarray(y_score)
    classes = np.unique(y_true)

    if scores.ndim == 1 or scores.shape[1] == 1:
        fpr, tpr, thresholds = roc_curve(y_true, scores.ravel(), pos_label=classes[-1])
    elif scores.shape[1] == 2:
        fpr, tpr, thresholds = roc_curve(y_true, scores[:, 1], pos_label=classes[-1])
    else:
        if len(classes) != scores.shape[1]:
            classes = np.arange(scores.shape[1])
        y_bin = label_binarize(y_true, classes=classes)
        fpr, tpr, thresholds = roc_curve(y_bin.ravel(), scores.ravel())

    # roc_curve's first threshold is +inf, which is not valid JSON.
    thresholds = np.where(np.isfinite(thresholds), thresholds, np.finfo(float).max)
    return {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": thresholds.tolist(),
        "roc_auc": float(auc(fpr, tpr)),
    }

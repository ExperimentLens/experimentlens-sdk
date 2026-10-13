from __future__ import annotations

import json
import math
import re
import urllib.request
from dataclasses import dataclass
from typing import Iterable, Optional

from .runner import _t_critical_95, summarize

# Metrics whose name matches this are treated as lower-is-better unless told otherwise.
_LOWER_IS_BETTER = re.compile(r"latency|duration|cost|tokens|error|_ms$|_s$", re.IGNORECASE)


@dataclass
class MetricComparison:
    """How one metric moved from the baseline group to the candidate group."""

    metric: str
    baseline_mean: Optional[float]
    candidate_mean: Optional[float]
    delta: Optional[float]  # candidate - baseline
    lower_is_better: bool
    significant: Optional[bool]  # None when either side has fewer than 2 values
    regressed: bool

    def describe(self) -> str:
        if self.delta is None:
            return f"{self.metric}: no data"
        arrow = "↑" if self.delta > 0 else "↓" if self.delta < 0 else "="
        note = "" if self.significant is None else (" (significant)" if self.significant else " (within noise)")
        return f"{self.metric}: {self.baseline_mean:.3f} → {self.candidate_mean:.3f} {arrow}{abs(self.delta):.3f}{note}"


class RegressionError(AssertionError):
    def __init__(self, regressions: list[MetricComparison], baseline: str, candidate: str) -> None:
        self.regressions = regressions
        lines = "\n".join(f"  - {r.describe()}" for r in regressions)
        super().__init__(f"{candidate} regressed against {baseline} on {len(regressions)} metric(s):\n{lines}")


def compare(
    results: list[dict],
    baseline: str,
    candidate: str,
    metrics: Iterable[str],
    by: str = "variant",
    tolerance: float = 0.0,
    lower_is_better: Optional[Iterable[str]] = None,
) -> list[MetricComparison]:
    """Compare two groups of runs (by default two variants) metric by metric.

    `results` is what run_experiment() returns, or what load_results() reads back from
    MLflow. A metric counts as regressed when the candidate is worse than the baseline
    by more than `tolerance` (absolute), unless repeats show the difference is within
    noise: with at least two values on each side, the change must also be significant
    at 95% (Welch's t-test). Run with `repeats` to get that check.

    Metrics named like latency/cost/tokens/errors are lower-is-better; pass
    `lower_is_better` to set the list yourself.
    """
    metrics = list(metrics)
    lower = set(lower_is_better) if lower_is_better is not None else {m for m in metrics if _LOWER_IS_BETTER.search(m)}
    groups = {row[by]: row for row in summarize(results, metrics, by=by)}
    for name in (baseline, candidate):
        if name not in groups:
            raise ValueError(f"No runs with {by}={name!r}. Found: {', '.join(map(str, groups)) or 'none'}")

    comparisons = []
    for metric in metrics:
        b, c = groups[baseline][metric], groups[candidate][metric]
        is_lower = metric in lower
        if b["mean"] is None or c["mean"] is None:
            comparisons.append(MetricComparison(metric, b["mean"], c["mean"], None, is_lower, None, False))
            continue
        delta = c["mean"] - b["mean"]
        significant = _significant(b, c)
        worse_by = delta if is_lower else -delta
        regressed = worse_by > tolerance and significant is not False
        comparisons.append(MetricComparison(metric, b["mean"], c["mean"], delta, is_lower, significant, regressed))
    return comparisons


def _significant(b: dict, c: dict) -> Optional[bool]:
    if b["n"] < 2 or c["n"] < 2:
        return None
    vb, vc = b["std"] ** 2 / b["n"], c["std"] ** 2 / c["n"]
    se = math.sqrt(vb + vc)
    if se == 0:
        return c["mean"] != b["mean"]
    # Welch-Satterthwaite degrees of freedom.
    df = (vb + vc) ** 2 / ((vb**2 / (b["n"] - 1) if vb else 0) + (vc**2 / (c["n"] - 1) if vc else 0))
    return abs(c["mean"] - b["mean"]) / se > _t_critical_95(max(1, int(df)))


def find_regressions(results: list[dict], baseline: str, candidate: str, metrics: Iterable[str], **kwargs) -> list[MetricComparison]:
    """The subset of compare() that regressed."""
    return [c for c in compare(results, baseline, candidate, metrics, **kwargs) if c.regressed]


def assert_no_regression(
    results: list[dict],
    baseline: str,
    candidate: str,
    metrics: Iterable[str],
    webhook_url: Optional[str] = None,
    **kwargs,
) -> list[MetricComparison]:
    """Raise RegressionError if `candidate` is worse than `baseline` on any metric, so a
    CI job or pytest test fails. With `webhook_url`, also post the regressions to a
    Slack-compatible incoming webhook before raising. Returns the full comparison
    when nothing regressed."""
    comparisons = compare(results, baseline, candidate, metrics, **kwargs)
    regressions = [c for c in comparisons if c.regressed]
    if regressions:
        if webhook_url:
            notify(webhook_url, regressions, baseline=baseline, candidate=candidate)
        raise RegressionError(regressions, baseline, candidate)
    return comparisons


def notify(
    webhook_url: str,
    regressions: list[MetricComparison],
    baseline: str,
    candidate: str,
    experiment_url: Optional[str] = None,
    timeout: float = 10.0,
) -> None:
    """Post a regression summary as `{"text": ...}`, the payload Slack, Mattermost and
    Rocket.Chat incoming webhooks accept."""
    lines = [f":warning: *{candidate}* regressed against *{baseline}* on {len(regressions)} metric(s)"]
    lines += [f"• {r.describe()}" for r in regressions]
    if experiment_url:
        lines.append(f"<{experiment_url}|Open in ExperimentLens>")
    request = urllib.request.Request(
        webhook_url,
        data=json.dumps({"text": "\n".join(lines)}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout):
        pass


def load_results(
    experiment_name: str,
    tracking_uri: Optional[str] = None,
    max_results: int = 5000,
) -> list[dict]:
    """Read an experiment's runs back from MLflow in the same shape run_experiment()
    returns: one dict per run with `variant`, `case_id`, `status` and every metric.
    Use it to compare against a baseline from an earlier session, e.g. in CI:

        results = load_results("rag-sweep")
        assert_no_regression(results, "prod_prompt", "new_prompt", ["eval_faithfulness"])
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"No MLflow experiment named {experiment_name!r}")

    runs = MlflowClient().search_runs([experiment.experiment_id], max_results=max_results)
    rows = []
    for run in runs:
        params = run.data.params
        rows.append({
            "variant": params.get("variant", run.info.run_name),
            "case_id": params.get("case_id"),
            "run_id": run.info.run_id,
            "status": "error" if run.info.status == "FAILED" or "error" in run.data.tags else "ok",
            **run.data.metrics,
        })
    return rows

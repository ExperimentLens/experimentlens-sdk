from __future__ import annotations

import hashlib
import json
import math
import statistics
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Iterable, Optional, Union

from .datasets import Dataset, link_run_to_item, load_dataset
from .tracking import ExperimentTracker, RunContext

RunCaseFn = Callable[[str, dict, dict, RunContext], dict]

# Two-sided 95% critical values of Student's t for 1..30 degrees of freedom.
_T_CRITICAL_95 = [
    12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
    2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
    2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042,
]


def run_experiment(
    tracker: ExperimentTracker,
    variants: dict[str, dict],
    dataset: Union[Iterable[dict], Dataset, str],
    run_case: RunCaseFn,
    case_id_key: str = "id",
    log_dataset: bool = False,
    repeats: int = 1,
    max_workers: int = 1,
    baseline: Optional[str] = None,
    run_unit: str = "case",
) -> list[dict]:
    """Loop every variant x every dataset case (x every repeat), dual-logging each as
    one MLflow run with a linked Langfuse trace.

    `run_unit="dataset"` logs one MLflow run per variant (x repeat) instead, holding
    the whole dataset: each case is its own Langfuse trace in the run's session, and
    the run's metrics are the cases' means (plus `latency_p95_ms`, `n_cases` and
    `error_rate`). Each case's own metrics go on its trace as NUMERIC scores, which is
    where ExperimentLens compares configurations case by case. `run_case` is called
    the same way, with a context aimed at the case's trace; `max_workers` then runs a
    variant's cases in parallel. Results stay one row per case, ordered variant,
    repeat, case. A run whose cases all fail is marked failed.

    `run_case(variant_name, variant_config, case, ctx)` does the actual work (call an
    LLM, run an agent step, score the output, call ctx.observe/log_generation/score as
    it goes) and returns a dict of metric_name -> value to log against the run.

    `repeats` runs every (variant, case) pair several times. LLM outputs vary from call
    to call, so with repeats the summary and the ExperimentLens Insights tab can show
    confidence intervals and tell a real difference between variants from noise. Each
    run is tagged `experimentlens.repeat` (a tag, not a param, so it isn't mistaken for
    a factor you varied).

    `max_workers` > 1 runs cases in parallel threads. LLM calls spend most of their
    time waiting on the network, so this usually cuts wall-clock time close to
    linearly; keep it within your provider's rate limits. Results are returned in the
    same order as the sequential loop.

    `log_dataset=True` additionally logs each run's single case via
    `ctx.log_dataset(...)` (an MLflow input dataset + a downloadable CSV/Parquet
    artifact) — off by default; call `ctx.log_dataset(...)` yourself from
    `run_case` for full control over what's logged, when, and in what format.

    `dataset` is a list of case dicts, a `Dataset` from `load_dataset()`, or the name
    of a dataset stored in Langfuse (loaded as it is now). With a stored dataset, every
    run is tagged `experimentlens.dataset` / `experimentlens.dataset_version`, so
    ExperimentLens can tell whether two experiments ran on the same items, and each
    trace is linked to its dataset item in Langfuse.

    `baseline` names the variant the UI compares the others against (default: one
    named "baseline", else the first alphabetically).

    A case that raises is recorded with status "error" (and its MLflow run marked
    failed) without stopping the rest of the experiment.
    """
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    if max_workers < 1:
        raise ValueError("max_workers must be at least 1")
    if run_unit not in ("case", "dataset"):
        raise ValueError('run_unit must be "case" or "dataset"')

    # Tell the UI which params group runs and identify cases, and what to compare against.
    tracker.set_conventions(group_param="variant", case_param="case_id", run_unit=run_unit)
    if baseline is not None:
        tracker.set_baseline(baseline)

    if isinstance(dataset, str):
        dataset = load_dataset(dataset, config=tracker.config)
    stored = dataset if isinstance(dataset, Dataset) else None
    dataset_tags = {}
    if stored is not None:
        dataset_tags = {
            "experimentlens.dataset": stored.name,
            "experimentlens.dataset_version": stored.version,
            "experimentlens.dataset_snapshot": stored.snapshot.isoformat(),
        }
        tracker.add_dataset_tag(stored.name)

    cases = list(dataset)

    if run_unit == "dataset":
        return _run_dataset_per_variant(
            tracker, variants, cases, run_case, case_id_key, log_dataset, repeats, max_workers, stored, dataset_tags,
        )

    jobs = [
        (variant_name, variant_config, case, repeat)
        for variant_name, variant_config in variants.items()
        for case in cases
        for repeat in range(repeats)
    ]

    def run_one(job: tuple[str, dict, dict, int]) -> dict:
        variant_name, variant_config, case, repeat = job
        case_id = case.get(case_id_key, "case")
        run_name = f"{variant_name}__{case_id}" + (f"__r{repeat}" if repeats > 1 else "")
        params = {"variant": variant_name, "case_id": str(case_id), **_flatten_params(variant_config)}
        tags = {"experimentlens.config_hash": config_hash(variant_config), **dataset_tags}
        if repeats > 1:
            tags["experimentlens.repeat"] = str(repeat)
        row = {"variant": variant_name, "case_id": case_id, "repeat": repeat}
        try:
            # The expected answer travels on the trace, so ExperimentLens can show it next
            # to the actual answer.
            expected = {"expected": case["expected"]} if case.get("expected") is not None else None
            with tracker.start_run(run_name, params=params, tags=tags, variant=variant_name, trace_metadata=expected) as ctx:
                if log_dataset:
                    ctx.log_dataset([case], name="case", context="evaluation")
                metrics = run_case(variant_name, variant_config, case, ctx) or {}
                ctx.log_metrics(metrics)
                item_id = stored.item_id(case) if stored is not None else None
                if item_id and tracker._langfuse is not None and ctx.root_span is not None:
                    link_run_to_item(
                        tracker.config,
                        run_name=f"{tracker.experiment_name} · {variant_name}",
                        item_id=item_id,
                        trace_id=ctx.root_span.trace_id,
                        metadata={"mlflow_run_id": ctx.run_id, "repeat": repeat},
                    )
            return {**row, "status": "ok", **metrics}
        except Exception as exc:
            return {**row, "status": "error", "error": str(exc)}

    if max_workers == 1:
        return [run_one(job) for job in jobs]
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="experimentlens") as pool:
        return list(pool.map(run_one, jobs))


# Keys of a result row that aren't metrics.
_ROW_KEYS = {"variant", "case_id", "repeat", "status", "error"}


def _run_dataset_per_variant(
    tracker: ExperimentTracker,
    variants: dict[str, dict],
    cases: list[dict],
    run_case: RunCaseFn,
    case_id_key: str,
    log_dataset: bool,
    repeats: int,
    max_workers: int,
    stored: Optional[Dataset],
    dataset_tags: dict,
) -> list[dict]:
    """run_experiment with run_unit="dataset": one run per variant (x repeat), one trace per case."""

    def run_variant(variant_name: str, variant_config: dict, repeat: int) -> list[dict]:
        run_name = variant_name + (f"__r{repeat}" if repeats > 1 else "")
        params = {"variant": variant_name, **_flatten_params(variant_config)}
        tags = {"experimentlens.config_hash": config_hash(variant_config), **dataset_tags}
        if repeats > 1:
            tags["experimentlens.repeat"] = str(repeat)
        rows: list[dict] = []
        try:
            with tracker.start_run(run_name, params=params, tags=tags, variant=variant_name, trace_per_case=True) as ctx:
                if log_dataset:
                    ctx.log_dataset(cases, name="dataset", context="evaluation")

                def run_one(case: dict) -> dict:
                    case_id = case.get(case_id_key, "case")
                    row = {"variant": variant_name, "case_id": case_id, "repeat": repeat}
                    metadata = {"case_id": case_id}
                    if case.get("expected") is not None:
                        metadata["expected"] = case["expected"]
                    try:
                        with ctx.case(f"{run_name}__{case_id}", metadata=metadata) as case_ctx:
                            case_ctx.log_metrics(run_case(variant_name, variant_config, case, case_ctx) or {})
                            item_id = stored.item_id(case) if stored is not None else None
                            if item_id and case_ctx.root_span is not None:
                                link_run_to_item(
                                    tracker.config,
                                    run_name=f"{tracker.experiment_name} · {variant_name}",
                                    item_id=item_id,
                                    trace_id=case_ctx.root_span.trace_id,
                                    metadata={"mlflow_run_id": ctx.run_id, "repeat": repeat},
                                )
                        return {**row, "status": "ok", **case_ctx.metrics}
                    except Exception as exc:
                        return {**row, "status": "error", "error": str(exc)}

                if max_workers == 1:
                    rows = [run_one(case) for case in cases]
                else:
                    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="experimentlens") as pool:
                        rows = list(pool.map(run_one, cases))

                ok = [row for row in rows if row["status"] == "ok"]
                if not ok:
                    raise RuntimeError(f"every case failed, e.g. {rows[0]['error']}" if rows else "the dataset is empty")
                ctx.log_metrics(_aggregate(ok, len(rows)))
        except Exception as exc:
            if not rows:
                rows = [
                    {"variant": variant_name, "case_id": case.get(case_id_key, "case"), "repeat": repeat,
                     "status": "error", "error": str(exc)}
                    for case in cases
                ]
        return rows

    return [
        row
        for variant_name, variant_config in variants.items()
        for repeat in range(repeats)
        for row in run_variant(variant_name, variant_config, repeat)
    ]


def _percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile, the same as the ExperimentLens overview's."""
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(p / 100 * len(ordered)) - 1))]


def _aggregate(rows: list[dict], n_cases: int) -> dict[str, float]:
    """A dataset run's metrics from its successful cases: each metric's mean, latency's
    p95 as well, how many cases ran and the share that failed."""
    values: dict[str, list[float]] = {}
    for row in rows:
        for name, value in row.items():
            if name not in _ROW_KEYS and isinstance(value, (int, float)) and math.isfinite(value):
                values.setdefault(name, []).append(float(value))
    metrics = {name: statistics.fmean(vals) for name, vals in values.items()}
    if "latency_ms" in values:
        metrics["latency_p95_ms"] = _percentile(values["latency_ms"], 95)
    metrics["n_cases"] = float(n_cases)
    metrics["error_rate"] = (n_cases - len(rows)) / n_cases
    return metrics


def _flatten_params(config: dict, prefix: str = "") -> dict:
    """MLflow params are flat strings: nested dicts become dotted keys
    ({"retriever": {"k": 5}} -> {"retriever.k": "5"}) and lists become JSON."""
    flat: dict[str, str] = {}
    for key, value in config.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten_params(value, prefix=f"{name}."))
        elif isinstance(value, (list, tuple)):
            flat[name] = json.dumps(value, default=str)
        else:
            flat[name] = str(value)
    return flat


def config_hash(config: dict) -> str:
    """Short stable hash of a variant config, so runs of an identical configuration can
    be grouped even when the variant names differ."""
    canonical = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def _t_critical_95(df: int) -> float:
    if df <= 0:
        return math.nan
    if df <= len(_T_CRITICAL_95):
        return _T_CRITICAL_95[df - 1]
    # Normal approximation with a small-sample correction; within 0.1% for df > 30.
    z = 1.959964
    return z + (z**3 + z) / (4 * df)


def summarize(
    results: list[dict],
    metric_names: list[str],
    by: str = "variant",
) -> list[dict]:
    """Per-group statistics for each metric: mean, standard deviation, 95% confidence
    interval and the number of successful runs, e.g. after run_experiment().

    Returns one dict per group, such as
    {"variant": "concise", "n": 10, "errors": 0,
     "latency_ms": {"mean": ..., "std": ..., "ci_low": ..., "ci_high": ..., "n": 10}}.
    The interval needs at least two values; with one, `ci_low`/`ci_high` are None.
    """
    groups: dict[Any, list[dict]] = {}
    for row in results:
        groups.setdefault(row.get(by), []).append(row)

    summary = []
    for group, rows in groups.items():
        entry: dict[str, Any] = {
            by: group,
            "n": sum(1 for r in rows if r.get("status", "ok") == "ok"),
            "errors": sum(1 for r in rows if r.get("status") == "error"),
        }
        for metric in metric_names:
            # bool counts too (it is an int), so pass/fail metrics average to a pass rate.
            values = [
                float(r[metric])
                for r in rows
                if isinstance(r.get(metric), (int, float)) and not math.isnan(r[metric])
            ]
            entry[metric] = _describe(values)
        summary.append(entry)
    return summary


def _describe(values: list[float]) -> dict[str, Optional[float]]:
    n = len(values)
    if n == 0:
        return {"mean": None, "std": None, "ci_low": None, "ci_high": None, "n": 0}
    mean = statistics.fmean(values)
    if n == 1:
        return {"mean": mean, "std": 0.0, "ci_low": None, "ci_high": None, "n": 1}
    std = statistics.stdev(values)
    half = _t_critical_95(n - 1) * std / math.sqrt(n)
    return {"mean": mean, "std": std, "ci_low": mean - half, "ci_high": mean + half, "n": n}


def print_summary(results: list[dict], metric_names: list[str], by: str = "variant") -> None:
    """Print a per-variant table of mean ± 95% CI half-width for the given metrics,
    e.g. after run_experiment(). Overlapping intervals mean the variants are not
    clearly different; add `repeats` to narrow them."""
    rows = summarize(results, metric_names, by=by)

    def cell(stats: dict) -> str:
        if stats["mean"] is None:
            return "—"
        if stats["ci_low"] is None:
            return f"{stats['mean']:.3f}"
        return f"{stats['mean']:.3f} ± {stats['mean'] - stats['ci_low']:.3f}"

    width = max([16, *(len(m) + 2 for m in metric_names)])
    header = f"{by:<24}" + "".join(f"{m:>{width}}" for m in metric_names) + f"{'n':>6}{'errors':>8}"
    print(header)
    print("-" * len(header))
    for row in rows:
        line = f"{str(row[by]):<24}" + "".join(f"{cell(row[m]):>{width}}" for m in metric_names)
        print(line + f"{row['n']:>6}{row['errors']:>8}")
    print("(mean ± 95% confidence interval)")

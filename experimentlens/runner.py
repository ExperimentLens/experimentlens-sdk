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
) -> list[dict]:
    """Loop every variant x every dataset case (x every repeat), dual-logging each as
    one MLflow run with a linked Langfuse trace.

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

    A case that raises is recorded with status "error" (and its MLflow run marked
    failed) without stopping the rest of the experiment.
    """
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    if max_workers < 1:
        raise ValueError("max_workers must be at least 1")

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

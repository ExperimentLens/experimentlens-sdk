"""What the SDK declares about an experiment's metrics and params, so the ExperimentLens
UI reads it instead of guessing from names. Stored as MLflow experiment tags:

    experimentlens.metric.<name>  {"pass": 0.75, "direction": "maximize", "kind": "safety"}
    experimentlens.group_param    param that names a configuration ("variant")
    experimentlens.case_param     param that names a dataset case ("case_id")
    experimentlens.baseline       configuration others are compared against
"""

from __future__ import annotations

import re

METRIC_TAG_PREFIX = "experimentlens.metric."

# Metrics named like this are declared as cost metrics (lower is better) when logged.
COST_METRIC = re.compile(r"latency|duration|cost|tokens|_ms$|_s$", re.IGNORECASE)

# The same name-based guess as the UI (vis-frontend run-statistics.ts inferDirection), used
# only for metrics nobody declared. Checked in order: unmistakable "bad" words, then
# quality words, then cost-like words.
_ALWAYS_MINIMIZE = re.compile(
    r"(loss|error|err$|_err_|mse|mae|rmse|mape|perplexity|wer|cer|fpr|fnr|toxicity|hallucination)", re.IGNORECASE
)
_MAXIMIZE = re.compile(
    r"(f1|accuracy|precision|recall|auc|score|pass|hit|mrr|ndcg|coverage|correct|grounded|faithful|relevan"
    r"|success|bleu|rouge|exact_match|truth)",
    re.IGNORECASE,
)
_MINIMIZE = re.compile(r"(latency|duration|time|_ms$|_s$|seconds|cost|token)", re.IGNORECASE)


def infer_direction(metric: str) -> str:
    """"maximize" or "minimize", guessed from the metric's name."""
    # Eval scores are 0..1 and higher is better, also for safety (eval_toxicity = 1 means none found).
    if metric.startswith("eval_"):
        return "maximize"
    if _ALWAYS_MINIMIZE.search(metric):
        return "minimize"
    if _MAXIMIZE.search(metric):
        return "maximize"
    return "minimize" if _MINIMIZE.search(metric) else "maximize"

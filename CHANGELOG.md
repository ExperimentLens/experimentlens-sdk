# Changelog

## 0.4.0

### Added
- Datasets stored in Langfuse: `upload_dataset(name, cases)` creates or updates the
  items, `load_dataset(name, version=None)` reads them (optionally as they were at a
  point in time), and `run_experiment(..., dataset="name")` runs on a stored dataset.
  Runs are tagged `experimentlens.dataset` / `experimentlens.dataset_version` (a
  content fingerprint) / `experimentlens.dataset_snapshot`; the experiment gets
  `experimentlens.datasets`; each trace is linked to its dataset item.
- `examples/dataset_experiment.py`.

### Changed
- The judge prompt sent to an LLM judge is logged on its `judge_<name>` trace step.

## 0.3.0

### Added
- Built-in evaluation metrics: `ctx.evaluate([...], question=, answer=, context=, expected=)`
  scores a run on `faithfulness`, `answer_relevance`, `context_relevance`,
  `correctness`, `toxicity`, `bias`, `prompt_injection` (LLM judges on a 1-5 rubric)
  and `pii_leakage` (rule-based). Scores are 0-1, higher is better, logged as
  `eval_<name>` MLflow metrics and as `judge_<name>` verdicts on the trace.
  `evaluate(...)` computes scores without logging; `Metric(...)` defines your own.
- Regression checks: `compare`, `find_regressions` and `assert_no_regression`
  compare a candidate variant against a baseline, account for noise when runs are
  repeated (Welch's t-test), and raise `RegressionError` for CI. `notify` posts a
  summary to a Slack-compatible webhook; `load_results` reads past runs from MLflow.

## 0.2.0

### Added
- `run_experiment(..., repeats=N)` runs every (variant, case) pair N times, so
  results come with confidence intervals. Each run is tagged `experimentlens.repeat`.
- `run_experiment(..., max_workers=N)` runs cases in parallel threads. Results keep
  the sequential order.
- `summarize(results, metric_names)` returns mean, standard deviation and a 95%
  confidence interval per variant; `print_summary` now shows mean ± CI and error counts.
- `config_hash(config)` and an `experimentlens.config_hash` tag on every run, to
  group identical configurations.
- `ExperimentTracker` works as a context manager and closes itself.
- Every run is tagged with `experimentlens.sdk_version`.
- Test suite (`pip install -e ".[dev]" && pytest`).

### Changed
- Nested variant configs are logged as dotted params (`retriever.k`) and lists as
  JSON, instead of being dropped.
- A failed run's `error` tag includes the exception type and is capped at 5000
  characters, so an oversized message no longer hides the original error.

## 0.1.0

- First release: MLflow + Langfuse logging in the shape ExperimentLens reads,
  LLM-as-judge, LiteLLM calls and explainability artifacts.

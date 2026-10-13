# Changelog

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

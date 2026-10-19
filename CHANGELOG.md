# Changelog

## 0.6.0

### Added
- Multi-turn conversations. `with ctx.turn(message) as turn: ... turn.reply(text)` records
  one turn as a `turn N` span (metadata `turn: N`) in the run's trace, and
  `ctx.messages(system_prompt)` returns the history so far as chat messages for `call_llm`.
- Conversation-level evals: `CONVERSATION_METRICS` (conversation_relevance,
  knowledge_retention, conversation_completeness, role_adherence), `evaluate_conversation()`
  and `ctx.evaluate_conversation(metrics, role=...)`. Scores are 0..1, logged as
  `eval_<name>` with `judge_<name>` verdicts on the trace, like `ctx.evaluate()`. The role
  is saved as the run tag `experimentlens.conversation_role`, so the UI can re-run
  role_adherence with the same text.
- `examples/conversation_experiment.py`: a support bot with and without conversation history.

## 0.5.0

### Added
- The SDK now tells ExperimentLens what its metrics and params mean, as experiment tags,
  so the UI stops keeping its own copy of these rules:
  - `ctx.evaluate()` records each `eval_*` metric's pass mark (including custom
    `thresholds`), direction and kind (`quality` / `safety`).
  - Metrics named like latency/cost/tokens are recorded as cost metrics when first
    logged, so they appear in the UI's baseline comparison.
  - `tracker.describe_metric(name, pass_mark=, direction=, kind=)` declares any other
    metric, e.g. one whose name would be guessed the wrong way.
  - `run_experiment(..., baseline="v1")` and `tracker.set_baseline()` set the variant
    the UI compares against; `tracker.set_conventions(group_param=, case_param=)`
    names the params for runs logged without `run_experiment`.

### Changed
- The regression checks guess "lower is better" with the same rules as the UI.

## 0.4.1

### Fixed
- An exception raised inside `ctx.observe(...)` now reaches the caller unchanged. It was
  replaced by `RuntimeError: generator didn't stop after throw()`, so a failed run's
  `error` tag hid the real cause (e.g. a `TimeoutError`).

### Added
- `run_experiment` puts each case's `expected` answer on its trace's metadata, so
  ExperimentLens can show it next to the actual answer. `start_run(...,
  trace_metadata={...})` adds any other trace metadata.

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

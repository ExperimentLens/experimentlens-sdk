# experimentlens

The Python SDK for [ExperimentLens](https://experimentlens.imsi.athenarc.gr/), a
visual dashboard for exploring, monitoring and explaining AI experiments.

Use it to log **ML experiments** and **LLM/agent experiments** to MLflow and
Langfuse in the exact shape ExperimentLens reads. Runs, metrics, traces, judge
checks and explainability data then appear in the dashboard, with no hand-written
MLflow + Langfuse plumbing.

- Website: https://experimentlens.imsi.athenarc.gr/
- Documentation: https://experimentlens.imsi.athenarc.gr/docs
- ExperimentLens source: https://github.com/ExperimentLens

## What it does for you

| | ML Experiments | Agents & LLMs |
|---|---|---|
| Creates the MLflow experiment and tags it (`experiment_type`) | ✓ | ✓ |
| Logs params, metrics, datasets and artifacts per run | ✓ | ✓ |
| Logs the `explainability/` files the explainability views read | ✓ | |
| Opens one Langfuse trace per MLflow run and links them both ways | | ✓ |
| Traces LLM calls, tools and pipeline steps; LLM-as-judge checks | | ✓ |
| Built-in quality and safety metrics (faithfulness, toxicity, PII, ...) | | ✓ |
| Fails CI and alerts when a variant regresses against a baseline | ✓ | ✓ |
| Versioned test datasets stored in Langfuse, linked to each run | | ✓ |
| Calls any LLM provider through one function (via LiteLLM) | | ✓ |

## Requirements

- Python 3.10+
- A running ExperimentLens stack. Follow
  [Build the app](https://experimentlens.imsi.athenarc.gr/docs) in the docs:
  `docker compose up` starts ExperimentLens (http://localhost:5173) together with
  MLflow (http://localhost:5000) and Langfuse (http://localhost:3000).
- For LLM experiments: a model to call. The default is a local
  [Ollama](https://ollama.com) model (`ollama pull llama3.2`); any
  [LiteLLM-supported](https://docs.litellm.ai/docs/providers) provider works.

## Install

```bash
pip install experimentlens
```

## Configure

Create a `.env` file in your project. The SDK finds it in the current folder,
any folder above it, or next to the script you run:

```bash
MLFLOW_TRACKING_URI=http://127.0.0.1:5000
# The SAME key pair as in ExperimentLens' root .env, so traces land in the
# Langfuse project ExperimentLens reads.
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=http://localhost:3000
```

All settings:

| Variable | Default | Purpose |
|---|---|---|
| `MLFLOW_TRACKING_URI` | `http://127.0.0.1:5000` | MLflow server ExperimentLens reads from |
| `MLFLOW_TRACKING_TOKEN` | unset | Token, if your MLflow server requires one |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | unset (tracing off) | Langfuse credentials |
| `LANGFUSE_HOST` (or `LANGFUSE_BASE_URL`) | `http://localhost:3000` | Langfuse instance |
| `LANGFUSE_PROJECT_ID` | looked up from the keys | Langfuse project to link experiments to |
| `EXPERIMENTLENS_MODEL` (or `LLM_MODEL`) | `ollama/llama3.2` | Default model for `call_llm` |
| `EXPERIMENTLENS_JUDGE_MODEL` | `EXPERIMENTLENS_MODEL` | Default model for LLM-as-judge |
| `OLLAMA_API_BASE` | `http://localhost:11434` | Where Ollama runs |

You can also configure in code, which takes precedence over the environment:

```python
from experimentlens import ExperimentTracker, SDKConfig

config = SDKConfig.from_env(mlflow_tracking_uri="http://mlflow.example.org:5000")
tracker = ExperimentTracker("my-experiment", config=config)
```

## Quickstart: an LLM experiment

Compare two system prompts on a small dataset. Each (variant, question) pair
becomes one MLflow run with its own linked Langfuse trace.

```python
from experimentlens import ExperimentTracker, call_llm, messages_to_prompt, print_summary, run_experiment

VARIANTS = {
    "short": {"system_prompt": "Answer in one sentence."},
    "long": {"system_prompt": "Explain your reasoning."},
}
DATASET = [
    {"id": "q1", "question": "What is the capital of France?"},
    {"id": "q2", "question": "What is 12 * 12?"},
]

def run_case(variant_name, variant, case, ctx):
    messages = [
        {"role": "system", "content": variant["system_prompt"]},
        {"role": "user", "content": case["question"]},
    ]
    # The trace view shows `input.prompt` as the generation's prompt.
    with ctx.observe("llm_call", as_type="generation", input={"prompt": messages_to_prompt(messages)}) as obs:
        result = call_llm(messages)
        obs.update(output=result.text)

    verdict = ctx.log_judge(
        "correct",
        question=case["question"],
        answer=result.text,
        criteria="The answer is factually correct.",
    )
    ctx.set_trace_io(input={"question": case["question"]}, output={"answer": result.text})

    # Returned values are logged as MLflow metrics on the run.
    return {"judge_passed": float(verdict.passed), "latency_ms": result.latency_ms}

# experiment_type="llm" by default; the `with` block flushes traces at the end.
with ExperimentTracker("prompt-ab-test") as tracker:
    results = run_experiment(tracker, VARIANTS, DATASET, run_case, repeats=5, max_workers=4)
print_summary(results, metric_names=["judge_passed", "latency_ms"])
```

`print_summary` shows each metric as mean ± 95% confidence interval per variant.
If the intervals of two variants overlap, the difference may just be noise.

Open ExperimentLens, go to **Agents & LLMs** and select `prompt-ab-test`. The
**Insights** tab tests which params really changed each metric and whether one
variant is significantly better than another.

### Repeats and parallel runs

- `repeats=N` runs every (variant, case) pair N times. LLM answers change from
  call to call, so a single run per pair can't tell a better variant from a lucky
  one. Repeats are tagged `experimentlens.repeat` on each run.
- `max_workers=N` runs N cases at once in threads. LLM calls mostly wait on the
  network, so this saves a lot of time. Stay within your provider's rate limits.
- Nested variant configs are logged as dotted params:
  `{"retriever": {"k": 5}}` becomes `retriever.k = 5`.

## Built-in evaluation metrics

`ctx.evaluate(...)` scores a run's output on ready-made metrics. Every score is
between 0 and 1, and higher is always better: for the safety metrics, 1.0 means
nothing harmful was found.

```python
def run_case(variant_name, variant, case, ctx):
    chunks = retrieve(case["question"], k=variant["top_k"])
    answer = generate(case["question"], chunks)

    report = ctx.evaluate(
        ["faithfulness", "answer_relevance", "correctness", "toxicity", "pii_leakage"],
        question=case["question"], answer=answer, context=chunks, expected=case["expected"],
    )
    return {"latency_ms": ..., **report.metrics()}  # eval_faithfulness, eval_toxicity, ...
```

| Metric | Needs | Kind | Passes at |
|---|---|---|---|
| `faithfulness` | answer, context | LLM judge | 0.5 |
| `answer_relevance` | question, answer | LLM judge | 0.5 |
| `context_relevance` | question, context | LLM judge | 0.5 |
| `correctness` | question, answer, expected | LLM judge | 0.5 |
| `toxicity` | answer | LLM judge, safety | 0.75 |
| `bias` | answer | LLM judge, safety | 0.75 |
| `prompt_injection` | question | LLM judge, safety | 0.75 |
| `pii_leakage` | answer | Rule-based (emails, phones, cards, IBANs, SSNs), safety | 1.0 |

LLM judges rate on a 1–5 rubric, normalized to 0–1, using
`EXPERIMENTLENS_JUDGE_MODEL`. Pick a stronger or different model than the one under
test. Change a pass mark with `thresholds={"faithfulness": 0.75}`.

What gets logged:
- each score as an MLflow metric `eval_<name>`, so you can compare runs and check
  them for regressions;
- LLM-judged metrics as `judge_<name>` verdicts on the trace, which the Judges
  views already show;
- rule-based metrics as boolean checker scores.

Add your own metric with `Metric(...)`: give it a `criterion` and `scale` for an
LLM judge, or a `check` function for a rule. `evaluate(...)` computes scores
without logging them.

## Catch regressions

Compare a candidate variant against a baseline and fail when it got worse:

```python
from experimentlens import assert_no_regression, load_results

results = run_experiment(tracker, VARIANTS, DATASET, run_case, repeats=5)
assert_no_regression(
    results, baseline="prod_prompt", candidate="new_prompt",
    metrics=["eval_faithfulness", "eval_answer_relevance", "latency_ms"],
    webhook_url=os.environ.get("SLACK_WEBHOOK_URL"),  # optional alert
)
```

- **Direction:** metrics named like latency, cost, tokens or errors count as lower
  is better. Everything else counts as higher is better. Override this with
  `lower_is_better=[...]`.
- **Noise:** with repeats, a drop only counts when it is statistically significant
  at 95%. With a single run per variant, any drop larger than `tolerance` counts.
- **Failure:** `RegressionError` is raised, so a CI job or pytest test fails.
  With `webhook_url`, a summary is also posted to a Slack-compatible webhook
  first.
- **Earlier baselines:** `load_results("experiment-name")` reads runs back from
  MLflow, so you can compare against a baseline logged in an earlier session.
- **Just looking:** `compare(...)` and `find_regressions(...)` return the per-metric
  comparison without raising.

## Datasets

Store a test set once and run every experiment on it. Datasets live in Langfuse,
next to the traces.

```python
from experimentlens import load_dataset, run_experiment, upload_dataset

upload_dataset("capitals-golden", [
    {"id": "fr", "question": "What is the capital of France?", "expected": "Paris"},
    {"id": "au", "question": "What is the capital of Australia?", "expected": "Canberra"},
])

with ExperimentTracker("capitals-on-dataset") as tracker:
    run_experiment(tracker, VARIANTS, "capitals-golden", run_case)
```

- **Items:** each case's `expected` becomes the item's expected output, and the rest
  becomes its input. Cases are keyed by `id`, so uploading a case again updates it.
- **Versions:** every run is tagged with `experimentlens.dataset` and
  `experimentlens.dataset_version`, a fingerprint of the items' content.
  ExperimentLens compares two experiments cleanly only when the versions match.
- **Reproducing:** `dataset.snapshot` is when the items were read.
  `load_dataset(name, version=dataset.snapshot)` returns exactly those items again.
- **Links:** each trace is linked to its dataset item, so the experiment also shows
  up as a dataset run in Langfuse.

## Quickstart: an ML experiment with explainability

```python
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from experimentlens import ExperimentTracker

X, y = load_iris(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

tracker = ExperimentTracker("iris-rf", experiment_type="ml")  # MLflow only, no Langfuse
for n_estimators in (50, 100, 200):
    with tracker.start_run(f"rf-{n_estimators}", params={"n_estimators": n_estimators}) as ctx:
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42).fit(X_train, y_train)
        predictions = model.predict(X_test)
        ctx.log_metrics({"accuracy": accuracy_score(y_test, predictions)})

        # X_train.csv, X_test.csv, Y_train.csv, Y_test.csv, Y_pred.csv, model.pkl and
        # roc_data.json under the run's explainability/ folder.
        ctx.log_explainability_data(
            X_train=X_train, X_test=X_test,
            y_train=y_train, y_test=y_test, y_pred=predictions,
            model=model,
            y_score=model.predict_proba(X_test),
        )
tracker.close()
```

Open ExperimentLens, go to **ML Experiments** and select `iris-rf`.

## How runs are linked in ExperimentLens

The SDK applies the conventions ExperimentLens uses, so you don't have to:

| Where | What | Why |
|---|---|---|
| MLflow experiment tag | `experiment_type` = `llm` or `ml` | Which area of ExperimentLens shows the experiment |
| MLflow experiment tag | `langfuse.project_id`, `langfuse.user_id` | Which Langfuse project and traces belong to it |
| Langfuse trace | `user_id` = MLflow experiment id, `session_id` = MLflow run id | Links each trace to its run |
| MLflow run tag | `langfuse.session_id`, `langfuse.trace_id`, `langfuse.trace_url` | Opens the matching trace from the run |
| Trace input / output | `question` / `answer` keys (via `ctx.set_trace_io`) | Shown at the top of the trace view |
| Judge observations | named `judge_<name>`, output `{answer, passed, rationale}` | Shown as judge checks on the trace |
| Run artifacts | `explainability/` folder | Powers the explainability views |
| MLflow run tag | `experimentlens.repeat`, `experimentlens.config_hash`, `experimentlens.sdk_version` | Groups repeats and identical configs; records which SDK logged the run |

## API overview

**`ExperimentTracker(name, experiment_type="llm", config=None)`** creates or
reuses an MLflow experiment. `tracker.start_run(run_name, params=..., tags=...)`
is a context manager yielding a `RunContext`. Call `tracker.close()` at the end
to flush traces.

**`run_experiment(tracker, variants, dataset, run_case, repeats=1, max_workers=1)`**
runs `run_case` for every variant, dataset item and repeat, one MLflow run each,
and logs the metrics it returns. A case that raises is recorded with status
`error` and the rest continue. `summarize(results, metric_names)` returns the
mean, standard deviation and 95% confidence interval per variant, and
`print_summary(results, metric_names)` prints them as a table.

`ExperimentTracker` can be used as a context manager (`with ExperimentTracker(...) as tracker:`),
which calls `close()` for you.

**`RunContext`** (the `ctx` in a run):

- `observe(name, as_type="span" | "generation" | "tool" | "retriever" | ...)`:
  context manager that traces one real step with accurate timing.
- `log_generation(...)`, `log_tool_call(...)`, `log_span(...)`, `log_event(...)`:
  record steps that already happened. They show about 0 ms in the trace view,
  so prefer `observe()` when timing matters.
- `log_judge(name, question=, answer=, criteria=, model=None)`: LLM-as-judge
  check, logged as a judge observation plus a boolean trace score. Returns
  `JudgeResult(passed, rationale, llm_result)`.
- `evaluate(metrics, question=, answer=, context=, expected=, model=None, thresholds=None)`:
  score the output on built-in or custom metrics and log them. Returns an
  `EvalReport`; `report.metrics()` gives `{"eval_<name>": score}`.
- `score(name, value, data_type="NUMERIC")`: attach a score to the trace.
- `set_trace_io(input=, output=)`: trace-level input and output.
- `log_metrics`, `log_params`, `set_tags`, `log_artifact_dict`: MLflow logging
  for the current run.
- `log_dataset(data, name=, context=, targets=, file_format="csv")`: log a list of
  dicts or a DataFrame as an MLflow input dataset plus a downloadable file.
- `log_explainability_data(X_train=, X_test=, y_train=, y_test=, y_pred=, model=,
  y_score=, roc_data=, extra=)`: log the explainability files; pass whichever
  you have.

**`call_llm(messages, model=None, **kwargs)`** calls any LiteLLM model and
returns an `LLMResult` with `text`, token counts, `latency_ms` and `cost_usd`.
`llm_judge(...)` is the judge without logging.

**Regressions:** `compare`, `find_regressions` and `assert_no_regression(results,
baseline, candidate, metrics, webhook_url=None, tolerance=0.0)` compare two
variants. `load_results(experiment_name)` reads past runs from MLflow, and
`notify(webhook_url, ...)` posts a summary to a webhook.

## Examples

The source distribution includes runnable examples in `examples/`:

- `quickstart_experiment.py`: prompt A/B test with LLM-as-judge.
- `rag_experiment.py`: retrieval + generation traced as separate steps.
- `ml_experiment.py`: hyperparameter grid with explainability data
  (`pip install "experimentlens[examples]"` for scikit-learn).

## Troubleshooting

- **Runs appear but no traces.** Look for a `Langfuse disabled` warning: the
  keys were not found. Check that your `.env` is in your project folder, or
  pass `SDKConfig.from_env(env_file="path/to/.env")`.
- **Traces exist in Langfuse but ExperimentLens does not show them.** The SDK
  must use the same Langfuse key pair as ExperimentLens' root `.env`, and the
  same MLflow server.

## Development

```bash
git clone <this repository> && cd experimentlens
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
pytest
```

The tests use a throwaway local MLflow database, so they need no running
ExperimentLens, MLflow or Langfuse.

## License

Apache License 2.0, the same license as the rest of ExperimentLens. The full text is in the `LICENSE` file shipped with the package.

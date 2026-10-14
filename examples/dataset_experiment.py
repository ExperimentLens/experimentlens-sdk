"""Dataset example: store a test set in Langfuse once, then run experiments on it.

  1. upload_dataset(...) creates the dataset (or updates its items) in Langfuse
  2. run_experiment(..., dataset="capitals-golden") loads it and tags every run with
     the dataset name and version, so ExperimentLens only compares experiments that
     ran on the same items
  3. each trace is linked to its dataset item, so the run also shows up under the
     dataset in Langfuse

Prerequisites: same as examples/quickstart_experiment.py.

Run with:
    python examples/dataset_experiment.py
"""
from experimentlens import ExperimentTracker, call_llm, messages_to_prompt, print_summary, run_experiment, upload_dataset

MODEL = "ollama/llama3.2"

CASES = [
    {"id": "fr", "question": "What is the capital of France?", "expected": "Paris"},
    {"id": "au", "question": "What is the capital of Australia?", "expected": "Canberra"},
    {"id": "ca", "question": "What is the capital of Canada?", "expected": "Ottawa"},
    {"id": "br", "question": "What is the capital of Brazil?", "expected": "Brasília"},
]

VARIANTS = {
    "one_word": {"system_prompt": "Answer with the city name only."},
    "explain": {"system_prompt": "Answer in one sentence and say why the city is the capital."},
}


def run_case(variant_name, variant, case, ctx):
    messages = [
        {"role": "system", "content": variant["system_prompt"]},
        {"role": "user", "content": case["question"]},
    ]
    with ctx.observe("answer", as_type="generation", input={"prompt": messages_to_prompt(messages)}, model=MODEL) as obs:
        result = call_llm(messages, model=MODEL)
        obs.update(output=result.text)

    report = ctx.evaluate(["correctness", "answer_relevance"], question=case["question"], answer=result.text, expected=case["expected"], model=MODEL)
    ctx.set_trace_io(input={"question": case["question"]}, output={"answer": result.text})
    return {"latency_ms": result.latency_ms, **report.metrics()}


def main():
    dataset = upload_dataset("capitals-golden", CASES, description="World capitals with expected answers")
    print(f"Dataset {dataset.name}: {len(dataset)} items, version {dataset.version}")

    with ExperimentTracker("capitals-on-dataset") as tracker:
        results = run_experiment(tracker, VARIANTS, "capitals-golden", run_case, max_workers=2)
    print_summary(results, metric_names=["eval_correctness", "eval_answer_relevance", "latency_ms"])


if __name__ == "__main__":
    main()

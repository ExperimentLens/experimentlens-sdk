"""Dataset sweep: the RAG config sweep with one run per configuration.

examples/config_sweep_experiment.py logs one MLflow run per configuration x question.
This one logs one run per configuration, which answers the whole dataset inside it:
every question is its own Langfuse trace in the run's session, the run's metrics are
the means over the questions (plus latency_p95_ms, n_cases and error_rate), and each
question's own metrics are scores on its trace, where ExperimentLens compares the
configurations question by question.

The per-question code is config_sweep_experiment.run_case, unchanged; the only switch
is `run_experiment(..., run_unit="dataset")`. The grid is smaller and the dataset
larger, since a configuration's score is now a mean over the questions:

  - top_k         1, 3
  - temperature   0.0, 1.0
  - prompt_style  grounded, detailed
  - max_tokens    48, 256

That is 16 configurations x 20 questions = 16 runs and 320 traces:

    python examples/dataset_sweep_experiment.py
    python examples/dataset_sweep_experiment.py --configs "k1_t0.0_grounded_m48,k3_t0.0_grounded_m48"

Prerequisites: same as examples/eval_showcase_experiment.py (Ollama with llama3.2,
MLflow and Langfuse). The full grid takes about 30-45 minutes on a laptop.
"""
import argparse
import itertools

from config_sweep_experiment import METRICS, run_case, variant_name
from experimentlens import ExperimentTracker, print_summary, run_experiment

EXPERIMENT = "rag-dataset-sweep"

GRID = {
    "top_k": [1, 3],
    "temperature": [0.0, 1.0],
    "prompt_style": ["grounded", "detailed"],
    "max_tokens": [48, 256],
}

# Every answer is in config_sweep_experiment.CORPUS; some need the right chunk retrieved
# among similar ones (largest city vs capital, two Tokyo facts, two Athens facts).
DATASET = [
    {"id": "fr", "question": "What is the capital of France?", "expected": "Paris"},
    {"id": "seine", "question": "Which river does Paris lie on?", "expected": "The Seine"},
    {"id": "jp", "question": "What is the capital of Japan?", "expected": "Tokyo"},
    {"id": "jp_city", "question": "What is the most populous city in Japan?", "expected": "Tokyo"},
    {"id": "au", "question": "What is the capital of Australia?", "expected": "Canberra (not Sydney)"},
    {"id": "au_built", "question": "In which year was Canberra purpose-built as the capital?", "expected": "1913"},
    {"id": "au_largest", "question": "What is the largest city in Australia?", "expected": "Sydney"},
    {"id": "ca", "question": "What is the capital of Canada?", "expected": "Ottawa (not Toronto)"},
    {"id": "ca_province", "question": "In which province is Ottawa?", "expected": "Ontario"},
    {"id": "ca_largest", "question": "What is the largest city in Canada?", "expected": "Toronto"},
    {"id": "gr", "question": "What is the capital of Greece?", "expected": "Athens"},
    {"id": "parthenon", "question": "When was the Parthenon completed?", "expected": "432 BC"},
    {"id": "oldest", "question": "Which capital is one of the oldest cities in the world?", "expected": "Athens"},
    {"id": "eiffel", "question": "When was the Eiffel Tower completed?", "expected": "1889"},
    {"id": "fuji", "question": "What is the tallest mountain in Japan?", "expected": "Mount Fuji"},
    {"id": "fuji_near", "question": "Which city is Mount Fuji near?", "expected": "Tokyo"},
    {"id": "de", "question": "What is the capital of Germany?", "expected": "Berlin"},
    {"id": "wall", "question": "In which year did the Berlin Wall fall?", "expected": "1989"},
    {"id": "br", "question": "What is the capital of Brazil and when did it become the capital?", "expected": "Brasilia, in 1960"},
    {"id": "br_before", "question": "Which city was the capital of Brazil before Brasilia?", "expected": "Rio de Janeiro"},
]

ALL_CONFIGS = {
    variant_name(config): config
    for config in (dict(zip(GRID, values)) for values in itertools.product(*GRID.values()))
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--configs", help="comma-separated variant names to run instead of the whole grid")
    parser.add_argument("--workers", type=int, default=4, help="questions answered in parallel within a run")
    args = parser.parse_args()

    names = sorted(ALL_CONFIGS)
    if args.configs:
        names = [name.strip() for name in args.configs.split(",") if name.strip()]
        unknown = [name for name in names if name not in ALL_CONFIGS]
        if unknown:
            parser.error(f"unknown configurations: {', '.join(unknown)}")
    variants = {name: ALL_CONFIGS[name] for name in names}

    with ExperimentTracker(EXPERIMENT) as tracker:
        results = run_experiment(tracker, variants, DATASET, run_case, max_workers=args.workers, run_unit="dataset")

    print_summary(results, metric_names=[f"eval_{m}" for m in METRICS] + ["total_tokens", "latency_ms"])


if __name__ == "__main__":
    main()

"""Config sweep: a RAG grid for the ExperimentLens Config Advisor.

The eval showcase compares six hand-made variants. This sweeps four knobs instead,
so the Advisor (Overview -> Optimize) has enough configurations to find the
quality/cost frontier and to predict configurations nobody ran yet:

  - top_k         1, 2, 3, 4      chunks of context (more context = more prompt tokens)
  - temperature   0.0, 0.7, 1.2
  - prompt_style  grounded, concise, detailed
  - max_tokens    48, 256         cap on the answer length

That is 72 configurations. A round runs a random sample of them on every case, so the
Advisor's "try next" suggestions are configurations that really were left out. Run a
second round with the ones it suggests to check its predictions:

    python examples/config_sweep_experiment.py --sample 24
    python examples/config_sweep_experiment.py --configs "k2_t0.0_concise_m48,k3_t0.0_grounded_m48"

Prerequisites: same as examples/eval_showcase_experiment.py (Ollama with llama3.2,
MLflow and Langfuse). A 24-config round is 192 runs and takes about 15-25 minutes on
a laptop.
"""
import argparse
import itertools
import random

from experimentlens import ExperimentTracker, call_llm, messages_to_prompt, print_summary, run_experiment

MODEL = "ollama/llama3.2"
EXPERIMENT = "rag-config-sweep"

CORPUS = [
    "Paris is the capital of France. It lies on the Seine river.",
    "Tokyo is the capital of Japan and its most populous city.",
    "Canberra is the capital of Australia. It was purpose-built as the capital in 1913.",
    "Ottawa is the capital of Canada, located in the province of Ontario.",
    "Athens is the capital of Greece and one of the oldest cities in the world.",
    "The Eiffel Tower in Paris was completed in 1889.",
    "Mount Fuji, near Tokyo, is the tallest mountain in Japan.",
    "Sydney is the largest city in Australia, but it is not the capital.",
    "Toronto is the largest city in Canada, but Ottawa is the capital.",
    "The Parthenon in Athens was completed in 432 BC.",
    "Berlin is the capital of Germany. The Berlin Wall fell in 1989.",
    "Brasilia is the capital of Brazil; it replaced Rio de Janeiro as the capital in 1960.",
]

PROMPTS = {
    "grounded": "Answer the question using only the provided context. Be concise.",
    "concise": "Answer using only the provided context, in as few words as possible. No explanation.",
    "detailed": "Answer the question using the provided context. Explain your answer thoroughly and add relevant background.",
}

GRID = {
    "top_k": [1, 2, 3, 4],
    "temperature": [0.0, 0.7, 1.2],
    "prompt_style": list(PROMPTS),
    "max_tokens": [48, 256],
}

DATASET = [
    {"id": "fr", "question": "What is the capital of France?", "expected": "Paris"},
    {"id": "au", "question": "What is the capital of Australia?", "expected": "Canberra (not Sydney)"},
    {"id": "ca", "question": "What is the capital of Canada?", "expected": "Ottawa (not Toronto)"},
    {"id": "gr", "question": "What is the capital of Greece?", "expected": "Athens"},
    {"id": "br", "question": "What is the capital of Brazil and when did it become the capital?", "expected": "Brasilia, in 1960"},
    {"id": "eiffel", "question": "When was the Eiffel Tower completed?", "expected": "1889"},
    {"id": "fuji", "question": "What is the tallest mountain in Japan?", "expected": "Mount Fuji"},
    {"id": "wall", "question": "In which year did the Berlin Wall fall?", "expected": "1989"},
]

METRICS = ["correctness", "faithfulness", "answer_relevance"]


def variant_name(config: dict) -> str:
    return f"k{config['top_k']}_t{config['temperature']}_{config['prompt_style']}_m{config['max_tokens']}"


ALL_CONFIGS = {
    variant_name(config): config
    for config in (dict(zip(GRID, values)) for values in itertools.product(*GRID.values()))
}


def retrieve(question: str, top_k: int) -> list[str]:
    """Word-overlap retrieval over the in-memory corpus; stands in for a vector store."""
    words = {w.strip("?.,").lower() for w in question.split()}
    scored = sorted(CORPUS, key=lambda doc: -len(words & {w.strip(".,;").lower() for w in doc.split()}))
    return scored[:top_k]


def run_case(variant_name, variant, case, ctx):
    with ctx.observe("retrieve", as_type="retriever", input=case["question"]) as obs:
        chunks = retrieve(case["question"], variant["top_k"])
        obs.update(output=chunks)

    context = "\n".join(chunks)
    messages = [
        {"role": "system", "content": PROMPTS[variant["prompt_style"]]},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {case['question']}"},
    ]
    with ctx.observe("generate", as_type="generation", input={"prompt": messages_to_prompt(messages)}, model=MODEL) as obs:
        result = call_llm(messages, model=MODEL, temperature=variant["temperature"], max_tokens=variant["max_tokens"])
        obs.update(output=result.text)

    report = ctx.evaluate(
        METRICS,
        question=case["question"],
        answer=result.text,
        context=chunks,
        expected=case["expected"],
        model=MODEL,
    )
    ctx.set_trace_io(input={"question": case["question"]}, output={"answer": result.text})

    return {
        "latency_ms": result.latency_ms,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "total_tokens": result.total_tokens,
        **report.metrics(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample", type=int, default=24, help="random configurations to run (default 24 of 72)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--configs", help="comma-separated variant names to run instead of a random sample")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    if args.configs:
        names = [name.strip() for name in args.configs.split(",") if name.strip()]
        unknown = [name for name in names if name not in ALL_CONFIGS]
        if unknown:
            parser.error(f"unknown configurations: {', '.join(unknown)}")
    else:
        names = random.Random(args.seed).sample(sorted(ALL_CONFIGS), min(args.sample, len(ALL_CONFIGS)))
    variants = {name: ALL_CONFIGS[name] for name in names}

    with ExperimentTracker(EXPERIMENT) as tracker:
        results = run_experiment(tracker, variants, DATASET, run_case, max_workers=args.workers)

    print_summary(results, metric_names=[f"eval_{m}" for m in METRICS] + ["total_tokens", "latency_ms"])


if __name__ == "__main__":
    main()

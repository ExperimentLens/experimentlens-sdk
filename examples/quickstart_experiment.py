"""Minimal first experiment: A/B test two system prompts on a tiny dataset,
logged to MLflow + Langfuse so the runs and their traces show up together in
ExperimentLens (Agents & LLMs).

Prerequisites (see ../.env.example):
  - ExperimentLens running (`docker compose up`), which also starts MLflow
    (MLFLOW_TRACKING_URI) and Langfuse (LANGFUSE_HOST)
  - The same LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY as ExperimentLens' root .env
  - Ollama running locally with a model pulled (default here: llama3.2)
    -> or change `MODEL` below to e.g. "openai/gpt-4o-mini" (needs OPENAI_API_KEY)

Run with:
    python examples/quickstart_experiment.py
"""
from experimentlens import ExperimentTracker, call_llm, messages_to_prompt, print_summary, run_experiment

MODEL = "ollama/llama3.2"

VARIANTS = {
    "concise": {"system_prompt": "Answer in one short sentence."},
    "detailed": {"system_prompt": "Answer thoroughly, with your reasoning."},
}

DATASET = [
    {"id": "capital-fr", "question": "What is the capital of France?", "expected_keywords": ["paris"]},
    {"id": "capital-jp", "question": "What is the capital of Japan?", "expected_keywords": ["tokyo"]},
]


def run_case(variant_name, variant_config, case, ctx):
    messages = [
        {"role": "system", "content": variant_config["system_prompt"]},
        {"role": "user", "content": case["question"]},
    ]

    generation_input = {"prompt": messages_to_prompt(messages), "model": MODEL, "messages": messages}
    with ctx.observe("llm_call", as_type="generation", input=generation_input, model=MODEL) as obs:
        result = call_llm(messages, model=MODEL)
        obs.update(
            output=result.text,
            usage_details={"input": result.prompt_tokens, "output": result.completion_tokens},
            cost_details={"total": result.cost_usd},
        )

    passed = any(kw in result.text.lower() for kw in case["expected_keywords"])
    ctx.score("keyword_match", passed, data_type="BOOLEAN")

    verdict = ctx.log_judge(
        "answer_quality",
        question=case["question"],
        answer=result.text,
        criteria=f"The answer must correctly state that the answer contains one of: {case['expected_keywords']}.",
        model=MODEL,
    )

    ctx.set_trace_io(
        input={"question": case["question"]},
        output={"answer": result.text, "judge_pass_rate": float(verdict.passed)},
    )

    return {
        "task_success": float(passed),
        "judge_passed": float(verdict.passed),
        "latency_ms": result.latency_ms,
        "total_tokens": result.total_tokens,
        "cost_usd": result.cost_usd,
    }


def main():
    with ExperimentTracker("quickstart-prompt-ab") as tracker:
        # repeats: LLM answers vary between calls, so run each pair a few times to get
        # confidence intervals. max_workers: run cases in parallel (mind rate limits).
        results = run_experiment(tracker, VARIANTS, DATASET, run_case, case_id_key="id", repeats=3, max_workers=2)
    print_summary(results, metric_names=["task_success", "judge_passed", "latency_ms", "total_tokens", "cost_usd"])


if __name__ == "__main__":
    main()

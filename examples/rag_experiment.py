"""RAG experiment example: retrieve context, then call an LLM, saving the
embedding, retrieved docs, and final answer together as a per-run dataset.

Demonstrates a pattern beyond the plain prompt A/B test in quickstart_experiment.py:
  - ctx.observe(as_type="retriever") for the retrieval step (its own place in
    the trace, distinct from the LLM call)
  - ctx.observe(as_type="generation") for the LLM call
  - ctx.log_judge(...) for an LLM-as-judge pass/fail check
  - ctx.evaluate(...) for built-in quality and safety metrics (eval_* metrics)
  - compare(...) to see how one variant moved against another
  - ctx.log_dataset(..., file_format="parquet") to save an enriched row per run
    (embedding + retrieved doc ids + answer), not just the raw case input --
    parquet keeps the embedding as a real list column instead of a stringified
    blob the way CSV would

Uses a tiny deterministic hashing "embedding" and a 4-document in-memory corpus
so this example has no extra dependencies (swap `embed`/`CORPUS`/`retrieve` for
a real embedding model and vector store in your own experiment).

Prerequisites: same as examples/quickstart_experiment.py.

Run with:
    python examples/rag_experiment.py
"""
import hashlib
import math

from experimentlens import ExperimentTracker, call_llm, compare, messages_to_prompt, print_summary, run_experiment

MODEL = "ollama/llama3.2"
EMBEDDING_DIM = 16

CORPUS = [
    {"id": "doc-1", "text": "The capital of France is Paris, a city on the Seine river."},
    {"id": "doc-2", "text": "Tokyo is the capital of Japan and its most populous city."},
    {"id": "doc-3", "text": "The Eiffel Tower is a famous landmark in Paris, France."},
    {"id": "doc-4", "text": "Mount Fuji is the tallest mountain in Japan, near Tokyo."},
]

VARIANTS = {
    "top1": {"top_k": 1},
    "top2": {"top_k": 2},
}

DATASET = [
    {"id": "capital-fr", "question": "What is the capital of France?", "expected_keywords": ["paris"]},
    {"id": "capital-jp", "question": "What is the capital of Japan?", "expected_keywords": ["tokyo"]},
]


def embed(text: str) -> list[float]:
    """Tiny deterministic hashing embedding, standing in for a real embedding
    model (e.g. sentence-transformers) so this example needs no extra deps."""
    vector = [0.0] * EMBEDDING_DIM
    for token in text.lower().split():
        idx = int(hashlib.md5(token.encode()).hexdigest(), 16) % EMBEDDING_DIM
        vector[idx] += 1.0
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def retrieve(query_embedding: list[float], top_k: int) -> list[dict]:
    scored = [(cosine_similarity(query_embedding, embed(doc["text"])), doc) for doc in CORPUS]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [doc for _, doc in scored[:top_k]]


def run_case(variant_name, variant_config, case, ctx):
    top_k = variant_config["top_k"]

    with ctx.observe("retrieve", as_type="retriever", input=case["question"]) as obs:
        query_embedding = embed(case["question"])
        retrieved_docs = retrieve(query_embedding, top_k)
        obs.update(output=[d["text"] for d in retrieved_docs])

    context = "\n".join(d["text"] for d in retrieved_docs)
    messages = [
        {"role": "system", "content": "Answer the question using only the provided context."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {case['question']}"},
    ]

    generation_input = {"prompt": messages_to_prompt(messages), "model": MODEL}
    with ctx.observe("generate", as_type="generation", input=generation_input, model=MODEL) as obs:
        result = call_llm(messages, model=MODEL)
        obs.update(output=result.text)

    passed = any(kw in result.text.lower() for kw in case["expected_keywords"])
    ctx.score("keyword_match", passed, data_type="BOOLEAN")

    verdict = ctx.log_judge(
        "answer_quality",
        question=case["question"],
        answer=result.text,
        criteria=f"The answer must correctly state one of: {case['expected_keywords']}.",
        model=MODEL,
    )

    report = ctx.evaluate(
        ["faithfulness", "answer_relevance", "context_relevance", "toxicity", "pii_leakage"],
        question=case["question"],
        answer=result.text,
        context=[d["text"] for d in retrieved_docs],
        model=MODEL,
    )

    ctx.set_trace_io(
        input={"question": case["question"]},
        output={"answer": result.text, "judge_pass_rate": float(verdict.passed)},
    )

    # One enriched row per run: the case, the retrieval, the embedding, and the
    # final answer. Build whatever shape you need here -- log_dataset takes any
    # list of dicts, it's not tied to the original dataset's schema.
    ctx.log_dataset(
        [{
            **case,
            "top_k": top_k,
            "query_embedding": query_embedding,
            "retrieved_doc_ids": [d["id"] for d in retrieved_docs],
            "answer": result.text,
        }],
        name="rag_result",
        file_format="csv",
    )

    return {
        "task_success": float(passed),
        "judge_passed": float(verdict.passed),
        "latency_ms": result.latency_ms,
        "total_tokens": result.total_tokens,
        "cost_usd": result.cost_usd,
        **report.metrics(),
    }


def main():
    tracker = ExperimentTracker("rag-topk-ab")
    results = run_experiment(tracker, VARIANTS, DATASET, run_case, case_id_key="id")
    metrics = ["task_success", "judge_passed", "eval_faithfulness", "eval_context_relevance", "latency_ms"]
    print_summary(results, metric_names=metrics)

    print("\ntop2 against top1:")
    for comparison in compare(results, baseline="top1", candidate="top2", metrics=metrics):
        print(("  REGRESSED " if comparison.regressed else "  ") + comparison.describe())
    tracker.close()


if __name__ == "__main__":
    main()

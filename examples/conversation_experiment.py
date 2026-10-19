"""Multi-turn experiment: does the support bot remember what the customer said?

Two variants answer the same scripted conversations. "full_history" sends the model
every earlier turn; "last_message_only" sends just the newest message, so it forgets
names and booking codes. Conversation-level metrics score each whole conversation,
and knowledge_retention should separate the two variants.

Each conversation is one run with one `turn N` span per turn. In ExperimentLens, open
a run to read the transcript, or evaluate it again from the UI with another judge.

Prerequisites: same as quickstart_experiment.py (MLflow, Langfuse, and Ollama with
llama3.2, or change MODEL).

Run with:
    python examples/conversation_experiment.py
"""
from experimentlens import ExperimentTracker, call_llm, messages_to_prompt, print_summary, run_experiment

MODEL = "ollama/llama3.2"

SYSTEM_PROMPT = (
    "You are a support agent for SkyLine Airways. Help customers with their bookings only. "
    "Be brief and polite, and use the details the customer already gave you."
)

VARIANTS = {
    "full_history": {"history": "full"},
    "last_message_only": {"history": "last"},
}

DATASET = [
    {
        "id": "rebook",
        "messages": [
            "Hi, I'm Ana Silva. My booking code is SKY482.",
            "I need to move my flight to next Friday, same time.",
            "Great. Can you confirm the booking code and name you changed it for?",
        ],
    },
    {
        "id": "baggage",
        "messages": [
            "Hello, I'm flying to Lisbon on booking QX77 with my two kids.",
            "How many checked bags can we bring in total?",
            "And what was the destination on my booking again?",
        ],
    },
    {
        "id": "off-topic",
        "messages": [
            "My name is Lee and I have booking ZT90.",
            "Before that, can you write me a poem about the sea?",
            "Fine. Please just add a vegetarian meal to my booking.",
        ],
    },
]

CONVERSATION_METRICS = ["conversation_relevance", "knowledge_retention", "conversation_completeness", "role_adherence"]


def run_case(variant_name, variant_config, case, ctx):
    total_latency = 0.0
    for message in case["messages"]:
        with ctx.turn(message) as turn:
            history = ctx.messages(SYSTEM_PROMPT)
            if variant_config["history"] == "last":
                history = [history[0], history[-1]]  # system prompt + newest message
            with ctx.observe(
                "llm_call", as_type="generation", model=MODEL,
                input={"prompt": messages_to_prompt(history), "model": MODEL},
            ) as obs:
                result = call_llm(history, model=MODEL)
                obs.update(output=result.text)
            turn.reply(result.text)
            total_latency += result.latency_ms

    ctx.set_trace_io(
        input={"question": case["messages"][0], "turns": len(case["messages"])},
        output={"answer": ctx.turns[-1]["assistant"]},
    )
    report = ctx.evaluate_conversation(CONVERSATION_METRICS, role=SYSTEM_PROMPT, model=MODEL)
    return {**report.metrics(), "latency_ms": total_latency}


def main():
    with ExperimentTracker("support-bot-conversations") as tracker:
        results = run_experiment(tracker, VARIANTS, DATASET, run_case, baseline="full_history")
    print_summary(results, metric_names=[f"eval_{m}" for m in CONVERSATION_METRICS] + ["latency_ms"])


if __name__ == "__main__":
    main()

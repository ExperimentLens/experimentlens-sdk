import uuid

import pytest
from mlflow.tracking import MlflowClient

from experimentlens import CONVERSATION_METRICS, ExperimentTracker, evaluate_conversation
from experimentlens import evals
from experimentlens.evals import format_conversation
from experimentlens.llm import LLMResult

TURNS = [
    {"user": "Hi, I'm Ana. My booking is LH123.", "assistant": "Hello Ana, how can I help with LH123?"},
    {"user": "Can I move it to Friday?", "assistant": "Yes, LH123 is now on Friday."},
]


@pytest.fixture
def judge_prompts(monkeypatch):
    prompts = []

    def call_llm(messages, model=None, **kwargs):
        prompts.append(messages[0]["content"])
        return LLMResult(text='{"score": 4, "reason": "ok"}', raw=None, latency_ms=1.0, total_tokens=5)

    monkeypatch.setattr(evals, "call_llm", call_llm)
    return prompts


def test_transcript_numbers_turns_and_names_speakers():
    assert format_conversation(TURNS) == (
        "Turn 1\nUser: Hi, I'm Ana. My booking is LH123.\nAssistant: Hello Ana, how can I help with LH123?\n\n"
        "Turn 2\nUser: Can I move it to Friday?\nAssistant: Yes, LH123 is now on Friday."
    )


def test_judges_see_the_whole_conversation_and_the_role(judge_prompts):
    report = evaluate_conversation(list(CONVERSATION_METRICS), TURNS, role="Airline support agent.")

    assert set(report) == set(CONVERSATION_METRICS)
    assert all(r.score == 0.75 for r in report.values())
    assert all("Turn 2\nUser: Can I move it to Friday?" in p for p in judge_prompts)
    role_prompt = next(p for p in judge_prompts if "stay within the role" in p)
    assert "Role: Airline support agent." in role_prompt


def test_role_adherence_needs_a_role(judge_prompts):
    with pytest.raises(ValueError, match="role"):
        evaluate_conversation(["role_adherence"], TURNS)


def test_unknown_or_single_turn_metrics_are_rejected(judge_prompts):
    with pytest.raises(ValueError, match="Unknown conversation metric"):
        evaluate_conversation(["faithfulness"], TURNS)
    with pytest.raises(ValueError, match="at least one turn"):
        evaluate_conversation(["knowledge_retention"], [])


def test_turns_build_the_history_and_are_evaluated_and_logged(config, judge_prompts):
    with pytest.warns(UserWarning):
        tracker = ExperimentTracker(f"conv-{uuid.uuid4().hex[:8]}", config=config)
    seen = []
    with tracker, tracker.start_run("r") as ctx:
        for turn_data in TURNS:
            with ctx.turn(turn_data["user"]) as turn:
                seen.append(ctx.messages("Be brief."))
                turn.reply(turn_data["assistant"])
        report = ctx.evaluate_conversation(["knowledge_retention", "conversation_completeness"], role="Be brief.")

    assert ctx.turns == TURNS
    assert seen[0] == [{"role": "system", "content": "Be brief."}, {"role": "user", "content": TURNS[0]["user"]}]
    assert seen[1][-2:] == [
        {"role": "assistant", "content": TURNS[0]["assistant"]},
        {"role": "user", "content": TURNS[1]["user"]},
    ]
    assert ctx.messages() == [
        {"role": "user", "content": TURNS[0]["user"]},
        {"role": "assistant", "content": TURNS[0]["assistant"]},
        {"role": "user", "content": TURNS[1]["user"]},
        {"role": "assistant", "content": TURNS[1]["assistant"]},
    ]
    assert set(report) == {"knowledge_retention", "conversation_completeness"}
    run = MlflowClient().get_run(ctx.run_id)
    assert run.data.metrics == {"eval_knowledge_retention": 0.75, "eval_conversation_completeness": 0.75}
    assert run.data.tags["experimentlens.conversation_role"] == "Be brief."
    experiment = MlflowClient().get_experiment(run.info.experiment_id)
    assert '"kind": "quality"' in experiment.tags["experimentlens.metric.eval_knowledge_retention"]


def test_a_failed_turn_is_not_recorded(config):
    with pytest.warns(UserWarning):
        tracker = ExperimentTracker(f"conv-{uuid.uuid4().hex[:8]}", config=config)
    with pytest.raises(TimeoutError), tracker, tracker.start_run("r") as ctx:
        with ctx.turn("hello"):
            raise TimeoutError("model timed out")
    assert ctx.turns == []
    assert ctx.messages() == []

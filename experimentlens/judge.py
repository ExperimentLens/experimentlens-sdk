from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

from .config import default_judge_model
from .llm import LLMResult, call_llm

DEFAULT_JUDGE_PROMPT = """You are an evaluation judge. Given a question, an answer, and \
criteria, decide whether the answer satisfies the criteria.

Question: {question}
Answer: {answer}
Criteria: {criteria}

Respond with ONLY a JSON object, no other text: {{"passed": true or false, "rationale": "one sentence explanation"}}"""


@dataclass
class JudgeResult:
    passed: bool
    rationale: str
    llm_result: LLMResult


def llm_judge(
    question: str,
    answer: str,
    criteria: str,
    model: Optional[str] = None,
    prompt_template: str = DEFAULT_JUDGE_PROMPT,
) -> JudgeResult:
    """Use an LLM to judge whether `answer` satisfies `criteria` for `question`.

    Works with any model litellm supports — prefer a stronger/different model than
    the one under test to reduce judge bias, e.g. model="openai/gpt-4o-mini".
    Usually called via RunContext.log_judge(...) rather than directly, so the
    verdict also gets logged/scored against the run's trace.

    If `model` is omitted, EXPERIMENTLENS_JUDGE_MODEL is used (falling back to
    EXPERIMENTLENS_MODEL, then "ollama/llama3.2").
    """
    model = model or default_judge_model()
    prompt = prompt_template.format(question=question, answer=answer, criteria=criteria)
    result = call_llm([{"role": "user", "content": prompt}], model=model)
    passed, rationale = _parse_verdict(result.text)
    return JudgeResult(passed=passed, rationale=rationale, llm_result=result)


def _parse_verdict(text: str) -> tuple[bool, str]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            return bool(data.get("passed", False)), str(data.get("rationale", "")).strip()
        except Exception:
            pass
    # Fallback if the judge model didn't return valid JSON.
    has_pass = re.search(r"\bpass(ed)?\b", text, re.IGNORECASE) is not None
    has_fail = re.search(r"\bfail(ed)?\b", text, re.IGNORECASE) is not None
    return (has_pass and not has_fail), text.strip()

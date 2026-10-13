from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional

from .config import default_model


@dataclass
class LLMResult:
    text: str
    raw: Any
    latency_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0


def messages_to_prompt(messages: list[dict]) -> str:
    """Flatten a chat-messages list into one string. The vis-frontend's Langfuse
    generation view reads a plain `input.prompt` string (see GenInput in
    agentic-conventions.ts) rather than a raw messages array, so this is what
    should go in `observe(..., input={"prompt": ...})`."""
    return "\n\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages)


def call_llm(messages: list[dict], model: Optional[str] = None, **kwargs) -> LLMResult:
    """Call any LLM provider through litellm and return a uniform result.

    `model` follows litellm's `<provider>/<model>` convention, e.g.:
      - "ollama/llama3.2"       (local Ollama, no API key needed)
      - "openai/gpt-4o-mini"    (needs OPENAI_API_KEY in the environment)
      - "anthropic/claude-sonnet-5" (needs ANTHROPIC_API_KEY)

    This is the one place experiment code should call an LLM, so latency/tokens/cost
    are captured consistently and can be handed straight to RunContext.log_generation
    or attached inside a RunContext.observe(...) block.

    If `model` is omitted, EXPERIMENTLENS_MODEL is used (falling back to
    "ollama/llama3.2").
    """
    import litellm

    model = model or default_model()

    start = time.perf_counter()
    response = litellm.completion(model=model, messages=messages, **kwargs)
    latency_ms = (time.perf_counter() - start) * 1000

    text = response["choices"][0]["message"]["content"] or ""
    usage = response.get("usage") or {}

    try:
        cost_usd = litellm.completion_cost(completion_response=response) or 0.0
    except Exception:
        cost_usd = 0.0

    return LLMResult(
        text=text,
        raw=response,
        latency_ms=latency_ms,
        prompt_tokens=usage.get("prompt_tokens", 0) or 0,
        completion_tokens=usage.get("completion_tokens", 0) or 0,
        total_tokens=usage.get("total_tokens", 0) or 0,
        cost_usd=cost_usd,
    )

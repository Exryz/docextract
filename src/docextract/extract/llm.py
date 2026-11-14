"""Provider-agnostic LLM client (LiteLLM) plus a deterministic `ScriptedLLM` for offline tests."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

Message = dict[str, Any]


@dataclass(frozen=True)
class LLMCallStats:
    """Token, cost, and latency telemetry for a single LLM completion call."""

    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency_ms: float


@dataclass(frozen=True)
class LLMCompletion:
    """Raw string content and telemetry returned by an LLM call."""

    content: str
    stats: LLMCallStats


class ExtractionLLM(Protocol):
    """Protocol for LLM clients used by the structured extraction pipeline."""

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.0,
        json_mode: bool = True,
    ) -> LLMCompletion: ...


class LiteLLMExtractorClient:
    """Production LLM client wrapping `litellm.acompletion` with JSON mode and cost tracking."""

    def __init__(
        self,
        model: str = "gemini/gemini-3.5-flash",
        timeout_s: float = 60.0,
    ) -> None:
        self.model = model
        self.timeout_s = timeout_s

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.0,
        json_mode: bool = True,
    ) -> LLMCompletion:
        import litellm

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "temperature": temperature,
            "timeout": self.timeout_s,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        t0 = time.perf_counter()
        resp: Any = await litellm.acompletion(**kwargs)
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        choice = resp.choices[0].message
        content = str(choice.content or "")

        usage = getattr(resp, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)

        try:
            cost_usd = float(litellm.completion_cost(completion_response=resp) or 0.0)
        except Exception:
            cost_usd = 0.0

        return LLMCompletion(
            content=content,
            stats=LLMCallStats(
                model=self.model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost_usd=cost_usd,
                latency_ms=latency_ms,
            ),
        )


Responder = Callable[[Sequence[Message]], str | LLMCompletion]


class ScriptedLLM:
    """Deterministic fake LLM for tests: replays a list of JSON strings or invokes a callable."""

    def __init__(self, script: Sequence[str | LLMCompletion] | Responder) -> None:
        self._script = script
        self._index = 0
        self.calls: list[list[Message]] = []

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.0,
        json_mode: bool = True,
    ) -> LLMCompletion:
        self.calls.append([dict(m) for m in messages])
        if callable(self._script):
            item = self._script(messages)
        else:
            if self._index >= len(self._script):
                raise RuntimeError(
                    f"ScriptedLLM exhausted after {self._index} calls (no more scripted responses)"
                )
            item = self._script[self._index]
            self._index += 1

        if isinstance(item, LLMCompletion):
            return item

        return LLMCompletion(
            content=item,
            stats=LLMCallStats(
                model="scripted-llm",
                prompt_tokens=120,
                completion_tokens=80,
                cost_usd=0.0,
                latency_ms=1.0,
            ),
        )

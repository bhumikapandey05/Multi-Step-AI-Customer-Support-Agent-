"""Model wrappers for evaluation runs.

* :class:`MockLLM` replays a scenario's scripted steps, so offline runs are
  deterministic and need no network or API key.
* :class:`InstrumentedLLM` wraps any client and records, per call, the latency,
  the tools the model proposed, and the token usage the provider reported.
* :class:`FreeTierGuard` keeps live runs inside free-tier limits: a minimum gap
  between requests (requests per minute) and a hard cap on total calls.
"""

import itertools
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from support_agent.agent.llm import (
    LLMClient,
    LLMResponse,
    LLMUnavailableError,
    Message,
    ToolCall,
    ToolDefinition,
)
from support_agent.evals.scenario import MockStep

MOCK_EXHAUSTED = "(mock script exhausted)"


class MockScriptExhausted(Exception):
    """The scenario asked the model more often than its script allows (an authoring error)."""


class MockLLM:
    def __init__(self, steps: Sequence[MockStep]) -> None:
        self._steps = list(steps)
        self._ids = itertools.count(1)
        self._lock = threading.Lock()  # the race scenario calls it from two threads
        self.exhausted = False

    def generate(
        self,
        *,
        system_prompt: str,
        messages: Sequence[Message],
        tools: Sequence[ToolDefinition],
    ) -> LLMResponse:
        with self._lock:
            if not self._steps:
                self.exhausted = True
                return LLMResponse(text=MOCK_EXHAUSTED)
            step = self._steps.pop(0)
        if step.error == "unavailable":
            raise LLMUnavailableError("mocked outage")
        calls = []
        for spec in step.calls:
            n = next(self._ids)
            calls.append(ToolCall(f"mock_{n}", spec.tool, dict(spec.args), f"mock_fc_{n}"))
        return LLMResponse(text=step.text, tool_calls=tuple(calls))

    @property
    def remaining(self) -> int:
        return len(self._steps)


@dataclass
class LLMCall:
    latency_ms: float  # model time only; free-tier pacing waits are excluded
    tools: list[str]
    usage: dict[str, int]
    error: str | None = None
    finish_reason: str | None = None
    paced_ms: float = 0.0  # time spent waiting for the free-tier request gap


@dataclass
class InstrumentedLLM:
    inner: LLMClient
    calls: list[LLMCall] = field(default_factory=list)
    clock: Callable[[], float] = time.perf_counter

    def generate(self, **kwargs: Any) -> LLMResponse:
        started = self.clock()
        try:
            response = self.inner.generate(**kwargs)
        except Exception as exc:
            paced = self._paced_ms()
            self.calls.append(
                LLMCall(self._ms(started, paced), [], {}, error=type(exc).__name__, paced_ms=paced)
            )
            raise
        paced = self._paced_ms()
        self.calls.append(
            LLMCall(
                self._ms(started, paced),
                [c.name for c in response.tool_calls],
                dict(response.usage),
                finish_reason=response.finish_reason,
                paced_ms=paced,
            )
        )
        return response

    def _paced_ms(self) -> float:
        return round(getattr(self.inner, "last_wait_s", 0.0) * 1000, 3)

    def _ms(self, started: float, paced_ms: float = 0.0) -> float:
        return round(max((self.clock() - started) * 1000 - paced_ms, 0.0), 3)

    @property
    def proposed_tools(self) -> list[str]:
        return [t for c in self.calls for t in c.tools]


class BudgetExhausted(LLMUnavailableError):
    """The live run reached its configured call budget."""


class FreeTierGuard:
    """Shared by every scenario in a live run: paces requests and caps their number."""

    def __init__(
        self,
        inner: LLMClient,
        *,
        requests_per_minute: float,
        max_calls: int,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if requests_per_minute <= 0 or max_calls < 1:
            raise ValueError("requests_per_minute must be > 0 and max_calls >= 1")
        self._inner = inner
        self._gap = 60.0 / requests_per_minute
        self._max_calls = max_calls
        self._sleep = sleep
        self._monotonic = monotonic
        self._lock = threading.Lock()
        self._last: float | None = None
        self.used = 0
        self.last_wait_s = 0.0  # pacing wait before the most recent call

    def generate(self, **kwargs: Any) -> LLMResponse:
        with self._lock:
            if self.used >= self._max_calls:
                raise BudgetExhausted(f"live evaluation budget of {self._max_calls} calls used")
            self.last_wait_s = 0.0
            if self._last is not None:
                wait = self._gap - (self._monotonic() - self._last)
                if wait > 0:
                    self._sleep(wait)
                    self.last_wait_s = wait
            self._last = self._monotonic()
            self.used += 1
        return self._inner.generate(**kwargs)

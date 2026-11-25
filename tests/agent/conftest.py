"""Offline agent fixtures: a scripted LLM stands in for Gemini."""

import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import pytest

from support_agent.agent.agent import AgentConfig, DatabaseApprovalAuditor, SupportAgent
from support_agent.agent.llm import LLMResponse, Message, ToolCall, ToolDefinition

_ids = itertools.count(1)

Step = LLMResponse | Exception | Callable[[Sequence[Message]], LLMResponse]


def call(name: str, **arguments: Any) -> ToolCall:
    n = next(_ids)
    return ToolCall(id=f"call_{n}", name=name, arguments=arguments, provider_id=f"fc_{n}")


def tools(*calls: ToolCall, text: str | None = None) -> LLMResponse:
    return LLMResponse(text=text, tool_calls=tuple(calls))


def say(text: str) -> LLMResponse:
    return LLMResponse(text=text)


@dataclass
class Request:
    system_prompt: str
    messages: tuple[Message, ...]
    tools: tuple[ToolDefinition, ...]

    @property
    def tool_names(self) -> set[str]:
        return {t.name for t in self.tools}

    def last_results(self) -> dict[str, dict[str, Any]]:
        """Tool results (by tool name) in the most recent tool message sent to the model."""
        message = self.messages[-1]
        assert message.role == "tool", f"expected tool results, got {message.role}"
        return {r.name: r.content for r in message.tool_results}


class ScriptedLLM:
    """Returns pre-programmed responses in order and records every request."""

    def __init__(self, *steps: Step) -> None:
        self.steps = list(steps)
        self.requests: list[Request] = []

    def generate(self, *, system_prompt, messages, tools) -> LLMResponse:
        self.requests.append(Request(system_prompt, tuple(messages), tuple(tools)))
        if not self.steps:
            raise AssertionError("ScriptedLLM ran out of responses")
        step = self.steps.pop(0)
        if isinstance(step, Exception):
            raise step
        if callable(step):
            return step(messages)
        return step


@pytest.fixture
def make_agent(crm, login, session_factory, clock):
    def _make(llm: ScriptedLLM, email: str | None = None, **config: Any) -> SupportAgent:
        return SupportAgent(
            llm,
            crm,
            ctx=login(email) if email else None,
            config=AgentConfig(**config),
            auditor=DatabaseApprovalAuditor(session_factory, clock=clock),
            clock=clock,
        )

    return _make

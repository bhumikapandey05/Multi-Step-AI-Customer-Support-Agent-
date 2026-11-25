"""Provider-neutral LLM interface.

The agent loop talks only to :class:`LLMClient`. Gemini is one implementation
(:mod:`support_agent.agent.gemini`); tests use a scripted fake. This keeps the
loop testable offline and makes the provider swappable.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol


@dataclass(frozen=True)
class ToolDefinition:
    """A tool as the model sees it: name, description, JSON-schema parameters."""

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(frozen=True)
class ToolCall:
    """A tool invocation requested by the model. Arguments are untrusted input."""

    id: str
    name: str
    arguments: dict[str, Any]
    provider_id: str | None = None  # id assigned by the provider, echoed back with the result


@dataclass(frozen=True)
class ToolResult:
    call_id: str
    name: str
    content: dict[str, Any]
    provider_id: str | None = None


@dataclass(frozen=True)
class Message:
    role: Literal["user", "assistant", "tool"]
    text: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    tool_results: tuple[ToolResult, ...] = ()
    # Opaque provider payload replayed verbatim (e.g. Gemini content with thought signatures).
    provider_data: Any = field(default=None, compare=False, repr=False)


@dataclass(frozen=True)
class LLMResponse:
    text: str | None
    tool_calls: tuple[ToolCall, ...] = ()
    finish_reason: str | None = None
    provider_data: Any = field(default=None, compare=False, repr=False)
    usage: dict[str, int] = field(default_factory=dict)


class LLMClient(Protocol):
    def generate(
        self,
        *,
        system_prompt: str,
        messages: Sequence[Message],
        tools: Sequence[ToolDefinition],
    ) -> LLMResponse: ...


class LLMError(Exception):
    """The model call failed and should not be retried as-is."""


class LLMUnavailableError(LLMError):
    """Transient failure (rate limit, timeout, 5xx) that persisted after retries."""


class LLMConfigurationError(LLMError):
    """Missing API key, unknown model, or another setup problem."""


# --------------------------------------------------------------------------- persistence
#
# Workflow checkpoints store conversations as plain JSON (no pickled objects), so
# they can be inspected and are safe to load. Provider payloads that are Pydantic
# models (Gemini ``Content``) are stored as their JSON form; adapters accept that
# form back (see ``gemini.to_gemini_contents``).


def _dump_provider_data(value: Any) -> Any:
    if value is None:
        return None
    dump_json = getattr(value, "model_dump_json", None)
    if callable(dump_json):
        return json.loads(dump_json(exclude_none=True))
    return value


def tool_call_to_dict(call: ToolCall) -> dict[str, Any]:
    return {
        "id": call.id,
        "name": call.name,
        "arguments": call.arguments,
        "provider_id": call.provider_id,
    }


def tool_call_from_dict(data: dict[str, Any]) -> ToolCall:
    return ToolCall(data["id"], data["name"], dict(data["arguments"]), data.get("provider_id"))


def tool_result_to_dict(result: ToolResult) -> dict[str, Any]:
    return {
        "call_id": result.call_id,
        "name": result.name,
        "content": result.content,
        "provider_id": result.provider_id,
    }


def tool_result_from_dict(data: dict[str, Any]) -> ToolResult:
    return ToolResult(data["call_id"], data["name"], data["content"], data.get("provider_id"))


def message_to_dict(message: Message) -> dict[str, Any]:
    return {
        "role": message.role,
        "text": message.text,
        "tool_calls": [tool_call_to_dict(c) for c in message.tool_calls],
        "tool_results": [tool_result_to_dict(r) for r in message.tool_results],
        "provider_data": _dump_provider_data(message.provider_data),
    }


def message_from_dict(data: dict[str, Any]) -> Message:
    return Message(
        role=data["role"],
        text=data.get("text"),
        tool_calls=tuple(tool_call_from_dict(c) for c in data.get("tool_calls", ())),
        tool_results=tuple(tool_result_from_dict(r) for r in data.get("tool_results", ())),
        provider_data=data.get("provider_data"),
    )

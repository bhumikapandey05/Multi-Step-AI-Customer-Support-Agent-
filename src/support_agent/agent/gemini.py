"""Gemini implementation of :class:`LLMClient` using the ``google-genai`` SDK.

Automatic function calling is disabled: the SDK never executes Python
functions. Gemini only *proposes* tool calls; the agent loop decides whether
and how to run them.
"""

import json
import logging
import random
import time
import uuid
from collections.abc import Callable, Sequence
from typing import Any

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from support_agent.agent.llm import (
    LLMConfigurationError,
    LLMError,
    LLMResponse,
    LLMUnavailableError,
    Message,
    ToolCall,
    ToolDefinition,
)

logger = logging.getLogger(__name__)

_RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


class GeminiClient:
    def __init__(
        self,
        *,
        api_key: str | None,
        model: str,
        temperature: float = 0.2,
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
        client: Any = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if client is None:
            if not api_key:
                raise LLMConfigurationError(
                    "GEMINI_API_KEY is not set. Add it to .env (free key: "
                    "https://aistudio.google.com/apikey)."
                )
            client = genai.Client(
                api_key=api_key,
                http_options=types.HttpOptions(timeout=int(timeout_seconds * 1000)),
            )
        self._client = client
        self._model = model
        self._temperature = temperature
        self._max_retries = max_retries
        self._sleep = sleep

    @property
    def model(self) -> str:
        return self._model

    def generate(
        self,
        *,
        system_prompt: str,
        messages: Sequence[Message],
        tools: Sequence[ToolDefinition],
    ) -> LLMResponse:
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=self._temperature,
            tools=[types.Tool(function_declarations=[_declaration(t) for t in tools])]
            if tools
            else None,
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode="AUTO")
            )
            if tools
            else None,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        contents = to_gemini_contents(messages)
        response = self._call_with_retries(contents, config)
        return parse_gemini_response(response)

    def _call_with_retries(self, contents: list[types.Content], config: Any) -> Any:
        attempt = 0
        while True:
            try:
                return self._client.models.generate_content(
                    model=self._model, contents=contents, config=config
                )
            except genai_errors.APIError as exc:
                retryable = exc.code in _RETRYABLE_STATUS
                if not retryable:
                    if exc.code in (400, 401, 403, 404):
                        raise LLMConfigurationError(
                            f"Gemini rejected the request ({exc.code}): {exc.message}"
                        ) from exc
                    raise LLMError(f"Gemini error {exc.code}: {exc.message}") from exc
                error: Exception = exc
            except (TimeoutError, ConnectionError, OSError) as exc:
                error = exc
            except Exception as exc:
                # httpx transport errors and similar network problems.
                if type(exc).__module__.startswith(("httpx", "httpcore")):
                    error = exc
                else:
                    raise LLMError(f"Unexpected Gemini client error: {exc}") from exc

            if attempt >= self._max_retries:
                raise LLMUnavailableError(
                    f"Gemini is unavailable after {attempt + 1} attempts: {error}"
                ) from error
            delay = min(2**attempt, 16) + random.uniform(0, 0.5)
            logger.warning(
                "Retrying Gemini call",
                extra={"attempt": attempt + 1, "delay_s": round(delay, 2), "error": str(error)},
            )
            self._sleep(delay)
            attempt += 1


# --------------------------------------------------------------------------- conversion


def _declaration(tool: ToolDefinition) -> types.FunctionDeclaration:
    return types.FunctionDeclaration(
        name=tool.name, description=tool.description, parameters_json_schema=tool.parameters
    )


def to_gemini_contents(messages: Sequence[Message]) -> list[types.Content]:
    contents: list[types.Content] = []
    for message in messages:
        if message.role == "user":
            contents.append(types.Content(role="user", parts=[types.Part(text=message.text)]))
        elif message.role == "assistant":
            if isinstance(message.provider_data, types.Content):
                # Replay verbatim so Gemini thought signatures survive the round trip.
                contents.append(message.provider_data)
                continue
            if isinstance(message.provider_data, dict):
                # Restored from a JSON checkpoint. JSON validation base64-decodes the
                # thought signatures back to bytes.
                contents.append(
                    types.Content.model_validate_json(json.dumps(message.provider_data))
                )
                continue
            parts: list[types.Part] = []
            if message.text:
                parts.append(types.Part(text=message.text))
            parts.extend(
                types.Part(
                    function_call=types.FunctionCall(
                        id=call.provider_id, name=call.name, args=call.arguments
                    )
                )
                for call in message.tool_calls
            )
            contents.append(types.Content(role="model", parts=parts))
        else:  # tool results go back as a user turn of function_response parts
            contents.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part(
                            function_response=types.FunctionResponse(
                                id=result.provider_id, name=result.name, response=result.content
                            )
                        )
                        for result in message.tool_results
                    ],
                )
            )
    return contents


def parse_gemini_response(response: Any) -> LLMResponse:
    usage: dict[str, int] = {}
    meta = getattr(response, "usage_metadata", None)
    if meta is not None:
        for key in ("prompt_token_count", "candidates_token_count", "total_token_count"):
            value = getattr(meta, key, None)
            if value is not None:
                usage[key] = value

    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        feedback = getattr(response, "prompt_feedback", None)
        reason = getattr(feedback, "block_reason", None)
        return LLMResponse(
            text=None, finish_reason=f"blocked:{reason}" if reason else "empty", usage=usage
        )

    candidate = candidates[0]
    content = candidate.content
    texts: list[str] = []
    calls: list[ToolCall] = []
    for part in content.parts if content and content.parts else []:
        if part.function_call is not None:
            fc = part.function_call
            calls.append(
                ToolCall(
                    id=fc.id or f"call_{uuid.uuid4().hex[:12]}",
                    name=fc.name or "",
                    arguments=dict(fc.args or {}),
                    provider_id=fc.id,
                )
            )
        elif part.text and not part.thought:
            texts.append(part.text)

    finish = getattr(candidate, "finish_reason", None)
    return LLMResponse(
        text="".join(texts).strip() or None,
        tool_calls=tuple(calls),
        finish_reason=str(finish.value if hasattr(finish, "value") else finish) if finish else None,
        provider_data=content,
        usage=usage,
    )

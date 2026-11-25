"""Gemini adapter, tested offline with real SDK types and a fake transport."""

import pytest
from google.genai import errors as genai_errors
from google.genai import types

from support_agent.agent.gemini import GeminiClient, parse_gemini_response, to_gemini_contents
from support_agent.agent.llm import (
    LLMConfigurationError,
    LLMUnavailableError,
    Message,
    ToolCall,
    ToolDefinition,
    ToolResult,
)
from support_agent.agent.tools import default_tools

TOOLS = [t.definition() for t in default_tools()]


def response(*parts: dict, finish: str = "STOP") -> types.GenerateContentResponse:
    return types.GenerateContentResponse.model_validate(
        {
            "candidates": [
                {"content": {"role": "model", "parts": list(parts)}, "finish_reason": finish}
            ],
            "usage_metadata": {
                "prompt_token_count": 100,
                "candidates_token_count": 20,
                "total_token_count": 120,
            },
        }
    )


class FakeModels:
    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls: list[dict] = []

    def generate_content(self, *, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeClient:
    def __init__(self, *outcomes):
        self.models = FakeModels(*outcomes)


def client(*outcomes, **kwargs) -> tuple[GeminiClient, FakeClient]:
    fake = FakeClient(*outcomes)
    sleeps: list[float] = []
    gemini = GeminiClient(
        api_key=None, model="gemini-test", client=fake, sleep=sleeps.append, **kwargs
    )
    gemini.sleeps = sleeps  # type: ignore[attr-defined]
    return gemini, fake


def api_error(code: int) -> genai_errors.APIError:
    return genai_errors.APIError(
        code, {"error": {"code": code, "message": f"err {code}", "status": "X"}}
    )


def test_parses_function_calls_and_text():
    parsed = parse_gemini_response(
        response(
            {"text": "Checking."},
            {"function_call": {"id": "fc1", "name": "list_orders", "args": {"limit": 5}}},
            {"function_call": {"name": "get_customer_profile", "args": {}}},
        )
    )
    assert parsed.text == "Checking."
    first, second = parsed.tool_calls
    assert (first.name, first.arguments, first.provider_id) == ("list_orders", {"limit": 5}, "fc1")
    assert second.provider_id is None and second.id.startswith("call_")
    assert parsed.usage["total_token_count"] == 120
    assert isinstance(parsed.provider_data, types.Content)


def test_thought_parts_are_not_shown_as_text():
    parsed = parse_gemini_response(
        response({"text": "internal reasoning", "thought": True}, {"text": "Hello!"})
    )
    assert parsed.text == "Hello!"


def test_blocked_prompt_yields_empty_response():
    blocked = types.GenerateContentResponse.model_validate(
        {"prompt_feedback": {"block_reason": "SAFETY"}}
    )
    parsed = parse_gemini_response(blocked)
    assert parsed.text is None and parsed.tool_calls == ()
    assert parsed.finish_reason.startswith("blocked")


def test_history_conversion_round_trip():
    raw = types.Content(
        role="model",
        parts=[
            types.Part(
                function_call=types.FunctionCall(id="fc1", name="list_orders", args={}),
                thought_signature=b"sig",
            )
        ],
    )
    messages = [
        Message(role="user", text="my orders?"),
        Message(
            role="assistant",
            tool_calls=(ToolCall("c1", "list_orders", {}, "fc1"),),
            provider_data=raw,
        ),
        Message(role="tool", tool_results=(ToolResult("c1", "list_orders", {"ok": True}, "fc1"),)),
        Message(role="assistant", text="You have 3 orders."),
    ]
    contents = to_gemini_contents(messages)

    assert [c.role for c in contents] == ["user", "model", "user", "model"]
    assert contents[1] is raw  # replayed verbatim, thought signature intact
    fr = contents[2].parts[0].function_response
    assert (fr.id, fr.name, fr.response) == ("fc1", "list_orders", {"ok": True})
    assert contents[3].parts[0].text == "You have 3 orders."


def test_assistant_without_provider_data_is_rebuilt():
    message = Message(
        role="assistant",
        text="Let me check",
        tool_calls=(ToolCall("c1", "list_orders", {"limit": 3}, None),),
    )
    (content,) = to_gemini_contents([message])
    assert content.role == "model"
    assert content.parts[0].text == "Let me check"
    assert content.parts[1].function_call.args == {"limit": 3}


def test_request_config_disables_automatic_function_calling():
    gemini, fake = client(response({"text": "hi"}))
    gemini.generate(
        system_prompt="be nice", messages=[Message(role="user", text="hi")], tools=TOOLS
    )
    sent = fake.models.calls[0]
    config = sent["config"]
    assert sent["model"] == "gemini-test"
    assert config.automatic_function_calling.disable is True
    assert config.system_instruction == "be nice"
    names = {d.name for d in config.tools[0].function_declarations}
    assert names == {t.name for t in TOOLS}


def test_no_tools_sends_no_tool_config():
    gemini, fake = client(response({"text": "hi"}))
    gemini.generate(system_prompt="x", messages=[Message(role="user", text="hi")], tools=[])
    assert fake.models.calls[0]["config"].tools is None


@pytest.mark.parametrize("code", [429, 500, 503])
def test_transient_errors_are_retried(code):
    gemini, fake = client(api_error(code), api_error(code), response({"text": "ok"}), max_retries=3)
    result = gemini.generate(
        system_prompt="x", messages=[Message(role="user", text="hi")], tools=[]
    )
    assert result.text == "ok"
    assert len(fake.models.calls) == 3
    assert len(gemini.sleeps) == 2 and gemini.sleeps[1] > gemini.sleeps[0]


def test_retries_are_bounded():
    gemini, fake = client(*(api_error(429) for _ in range(3)), max_retries=2)
    with pytest.raises(LLMUnavailableError):
        gemini.generate(system_prompt="x", messages=[Message(role="user", text="hi")], tools=[])
    assert len(fake.models.calls) == 3


def test_network_errors_are_retried():
    gemini, _ = client(ConnectionError("reset"), response({"text": "ok"}))
    assert (
        gemini.generate(
            system_prompt="x", messages=[Message(role="user", text="hi")], tools=[]
        ).text
        == "ok"
    )


@pytest.mark.parametrize("code", [400, 401, 403, 404])
def test_configuration_errors_are_not_retried(code):
    gemini, fake = client(api_error(code))
    with pytest.raises(LLMConfigurationError):
        gemini.generate(system_prompt="x", messages=[Message(role="user", text="hi")], tools=[])
    assert len(fake.models.calls) == 1


def test_missing_api_key_is_a_configuration_error():
    with pytest.raises(LLMConfigurationError, match="GEMINI_API_KEY"):
        GeminiClient(api_key=None, model="gemini-test")


def test_tool_schemas_are_accepted_by_sdk():
    for tool in TOOLS:
        declaration = types.FunctionDeclaration(
            name=tool.name, description=tool.description, parameters_json_schema=tool.parameters
        )
        assert declaration.name == tool.name


def test_schema_simplification():
    schemas = {t.name: t.parameters for t in TOOLS}
    refund = schemas["request_refund"]
    assert refund["required"] == ["order_number", "reason"]
    assert "enum" in refund["properties"]["reason"]  # $ref inlined
    assert refund["properties"]["amount_cents"]["type"] == "integer"  # Optional collapsed
    text = str(schemas)
    assert "$ref" not in text and "anyOf" not in text and "additionalProperties" not in text
    assert schemas["get_customer_profile"] == {"properties": {}, "type": "object"}


def test_definitions_are_plain_tooldefinitions():
    assert all(isinstance(t, ToolDefinition) for t in TOOLS)

"""Checkpointed messages are plain JSON and replay to Gemini without losing thought signatures."""

import json

import pytest
from google.genai import types

from support_agent.agent.gemini import to_gemini_contents
from support_agent.agent.llm import Message, ToolCall, message_from_dict, message_to_dict
from support_agent.db.models import RefundStatus
from support_agent.workflow.checkpoint import strict_serializer


def test_gemini_content_round_trips_through_json():
    signature = b"\x00\x9f-opaque-signature-\xff"
    content = types.Content(
        role="model",
        parts=[
            types.Part(
                function_call=types.FunctionCall(id="fc_1", name="list_orders", args={}),
                thought_signature=signature,
            )
        ],
    )
    message = Message(
        role="assistant",
        tool_calls=(ToolCall("call_1", "list_orders", {}, "fc_1"),),
        provider_data=content,
    )

    stored = json.loads(json.dumps(message_to_dict(message)))  # what the checkpoint keeps
    (replayed,) = to_gemini_contents([message_from_dict(stored)])

    assert replayed.parts[0].thought_signature == signature
    assert replayed.parts[0].function_call.name == "list_orders"


def test_strict_serializer_never_revives_python_objects():
    serde = strict_serializer()
    assert serde.loads_typed(serde.dumps_typed({"plain": [1, "two", None]})) == {
        "plain": [1, "two", None]
    }

    class Custom:
        pass

    with pytest.raises(TypeError):
        serde.dumps_typed({"obj": Custom()})

    # Even a known enum comes back as its plain value, not as the class.
    revived = serde.loads_typed(serde.dumps_typed({"status": RefundStatus.APPROVED}))
    assert type(revived["status"]) is str

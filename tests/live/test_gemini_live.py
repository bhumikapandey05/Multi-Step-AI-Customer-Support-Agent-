"""Optional smoke tests against the real Gemini API (uses free-tier quota).

Skipped unless both are set:  RUN_LIVE_TESTS=1  and  GEMINI_API_KEY=...
Run with:  RUN_LIVE_TESTS=1 pytest -m live -v
"""

import os

import pytest
from sqlalchemy import func, select

from support_agent.agent.agent import AgentConfig, SupportAgent
from support_agent.agent.factory import build_llm
from support_agent.config import Settings
from support_agent.db.engine import transaction
from support_agent.db.models import RefundRequest
from tests.conftest import ALICE

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        os.getenv("RUN_LIVE_TESTS") != "1" or not Settings().gemini_api_key,
        reason="set RUN_LIVE_TESTS=1 and GEMINI_API_KEY to run live Gemini tests",
    ),
]


@pytest.fixture
def live_agent(crm, login, clock):
    return SupportAgent(
        build_llm(Settings()), crm, ctx=login(ALICE), config=AgentConfig(), clock=clock
    )


def test_model_uses_tools_to_answer(live_agent):
    reply = live_agent.send("Which of my orders have been delivered?")
    assert any(e.tool in {"list_orders", "get_order_details"} for e in reply.tool_executions)
    assert "ORD-10001" in reply.text


def test_refund_request_waits_for_human_approval(live_agent, session_factory):
    reply = live_agent.send(
        "My headphones from order ORD-10001 are defective. Please submit a refund request."
    )
    with transaction(session_factory) as s:
        assert s.scalar(select(func.count()).select_from(RefundRequest)) == 2  # nothing written
    if reply.pending_approvals:
        assert reply.pending_approvals[0].tool == "request_refund"

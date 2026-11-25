"""Write tools require explicit human approval and are idempotent."""

from datetime import timedelta

import pytest
from sqlalchemy import func, select

from support_agent.agent.agent import AgentError, ExecutionStatus, StopReason
from support_agent.agent.tools import RefundArgs
from support_agent.db.engine import transaction
from support_agent.db.models import AuditOutcome, RefundRequest, SupportTicket
from tests.agent.conftest import ScriptedLLM, call, say, tools
from tests.conftest import ALICE, CARMEN, EMMA

pytestmark = pytest.mark.integration

REFUND = {"order_number": "ORD-10001", "reason": "defective", "details": "Left side crackles"}
TICKET = {
    "category": "shipping",
    "order_number": "ORD-10003",
    "subject": "Package is late",
    "description": "Tracking has not updated in days.",
}


def count(session_factory, model) -> int:
    with transaction(session_factory) as s:
        return s.scalar(select(func.count()).select_from(model))


def test_refund_pauses_for_approval_and_writes_nothing(make_agent, session_factory):
    llm = ScriptedLLM(tools(call("request_refund", **REFUND), text="Let me set that up."))
    reply = make_agent(llm, ALICE).send("Please refund my headphones")

    assert reply.stop_reason is StopReason.AWAITING_APPROVAL
    (pending,) = reply.pending_approvals
    assert "$129.99" in pending.summary and "ORD-10001" in pending.summary
    assert pending.details["amount_cents"] == 12_999
    assert reply.tool_executions[0].status is ExecutionStatus.AWAITING_APPROVAL
    assert "Please confirm" in reply.text
    assert count(session_factory, RefundRequest) == 2  # seeded only
    assert len(llm.requests) == 1  # model waits until a human decides


def test_approve_executes_and_model_sees_result(make_agent, session_factory, audit_rows):
    llm = ScriptedLLM(tools(call("request_refund", **REFUND)), say("Submitted for review."))
    agent = make_agent(llm, ALICE)
    pending = agent.send("refund please").pending_approvals[0]

    reply = agent.approve(pending.action_id)

    assert reply.stop_reason is StopReason.COMPLETED
    execution = reply.tool_executions[0]
    assert execution.approved and execution.status is ExecutionStatus.SUCCESS
    result = llm.requests[1].last_results()["request_refund"]
    assert result["ok"] and result["data"]["created"]
    assert count(session_factory, RefundRequest) == 3

    (approval,) = audit_rows("agent.approval.granted")
    assert approval.outcome is AuditOutcome.SUCCESS and approval.entity_id == "request_refund"
    assert audit_rows("refund.request")[-1].outcome is AuditOutcome.SUCCESS


def test_reject_writes_nothing_and_tells_model_not_to_retry(
    make_agent, session_factory, audit_rows
):
    llm = ScriptedLLM(tools(call("request_refund", **REFUND)), say("No problem."))
    agent = make_agent(llm, ALICE)
    pending = agent.send("refund").pending_approvals[0]

    reply = agent.reject(pending.action_id)

    assert reply.text == "No problem."
    assert llm.requests[1].last_results()["request_refund"]["error"] == "declined_by_customer"
    assert count(session_factory, RefundRequest) == 2
    assert audit_rows("agent.approval.declined")


def test_approval_cannot_be_used_twice_or_guessed(make_agent):
    llm = ScriptedLLM(tools(call("request_refund", **REFUND)), say("Done."))
    agent = make_agent(llm, ALICE)
    action_id = agent.send("refund").pending_approvals[0].action_id
    agent.approve(action_id)
    with pytest.raises(AgentError):
        agent.approve(action_id)
    with pytest.raises(AgentError):
        agent.approve("act_madeup")


def test_new_message_instead_of_decision_cancels_pending(make_agent, session_factory):
    llm = ScriptedLLM(tools(call("request_refund", **REFUND)), say("Okay, I won't submit it."))
    agent = make_agent(llm, ALICE)
    agent.send("refund")

    reply = agent.send("actually, wait")

    assert agent.pending_approvals == []
    assert reply.tool_executions[0].status is ExecutionStatus.DECLINED
    tool_msg = llm.requests[1].messages[-2]
    assert tool_msg.role == "tool"
    assert tool_msg.tool_results[0].content["error"] == "action_cancelled"
    assert llm.requests[1].messages[-1].text == "actually, wait"
    assert count(session_factory, RefundRequest) == 2


def test_expired_approval_is_not_executed(make_agent, session_factory, clock):
    llm = ScriptedLLM(tools(call("request_refund", **REFUND)), say("It expired."))
    agent = make_agent(llm, ALICE, approval_ttl=timedelta(minutes=5))
    action_id = agent.send("refund").pending_approvals[0].action_id
    clock.advance(timedelta(minutes=6))

    reply = agent.approve(action_id)

    assert reply.tool_executions[0].result["error"] == "approval_expired"
    assert count(session_factory, RefundRequest) == 2


def test_ineligible_refund_is_reported_without_asking(make_agent):
    llm = ScriptedLLM(
        tools(call("request_refund", order_number="ORD-10002", reason="damaged")),
        say("That order is outside the refund window."),
    )
    reply = make_agent(llm, ALICE).send("refund my jacket")
    assert reply.stop_reason is StopReason.COMPLETED and reply.pending_approvals == []
    result = llm.requests[1].last_results()["request_refund"]
    assert result["error"] == "refund_not_eligible" and "window" in result["message"]


def test_amount_above_refundable_is_reported_without_asking(make_agent):
    llm = ScriptedLLM(
        tools(
            call("request_refund", order_number="ORD-10007", reason="damaged", amount_cents=7_900)
        ),
        say("Only the dress can be refunded."),
    )
    reply = make_agent(llm, CARMEN).send("refund the whole order")
    assert reply.pending_approvals == []
    assert reply.tool_executions[0].result["details"]["refundable_amount_cents"] == 6_400


def test_refund_for_someone_elses_order_never_reaches_approval(make_agent):
    llm = ScriptedLLM(
        tools(call("request_refund", order_number="ORD-10004", reason="damaged")),
        say("I can't find that order."),
    )
    reply = make_agent(llm, ALICE).send("refund ORD-10004")
    assert reply.pending_approvals == []
    assert reply.tool_executions[0].result["error"] == "not_found"


def test_suspended_account_is_blocked_by_crm_even_after_approval(make_agent, session_factory):
    llm = ScriptedLLM(
        tools(call("request_refund", order_number="ORD-10012", reason="damaged")),
        say("A human agent will need to help."),
    )
    agent = make_agent(llm, EMMA)
    pending = agent.send("refund my yoga mat").pending_approvals[0]
    # The preview is a read, so it succeeds. The CRM itself blocks the write.
    reply = agent.approve(pending.action_id)
    assert reply.tool_executions[0].result["error"] == "not_authorized"
    assert count(session_factory, RefundRequest) == 2


def test_ticket_requires_approval_by_default(make_agent, session_factory):
    llm = ScriptedLLM(tools(call("create_support_ticket", **TICKET)), say("Ticket opened."))
    agent = make_agent(llm, ALICE)
    reply = agent.send("my package is late")
    assert reply.stop_reason is StopReason.AWAITING_APPROVAL
    assert "shipping ticket about order ORD-10003" in reply.pending_approvals[0].summary
    agent.approve(reply.pending_approvals[0].action_id)
    assert count(session_factory, SupportTicket) == 3


def test_ticket_approval_can_be_disabled(make_agent, session_factory):
    llm = ScriptedLLM(tools(call("create_support_ticket", **TICKET)), say("Ticket opened."))
    reply = make_agent(llm, ALICE, require_ticket_approval=False).send("my package is late")
    assert reply.stop_reason is StopReason.COMPLETED
    assert reply.tool_executions[0].result["data"]["created"] is True
    assert count(session_factory, SupportTicket) == 3


def test_two_writes_in_one_round_are_approved_individually(make_agent, session_factory):
    llm = ScriptedLLM(
        tools(
            call("request_refund", **REFUND),
            call("create_support_ticket", **TICKET),
            call("list_orders"),
        ),
        say("Refund submitted; ticket skipped."),
    )
    agent = make_agent(llm, ALICE)
    reply = agent.send("refund my headphones and open a ticket about my late package")
    refund_action, ticket_action = reply.pending_approvals
    assert reply.tool_executions[2].status is ExecutionStatus.SUCCESS  # the read ran right away

    partial = agent.approve(refund_action.action_id)
    assert partial.stop_reason is StopReason.AWAITING_APPROVAL
    assert len(llm.requests) == 1  # still waiting on the second decision

    final = agent.reject(ticket_action.action_id)
    assert final.stop_reason is StopReason.COMPLETED
    results = llm.requests[1].last_results()
    assert results["request_refund"]["ok"] is True
    assert results["create_support_ticket"]["error"] == "declined_by_customer"
    assert results["list_orders"]["ok"] is True
    assert count(session_factory, RefundRequest) == 3
    assert count(session_factory, SupportTicket) == 2


def test_repeated_identical_write_is_not_repeated(make_agent, session_factory):
    llm = ScriptedLLM(
        tools(call("create_support_ticket", **TICKET)),
        say("Opened."),
        tools(call("create_support_ticket", **TICKET)),
        say("It's already open."),
    )
    agent = make_agent(llm, ALICE)
    agent.approve(agent.send("late package").pending_approvals[0].action_id)

    reply = agent.send("open it again please")

    assert reply.pending_approvals == []  # no second confirmation for an identical action
    assert reply.tool_executions[0].status is ExecutionStatus.CACHED
    assert count(session_factory, SupportTicket) == 3


def test_write_is_idempotent_even_if_executed_twice(make_agent, crm, session_factory):
    """The agent's derived idempotency key turns a re-execution into a CRM replay."""
    llm = ScriptedLLM(tools(call("request_refund", **REFUND)), say("ok"))
    agent = make_agent(llm, ALICE)
    agent.approve(agent.send("refund").pending_approvals[0].action_id)

    tool = agent._tools["request_refund"]
    args = RefundArgs.model_validate(REFUND)
    again = tool.handler(crm, agent._ctx, args, agent._idempotency_key(tool, args))
    assert again.created is False
    assert count(session_factory, RefundRequest) == 3


def test_approval_that_cannot_be_recorded_fails_closed(make_agent, session_factory, monkeypatch):
    llm = ScriptedLLM(tools(call("request_refund", **REFUND)), say("Please try again."))
    agent = make_agent(llm, ALICE)
    action_id = agent.send("refund").pending_approvals[0].action_id

    def broken(*a, **k):
        raise RuntimeError("audit db down")

    monkeypatch.setattr(agent._auditor, "record", broken)
    reply = agent.approve(action_id)
    assert reply.tool_executions[0].result["error"] == "approval_not_recorded"
    assert count(session_factory, RefundRequest) == 2


def test_model_has_no_tool_that_grants_approval(make_agent):
    llm = ScriptedLLM(say("ok"))
    make_agent(llm, ALICE).send("hi")
    assert not any("approv" in n or "confirm" in n for n in llm.requests[0].tool_names)

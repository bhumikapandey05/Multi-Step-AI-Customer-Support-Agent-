"""Conversations and pending confirmations survive restarts, and resuming never duplicates."""

import json
from datetime import UTC, datetime, timedelta

import pytest
from langgraph.types import Command
from sqlalchemy.exc import OperationalError

from support_agent.agent.agent import (
    RETRY_LATER_TEXT,
    AgentError,
    ExecutionStatus,
    StopReason,
    ThreadOwnershipError,
)
from support_agent.db.engine import transaction
from support_agent.db.models import AuditLog, Order, RefundReason, RefundRequest, RefundStatus
from tests.agent.conftest import ScriptedLLM, call, say, tools
from tests.conftest import ALICE, BRIAN
from tests.workflow.conftest import REFUND, SimulatedCrash, count

pytestmark = pytest.mark.integration


def pause_for_refund(make_wf_agent):
    """Alice asks for a refund; the workflow stops at the confirmation checkpoint."""
    first = make_wf_agent(ScriptedLLM(tools(call("request_refund", **REFUND))), ALICE)
    reply = first.send("Please refund my headphones")
    assert reply.stop_reason is StopReason.AWAITING_APPROVAL
    return first.thread_id, reply.pending_approvals[0]


def test_pending_confirmation_survives_restart_and_approval_runs_once(
    make_wf_agent, checkpoints, session_factory
):
    thread_id, pending = pause_for_refund(make_wf_agent)
    saver = checkpoints.restart()

    llm = ScriptedLLM(say("Submitted for review."))
    resumed = make_wf_agent(llm, ALICE, thread_id=thread_id, saver=saver)
    assert [p.action_id for p in resumed.pending_approvals] == [pending.action_id]
    assert resumed.pending_approvals[0].summary == pending.summary

    reply = resumed.approve(pending.action_id)

    assert reply.stop_reason is StopReason.COMPLETED and reply.text == "Submitted for review."
    assert reply.tool_executions[0].approved
    assert count(session_factory, RefundRequest) == 3
    # The model after the restart sees the whole conversation, including the result.
    roles = [m.role for m in llm.requests[0].messages]
    assert roles == ["user", "assistant", "tool"]
    assert llm.requests[0].last_results()["request_refund"]["data"]["created"] is True
    with pytest.raises(AgentError):
        resumed.approve(pending.action_id)


def test_rejection_after_restart_writes_nothing(
    make_wf_agent, checkpoints, session_factory, audit_rows
):
    thread_id, pending = pause_for_refund(make_wf_agent)
    llm = ScriptedLLM(say("Okay, not submitted."))
    resumed = make_wf_agent(llm, ALICE, thread_id=thread_id, saver=checkpoints.restart())

    resumed.reject(pending.action_id)

    assert count(session_factory, RefundRequest) == 2
    assert llm.requests[0].last_results()["request_refund"]["error"] == "declined_by_customer"
    (declined,) = audit_rows("agent.approval.declined")
    assert declined.details["thread_id"] == thread_id


def test_new_message_after_restart_cancels_pending(make_wf_agent, checkpoints, session_factory):
    thread_id, _ = pause_for_refund(make_wf_agent)
    llm = ScriptedLLM(say("Sure, I won't submit it."))
    resumed = make_wf_agent(llm, ALICE, thread_id=thread_id, saver=checkpoints.restart())

    reply = resumed.send("never mind")

    assert reply.tool_executions[0].status is ExecutionStatus.DECLINED
    assert resumed.pending_approvals == []
    assert llm.requests[0].messages[-2].tool_results[0].content["error"] == "action_cancelled"
    assert count(session_factory, RefundRequest) == 2


def test_conversation_memory_survives_restart(make_wf_agent, checkpoints):
    agent = make_wf_agent(ScriptedLLM(say("Hi Alice!")), ALICE)
    agent.send("Hello")
    llm = ScriptedLLM(say("You said hello."))
    resumed = make_wf_agent(llm, ALICE, thread_id=agent.thread_id, saver=checkpoints.restart())
    resumed.send("What did I say?")
    assert [m.text for m in llm.requests[0].messages] == ["Hello", "Hi Alice!", "What did I say?"]


def test_another_customer_cannot_resume_the_thread(make_wf_agent, audit_rows):
    thread_id, _ = pause_for_refund(make_wf_agent)
    with pytest.raises(ThreadOwnershipError):
        make_wf_agent(ScriptedLLM(), BRIAN, thread_id=thread_id)
    with pytest.raises(ThreadOwnershipError):
        make_wf_agent(ScriptedLLM(), None, thread_id=thread_id)  # guest
    denied = audit_rows("workflow.resume")
    assert len(denied) == 2 and denied[0].details["reason"] == "not_thread_owner"


def test_checkpoints_hold_no_credentials(make_wf_agent, checkpoints, login):
    agent = make_wf_agent(ScriptedLLM(tools(call("request_refund", **REFUND))), ALICE)
    agent.send("refund please")
    checkpoints.saver.conn.commit()
    raw = b"".join(
        p.read_bytes() for p in checkpoints.path.parent.glob("checkpoints.db*") if p.is_file()
    )
    assert agent._ctx.signature.encode() not in raw
    assert b"246810" not in raw  # Alice's PIN
    assert b"customer:" in raw  # the owner binding is stored; it is not a credential


def test_invalid_resume_payload_fails_closed(make_wf_agent, session_factory):
    agent = make_wf_agent(ScriptedLLM(tools(call("request_refund", **REFUND))), ALICE)
    pending = agent.send("refund").pending_approvals[0]
    # Bypass the public API and resume the graph directly with garbage.
    for payload in ({"decision": "approve", "action_id": "act_forged"}, {"approved": True}, "yes"):
        agent._graph.invoke(Command(resume=payload), agent._run_config)
        assert [p.action_id for p in agent.pending_approvals] == [pending.action_id]
    assert count(session_factory, RefundRequest) == 2


def test_approval_expires_across_a_long_restart(make_wf_agent, checkpoints, clock, session_factory):
    thread_id, pending = pause_for_refund(make_wf_agent)
    clock.advance(timedelta(minutes=11))
    resumed = make_wf_agent(
        ScriptedLLM(say("That expired.")), ALICE, thread_id=thread_id, saver=checkpoints.restart()
    )
    reply = resumed.approve(pending.action_id)
    assert reply.tool_executions[0].result["error"] == "approval_expired"
    assert count(session_factory, RefundRequest) == 2


def test_stale_confirmation_is_not_executed(make_wf_agent, session_factory):
    """If the CRM now computes a different refund, the old 'yes' does not apply to it."""
    agent = make_wf_agent(
        ScriptedLLM(tools(call("request_refund", **REFUND)), say("Changed.")), ALICE
    )
    pending = agent.send("refund").pending_approvals[0]
    assert pending.details["amount_cents"] == 12_999
    with transaction(session_factory) as s:  # meanwhile, staff approve a partial refund
        order = s.query(Order).filter_by(order_number="ORD-10001").one()
        s.add(
            RefundRequest(
                refund_number="RFD-0000AAAA",
                customer_id=order.customer_id,
                order_id=order.id,
                amount_cents=2_000,
                reason=RefundReason.DAMAGED,
                status=RefundStatus.APPROVED,
                policy_version="2026.1",
                created_at=datetime(2026, 6, 15, tzinfo=UTC),
                updated_at=datetime(2026, 6, 15, tzinfo=UTC),
            )
        )

    reply = agent.approve(pending.action_id)

    result = reply.tool_executions[0].result
    assert result["error"] == "confirmation_outdated"
    assert result["details"]["current"]["amount_cents"] == 10_999
    assert count(session_factory, RefundRequest, RefundRequest.idempotency_key.is_not(None)) == 0


# ------------------------------------------------------------------ crashes and retries


def test_crash_after_crm_write_resumes_without_duplicate(
    make_wf_agent, checkpoints, crm, monkeypatch, session_factory, audit_rows
):
    thread_id, pending = pause_for_refund(make_wf_agent)
    original = crm.request_refund

    def write_then_die(*args, **kwargs):
        original(*args, **kwargs)  # committed in the CRM...
        raise SimulatedCrash  # ...but the process dies before the checkpoint is saved

    monkeypatch.setattr(crm, "request_refund", write_then_die)
    agent = make_wf_agent(ScriptedLLM(), ALICE, thread_id=thread_id)
    with pytest.raises(SimulatedCrash):
        agent.approve(pending.action_id)
    monkeypatch.setattr(crm, "request_refund", original)
    assert count(session_factory, RefundRequest) == 3

    llm = ScriptedLLM(say("Your request is in."))
    restarted = make_wf_agent(llm, ALICE, thread_id=thread_id, saver=checkpoints.restart())
    assert restarted.needs_resume and restarted.pending_approvals == []
    reply = restarted.resume()

    assert reply.stop_reason is StopReason.COMPLETED
    assert count(session_factory, RefundRequest) == 3  # replayed, not duplicated
    assert [r.outcome.value for r in audit_rows("refund.request")] == ["success", "duplicate"]
    assert len(audit_rows("agent.approval.granted")) == 1  # the human decided once
    data = llm.requests[0].last_results()["request_refund"]["data"]
    assert data["refund"]["status"] == "pending_review"


def test_transient_database_errors_are_retried(
    make_wf_agent, crm, monkeypatch, sleeps, session_factory
):
    agent = make_wf_agent(ScriptedLLM(tools(call("request_refund", **REFUND)), say("Done.")), ALICE)
    pending = agent.send("refund").pending_approvals[0]
    original, failures = crm.request_refund, [2]

    def flaky(*args, **kwargs):
        if failures[0]:
            failures[0] -= 1
            raise OperationalError("INSERT ...", {}, Exception("database is locked"))
        return original(*args, **kwargs)

    monkeypatch.setattr(crm, "request_refund", flaky)
    reply = agent.approve(pending.action_id)

    assert reply.stop_reason is StopReason.COMPLETED
    # Bounded exponential backoff: base 0.01s, doubling, plus up to 50% jitter.
    assert len(sleeps) == 2
    assert 0.01 <= sleeps[0] <= 0.015 and 0.02 <= sleeps[1] <= 0.03
    assert count(session_factory, RefundRequest) == 3


def test_persistent_outage_stalls_safely_then_resumes_once(
    make_wf_agent, checkpoints, crm, monkeypatch, sleeps, session_factory
):
    thread_id, pending = pause_for_refund(make_wf_agent)
    original = crm.request_refund
    attempts = []

    def down(*args, **kwargs):
        attempts.append(1)
        raise OperationalError("INSERT ...", {}, Exception("disk I/O error"))

    monkeypatch.setattr(crm, "request_refund", down)
    agent = make_wf_agent(ScriptedLLM(), ALICE, thread_id=thread_id)
    reply = agent.approve(pending.action_id)

    assert reply.stop_reason is StopReason.RETRY_LATER and reply.text == RETRY_LATER_TEXT
    assert len(attempts) == 3  # 1 try + 2 retries, then stop
    assert agent.needs_resume and count(session_factory, RefundRequest) == 2

    monkeypatch.setattr(crm, "request_refund", original)
    llm = ScriptedLLM(say("Done now."), say("Anything else?"))
    restarted = make_wf_agent(llm, ALICE, thread_id=thread_id, saver=checkpoints.restart())
    # A new message first finishes the already-approved action, then is answered.
    reply = restarted.send("did it work?")
    assert reply.text == "Anything else?"
    assert count(session_factory, RefundRequest) == 3
    assert restarted.resume() is None


def test_transient_read_failure_is_reported_to_the_model(make_wf_agent, crm, monkeypatch):
    def down(*a, **k):
        raise OperationalError("SELECT ...", {}, Exception("database is locked"))

    monkeypatch.setattr(crm, "list_orders", down)
    llm = ScriptedLLM(tools(call("list_orders")), say("Please try again shortly."))
    reply = make_wf_agent(llm, ALICE).send("orders?")
    assert llm.requests[1].last_results()["list_orders"]["error"] == "temporarily_unavailable"
    assert reply.stop_reason is StopReason.COMPLETED


def test_crash_while_waiting_for_the_model_resumes_the_turn(make_wf_agent, checkpoints):
    def die(messages):
        raise SimulatedCrash

    agent = make_wf_agent(ScriptedLLM(tools(call("list_orders")), die), ALICE)
    with pytest.raises(SimulatedCrash):
        agent.send("orders?")
    llm = ScriptedLLM(say("You have 3 orders."))
    restarted = make_wf_agent(llm, ALICE, thread_id=agent.thread_id, saver=checkpoints.restart())
    assert restarted.needs_resume
    assert restarted.resume().text == "You have 3 orders."
    assert [m.role for m in llm.requests[0].messages] == ["user", "assistant", "tool"]


def test_audit_trail_links_decisions_to_the_thread(make_wf_agent, audit_rows):
    agent = make_wf_agent(ScriptedLLM(tools(call("request_refund", **REFUND)), say("ok")), ALICE)
    agent.approve(agent.send("refund").pending_approvals[0].action_id)
    (granted,) = audit_rows("agent.approval.granted")
    assert granted.details["thread_id"] == agent.thread_id
    assert "PIN" not in json.dumps(granted.details)
    assert audit_rows("refund.request")[-1].outcome.value == "success"
    assert isinstance(audit_rows()[0], AuditLog)

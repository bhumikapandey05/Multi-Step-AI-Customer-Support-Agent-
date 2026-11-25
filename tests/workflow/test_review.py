"""Staff approval of an actual refund: a separate, authorized, resumable human decision."""

import dataclasses

import pytest
from sqlalchemy import select
from sqlalchemy.exc import OperationalError

from support_agent.crm.errors import (
    AuthenticationError,
    ConflictError,
    IdempotencyConflictError,
)
from support_agent.db.engine import transaction
from support_agent.db.models import (
    AccountStatus,
    Customer,
    RefundDecision,
    RefundRequest,
    RefundStatus,
    StaffMember,
    StaffRole,
)
from support_agent.workflow.retry import TransientFailure
from support_agent.workflow.review import ReviewWorkflowError, review_thread_id
from tests.agent.conftest import ScriptedLLM, call, say, tools
from tests.conftest import ALICE, BRIAN, DANIEL, EMMA
from tests.workflow.conftest import (
    AGENT,
    APPROVER,
    REFUND,
    SUPERVISOR,
    SimulatedCrash,
    count,
)

pytestmark = pytest.mark.integration


def customer_confirms_refund(make_wf_agent, email=ALICE, **args) -> str:
    """Customer confirmation: creates a refund *request*. Returns its number."""
    llm = ScriptedLLM(tools(call("request_refund", **(args or REFUND))), say("Submitted."))
    agent = make_wf_agent(llm, email)
    reply = agent.approve(agent.send("refund please").pending_approvals[0].action_id)
    return reply.tool_executions[0].result["data"]["refund"]["refund_number"]


def refund_status(session_factory, number) -> RefundStatus:
    with transaction(session_factory) as s:
        return s.scalar(select(RefundRequest.status).where(RefundRequest.refund_number == number))


def test_customer_confirmation_is_not_a_refund_approval(
    make_wf_agent, make_review, session_factory
):
    number = customer_confirms_refund(make_wf_agent)

    # The customer said yes, but that only filed a request. Nothing is approved yet.
    assert refund_status(session_factory, number) is RefundStatus.PENDING_REVIEW
    assert count(session_factory, RefundDecision) == 0
    status = make_review().status(number)  # opened automatically by the conversation
    assert status.waiting_for_staff and status.stage == "awaiting_staff_decision"
    assert status.case["required_role"] == "refund_approver"


def test_approver_approves_a_standard_refund(
    make_wf_agent, make_review, staff_login, session_factory, audit_rows
):
    number = customer_confirms_refund(make_wf_agent)
    status = make_review().decide(staff_login(APPROVER), number, "approve", "Defect confirmed")

    assert status.stage == "decided" and not status.waiting_for_staff
    assert status.result["decision"] == "approved" and status.result["decided_by"] == "STF-002"
    assert "No payment was processed" in status.result["payout"]
    assert refund_status(session_factory, number) is RefundStatus.APPROVED
    (row,) = audit_rows("refund.review.approved")
    assert row.actor.startswith("staff:") and row.details["payout"] == "not_processed"
    assert row.details["workflow_thread_id"] == review_thread_id(number)


def test_rejection_requires_a_note(make_wf_agent, make_review, staff_login, session_factory):
    number = customer_confirms_refund(make_wf_agent)
    review, riley = make_review(), staff_login(APPROVER)

    status = review.decide(riley, number, "reject", None)
    assert status.waiting_for_staff and status.last_attempt["error"] == "invalid_input"

    status = review.decide(riley, number, "reject", "Item shows physical damage")
    assert status.result["decision"] == "rejected"
    assert refund_status(session_factory, number) is RefundStatus.REJECTED


def test_support_agent_role_cannot_decide(make_wf_agent, make_review, staff_login, audit_rows):
    number = customer_confirms_refund(make_wf_agent)
    status = make_review().decide(staff_login(AGENT), number, "approve", None)

    assert status.waiting_for_staff  # still paused; the refused attempt is on record
    assert status.last_attempt["error"] == "not_authorized"
    assert audit_rows("refund.review.approved")[-1].outcome.value == "denied"


def test_manual_review_refund_needs_a_supervisor(
    make_wf_agent, make_review, staff_login, session_factory
):
    number = customer_confirms_refund(
        make_wf_agent, BRIAN, order_number="ORD-10005", reason="defective"
    )
    review = make_review()
    assert review.status(number).case["required_role"] == "supervisor"

    denied = review.decide(staff_login(APPROVER), number, "approve", None)
    assert denied.waiting_for_staff and "supervisor" in denied.last_attempt["message"]
    assert refund_status(session_factory, number) is RefundStatus.PENDING_REVIEW

    approved = review.decide(staff_login(SUPERVISOR), number, "approve", "Laptop DOA")
    assert approved.result["required_role"] == "supervisor"
    assert refund_status(session_factory, number) is RefundStatus.APPROVED


def test_customer_session_is_not_a_staff_session(review_service, login):
    with pytest.raises(AuthenticationError):
        review_service.list_queue(login(ALICE))  # type: ignore[arg-type]


def test_forged_staff_role_is_rejected(review_service, staff_login):
    forged = dataclasses.replace(staff_login(APPROVER), role=StaffRole.SUPERVISOR)
    with pytest.raises(AuthenticationError):
        review_service.list_queue(forged)


def test_role_is_re_read_from_the_database(
    make_wf_agent, make_review, staff_login, session_factory
):
    number = customer_confirms_refund(make_wf_agent)
    riley = staff_login(APPROVER)
    with transaction(session_factory) as s:  # role revoked after sign-in
        s.scalar(
            select(StaffMember).where(StaffMember.email == APPROVER)
        ).role = StaffRole.SUPPORT_AGENT
    status = make_review().decide(riley, number, "approve", None)
    assert status.waiting_for_staff and status.last_attempt["error"] == "not_authorized"


def test_a_decision_cannot_be_made_twice(make_wf_agent, make_review, staff_login, session_factory):
    number = customer_confirms_refund(make_wf_agent)
    review = make_review()
    review.decide(staff_login(APPROVER), number, "approve", None)
    with pytest.raises(ReviewWorkflowError):
        review.decide(staff_login(SUPERVISOR), number, "reject", "Changed my mind")
    assert count(session_factory, RefundDecision) == 1
    assert refund_status(session_factory, number) is RefundStatus.APPROVED


def test_service_level_idempotency_and_conflicts(review_service, staff_login, session_factory):
    riley = staff_login(APPROVER)
    request = {
        "refund_number": "RFD-00000001",
        "decision": "approved",
        "idempotency_key": "review-key-0001",
    }
    first = review_service.decide(riley, request)
    again = review_service.decide(riley, request)
    assert first.created and not again.created
    with pytest.raises(IdempotencyConflictError):
        review_service.decide(riley, {**request, "decision": "rejected", "note": "nope nope"})
    with pytest.raises(ConflictError):
        review_service.decide(riley, {**request, "idempotency_key": "review-key-0002"})
    assert count(session_factory, RefundDecision) == 1


def test_approval_blocked_when_business_rules_fail(
    make_wf_agent, make_review, staff_login, session_factory
):
    number = customer_confirms_refund(make_wf_agent)
    with transaction(session_factory) as s:
        s.scalar(select(Customer).where(Customer.email == ALICE)).status = AccountStatus.SUSPENDED
    review = make_review()
    status = review.decide(staff_login(APPROVER), number, "approve", None)
    assert status.last_attempt["error"] == "refund_not_eligible"
    assert "customer_account_not_active" in status.last_attempt["message"]
    # Rejecting is still allowed.
    status = review.decide(staff_login(APPROVER), number, "reject", "Account under review")
    assert status.result["decision"] == "rejected"


def test_pending_staff_approval_survives_restart(
    make_wf_agent, make_review, checkpoints, staff_login, session_factory
):
    number = customer_confirms_refund(make_wf_agent)
    review = make_review(checkpoints.restart())
    assert review.status(number).waiting_for_staff
    review.decide(staff_login(APPROVER), number, "approve", None)
    assert refund_status(session_factory, number) is RefundStatus.APPROVED


def test_crash_after_decision_commit_resumes_without_duplicate(
    make_wf_agent, make_review, checkpoints, staff_login, review_service, monkeypatch,
    session_factory, audit_rows,
):  # fmt: skip
    number = customer_confirms_refund(make_wf_agent)
    original = review_service.decide

    def decide_then_die(*args, **kwargs):
        original(*args, **kwargs)
        raise SimulatedCrash

    monkeypatch.setattr(review_service, "decide", decide_then_die)
    riley = staff_login(APPROVER)
    with pytest.raises(SimulatedCrash):
        make_review().decide(riley, number, "approve", "ok")
    monkeypatch.setattr(review_service, "decide", original)

    review = make_review(checkpoints.restart())
    status = review.status(number)
    assert status.needs_resume and status.stage == "deciding"
    with pytest.raises(ReviewWorkflowError):
        review.decide(riley, number, "approve", "ok")  # must resume, not decide again
    with pytest.raises(ReviewWorkflowError):
        review.resume(staff_login(SUPERVISOR), number)  # only the deciding staff member

    status = review.resume(staff_login(APPROVER), number)
    assert status.stage == "decided" and status.result["created"] is False
    assert count(session_factory, RefundDecision) == 1
    assert [r.outcome.value for r in audit_rows("refund.review.approved")] == [
        "success",
        "duplicate",
    ]


def test_transient_failure_on_decision_is_retried_then_stalls(
    make_wf_agent, make_review, staff_login, review_service, monkeypatch, sleeps, session_factory
):
    number = customer_confirms_refund(make_wf_agent)
    original = review_service.decide

    def down(*a, **k):
        raise OperationalError("UPDATE ...", {}, Exception("database is locked"))

    monkeypatch.setattr(review_service, "decide", down)
    review = make_review()
    with pytest.raises(TransientFailure):
        review.decide(staff_login(APPROVER), number, "approve", None)
    assert len(sleeps) == 2 and review.status(number).needs_resume

    monkeypatch.setattr(review_service, "decide", original)
    assert review.resume(staff_login(APPROVER), number).stage == "decided"
    assert refund_status(session_factory, number) is RefundStatus.APPROVED


def test_review_can_be_opened_for_requests_filed_outside_the_chat(
    make_review, staff_login, session_factory
):
    review = make_review()
    status = review.open("RFD-00000001")  # seeded request for Daniel
    assert status.waiting_for_staff and status.case["customer_ref"] == "CUST-1004"
    assert review.open("RFD-00000001").thread_id == status.thread_id  # idempotent
    closed = review.open("RFD-00000002")  # already approved in the seed data
    assert closed.stage == "closed" and not closed.waiting_for_staff


def test_queue_lists_pending_requests_with_required_role(review_service, staff_login):
    queue = review_service.list_queue(staff_login(AGENT))  # viewing is allowed
    assert [c.refund_number for c in queue] == ["RFD-00000001"]
    assert queue[0].customer_details == "Tracker stopped charging after a week."


def test_suspended_customer_still_cannot_file_through_the_agent(make_wf_agent, session_factory):
    llm = ScriptedLLM(
        tools(call("request_refund", order_number="ORD-10012", reason="damaged")), say("Sorry.")
    )
    agent = make_wf_agent(llm, EMMA)
    reply = agent.approve(agent.send("refund").pending_approvals[0].action_id)
    assert reply.tool_executions[0].result["error"] == "not_authorized"
    assert count(session_factory, RefundRequest) == 2
    assert DANIEL  # Daniel's seeded request is untouched

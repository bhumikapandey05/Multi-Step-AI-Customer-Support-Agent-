"""Staff routes: role checks in the API, decisions with reasons, audit log access."""

import threading

import pytest
from sqlalchemy import select

from support_agent.db.engine import transaction
from support_agent.db.models import (
    AuditLog,
    RefundDecision,
    RefundRequest,
    RefundStatus,
    StaffMember,
    StaffRole,
)
from tests.agent.conftest import call, say, tools
from tests.api.conftest import error_code
from tests.conftest import ALICE, BRIAN
from tests.workflow.conftest import AGENT, APPROVER, SUPERVISOR

pytestmark = pytest.mark.integration

SEEDED = "RFD-00000001"  # Daniel's pending request in the seed data


def file_refund(client, llm, headers, order="ORD-10001") -> str:
    llm.steps.extend(
        [tools(call("request_refund", order_number=order, reason="defective")), say("Done.")]
    )
    r = client.post("/v1/conversations", headers=headers, json={"message": "refund"})
    thread, action = r.json()["thread_id"], r.json()["pending_approvals"][0]["action_id"]
    done = client.post(f"/v1/conversations/{thread}/approvals/{action}/approve", headers=headers)
    return done.json()["tool_executions"][0]["data"]["refund"]["refund_number"]


def status_of(db, number) -> RefundStatus:
    with transaction(db.session_factory) as s:
        return s.scalar(select(RefundRequest.status).where(RefundRequest.refund_number == number))


def test_queue_shows_cases_and_workflow(client, llm, customer, staff):
    number = file_refund(client, llm, customer(ALICE))
    rows = client.get("/v1/staff/refunds", headers=staff(AGENT)).json()
    by_number = {r["case"]["refund_number"]: r for r in rows}
    assert by_number[number]["workflow"]["status"] == "awaiting_staff_decision"
    assert by_number[SEEDED]["workflow"] is None  # filed outside the chat; opened on view
    detail = client.get(f"/v1/staff/refunds/{SEEDED}", headers=staff(AGENT)).json()
    assert detail["workflow"]["status"] == "awaiting_staff_decision"
    assert detail["case"]["customer_details"] == "Tracker stopped charging after a week."


def test_approver_approves_with_a_note(client, staff, db):
    r = client.post(f"/v1/staff/refunds/{SEEDED}/approve", headers=staff(APPROVER),
                    json={"note": "Defect confirmed"})  # fmt: skip
    body = r.json()
    assert r.status_code == 200 and body["status"] == "decided"
    assert body["decision"]["decided_by"] == "STF-002" and body["refund_status"] == "approved"
    assert "No payment was processed" in body["decision"]["payout"]
    assert status_of(db, SEEDED) is RefundStatus.APPROVED
    again = client.post(f"/v1/staff/refunds/{SEEDED}/approve", headers=staff(SUPERVISOR))
    assert again.status_code == 409 and error_code(again) == "workflow_state"


def test_reject_requires_a_reason(client, staff, db):
    url = f"/v1/staff/refunds/{SEEDED}/reject"
    for body in (None, {}, {"note": "no"}, {"note": "valid note", "staff_id": 3}):
        r = client.post(url, headers=staff(APPROVER), json=body)
        assert r.status_code == 422 and error_code(r) == "invalid_request"
    r = client.post(url, headers=staff(APPROVER), json={"note": "Water damage, not covered"})
    assert r.status_code == 200 and r.json()["refund_status"] == "rejected"


def test_support_agent_cannot_decide(client, staff, db):
    r = client.post(f"/v1/staff/refunds/{SEEDED}/approve", headers=staff(AGENT))
    assert r.status_code == 403 and error_code(r) == "not_authorized"
    assert r.json()["error"]["details"]["workflow"]["status"] == "awaiting_staff_decision"
    assert status_of(db, SEEDED) is RefundStatus.PENDING_REVIEW


def test_manual_review_needs_a_supervisor(client, llm, customer, staff, db):
    number = file_refund(client, llm, customer(BRIAN), order="ORD-10005")
    r = client.post(f"/v1/staff/refunds/{number}/approve", headers=staff(APPROVER))
    assert r.status_code == 403 and "supervisor" in r.json()["error"]["message"]
    r = client.post(f"/v1/staff/refunds/{number}/approve", headers=staff(SUPERVISOR))
    assert r.status_code == 200 and status_of(db, number) is RefundStatus.APPROVED


def test_audit_log_is_role_restricted_and_filterable(client, staff, db):
    r = client.get("/v1/staff/audit", headers=staff(AGENT))
    assert r.status_code == 403 and r.json()["error"]["details"]["required_roles"]
    assert db(AuditLog, AuditLog.outcome == "denied", AuditLog.actor == "staff:1") >= 1

    events = client.get("/v1/staff/audit?action=staff.login&limit=5", headers=staff(APPROVER))
    assert events.status_code == 200
    assert {e["action"] for e in events.json()} == {"staff.login"}
    denied = client.get("/v1/staff/audit?outcome=denied", headers=staff(SUPERVISOR)).json()
    assert denied and all(e["outcome"] == "denied" for e in denied)
    assert client.get("/v1/staff/audit?outcome=bogus", headers=staff(APPROVER)).status_code == 422


def test_role_comes_from_the_database_not_the_token(client, staff, db):
    headers = staff(APPROVER)
    with transaction(db.session_factory) as s:
        s.scalar(
            select(StaffMember).where(StaffMember.staff_ref == "STF-002")
        ).role = StaffRole.SUPPORT_AGENT
    assert client.get("/v1/staff/audit", headers=headers).status_code == 403
    r = client.post(f"/v1/staff/refunds/{SEEDED}/approve", headers=headers)
    assert r.status_code == 403


def test_unknown_and_malformed_refund_numbers(client, staff):
    headers = staff(APPROVER)
    assert client.get("/v1/staff/refunds/RFD-FFFFFFFF", headers=headers).status_code == 404
    assert client.get("/v1/staff/refunds/12345", headers=headers).status_code == 422


def test_concurrent_staff_decisions_record_one(make_client, staff, db):
    """Approver and supervisor click at the same moment on two servers."""
    a, b = make_client(), make_client()
    calls = [
        (a, staff(APPROVER, a), "approve", None),
        (b, staff(SUPERVISOR, b), "reject", {"note": "Duplicate of an older claim"}),
    ]
    barrier = threading.Barrier(2, timeout=10)
    results = []

    def run(c, headers, verb, body):
        barrier.wait()
        results.append(c.post(f"/v1/staff/refunds/{SEEDED}/{verb}", headers=headers, json=body))

    threads = [threading.Thread(target=run, args=args) for args in calls]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert db(RefundDecision) == 1
    assert sorted(r.status_code for r in results) == [200, 409]
    winner = next(r for r in results if r.status_code == 200).json()
    assert status_of(db, SEEDED).value == winner["refund_status"]

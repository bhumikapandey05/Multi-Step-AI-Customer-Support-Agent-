"""Chat over HTTP: confirmation, resumption, ownership, guests, and concurrent approvals."""

import threading

import pytest

from support_agent.agent.agent import DatabaseApprovalAuditor
from support_agent.db.models import ApprovalDecision, AuditLog, RefundRequest
from tests.agent.conftest import call, say, tools
from tests.api.conftest import error_code
from tests.conftest import ALICE, BRIAN

pytestmark = pytest.mark.integration

REFUND_CALL = {"order_number": "ORD-10001", "reason": "defective"}


def start_refund(client, llm, headers) -> tuple[str, str]:
    llm.steps.append(tools(call("request_refund", **REFUND_CALL), text="Let me set that up."))
    r = client.post("/v1/conversations", headers=headers, json={"message": "Refund ORD-10001"})
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["stop_reason"] == "awaiting_approval"
    (pending,) = body["pending_approvals"]
    assert "$129.99" in pending["summary"]
    return body["thread_id"], pending["action_id"]


def approve_url(thread_id: str, action_id: str, verb: str = "approve") -> str:
    return f"/v1/conversations/{thread_id}/approvals/{action_id}/{verb}"


def test_refund_request_needs_customer_confirmation(client, llm, customer, db):
    alice = customer(ALICE)
    thread_id, action_id = start_refund(client, llm, alice)
    assert db(RefundRequest) == 2  # nothing written yet

    view = client.get(f"/v1/conversations/{thread_id}", headers=alice).json()
    assert view["status"] == "awaiting_customer_confirmation"
    assert view["pending_approvals"][0]["action_id"] == action_id

    llm.steps.append(say("Submitted for staff review."))
    r = client.post(approve_url(thread_id, action_id), headers=alice)
    body = r.json()
    assert r.status_code == 200 and body["stop_reason"] == "completed"
    execution = body["tool_executions"][0]
    assert execution["approved"] and execution["data"]["refund"]["status"] == "pending_review"
    assert db(RefundRequest) == 3

    again = client.post(approve_url(thread_id, action_id), headers=alice)
    assert again.status_code == 409 and error_code(again) == "approval_not_pending"
    listed = client.get("/v1/conversations", headers=alice).json()
    assert [c["thread_id"] for c in listed] == [thread_id]
    assert listed[0]["preview"] == "Refund ORD-10001"


def test_reject_writes_nothing(client, llm, customer, db):
    alice = customer(ALICE)
    thread_id, action_id = start_refund(client, llm, alice)
    llm.steps.append(say("Okay."))
    r = client.post(approve_url(thread_id, action_id, "reject"), headers=alice)
    assert r.json()["tool_executions"][0]["error"] == "declined_by_customer"
    assert db(RefundRequest) == 2


def test_conversation_survives_an_api_restart(make_client, llm, customer, db):
    first = make_client()
    alice = customer(ALICE, first)
    thread_id, action_id = start_refund(first, llm, alice)
    first.close()
    first.app.state.services.close()  # the server process stops

    second = make_client()  # a new server on the same database and checkpoint files
    alice_again = customer(ALICE, second)  # a fresh sign-in
    view = second.get(f"/v1/conversations/{thread_id}", headers=alice_again).json()
    assert view["pending_approvals"][0]["action_id"] == action_id
    llm.steps.append(say("Done."))
    assert second.post(approve_url(thread_id, action_id), headers=alice_again).status_code == 200
    assert db(RefundRequest) == 3


def test_other_customers_and_guests_get_404(client, llm, customer, guest, db):
    thread_id, action_id = start_refund(client, llm, customer(ALICE))
    for headers in (customer(BRIAN), guest()):
        assert client.get(f"/v1/conversations/{thread_id}", headers=headers).status_code == 404
        r = client.post(approve_url(thread_id, action_id), headers=headers)
        assert r.status_code == 404 and error_code(r) == "not_found"
        assert client.get("/v1/conversations", headers=headers).json() == []
    assert db(RefundRequest) == 2
    assert db(AuditLog, AuditLog.action == "workflow.resume") >= 2  # denied attempts audited


def test_unknown_and_malformed_thread_ids(client, customer):
    alice = customer(ALICE)
    r = client.post("/v1/conversations/conv-0000000000000000/messages", headers=alice,
                    json={"message": "hi"})  # fmt: skip
    assert r.status_code == 404  # a thread cannot be claimed by guessing an id
    assert client.get("/v1/conversations/../../etc", headers=alice).status_code in (404, 422)
    assert client.get("/v1/conversations/not-a-thread", headers=alice).status_code == 422


@pytest.mark.parametrize(
    "body",
    [{"message": ""}, {"message": "x" * 4001}, {"message": "hi", "customer_id": 2}, {}],
)
def test_message_validation(client, customer, body):
    r = client.post("/v1/conversations", headers=customer(ALICE), json=body)
    assert r.status_code == 422 and error_code(r) == "invalid_request"


def test_guest_chat_is_public_only_and_private_to_that_guest(client, llm, guest, customer):
    me = guest()
    llm.steps.extend([tools(call("list_orders")), say("Please sign in to see orders.")])
    r = client.post("/v1/conversations", headers=me, json={"message": "show my orders"})
    body = r.json()
    assert body["tool_executions"][0]["error"] == "authentication_required"
    assert llm.requests[-2].tool_names == {"list_support_policies", "get_support_policy"}
    thread_id = body["thread_id"]

    assert client.get(f"/v1/conversations/{thread_id}", headers=me).status_code == 200
    llm.steps.append(say("Hello again."))
    again = client.post(f"/v1/conversations/{thread_id}/messages", headers=me,
                        json={"message": "hi"})  # fmt: skip
    assert again.status_code == 200
    # A different guest session, or a signed-in customer, cannot read or continue it.
    for other in (guest(), customer(ALICE)):
        assert client.get(f"/v1/conversations/{thread_id}", headers=other).status_code == 404


def test_reads_work_without_an_llm_key(make_client, settings, customer):
    from support_agent.agent.factory import build_llm

    c = make_client(llm_factory=build_llm)  # no GEMINI_API_KEY in the test settings
    alice = customer(ALICE, c)
    assert c.get("/v1/customer/orders", headers=alice).status_code == 200
    r = c.post("/v1/conversations", headers=alice, json={"message": "hi"})
    assert r.status_code == 503 and error_code(r) == "llm_unavailable"
    assert "GEMINI_API_KEY" not in r.text


# ------------------------------------------------------------------ concurrency


def test_parallel_approvals_in_one_server_record_one_decision(client, llm, customer, db):
    alice = customer(ALICE)
    thread_id, action_id = start_refund(client, llm, alice)
    llm.steps.extend([say("Submitted."), say("Submitted.")])
    results: list[int] = []

    def click() -> None:
        results.append(client.post(approve_url(thread_id, action_id), headers=alice).status_code)

    threads = [threading.Thread(target=click) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sorted(results) == [200, 409]
    assert db(ApprovalDecision) == 1 and db(RefundRequest) == 3
    assert db(AuditLog, AuditLog.action == "agent.approval.granted") == 1


@pytest.mark.parametrize("second", ["approve", "reject"])
def test_racing_servers_record_exactly_one_decision(
    make_client, llm, customer, db, monkeypatch, second
):
    """Two API processes answer the same confirmation at the same instant.

    A barrier makes both reach the database insert together. The unique action id
    lets exactly one decision through; the other request sees what was recorded.
    """
    a, b = make_client(), make_client()  # separate checkpointer connections, like 2 processes
    alice_a, alice_b = customer(ALICE, a), customer(ALICE, b)
    thread_id, action_id = start_refund(a, llm, alice_a)
    llm.steps.extend([say("Submitted."), say("Submitted.")])

    barrier = threading.Barrier(2, timeout=10)
    original = DatabaseApprovalAuditor._record

    def synchronized(self, *args, **kwargs):
        barrier.wait()
        return original(self, *args, **kwargs)

    monkeypatch.setattr(DatabaseApprovalAuditor, "_record", synchronized)
    responses: dict[str, object] = {}

    def run(name, c, headers, verb):
        responses[name] = c.post(approve_url(thread_id, action_id, verb), headers=headers)

    threads = [
        threading.Thread(target=run, args=("first", a, alice_a, "approve")),
        threading.Thread(target=run, args=("second", b, alice_b, second)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert db(ApprovalDecision) == 1  # one decision, whatever the timing
    decision_audits = db(AuditLog, AuditLog.action.startswith("agent.approval."))
    assert decision_audits == 1
    statuses = sorted(r.status_code for r in responses.values())
    if second == "approve":
        # Same answer twice: it counts once, and the write is idempotent.
        assert statuses == [200, 200] and db(RefundRequest) == 3
    else:
        assert statuses == [200, 409]
        loser = next(r for r in responses.values() if r.status_code == 409)
        assert error_code(loser) == "approval_conflict"
        granted = db(ApprovalDecision, ApprovalDecision.decision == "granted")
        assert db(RefundRequest) == 2 + granted  # a refund exists only if "approve" won

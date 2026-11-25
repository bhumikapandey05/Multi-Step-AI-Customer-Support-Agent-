"""Authentication, token kinds, structured errors, and customer data isolation."""

import base64
import json

import pytest

from support_agent.db.models import AuditLog
from tests.api.conftest import bearer, error_code
from tests.conftest import ALICE, BRIAN
from tests.workflow.conftest import AGENT, APPROVER

pytestmark = pytest.mark.integration


def test_health_and_request_id(client):
    r = client.get("/health", headers={"X-Request-ID": "demo-123"})
    assert r.json()["status"] == "ok" and r.headers["X-Request-ID"] == "demo-123"


def test_customer_login_returns_token_and_profile(client):
    r = client.post("/v1/auth/customer/login", json={"email": ALICE, "pin": "246810"})
    body = r.json()
    assert r.status_code == 200 and body["kind"] == "customer"
    assert body["token"].startswith("v1.") and body["profile"]["name"] == "Alice Nguyen"
    assert "246810" not in r.text


@pytest.mark.parametrize(
    ("payload", "status", "code"),
    [
        ({"email": ALICE, "pin": "000000"}, 401, "authentication_failed"),
        ({"email": ALICE, "pin": "12ab"}, 422, "invalid_request"),
        ({"email": ALICE}, 422, "invalid_request"),
        ({"email": ALICE, "pin": "246810", "customer_id": 2}, 422, "invalid_request"),
    ],
)
def test_login_errors_are_structured(client, payload, status, code):
    r = client.post("/v1/auth/customer/login", json=payload)
    assert r.status_code == status and error_code(r) == code
    assert set(r.json()) == {"error", "request_id"}
    assert set(r.json()["error"]) == {"code", "message", "details"}


def test_missing_or_bad_tokens_are_rejected(client, customer):
    assert error_code(client.get("/v1/customer/orders")) == "authentication_required"
    r = client.get("/v1/customer/orders", headers=bearer("v1.garbage.token"))
    assert r.status_code == 401 and r.headers["WWW-Authenticate"] == "Bearer"


def test_tampered_customer_id_in_token_is_rejected(client, customer):
    token = customer(ALICE)["Authorization"].removeprefix("Bearer ")
    version, payload, signature = token.split(".")
    data = json.loads(base64.urlsafe_b64decode(payload + "=="))
    data["cid"] = 2  # try to become Brian
    forged = base64.urlsafe_b64encode(json.dumps(data).encode()).rstrip(b"=").decode()
    r = client.get("/v1/customer/orders", headers=bearer(f"{version}.{forged}.{signature}"))
    assert r.status_code == 401 and error_code(r) == "authentication_failed"


def test_token_kinds_cannot_be_mixed(client, customer, staff, guest):
    assert client.get("/v1/staff/refunds", headers=customer(ALICE)).status_code == 403
    assert client.get("/v1/customer/orders", headers=staff(APPROVER)).status_code == 403
    assert client.get("/v1/customer/orders", headers=guest()).status_code == 403
    r = client.post("/v1/conversations", headers=staff(APPROVER), json={"message": "hi"})
    assert r.status_code == 403


def test_customer_sees_only_their_own_orders(client, customer):
    alice = customer(ALICE)
    orders = client.get("/v1/customer/orders", headers=alice).json()
    assert {o["order_number"] for o in orders} == {"ORD-10001", "ORD-10002", "ORD-10003"}
    # Client-supplied ids are ignored: identity only comes from the token.
    spoofed = client.get("/v1/customer/orders?customer_id=2", headers=alice).json()
    assert spoofed == orders
    r = client.get("/v1/customer/orders/ORD-10004", headers=alice)  # Brian's order
    assert r.status_code == 404 and error_code(r) == "not_found"
    assert "Dumbbell" not in r.text


def test_customer_reads(client, customer):
    brian = customer(BRIAN)
    assert client.get("/v1/customer/profile", headers=brian).json()["tier"] == "gold"
    e = client.get("/v1/customer/orders/ORD-10005/refund-eligibility", headers=brian).json()
    assert e["eligible"] and e["requires_manual_review"]
    assert client.get("/v1/customer/refunds", headers=brian).json() == []
    assert client.get("/v1/customer/tickets", headers=brian).status_code == 200
    r = client.get("/v1/customer/orders/10005", headers=brian)
    assert r.status_code == 422 and error_code(r) == "invalid_input"
    assert client.get("/v1/customer/orders?limit=0", headers=brian).status_code == 422


def test_staff_login_and_profile(client, staff):
    me = client.get("/v1/staff/me", headers=staff(AGENT)).json()
    assert me == {"staff_ref": "STF-001", "name": "Sam Patel", "role": "support_agent"}
    r = client.post("/v1/auth/staff/login", json={"email": ALICE, "pin": "246810"})
    assert r.status_code == 401  # customers are not staff


def test_unknown_route_and_unexpected_errors_are_structured(client, customer, monkeypatch):
    assert error_code(client.get("/v1/nope")) == "not_found"
    crm = client.app.state.services.app.crm

    def boom(*a, **k):
        raise RuntimeError("secret internal path /var/db")

    monkeypatch.setattr(crm, "list_orders", boom)
    r = client.get("/v1/customer/orders", headers=customer(ALICE))
    assert r.status_code == 500 and error_code(r) == "internal_error"
    assert "secret" not in r.text


def test_logins_are_audited(client, customer, db):
    customer(ALICE)
    client.post("/v1/auth/customer/login", json={"email": ALICE, "pin": "000000"})
    assert db(AuditLog, AuditLog.action == "auth.login") == 2

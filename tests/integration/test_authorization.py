"""Customers can only see and change their own data."""

import dataclasses

import pytest

from support_agent.crm.errors import AuthenticationError, AuthorizationError, NotFoundError
from support_agent.db.models import AuditOutcome
from tests.conftest import ALICE, BRIAN, DANIEL, EMMA

pytestmark = pytest.mark.integration


def test_list_orders_returns_only_own_orders(crm, login):
    numbers = {o.order_number for o in crm.list_orders(login(ALICE))}
    assert numbers == {"ORD-10001", "ORD-10002", "ORD-10003"}


def test_other_customers_order_looks_identical_to_missing_order(crm, login):
    alice = login(ALICE)
    with pytest.raises(NotFoundError) as foreign:
        crm.get_order(alice, "ORD-10004")  # Brian's order
    with pytest.raises(NotFoundError) as missing:
        crm.get_order(alice, "ORD-99999")
    assert foreign.value.to_dict()["details"] == missing.value.to_dict()["details"] == {}
    assert "ORD-10004" in foreign.value.message and "ORD-99999" in missing.value.message
    assert "owner" not in str(foreign.value.to_dict())


def test_cross_customer_access_is_audited_as_denied(crm, login, audit_rows):
    with pytest.raises(NotFoundError):
        crm.get_order(login(ALICE), "ORD-10004")
    (row,) = audit_rows("order.view")
    assert row.outcome is AuditOutcome.DENIED
    assert row.entity_id == "ORD-10004"
    assert row.details["reason"] == "owned_by_another_customer"


@pytest.mark.parametrize(
    "call",
    [
        lambda crm, ctx: crm.check_refund_eligibility(ctx, "ORD-10004"),
        lambda crm, ctx: crm.request_refund(
            ctx, {"order_number": "ORD-10004", "reason": "damaged"}
        ),
        lambda crm, ctx: crm.create_ticket(
            ctx,
            {
                "category": "shipping",
                "order_number": "ORD-10004",
                "subject": "Where is it",
                "description": "Please check my order.",
            },
        ),
        lambda crm, ctx: crm.get_ticket(ctx, "TKT-00000001"),  # Daniel's ticket
    ],
)
def test_every_tool_enforces_ownership(crm, login, call):
    with pytest.raises(NotFoundError):
        call(crm, login(ALICE))


def test_ticket_and_refund_lists_are_scoped(crm, login):
    assert {t.ticket_number for t in crm.list_tickets(login(ALICE))} == {"TKT-00000002"}
    assert {r.refund_number for r in crm.list_refund_requests(login(DANIEL))} == {
        "RFD-00000001",
        "RFD-00000002",
    }
    assert crm.list_refund_requests(login(ALICE)) == []


def test_forged_customer_id_is_rejected_and_audited(crm, login, audit_rows):
    alice = login(ALICE)
    brian_id = login(BRIAN).customer_id
    forged = dataclasses.replace(alice, customer_id=brian_id)

    with pytest.raises(AuthenticationError):
        crm.list_orders(forged)
    row = audit_rows("order.list")[-1]
    assert row.outcome is AuditOutcome.DENIED
    assert row.actor == f"unverified:customer:{brian_id}"


def test_bare_customer_id_is_not_accepted_as_identity(crm):
    with pytest.raises(AuthenticationError):
        crm.list_orders(1)  # type: ignore[arg-type]


def test_suspended_account_can_read_but_not_write(crm, login, audit_rows):
    emma = login(EMMA)
    assert [o.order_number for o in crm.list_orders(emma)] == ["ORD-10012"]

    with pytest.raises(AuthorizationError):
        crm.request_refund(emma, {"order_number": "ORD-10012", "reason": "damaged"})
    with pytest.raises(AuthorizationError):
        crm.create_ticket(
            emma,
            {
                "category": "account",
                "subject": "Unlock my account",
                "description": "Please reactivate my account.",
            },
        )
    assert audit_rows("refund.request")[-1].outcome is AuditOutcome.DENIED
    assert crm.list_refund_requests(emma) == []

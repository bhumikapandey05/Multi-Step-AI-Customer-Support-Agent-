import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from support_agent.crm.errors import IdempotencyConflictError, InvalidInputError
from support_agent.crm.service import CRMService
from support_agent.db.engine import transaction
from support_agent.db.models import (
    AuditOutcome,
    SupportTicket,
    TicketCategory,
    TicketStatus,
)
from tests.conftest import ALICE, BRIAN, DANIEL

pytestmark = pytest.mark.integration

TICKET = {
    "category": "product_issue",
    "order_number": "ORD-10001",
    "subject": "Headphones crackle",
    "description": "The left earcup crackles at high volume.",
}


def count_tickets(session_factory) -> int:
    with transaction(session_factory) as s:
        return s.scalar(select(func.count()).select_from(SupportTicket))


def test_create_ticket(crm, login, audit_rows):
    result = crm.create_ticket(login(ALICE), {**TICKET, "priority": "high"})
    assert result.created
    ticket = result.ticket
    assert ticket.ticket_number.startswith("TKT-")
    assert ticket.order_number == "ORD-10001"
    assert ticket.status is TicketStatus.OPEN
    assert crm.get_ticket(login(ALICE), ticket.ticket_number) == ticket

    (row,) = audit_rows("ticket.create")
    assert row.outcome is AuditOutcome.SUCCESS and row.entity_id == ticket.ticket_number


def test_ticket_without_order(crm, login):
    result = crm.create_ticket(
        login(ALICE),
        {
            "category": "account",
            "subject": "Change my email",
            "description": "I would like to update my email address.",
        },
    )
    assert result.created and result.ticket.order_number is None


def test_same_issue_on_same_order_is_deduplicated(crm, login, session_factory, audit_rows):
    alice = login(ALICE)
    first = crm.create_ticket(alice, TICKET)
    second = crm.create_ticket(
        alice,
        {**TICKET, "subject": "Still crackling", "description": "Reworded but the same problem."},
    )
    assert not second.created
    assert second.ticket.ticket_number == first.ticket.ticket_number
    assert count_tickets(session_factory) == 3  # 2 seeded + 1
    assert audit_rows("ticket.create")[-1].outcome is AuditOutcome.DUPLICATE


def test_different_category_on_same_order_is_allowed(crm, login):
    alice = login(ALICE)
    crm.create_ticket(alice, TICKET)
    assert crm.create_ticket(alice, {**TICKET, "category": "billing"}).created


def test_existing_seeded_open_ticket_is_detected(crm, login):
    result = crm.create_ticket(
        login(DANIEL),
        {
            "category": "refund",
            "order_number": "ORD-10010",
            "subject": "Refund status",
            "description": "Any update on my refund?",
        },
    )
    assert not result.created and result.ticket.ticket_number == "TKT-00000001"


def test_resolved_ticket_does_not_block_a_new_one(crm, login):
    # TKT-00000002 (shipping, ORD-10002) is resolved.
    result = crm.create_ticket(
        login(ALICE),
        {
            "category": "shipping",
            "order_number": "ORD-10002",
            "subject": "Another shipping question",
            "description": "Second question about it.",
        },
    )
    assert result.created


def test_orderless_duplicates_match_on_normalized_subject(crm, login):
    alice = login(ALICE)
    base = {"category": "account", "description": "Please help me with my account."}
    first = crm.create_ticket(alice, {**base, "subject": "Reset my PIN"})
    again = crm.create_ticket(alice, {**base, "subject": "  reset   MY pin "})
    other = crm.create_ticket(alice, {**base, "subject": "Close my account"})
    assert not again.created and again.ticket.ticket_number == first.ticket.ticket_number
    assert other.created


def test_idempotent_retry_returns_original(crm, login, session_factory):
    alice = login(ALICE)
    request = {**TICKET, "idempotency_key": "retry-key-0001"}
    first = crm.create_ticket(alice, request)
    retry = crm.create_ticket(alice, request)
    assert not retry.created and retry.ticket.ticket_number == first.ticket.ticket_number
    assert count_tickets(session_factory) == 3


def test_idempotency_key_reuse_with_different_payload_conflicts(crm, login, audit_rows):
    alice = login(ALICE)
    crm.create_ticket(alice, {**TICKET, "idempotency_key": "retry-key-0001"})
    with pytest.raises(IdempotencyConflictError):
        crm.create_ticket(
            alice, {**TICKET, "category": "billing", "idempotency_key": "retry-key-0001"}
        )
    assert audit_rows("ticket.create")[-1].outcome is AuditOutcome.REJECTED


def test_idempotency_keys_are_scoped_per_customer(crm, login):
    key = {"idempotency_key": "shared-key-0001"}
    assert crm.create_ticket(login(ALICE), {**TICKET, **key}).created
    assert crm.create_ticket(login(BRIAN), {**TICKET, "order_number": "ORD-10004", **key}).created


def test_invalid_input_is_rejected_and_audited(crm, login, audit_rows):
    with pytest.raises(InvalidInputError):
        crm.create_ticket(login(ALICE), {**TICKET, "customer_id": 2})
    row = audit_rows("ticket.create")[-1]
    assert row.outcome is AuditOutcome.REJECTED and row.details["error"] == "invalid_input"


def test_race_condition_is_caught_by_unique_index(crm, login, monkeypatch, session_factory):
    """Simulate two concurrent requests that both passed the duplicate check."""
    alice = login(ALICE)
    first = crm.create_ticket(alice, TICKET)

    calls = {"n": 0}
    original = CRMService._find_active_duplicate_ticket

    def miss_first_time(op, req, order):
        calls["n"] += 1
        return None if calls["n"] == 1 else original(op, req, order)

    monkeypatch.setattr(CRMService, "_find_active_duplicate_ticket", staticmethod(miss_first_time))
    result = crm.create_ticket(alice, TICKET)
    assert not result.created
    assert result.ticket.ticket_number == first.ticket.ticket_number
    assert count_tickets(session_factory) == 3


def test_database_rejects_second_active_ticket_directly(crm, login, session_factory):
    crm.create_ticket(login(ALICE), TICKET)
    with pytest.raises(IntegrityError), transaction(session_factory) as s:
        existing = s.scalar(
            select(SupportTicket).where(SupportTicket.category == TicketCategory.PRODUCT_ISSUE)
        )
        s.add(
            SupportTicket(
                ticket_number="TKT-FFFFFFFF",
                customer_id=existing.customer_id,
                order_id=existing.order_id,
                category=existing.category,
                subject="dup",
                description="dup",
            )
        )

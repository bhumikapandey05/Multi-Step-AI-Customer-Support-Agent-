"""Writes are atomic: a failure anywhere in an operation leaves no partial data."""

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from support_agent.crm import service as service_module
from support_agent.db.engine import transaction
from support_agent.db.models import AuditLog, AuditOutcome, Order, RefundRequest, SupportTicket
from support_agent.db.seed import seed_database
from tests.conftest import ALICE, FIXED_NOW

pytestmark = pytest.mark.integration


def count(session_factory, model) -> int:
    with transaction(session_factory) as s:
        return s.scalar(select(func.count()).select_from(model))


def test_failure_after_insert_rolls_back_ticket(crm, login, session_factory, monkeypatch):
    alice = login(ALICE)
    before = count(session_factory, SupportTicket)

    def boom(*args, **kwargs):
        raise RuntimeError("audit store unavailable")

    # The success-audit write happens after the ticket insert in the same transaction.
    monkeypatch.setattr(service_module, "record_audit", boom)
    with pytest.raises(RuntimeError):
        crm.create_ticket(
            alice,
            {
                "category": "shipping",
                "order_number": "ORD-10003",
                "subject": "Where is my order",
                "description": "Tracking has not updated.",
            },
        )
    monkeypatch.undo()

    assert count(session_factory, SupportTicket) == before


def test_failure_after_insert_rolls_back_refund(crm, login, session_factory, monkeypatch):
    alice = login(ALICE)
    before = count(session_factory, RefundRequest)
    monkeypatch.setattr(
        service_module, "record_audit", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    with pytest.raises(RuntimeError):
        crm.request_refund(alice, {"order_number": "ORD-10001", "reason": "defective"})
    monkeypatch.undo()
    assert count(session_factory, RefundRequest) == before


def test_unexpected_errors_are_audited_as_error(crm, login, session_factory, monkeypatch):
    alice = login(ALICE)
    real = service_module.record_audit

    def fail_on_success(session, **kwargs):
        if kwargs["outcome"] is AuditOutcome.SUCCESS:
            raise RuntimeError("boom")
        return real(session, **kwargs)

    monkeypatch.setattr(service_module, "record_audit", fail_on_success)
    with pytest.raises(RuntimeError):
        crm.list_orders(alice)

    with transaction(session_factory) as s:
        row = s.scalars(select(AuditLog).order_by(AuditLog.id.desc())).first()
        assert row.action == "order.list" and row.outcome is AuditOutcome.ERROR


def test_seeding_twice_is_refused(session_factory, seeded):
    with pytest.raises(RuntimeError), transaction(session_factory) as s:
        seed_database(s, now=FIXED_NOW, pin_iterations=1_000)
    assert count(session_factory, Order) == 14


def test_foreign_keys_are_enforced(session_factory, seeded):
    with pytest.raises(IntegrityError), transaction(session_factory) as s:
        s.add(
            SupportTicket(
                ticket_number="TKT-ABCDEF01",
                customer_id=9_999,
                category="other",
                subject="x" * 5,
                description="y" * 10,
            )
        )

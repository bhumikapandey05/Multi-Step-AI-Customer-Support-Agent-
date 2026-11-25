import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from support_agent.crm.errors import IdempotencyConflictError, RefundNotEligibleError
from support_agent.db.engine import transaction
from support_agent.db.models import AuditOutcome, RefundRequest, RefundStatus
from tests.conftest import ALICE, BRIAN, CARMEN, DANIEL

pytestmark = pytest.mark.integration


def count_refunds(session_factory) -> int:
    with transaction(session_factory) as s:
        return s.scalar(select(func.count()).select_from(RefundRequest))


def test_full_refund_request_is_recorded_for_review(crm, login, audit_rows):
    result = crm.request_refund(login(ALICE), {"order_number": "ORD-10001", "reason": "defective"})
    refund = result.refund
    assert result.created
    assert refund.amount_cents == 12_999
    assert refund.status is RefundStatus.PENDING_REVIEW
    assert not refund.requires_manual_review
    assert "No payment has been issued" in result.message

    (row,) = audit_rows("refund.request")
    assert row.outcome is AuditOutcome.SUCCESS
    assert row.details["amount_cents"] == 12_999


def test_partial_refund(crm, login):
    result = crm.request_refund(
        login(ALICE),
        {"order_number": "ORD-10001", "reason": "not_as_described", "amount_cents": 5_000},
    )
    assert result.refund.amount_cents == 5_000


def test_amount_above_refundable_is_rejected(crm, login, session_factory):
    # ORD-10007: only the dress (6,400) is refundable; the hat is final sale.
    with pytest.raises(RefundNotEligibleError) as exc:
        crm.request_refund(
            login(CARMEN), {"order_number": "ORD-10007", "reason": "damaged", "amount_cents": 7_900}
        )
    assert exc.value.details["refundable_amount_cents"] == 6_400
    assert count_refunds(session_factory) == 2


def test_ineligible_order_writes_nothing_but_is_audited(crm, login, session_factory, audit_rows):
    with pytest.raises(RefundNotEligibleError) as exc:
        crm.request_refund(login(ALICE), {"order_number": "ORD-10002", "reason": "damaged"})
    assert exc.value.details["eligibility"]["reasons"][0]["code"] == "window_expired"
    assert count_refunds(session_factory) == 2
    assert audit_rows("refund.request")[-1].outcome is AuditOutcome.REJECTED


def test_high_value_refund_is_flagged_for_manual_review(crm, login):
    result = crm.request_refund(login(BRIAN), {"order_number": "ORD-10005", "reason": "defective"})
    assert result.refund.requires_manual_review


def test_second_request_for_same_order_returns_existing(crm, login, session_factory):
    alice = login(ALICE)
    first = crm.request_refund(alice, {"order_number": "ORD-10001", "reason": "defective"})
    second = crm.request_refund(
        alice, {"order_number": "ORD-10001", "reason": "damaged", "amount_cents": 100}
    )
    assert not second.created
    assert second.refund.refund_number == first.refund.refund_number
    assert count_refunds(session_factory) == 3


def test_seeded_pending_refund_is_detected(crm, login):
    result = crm.request_refund(login(DANIEL), {"order_number": "ORD-10010", "reason": "defective"})
    assert not result.created and result.refund.refund_number == "RFD-00000001"


def test_idempotency_key_replay_and_conflict(crm, login):
    alice = login(ALICE)
    request = {
        "order_number": "ORD-10001",
        "reason": "defective",
        "idempotency_key": "refund-key-0001",
    }
    first = crm.request_refund(alice, request)
    assert crm.request_refund(alice, request).refund == first.refund
    with pytest.raises(IdempotencyConflictError):
        crm.request_refund(alice, {**request, "amount_cents": 100})


def test_eligibility_reflects_new_pending_request(crm, login):
    alice = login(ALICE)
    crm.request_refund(alice, {"order_number": "ORD-10001", "reason": "defective"})
    result = crm.check_refund_eligibility(alice, "ORD-10001")
    assert not result.eligible
    assert [r.code for r in result.reasons] == ["refund_in_progress"]


def test_database_rejects_second_pending_refund_directly(crm, login, session_factory):
    crm.request_refund(login(ALICE), {"order_number": "ORD-10001", "reason": "defective"})
    with pytest.raises(IntegrityError), transaction(session_factory) as s:
        existing = s.scalar(
            select(RefundRequest).where(
                RefundRequest.status == RefundStatus.PENDING_REVIEW,
                RefundRequest.refund_number != "RFD-00000001",
            )
        )
        s.add(
            RefundRequest(
                refund_number="RFD-FFFFFFFF",
                customer_id=existing.customer_id,
                order_id=existing.order_id,
                amount_cents=1,
                reason=existing.reason,
                policy_version="test",
            )
        )


def test_database_rejects_non_positive_amount(session_factory, seeded):
    with pytest.raises(IntegrityError), transaction(session_factory) as s:
        s.add(
            RefundRequest(
                refund_number="RFD-EEEEEEEE",
                customer_id=1,
                order_id=1,
                amount_cents=0,
                reason="damaged",
                policy_version="test",
            )
        )

"""Pure refund-policy rules (no database)."""

from datetime import UTC, datetime, timedelta

import pytest

from support_agent.crm.policies import (
    LineItemFacts,
    OrderFacts,
    RefundPolicy,
    evaluate_refund_eligibility,
)
from support_agent.db.models import CustomerTier, OrderStatus

NOW = datetime(2026, 6, 15, 12, 0, tzinfo=UTC)
POLICY = RefundPolicy()
SHOES = LineItemFacts("Running Shoes", "apparel", 1, 10_000, False)


def facts(**overrides) -> OrderFacts:
    base = dict(
        order_number="ORD-00001",
        status=OrderStatus.DELIVERED,
        delivered_at=NOW - timedelta(days=5),
        customer_tier=CustomerTier.STANDARD,
        items=(SHOES,),
    )
    base.update(overrides)
    return OrderFacts(**base)


def evaluate(f: OrderFacts):
    return evaluate_refund_eligibility(f, now=NOW, policy=POLICY)


def codes(result) -> list[str]:
    return [r.code for r in result.reasons]


def test_recent_delivery_is_eligible_for_full_amount():
    result = evaluate(facts())
    assert result.eligible
    assert result.refundable_amount_cents == 10_000
    assert result.window_ends_at == NOW - timedelta(days=5) + timedelta(days=30)
    assert not result.requires_manual_review


@pytest.mark.parametrize(
    ("status", "code"),
    [
        (OrderStatus.PENDING, "not_shipped"),
        (OrderStatus.PROCESSING, "not_shipped"),
        (OrderStatus.SHIPPED, "not_delivered"),
        (OrderStatus.CANCELLED, "order_cancelled"),
    ],
)
def test_undelivered_orders_are_not_eligible(status, code):
    result = evaluate(facts(status=status, delivered_at=None))
    assert not result.eligible
    assert codes(result) == [code]
    assert result.refundable_amount_cents == 0


def test_window_boundary_is_inclusive():
    on_boundary = evaluate(facts(delivered_at=NOW - timedelta(days=30)))
    just_past = evaluate(facts(delivered_at=NOW - timedelta(days=30, seconds=1)))
    assert on_boundary.eligible
    assert codes(just_past) == ["window_expired"]


@pytest.mark.parametrize(
    ("tier", "eligible"), [(CustomerTier.STANDARD, False), (CustomerTier.GOLD, True)]
)
def test_gold_members_get_extended_window(tier, eligible):
    result = evaluate(facts(customer_tier=tier, delivered_at=NOW - timedelta(days=45)))
    assert result.eligible is eligible


def test_final_sale_and_digital_items_are_excluded():
    items = (
        SHOES,
        LineItemFacts("Clearance Hat", "apparel", 1, 1_500, True),
        LineItemFacts("Software", "digital", 1, 5_000, False),
        LineItemFacts("Gift Card", "gift_card", 2, 2_500, False),
    )
    result = evaluate(facts(items=items))
    assert result.eligible
    assert result.refundable_amount_cents == 10_000
    assert len(result.notes) == 3


def test_only_non_refundable_items_is_not_eligible():
    items = (LineItemFacts("Software", "digital", 1, 5_000, False),)
    assert codes(evaluate(facts(items=items))) == ["no_refundable_items"]


def test_previous_approved_refunds_reduce_remaining_amount():
    partial = evaluate(facts(approved_refunds_cents=4_000))
    assert partial.refundable_amount_cents == 6_000
    assert partial.already_refunded_cents == 4_000

    full = evaluate(facts(approved_refunds_cents=10_000))
    assert codes(full) == ["fully_refunded"]


def test_pending_refund_blocks_new_request():
    assert codes(evaluate(facts(has_pending_refund=True))) == ["refund_in_progress"]


def test_high_value_refund_requires_manual_review():
    laptop = LineItemFacts("Laptop", "electronics", 1, POLICY.manual_review_threshold_cents, False)
    result = evaluate(facts(items=(laptop,)))
    assert result.eligible and result.requires_manual_review


def test_delivered_without_date_fails_closed():
    assert codes(evaluate(facts(delivered_at=None))) == ["missing_delivery_date"]


def test_multiple_blocking_reasons_are_all_reported():
    result = evaluate(facts(delivered_at=NOW - timedelta(days=90), has_pending_refund=True))
    assert set(codes(result)) == {"window_expired", "refund_in_progress"}

"""Order lookups and refund eligibility against the seeded scenarios."""

import pytest

from support_agent.crm.errors import InvalidInputError, NotFoundError
from tests.conftest import ALICE, BRIAN, CARMEN, DANIEL, FARAH

pytestmark = pytest.mark.integration


def test_order_detail_includes_items_and_totals(crm, login):
    order = crm.get_order(login(CARMEN), "ord-10007")  # normalized
    assert order.order_number == "ORD-10007"
    assert order.total_cents == 6_400 + 1_500
    assert [i.is_final_sale for i in order.items] == [False, True]


def test_orders_are_newest_first_and_limited(crm, login):
    orders = crm.list_orders(login(ALICE), limit=2)
    assert [o.order_number for o in orders] == ["ORD-10003", "ORD-10001"]


def test_invalid_limit_is_rejected(crm, login):
    with pytest.raises(InvalidInputError):
        crm.list_orders(login(ALICE), limit=1_000)


@pytest.mark.parametrize(
    ("email", "order", "eligible", "amount", "reason"),
    [
        (ALICE, "ORD-10001", True, 12_999, None),
        (ALICE, "ORD-10002", False, 0, "window_expired"),
        (ALICE, "ORD-10003", False, 0, "not_delivered"),
        (BRIAN, "ORD-10004", True, 24_900, None),  # gold: 60-day window
        (BRIAN, "ORD-10006", False, 0, "not_shipped"),
        (CARMEN, "ORD-10007", True, 6_400, None),  # final-sale hat excluded
        (CARMEN, "ORD-10008", False, 0, "no_refundable_items"),
        (CARMEN, "ORD-10009", False, 0, "order_cancelled"),
        (DANIEL, "ORD-10010", False, 0, "refund_in_progress"),
        (DANIEL, "ORD-10011", False, 0, "fully_refunded"),
        (FARAH, "ORD-10013", False, 0, "not_shipped"),
        (FARAH, "ORD-10014", True, 8_900, None),
    ],
)
def test_seeded_eligibility_scenarios(crm, login, email, order, eligible, amount, reason):
    result = crm.check_refund_eligibility(login(email), order)
    assert result.eligible is eligible
    assert result.refundable_amount_cents == amount
    if reason:
        assert reason in [r.code for r in result.reasons]


def test_high_value_order_requires_manual_review(crm, login):
    result = crm.check_refund_eligibility(login(BRIAN), "ORD-10005")
    assert result.eligible and result.requires_manual_review


def test_policies_are_public_and_match_code(crm):
    slugs = {p.slug for p in crm.list_policies()}
    assert {"refund-policy", "final-sale-items", "digital-goods"} <= slugs
    body = crm.get_policy("REFUND-POLICY").body
    assert f"{crm.policy.standard_window_days} days" in body
    assert f"{crm.policy.gold_window_days} days" in body
    with pytest.raises(NotFoundError):
        crm.get_policy("secret-policy")

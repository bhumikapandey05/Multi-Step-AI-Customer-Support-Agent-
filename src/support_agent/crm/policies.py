"""Refund policy rules as a pure, deterministic function.

The rules live in code (not in the LLM prompt) so eligibility decisions are
testable, explainable, and identical no matter how a customer phrases a request.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from support_agent.crm.schemas import EligibilityReason, RefundEligibility
from support_agent.db.models import CustomerTier, OrderStatus


@dataclass(frozen=True)
class RefundPolicy:
    version: str = "2026.1"
    standard_window_days: int = 30
    gold_window_days: int = 60
    # Requests at or above this amount are flagged for a human reviewer.
    manual_review_threshold_cents: int = 50_000
    non_refundable_categories: frozenset[str] = field(
        default_factory=lambda: frozenset({"digital", "gift_card"})
    )

    def window_days(self, tier: CustomerTier) -> int:
        return self.gold_window_days if tier is CustomerTier.GOLD else self.standard_window_days


@dataclass(frozen=True)
class LineItemFacts:
    product_name: str
    category: str
    quantity: int
    unit_price_cents: int
    is_final_sale: bool

    @property
    def line_total_cents(self) -> int:
        return self.quantity * self.unit_price_cents


@dataclass(frozen=True)
class OrderFacts:
    """Everything the policy needs to know about an order, decoupled from the ORM."""

    order_number: str
    status: OrderStatus
    delivered_at: datetime | None
    customer_tier: CustomerTier
    items: tuple[LineItemFacts, ...]
    approved_refunds_cents: int = 0
    has_pending_refund: bool = False


def evaluate_refund_eligibility(
    facts: OrderFacts, *, now: datetime, policy: RefundPolicy
) -> RefundEligibility:
    reasons: list[EligibilityReason] = []
    notes: list[str] = []
    window_ends_at: datetime | None = None

    def block(code: str, message: str) -> None:
        reasons.append(EligibilityReason(code=code, message=message))

    match facts.status:
        case OrderStatus.CANCELLED:
            block("order_cancelled", "Cancelled orders are not eligible for a refund request.")
        case OrderStatus.PENDING | OrderStatus.PROCESSING:
            block(
                "not_shipped",
                "The order has not shipped yet; it can be cancelled instead of refunded.",
            )
        case OrderStatus.SHIPPED:
            block(
                "not_delivered",
                "Refunds can be requested after delivery. Open a shipping ticket if it is late.",
            )
        case OrderStatus.DELIVERED:
            if facts.delivered_at is None:
                block(
                    "missing_delivery_date",
                    "The delivery date is missing; a support agent must review this order.",
                )
            else:
                days = policy.window_days(facts.customer_tier)
                window_ends_at = facts.delivered_at + timedelta(days=days)
                if now > window_ends_at:
                    block(
                        "window_expired",
                        f"The {days}-day refund window ended on {window_ends_at:%Y-%m-%d}.",
                    )

    eligible_items_cents = 0
    for item in facts.items:
        if item.is_final_sale:
            notes.append(f"'{item.product_name}' is a final-sale item and is not refundable.")
        elif item.category in policy.non_refundable_categories:
            notes.append(f"'{item.product_name}' ({item.category}) is not refundable.")
        else:
            eligible_items_cents += item.line_total_cents

    remaining_cents = max(eligible_items_cents - facts.approved_refunds_cents, 0)

    if facts.has_pending_refund:
        block("refund_in_progress", "A refund request for this order is already under review.")
    if not reasons and remaining_cents == 0:
        if facts.approved_refunds_cents > 0:
            block("fully_refunded", "All refundable items on this order were already refunded.")
        else:
            block("no_refundable_items", "This order contains no refundable items.")

    eligible = not reasons
    refundable_cents = remaining_cents if eligible else 0
    return RefundEligibility(
        order_number=facts.order_number,
        eligible=eligible,
        reasons=reasons,
        notes=notes,
        refundable_amount_cents=refundable_cents,
        already_refunded_cents=facts.approved_refunds_cents,
        window_ends_at=window_ends_at,
        requires_manual_review=eligible
        and refundable_cents >= policy.manual_review_threshold_cents,
        policy_version=policy.version,
    )

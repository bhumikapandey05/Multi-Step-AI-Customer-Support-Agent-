"""ORM models for the fictional e-commerce CRM.

Money is stored as integer cents. Timestamps are timezone-aware UTC.
Duplicate-prevention rules that must hold under concurrency are enforced by
unique (partial) indexes, not only by application code.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import (
    JSON,
    CheckConstraint,
    Date,
    ForeignKey,
    Index,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy import Enum as SAEnum
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from support_agent.clock import utcnow
from support_agent.db.types import UTCDateTime


class Base(DeclarativeBase):
    pass


def _enum(enum_cls: type[StrEnum]) -> SAEnum:
    """Store enum *values* (e.g. ``'delivered'``) as VARCHAR with a CHECK constraint."""
    return SAEnum(
        enum_cls,
        native_enum=False,
        create_constraint=True,
        length=32,
        values_callable=lambda members: [m.value for m in members],
        validate_strings=True,
    )


# --------------------------------------------------------------------------- enums


class CustomerTier(StrEnum):
    STANDARD = "standard"
    GOLD = "gold"


class AccountStatus(StrEnum):
    ACTIVE = "active"
    SUSPENDED = "suspended"


class OrderStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"


class TicketCategory(StrEnum):
    ORDER_STATUS = "order_status"
    SHIPPING = "shipping"
    REFUND = "refund"
    PRODUCT_ISSUE = "product_issue"
    BILLING = "billing"
    ACCOUNT = "account"
    OTHER = "other"


class TicketPriority(StrEnum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class TicketStatus(StrEnum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    WAITING_ON_CUSTOMER = "waiting_on_customer"
    RESOLVED = "resolved"
    CLOSED = "closed"


ACTIVE_TICKET_STATUSES = (
    TicketStatus.OPEN,
    TicketStatus.IN_PROGRESS,
    TicketStatus.WAITING_ON_CUSTOMER,
)


class RefundReason(StrEnum):
    DAMAGED = "damaged"
    DEFECTIVE = "defective"
    WRONG_ITEM = "wrong_item"
    NOT_AS_DESCRIBED = "not_as_described"
    NO_LONGER_NEEDED = "no_longer_needed"
    LATE_DELIVERY = "late_delivery"
    OTHER = "other"


class RefundStatus(StrEnum):
    # No money moves in this system. A request waits for a human reviewer.
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


class AuditOutcome(StrEnum):
    SUCCESS = "success"
    DUPLICATE = "duplicate"  # request matched an existing record; nothing new written
    REJECTED = "rejected"  # invalid input or business rule violation
    DENIED = "denied"  # authentication / authorization failure
    ERROR = "error"  # unexpected failure


def _sql_in(values: tuple[StrEnum, ...]) -> str:
    return ", ".join(f"'{v.value}'" for v in values)


# --------------------------------------------------------------------------- tables


class Customer(Base):
    __tablename__ = "customers"

    id: Mapped[int] = mapped_column(primary_key=True)
    customer_ref: Mapped[str] = mapped_column(String(16), unique=True)
    full_name: Mapped[str] = mapped_column(String(120))
    email: Mapped[str] = mapped_column(String(254), unique=True)
    pin_hash: Mapped[str] = mapped_column(String(200))
    tier: Mapped[CustomerTier] = mapped_column(_enum(CustomerTier), default=CustomerTier.STANDARD)
    status: Mapped[AccountStatus] = mapped_column(
        _enum(AccountStatus), default=AccountStatus.ACTIVE
    )
    failed_login_attempts: Mapped[int] = mapped_column(default=0)
    locked_until: Mapped[datetime | None] = mapped_column(UTCDateTime())
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)

    orders: Mapped[list[Order]] = relationship(back_populates="customer")

    @property
    def member_since(self) -> datetime:
        return self.created_at


class Order(Base):
    __tablename__ = "orders"

    id: Mapped[int] = mapped_column(primary_key=True)
    order_number: Mapped[str] = mapped_column(String(16), unique=True)
    customer_id: Mapped[int] = mapped_column(ForeignKey("customers.id"), index=True)
    status: Mapped[OrderStatus] = mapped_column(_enum(OrderStatus))
    currency: Mapped[str] = mapped_column(String(3), default="USD")
    placed_at: Mapped[datetime] = mapped_column(UTCDateTime())
    shipped_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    delivered_at: Mapped[datetime | None] = mapped_column(UTCDateTime())

    customer: Mapped[Customer] = relationship(back_populates="orders")
    items: Mapped[list[OrderItem]] = relationship(
        back_populates="order", cascade="all, delete-orphan", order_by="OrderItem.id"
    )

    @property
    def total_cents(self) -> int:
        return sum(item.line_total_cents for item in self.items)

    @property
    def item_count(self) -> int:
        return sum(item.quantity for item in self.items)


class OrderItem(Base):
    __tablename__ = "order_items"
    __table_args__ = (
        CheckConstraint("quantity > 0", name="ck_order_item_quantity_positive"),
        CheckConstraint("unit_price_cents >= 0", name="ck_order_item_price_non_negative"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    order_id: Mapped[int] = mapped_column(ForeignKey("orders.id", ondelete="CASCADE"), index=True)
    sku: Mapped[str] = mapped_column(String(32))
    product_name: Mapped[str] = mapped_column(String(160))
    category: Mapped[str] = mapped_column(String(32))
    quantity: Mapped[int]
    unit_price_cents: Mapped[int]
    is_final_sale: Mapped[bool] = mapped_column(default=False)

    order: Mapped[Order] = relationship(back_populates="items")

    @property
    def line_total_cents(self) -> int:
        return self.quantity * self.unit_price_cents


class SupportTicket(Base):
    __tablename__ = "support_tickets"
    __table_args__ = (
        UniqueConstraint("customer_id", "idempotency_key", name="uq_ticket_idempotency_key"),
        # At most one active ticket per (customer, order, category), even under races.
        Index(
            "uq_ticket_active_per_order_category",
            "customer_id",
            "order_id",
            "category",
            unique=True,
            sqlite_where=text(
                f"status IN ({_sql_in(ACTIVE_TICKET_STATUSES)}) AND order_id IS NOT NULL"
            ),
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    ticket_number: Mapped[str] = mapped_column(String(16), unique=True)
    customer_id: Mapped[int] = mapped_column(ForeignKey("customers.id"), index=True)
    order_id: Mapped[int | None] = mapped_column(ForeignKey("orders.id"), index=True)
    category: Mapped[TicketCategory] = mapped_column(_enum(TicketCategory))
    priority: Mapped[TicketPriority] = mapped_column(
        _enum(TicketPriority), default=TicketPriority.NORMAL
    )
    status: Mapped[TicketStatus] = mapped_column(_enum(TicketStatus), default=TicketStatus.OPEN)
    subject: Mapped[str] = mapped_column(String(120))
    description: Mapped[str] = mapped_column(Text)
    idempotency_key: Mapped[str | None] = mapped_column(String(64))
    request_fingerprint: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow, onupdate=utcnow)

    order: Mapped[Order | None] = relationship()

    @property
    def order_number(self) -> str | None:
        return self.order.order_number if self.order else None


class RefundRequest(Base):
    __tablename__ = "refund_requests"
    __table_args__ = (
        CheckConstraint("amount_cents > 0", name="ck_refund_amount_positive"),
        UniqueConstraint("customer_id", "idempotency_key", name="uq_refund_idempotency_key"),
        # At most one refund awaiting review per order, even under races.
        Index(
            "uq_refund_pending_per_order",
            "order_id",
            unique=True,
            sqlite_where=text(f"status = '{RefundStatus.PENDING_REVIEW.value}'"),
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    refund_number: Mapped[str] = mapped_column(String(16), unique=True)
    customer_id: Mapped[int] = mapped_column(ForeignKey("customers.id"), index=True)
    order_id: Mapped[int] = mapped_column(ForeignKey("orders.id"), index=True)
    amount_cents: Mapped[int]
    currency: Mapped[str] = mapped_column(String(3), default="USD")
    reason: Mapped[RefundReason] = mapped_column(_enum(RefundReason))
    details: Mapped[str | None] = mapped_column(Text)
    status: Mapped[RefundStatus] = mapped_column(
        _enum(RefundStatus), default=RefundStatus.PENDING_REVIEW
    )
    requires_manual_review: Mapped[bool] = mapped_column(default=False)
    policy_version: Mapped[str] = mapped_column(String(32))
    idempotency_key: Mapped[str | None] = mapped_column(String(64))
    request_fingerprint: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow, onupdate=utcnow)

    order: Mapped[Order] = relationship()

    @property
    def order_number(self) -> str:
        return self.order.order_number


class SupportPolicy(Base):
    """Human-readable policy documents (future grounding material for the LLM)."""

    __tablename__ = "support_policies"

    id: Mapped[int] = mapped_column(primary_key=True)
    slug: Mapped[str] = mapped_column(String(64), unique=True)
    title: Mapped[str] = mapped_column(String(160))
    category: Mapped[str] = mapped_column(String(32))
    body: Mapped[str] = mapped_column(Text)
    version: Mapped[str] = mapped_column(String(32))
    effective_date: Mapped[date] = mapped_column(Date)


class AuditLog(Base):
    """Append-only record of every CRM operation, including denied attempts."""

    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(primary_key=True)
    occurred_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow, index=True)
    actor: Mapped[str] = mapped_column(String(64))
    session_id: Mapped[str | None] = mapped_column(String(64))
    action: Mapped[str] = mapped_column(String(64), index=True)
    outcome: Mapped[AuditOutcome] = mapped_column(_enum(AuditOutcome))
    entity_type: Mapped[str | None] = mapped_column(String(32))
    entity_id: Mapped[str | None] = mapped_column(String(32))
    details: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


# --------------------------------------------------------------------------- staff (Phase 3)


class StaffRole(StrEnum):
    SUPPORT_AGENT = "support_agent"  # can view the review queue, cannot decide
    REFUND_APPROVER = "refund_approver"  # can decide refunds below the manual-review threshold
    SUPERVISOR = "supervisor"  # can decide any refund, including manual-review ones


class RefundDecisionType(StrEnum):
    APPROVED = "approved"
    REJECTED = "rejected"


class StaffMember(Base):
    """Store employees who may review refund requests. Separate from customers on purpose."""

    __tablename__ = "staff_members"

    id: Mapped[int] = mapped_column(primary_key=True)
    staff_ref: Mapped[str] = mapped_column(String(16), unique=True)
    full_name: Mapped[str] = mapped_column(String(120))
    email: Mapped[str] = mapped_column(String(254), unique=True)
    pin_hash: Mapped[str] = mapped_column(String(200))
    role: Mapped[StaffRole] = mapped_column(_enum(StaffRole))
    active: Mapped[bool] = mapped_column(default=True)
    failed_login_attempts: Mapped[int] = mapped_column(default=0)
    locked_until: Mapped[datetime | None] = mapped_column(UTCDateTime())
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)


class RefundDecision(Base):
    """A staff member's final decision on a refund request. At most one per request.

    No money moves: an approved decision records that a payout *would* be
    authorized. Payment processing is out of scope for this project.
    """

    __tablename__ = "refund_decisions"
    __table_args__ = (
        UniqueConstraint("refund_request_id", name="uq_refund_decision_per_request"),
        UniqueConstraint("idempotency_key", name="uq_refund_decision_idempotency_key"),
        CheckConstraint("amount_cents > 0", name="ck_refund_decision_amount_positive"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    refund_request_id: Mapped[int] = mapped_column(ForeignKey("refund_requests.id"))
    decision: Mapped[RefundDecisionType] = mapped_column(_enum(RefundDecisionType))
    decided_by_id: Mapped[int] = mapped_column(ForeignKey("staff_members.id"))
    required_role: Mapped[StaffRole] = mapped_column(_enum(StaffRole))
    amount_cents: Mapped[int]
    note: Mapped[str | None] = mapped_column(Text)
    idempotency_key: Mapped[str] = mapped_column(String(64))
    request_fingerprint: Mapped[str] = mapped_column(String(64))
    workflow_thread_id: Mapped[str | None] = mapped_column(String(64))
    decided_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)

    refund_request: Mapped[RefundRequest] = relationship()
    decided_by: Mapped[StaffMember] = relationship()


class ApprovalDecision(Base):
    """A customer's decision on one pending agent action. At most one per action.

    The unique ``action_id`` makes the decision an atomic transition: when two
    sessions answer the same confirmation at the same time, exactly one insert
    succeeds and the other sees the recorded decision.
    """

    __tablename__ = "agent_approval_decisions"

    id: Mapped[int] = mapped_column(primary_key=True)
    action_id: Mapped[str] = mapped_column(String(32), unique=True)
    thread_id: Mapped[str] = mapped_column(String(64), index=True)
    tool: Mapped[str] = mapped_column(String(64))
    decision: Mapped[str] = mapped_column(String(16))  # granted | declined | cancelled | expired
    actor: Mapped[str] = mapped_column(String(64))
    session_id: Mapped[str | None] = mapped_column(String(64))
    decided_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)

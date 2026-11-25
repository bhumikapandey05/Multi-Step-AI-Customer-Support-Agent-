"""Validated tool inputs and structured tool outputs.

Input models use ``extra="forbid"``: an LLM cannot smuggle fields such as
``customer_id`` or ``status`` into a write operation. Identity always comes
from the verified :class:`~support_agent.crm.identity.AuthContext`.
"""

import re
from datetime import date, datetime
from typing import Annotated, Any, TypeVar

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
)

from support_agent.crm.errors import InvalidInputError
from support_agent.db.models import (
    AccountStatus,
    CustomerTier,
    OrderStatus,
    RefundReason,
    RefundStatus,
    TicketCategory,
    TicketPriority,
    TicketStatus,
)

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _upper(value: Any) -> Any:
    return value.strip().upper() if isinstance(value, str) else value


def _lower(value: Any) -> Any:
    return value.strip().lower() if isinstance(value, str) else value


# Normalization runs *before* the pattern check (StringConstraints checks the pattern first).
OrderNumber = Annotated[str, BeforeValidator(_upper), StringConstraints(pattern=r"^ORD-\d{5}$")]
TicketNumber = Annotated[
    str, BeforeValidator(_upper), StringConstraints(pattern=r"^TKT-[0-9A-F]{8}$")
]
PolicySlug = Annotated[
    str, BeforeValidator(_lower), StringConstraints(pattern=r"^[a-z0-9-]{3,64}$")
]
IdempotencyKey = Annotated[
    str, StringConstraints(strip_whitespace=True, pattern=r"^[A-Za-z0-9_.:-]{8,64}$")
]


def _clean_single_line(value: Any) -> Any:
    if isinstance(value, str):
        return " ".join(_CONTROL_CHARS.sub("", value).split())
    return value


def _clean_multiline(value: Any) -> Any:
    if isinstance(value, str):
        return _CONTROL_CHARS.sub("", value).strip()
    return value


# --------------------------------------------------------------------------- inputs


class _Input(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class OrderLookupInput(_Input):
    order_number: OrderNumber


class TicketLookupInput(_Input):
    ticket_number: TicketNumber


class PolicyLookupInput(_Input):
    slug: PolicySlug


class ListInput(_Input):
    limit: int = Field(default=20, ge=1, le=100)


class CreateTicketInput(_Input):
    category: TicketCategory
    subject: str = Field(min_length=5, max_length=120)
    description: str = Field(min_length=10, max_length=2000)
    order_number: OrderNumber | None = None
    priority: TicketPriority = TicketPriority.NORMAL
    idempotency_key: IdempotencyKey | None = None

    _clean_subject = field_validator("subject", mode="before")(_clean_single_line)
    _clean_description = field_validator("description", mode="before")(_clean_multiline)


class RefundRequestInput(_Input):
    order_number: OrderNumber
    reason: RefundReason
    details: str | None = Field(default=None, max_length=1000)
    # Omit to request the full refundable amount.
    amount_cents: int | None = Field(default=None, gt=0, le=10_000_000)
    idempotency_key: IdempotencyKey | None = None

    _clean_details = field_validator("details", mode="before")(_clean_multiline)


InputT = TypeVar("InputT", bound=_Input)


def parse_input(model: type[InputT], data: InputT | dict[str, Any]) -> InputT:
    """Validate ``data`` into ``model``; Pydantic errors become :class:`InvalidInputError`."""
    if isinstance(data, model):
        return data
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        errors = [
            {"field": ".".join(str(p) for p in err["loc"]) or "(root)", "message": err["msg"]}
            for err in exc.errors()
        ]
        raise InvalidInputError(
            "The request has invalid fields.", details={"errors": errors}
        ) from exc


# --------------------------------------------------------------------------- outputs


class _Output(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class CustomerProfile(_Output):
    customer_ref: str
    full_name: str
    email: str
    tier: CustomerTier
    status: AccountStatus
    member_since: datetime


class OrderItemView(_Output):
    sku: str
    product_name: str
    category: str
    quantity: int
    unit_price_cents: int
    line_total_cents: int
    is_final_sale: bool


class OrderSummary(_Output):
    order_number: str
    status: OrderStatus
    placed_at: datetime
    delivered_at: datetime | None
    item_count: int
    total_cents: int
    currency: str


class OrderDetail(OrderSummary):
    shipped_at: datetime | None
    items: list[OrderItemView]


class TicketView(_Output):
    ticket_number: str
    order_number: str | None
    category: TicketCategory
    priority: TicketPriority
    status: TicketStatus
    subject: str
    description: str
    created_at: datetime
    updated_at: datetime


class TicketCreateResult(BaseModel):
    created: bool
    message: str
    ticket: TicketView


class EligibilityReason(BaseModel):
    code: str
    message: str


class RefundEligibility(BaseModel):
    order_number: str
    eligible: bool
    reasons: list[EligibilityReason] = Field(
        default_factory=list, description="Blocking reasons when not eligible."
    )
    notes: list[str] = Field(default_factory=list, description="Informational notes.")
    refundable_amount_cents: int
    already_refunded_cents: int
    window_ends_at: datetime | None
    requires_manual_review: bool
    policy_version: str


class RefundRequestView(_Output):
    refund_number: str
    order_number: str
    amount_cents: int
    currency: str
    reason: RefundReason
    status: RefundStatus
    requires_manual_review: bool
    created_at: datetime


class RefundRequestResult(BaseModel):
    created: bool
    message: str
    refund: RefundRequestView


class PolicySummary(_Output):
    slug: str
    title: str
    category: str
    version: str


class PolicyView(PolicySummary):
    body: str
    effective_date: date

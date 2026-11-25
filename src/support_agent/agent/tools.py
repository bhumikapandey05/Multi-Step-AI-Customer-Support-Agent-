"""Tools the LLM may call, each bound to an existing :class:`CRMService` method.

Design rules:

* No tool accepts ``customer_id``. The verified :class:`AuthContext` is bound by
  the agent session, never supplied by the model.
* Argument models use ``extra="forbid"``, so unknown or injected fields fail.
* The model never supplies idempotency keys. The agent derives them, so retries
  of the same action are safe.
* Write tools declare an approval preview. The agent runs it before asking the
  customer to confirm, and no write runs without that confirmation.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from support_agent.agent.llm import ToolDefinition
from support_agent.crm.errors import RefundNotEligibleError
from support_agent.crm.identity import AuthContext
from support_agent.crm.schemas import OrderNumber, PolicySlug, TicketNumber
from support_agent.crm.service import CRMService
from support_agent.db.models import RefundReason, TicketCategory, TicketPriority

# --------------------------------------------------------------------------- argument models


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class NoArgs(_Args):
    pass


class ListArgs(_Args):
    limit: int = Field(default=10, ge=1, le=50, description="Maximum number of records.")


class OrderArgs(_Args):
    order_number: OrderNumber = Field(description="Order number, format ORD-12345.")


class TicketArgs(_Args):
    ticket_number: TicketNumber = Field(description="Ticket number, format TKT-1A2B3C4D.")


class PolicyArgs(_Args):
    slug: PolicySlug = Field(description="Policy slug from list_support_policies.")


class CreateTicketArgs(_Args):
    category: TicketCategory = Field(description="Ticket category.")
    subject: str = Field(min_length=5, max_length=120, description="One-line summary.")
    description: str = Field(
        min_length=10, max_length=2000, description="The customer's issue in their own words."
    )
    order_number: OrderNumber | None = Field(
        default=None, description="Related order number, if any (format ORD-12345)."
    )
    priority: TicketPriority = Field(default=TicketPriority.NORMAL, description="Urgency.")


class RefundArgs(_Args):
    order_number: OrderNumber = Field(description="Order number, format ORD-12345.")
    reason: RefundReason = Field(description="Why the customer wants a refund.")
    amount_cents: int | None = Field(
        default=None,
        gt=0,
        description="Partial amount in cents. Omit to request the full refundable amount.",
    )
    details: str | None = Field(
        default=None, max_length=1000, description="Short explanation from the customer."
    )


# --------------------------------------------------------------------------- tool spec


@dataclass(frozen=True)
class ApprovalPreview:
    """What the customer sees before confirming a write."""

    summary: str
    details: dict[str, Any]


Handler = Callable[[CRMService, AuthContext | None, Any, str | None], Any]
Previewer = Callable[[CRMService, AuthContext, Any], ApprovalPreview]


@dataclass(frozen=True)
class AgentTool:
    name: str
    description: str
    args_model: type[_Args]
    handler: Handler
    requires_auth: bool = True
    is_write: bool = False
    preview: Previewer | None = None

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=llm_json_schema(self.args_model),
        )


# --------------------------------------------------------------------------- previews


def _dollars(cents: int) -> str:
    return f"${cents / 100:,.2f}"


def _preview_refund(crm: CRMService, ctx: AuthContext, args: RefundArgs) -> ApprovalPreview:
    # Runs the real eligibility check, so the customer is never asked to approve
    # something the CRM would reject. CRMErrors propagate to the model as results.
    eligibility = crm.check_refund_eligibility(ctx, args.order_number)
    if not eligibility.eligible:
        raise RefundNotEligibleError(
            f"Order {args.order_number} is not eligible for a refund: "
            f"{eligibility.reasons[0].message}",
            details={"eligibility": eligibility.model_dump(mode="json")},
        )
    amount = args.amount_cents or eligibility.refundable_amount_cents
    if amount > eligibility.refundable_amount_cents:
        raise RefundNotEligibleError(
            f"The requested amount exceeds the refundable amount of "
            f"{_dollars(eligibility.refundable_amount_cents)}.",
            details={"refundable_amount_cents": eligibility.refundable_amount_cents},
        )
    review = (
        " Because of the amount, a specialist will review it before any decision."
        if eligibility.requires_manual_review
        else ""
    )
    return ApprovalPreview(
        summary=(
            f"Submit a refund request of {_dollars(amount)} for order {args.order_number} "
            f"(reason: {args.reason.value.replace('_', ' ')}). Our team reviews every request "
            f"before any money is returned.{review}"
        ),
        details={
            "order_number": args.order_number,
            "amount_cents": amount,
            "reason": args.reason.value,
            "requires_manual_review": eligibility.requires_manual_review,
        },
    )


def _preview_ticket(crm: CRMService, ctx: AuthContext, args: CreateTicketArgs) -> ApprovalPreview:
    if args.order_number:
        crm.get_order(ctx, args.order_number)  # ownership check before asking
    about = f" about order {args.order_number}" if args.order_number else ""
    return ApprovalPreview(
        summary=(
            f"Open a {args.priority.value}-priority {args.category.value.replace('_', ' ')} "
            f'ticket{about}: "{args.subject}".'
        ),
        details=args.model_dump(mode="json", exclude_none=True),
    )


# --------------------------------------------------------------------------- handlers


def _create_ticket(crm: CRMService, ctx: Any, args: CreateTicketArgs, key: str | None) -> Any:
    payload = args.model_dump(exclude_none=True)
    if key:
        payload["idempotency_key"] = key
    return crm.create_ticket(ctx, payload)


def _request_refund(crm: CRMService, ctx: Any, args: RefundArgs, key: str | None) -> Any:
    payload = args.model_dump(exclude_none=True)
    if key:
        payload["idempotency_key"] = key
    return crm.request_refund(ctx, payload)


def default_tools() -> tuple[AgentTool, ...]:
    return (
        AgentTool(
            "list_support_policies",
            "List the store's published support policies (refunds, shipping, cancellation, ...).",
            NoArgs,
            lambda crm, ctx, a, k: crm.list_policies(),
            requires_auth=False,
        ),
        AgentTool(
            "get_support_policy",
            "Read the full text of one support policy. Use it to quote official rules.",
            PolicyArgs,
            lambda crm, ctx, a, k: crm.get_policy(a.slug),
            requires_auth=False,
        ),
        AgentTool(
            "get_customer_profile",
            "Get the verified customer's name, membership tier, and account status.",
            NoArgs,
            lambda crm, ctx, a, k: crm.get_profile(ctx),
        ),
        AgentTool(
            "list_orders",
            "List the verified customer's recent orders, newest first.",
            ListArgs,
            lambda crm, ctx, a, k: crm.list_orders(ctx, limit=a.limit),
        ),
        AgentTool(
            "get_order_details",
            "Get one order with its items, status, and shipping/delivery dates.",
            OrderArgs,
            lambda crm, ctx, a, k: crm.get_order(ctx, a.order_number),
        ),
        AgentTool(
            "check_refund_eligibility",
            "Check whether an order can be refunded under the refund policy, and for how much. "
            "Always call this before offering or requesting a refund.",
            OrderArgs,
            lambda crm, ctx, a, k: crm.check_refund_eligibility(ctx, a.order_number),
        ),
        AgentTool(
            "list_support_tickets",
            "List the verified customer's support tickets, newest first.",
            ListArgs,
            lambda crm, ctx, a, k: crm.list_tickets(ctx, limit=a.limit),
        ),
        AgentTool(
            "get_support_ticket",
            "Get one support ticket by its number.",
            TicketArgs,
            lambda crm, ctx, a, k: crm.get_ticket(ctx, a.ticket_number),
        ),
        AgentTool(
            "list_refund_requests",
            "List the verified customer's refund requests and their review status.",
            NoArgs,
            lambda crm, ctx, a, k: crm.list_refund_requests(ctx),
        ),
        AgentTool(
            "create_support_ticket",
            "Open a support ticket for an issue that needs a human (late package, damaged item, "
            "account problem, ...). The customer is asked to confirm before it is created.",
            CreateTicketArgs,
            _create_ticket,
            is_write=True,
            preview=_preview_ticket,
        ),
        AgentTool(
            "request_refund",
            "Submit a refund request for review. It does not issue money. The customer is asked "
            "to confirm first. Check eligibility before calling this.",
            RefundArgs,
            _request_refund,
            is_write=True,
            preview=_preview_refund,
        ),
    )


# --------------------------------------------------------------------------- JSON schema


# JSON-schema keywords Gemini function declarations handle reliably.
_ALLOWED_KEYS = {
    "type", "description", "enum", "properties", "required", "items",
    "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
    "minLength", "maxLength", "pattern", "default", "format", "nullable",
}  # fmt: skip


def llm_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Pydantic JSON schema, simplified for function calling.

    Inlines ``$ref``/``$defs``, collapses ``Optional[X]`` (``anyOf [X, null]``) to
    ``X``, and drops keywords some providers reject. Validation is still done
    by the full Pydantic model in Python, so this only guides the model.
    """
    schema = model.model_json_schema()
    defs = schema.pop("$defs", {})

    def resolve(node: Any) -> Any:
        if isinstance(node, list):
            return [resolve(n) for n in node]
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            target = resolve(defs[node["$ref"].rsplit("/", 1)[-1]])
            extra = {k: v for k, v in node.items() if k != "$ref"}
            return resolve({**target, **extra})
        if "anyOf" in node:
            options = [o for o in node["anyOf"] if o.get("type") != "null"]
            if len(options) == 1:
                extra = {k: v for k, v in node.items() if k != "anyOf"}
                return resolve({**options[0], **extra})
        out: dict[str, Any] = {}
        for key, value in node.items():
            if key == "properties":
                out[key] = {name: resolve(prop) for name, prop in value.items()}
            elif key in _ALLOWED_KEYS:
                out[key] = resolve(value)
        if out.get("default", ...) is None:
            del out["default"]
        return out

    result = resolve(schema)
    result.setdefault("type", "object")
    result.setdefault("properties", {})
    return result

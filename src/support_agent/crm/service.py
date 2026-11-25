"""CRM tools: the operations the support agent is allowed to perform.

Every authenticated method:

1. verifies the signed :class:`AuthContext` (identity never comes from tool arguments),
2. validates input with Pydantic,
3. runs inside a single database transaction,
4. scopes every query to the verified customer, and
5. writes an audit record -- in the same transaction on success, or in a
   separate transaction on failure so denied attempts are never lost.
"""

import hashlib
import logging
import secrets
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, selectinload, sessionmaker

from support_agent.clock import Clock, utcnow
from support_agent.crm.audit import record_audit
from support_agent.crm.errors import (
    AuthenticationError,
    AuthorizationError,
    ConflictError,
    CRMError,
    IdempotencyConflictError,
    NotFoundError,
    RefundNotEligibleError,
)
from support_agent.crm.identity import AuthContext, SessionSigner
from support_agent.crm.policies import (
    LineItemFacts,
    OrderFacts,
    RefundPolicy,
    evaluate_refund_eligibility,
)
from support_agent.crm.schemas import (
    CreateTicketInput,
    CustomerProfile,
    ListInput,
    OrderDetail,
    OrderLookupInput,
    OrderSummary,
    PolicyLookupInput,
    PolicySummary,
    PolicyView,
    RefundEligibility,
    RefundRequestInput,
    RefundRequestResult,
    RefundRequestView,
    TicketCreateResult,
    TicketLookupInput,
    TicketView,
    parse_input,
)
from support_agent.db.engine import transaction
from support_agent.db.models import (
    ACTIVE_TICKET_STATUSES,
    AccountStatus,
    AuditOutcome,
    Customer,
    Order,
    RefundRequest,
    RefundStatus,
    SupportPolicy,
    SupportTicket,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _Operation:
    action: str
    session: Session
    customer: Customer
    ctx: AuthContext


def _new_reference(prefix: str) -> str:
    return f"{prefix}-{secrets.token_hex(4).upper()}"


def _fingerprint(request: CreateTicketInput | RefundRequestInput) -> str:
    """Stable hash of a request's content, used to detect idempotency-key reuse."""
    canonical = request.model_dump_json(exclude={"idempotency_key"})
    return hashlib.sha256(canonical.encode()).hexdigest()


def _normalize_subject(subject: str) -> str:
    return " ".join(subject.casefold().split())


class CRMService:
    def __init__(
        self,
        session_factory: sessionmaker[Session],
        signer: SessionSigner,
        *,
        policy: RefundPolicy | None = None,
        clock: Clock = utcnow,
    ) -> None:
        self._session_factory = session_factory
        self._signer = signer
        self._policy = policy or RefundPolicy()
        self._clock = clock

    @property
    def policy(self) -> RefundPolicy:
        return self._policy

    # ------------------------------------------------------------ public reference data

    def list_policies(self) -> list[PolicySummary]:
        with transaction(self._session_factory) as session:
            rows = session.scalars(select(SupportPolicy).order_by(SupportPolicy.slug))
            return [PolicySummary.model_validate(row) for row in rows]

    def get_policy(self, slug: str) -> PolicyView:
        params = parse_input(PolicyLookupInput, {"slug": slug})
        with transaction(self._session_factory) as session:
            row = session.scalar(select(SupportPolicy).where(SupportPolicy.slug == params.slug))
            if row is None:
                raise NotFoundError(f"No support policy named '{params.slug}' exists.")
            return PolicyView.model_validate(row)

    # ------------------------------------------------------------ customer & orders

    def get_profile(self, ctx: AuthContext) -> CustomerProfile:
        with self._operation(ctx, "customer.profile.view") as op:
            self._audit_success(op, "customer", op.customer.customer_ref)
            return CustomerProfile.model_validate(op.customer)

    def list_orders(self, ctx: AuthContext, *, limit: int = 20) -> list[OrderSummary]:
        with self._operation(ctx, "order.list") as op:
            params = parse_input(ListInput, {"limit": limit})
            orders = op.session.scalars(
                select(Order)
                .where(Order.customer_id == op.customer.id)
                .options(selectinload(Order.items))
                .order_by(Order.placed_at.desc(), Order.id.desc())
                .limit(params.limit)
            ).all()
            self._audit_success(op, details={"count": len(orders)})
            return [OrderSummary.model_validate(order) for order in orders]

    def get_order(self, ctx: AuthContext, order_number: str) -> OrderDetail:
        with self._operation(ctx, "order.view") as op:
            params = parse_input(OrderLookupInput, {"order_number": order_number})
            order = self._get_owned_order(op, params.order_number)
            self._audit_success(op, "order", order.order_number)
            return OrderDetail.model_validate(order)

    def check_refund_eligibility(self, ctx: AuthContext, order_number: str) -> RefundEligibility:
        with self._operation(ctx, "refund.eligibility.check") as op:
            params = parse_input(OrderLookupInput, {"order_number": order_number})
            order = self._get_owned_order(op, params.order_number)
            result = self._evaluate(op, order)
            self._audit_success(
                op,
                "order",
                order.order_number,
                details={
                    "eligible": result.eligible,
                    "reasons": [r.code for r in result.reasons],
                    "refundable_amount_cents": result.refundable_amount_cents,
                },
            )
            return result

    # ------------------------------------------------------------ tickets

    def list_tickets(self, ctx: AuthContext, *, limit: int = 20) -> list[TicketView]:
        with self._operation(ctx, "ticket.list") as op:
            params = parse_input(ListInput, {"limit": limit})
            tickets = op.session.scalars(
                select(SupportTicket)
                .where(SupportTicket.customer_id == op.customer.id)
                .options(selectinload(SupportTicket.order))
                .order_by(SupportTicket.created_at.desc(), SupportTicket.id.desc())
                .limit(params.limit)
            ).all()
            self._audit_success(op, details={"count": len(tickets)})
            return [TicketView.model_validate(t) for t in tickets]

    def get_ticket(self, ctx: AuthContext, ticket_number: str) -> TicketView:
        with self._operation(ctx, "ticket.view") as op:
            params = parse_input(TicketLookupInput, {"ticket_number": ticket_number})
            ticket = op.session.scalar(
                select(SupportTicket).where(SupportTicket.ticket_number == params.ticket_number)
            )
            if ticket is None or ticket.customer_id != op.customer.id:
                raise self._not_found(
                    "ticket", params.ticket_number, owner_id=ticket.customer_id if ticket else None
                )
            self._audit_success(op, "ticket", ticket.ticket_number)
            return TicketView.model_validate(ticket)

    def create_ticket(
        self, ctx: AuthContext, request: CreateTicketInput | dict[str, Any]
    ) -> TicketCreateResult:
        with self._operation(ctx, "ticket.create", write=True) as op:
            req = parse_input(CreateTicketInput, request)
            fingerprint = _fingerprint(req)

            if req.idempotency_key:
                existing = op.session.scalar(
                    select(SupportTicket).where(
                        SupportTicket.customer_id == op.customer.id,
                        SupportTicket.idempotency_key == req.idempotency_key,
                    )
                )
                if existing is not None:
                    return self._ticket_replay(op, existing, fingerprint)

            order = self._get_owned_order(op, req.order_number) if req.order_number else None

            duplicate = self._find_active_duplicate_ticket(op, req, order)
            if duplicate is not None:
                return self._ticket_duplicate(op, duplicate)

            ticket = SupportTicket(
                ticket_number=_new_reference("TKT"),
                customer_id=op.customer.id,
                order_id=order.id if order else None,
                category=req.category,
                priority=req.priority,
                subject=req.subject,
                description=req.description,
                idempotency_key=req.idempotency_key,
                request_fingerprint=fingerprint,
                created_at=self._clock(),
                updated_at=self._clock(),
            )
            ticket.order = order
            try:
                # SAVEPOINT: a unique-index violation (a concurrent duplicate) only
                # rolls back this insert, not the whole operation.
                with op.session.begin_nested():
                    op.session.add(ticket)
                    op.session.flush()
            except IntegrityError:
                duplicate = self._find_active_duplicate_ticket(op, req, order)
                if duplicate is None and req.idempotency_key:
                    duplicate = op.session.scalar(
                        select(SupportTicket).where(
                            SupportTicket.customer_id == op.customer.id,
                            SupportTicket.idempotency_key == req.idempotency_key,
                        )
                    )
                if duplicate is None:
                    raise ConflictError("The ticket could not be created; please retry.") from None
                return self._ticket_duplicate(op, duplicate)

            self._audit_success(
                op,
                "ticket",
                ticket.ticket_number,
                details={
                    "category": req.category.value,
                    "priority": req.priority.value,
                    "order_number": req.order_number,
                },
            )
            return TicketCreateResult(
                created=True,
                message=f"Ticket {ticket.ticket_number} was created.",
                ticket=TicketView.model_validate(ticket),
            )

    # ------------------------------------------------------------ refunds

    def list_refund_requests(self, ctx: AuthContext) -> list[RefundRequestView]:
        with self._operation(ctx, "refund.list") as op:
            refunds = op.session.scalars(
                select(RefundRequest)
                .where(RefundRequest.customer_id == op.customer.id)
                .options(selectinload(RefundRequest.order))
                .order_by(RefundRequest.created_at.desc(), RefundRequest.id.desc())
            ).all()
            self._audit_success(op, details={"count": len(refunds)})
            return [RefundRequestView.model_validate(r) for r in refunds]

    def request_refund(
        self, ctx: AuthContext, request: RefundRequestInput | dict[str, Any]
    ) -> RefundRequestResult:
        """Record a refund request for human review. No money is moved."""
        with self._operation(ctx, "refund.request", write=True) as op:
            req = parse_input(RefundRequestInput, request)
            fingerprint = _fingerprint(req)

            if req.idempotency_key:
                existing = op.session.scalar(
                    select(RefundRequest).where(
                        RefundRequest.customer_id == op.customer.id,
                        RefundRequest.idempotency_key == req.idempotency_key,
                    )
                )
                if existing is not None:
                    return self._refund_replay(op, existing, fingerprint)

            order = self._get_owned_order(op, req.order_number)

            pending = self._find_pending_refund(op, order)
            if pending is not None:
                return self._refund_duplicate(op, pending)

            # Re-evaluate inside this transaction; never trust an earlier eligibility check.
            eligibility = self._evaluate(op, order)
            if not eligibility.eligible:
                raise RefundNotEligibleError(
                    f"Order {order.order_number} is not eligible for a refund: "
                    f"{eligibility.reasons[0].message}",
                    details={"eligibility": eligibility.model_dump(mode="json")},
                    internal={
                        "order_number": order.order_number,
                        "reasons": [r.code for r in eligibility.reasons],
                    },
                )

            amount = req.amount_cents or eligibility.refundable_amount_cents
            if amount > eligibility.refundable_amount_cents:
                raise RefundNotEligibleError(
                    f"The requested amount exceeds the refundable amount of "
                    f"{eligibility.refundable_amount_cents} cents.",
                    details={
                        "requested_amount_cents": amount,
                        "refundable_amount_cents": eligibility.refundable_amount_cents,
                    },
                    internal={"order_number": order.order_number, "reason": "amount_too_high"},
                )

            refund = RefundRequest(
                refund_number=_new_reference("RFD"),
                customer_id=op.customer.id,
                order_id=order.id,
                amount_cents=amount,
                currency=order.currency,
                reason=req.reason,
                details=req.details,
                status=RefundStatus.PENDING_REVIEW,
                requires_manual_review=amount >= self._policy.manual_review_threshold_cents,
                policy_version=self._policy.version,
                idempotency_key=req.idempotency_key,
                request_fingerprint=fingerprint,
                created_at=self._clock(),
                updated_at=self._clock(),
            )
            refund.order = order
            try:
                with op.session.begin_nested():
                    op.session.add(refund)
                    op.session.flush()
            except IntegrityError:
                pending = self._find_pending_refund(op, order)
                if pending is None:
                    raise ConflictError(
                        "The refund request could not be recorded; please retry."
                    ) from None
                return self._refund_duplicate(op, pending)

            self._audit_success(
                op,
                "refund_request",
                refund.refund_number,
                details={
                    "order_number": order.order_number,
                    "amount_cents": amount,
                    "reason": req.reason.value,
                    "requires_manual_review": refund.requires_manual_review,
                },
            )
            return RefundRequestResult(
                created=True,
                message=(
                    f"Refund request {refund.refund_number} for {amount} cents was submitted "
                    "and is awaiting review. No payment has been issued yet."
                ),
                refund=RefundRequestView.model_validate(refund),
            )

    # ------------------------------------------------------------ internals

    @contextmanager
    def _operation(
        self, ctx: AuthContext, action: str, *, write: bool = False
    ) -> Iterator[_Operation]:
        try:
            customer_id = self._signer.verify(ctx)
            with transaction(self._session_factory) as session:
                customer = session.get(Customer, customer_id)
                if customer is None:
                    raise AuthenticationError(
                        "The customer session is invalid.", internal={"reason": "unknown_customer"}
                    )
                if write and customer.status is not AccountStatus.ACTIVE:
                    raise AuthorizationError(
                        "This account cannot make changes right now. "
                        "A human support agent needs to help with this request.",
                        internal={"account_status": customer.status.value},
                    )
                yield _Operation(action=action, session=session, customer=customer, ctx=ctx)
        except Exception as exc:
            self._audit_failure(ctx, action, exc)
            raise

    def _audit_success(
        self,
        op: _Operation,
        entity_type: str | None = None,
        entity_id: str | None = None,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        record_audit(
            op.session,
            occurred_at=self._clock(),
            actor=op.ctx.actor,
            session_id=op.ctx.session_id,
            action=op.action,
            outcome=AuditOutcome.SUCCESS,
            entity_type=entity_type,
            entity_id=entity_id,
            details=details,
        )

    def _audit_failure(self, ctx: object, action: str, exc: Exception) -> None:
        if isinstance(exc, CRMError):
            outcome = AuditOutcome(exc.audit_outcome)
            details: dict[str, Any] = {"error": exc.code, "message": exc.message, **exc.internal}
        else:
            outcome = AuditOutcome.ERROR
            details = {"error": type(exc).__name__}

        if isinstance(ctx, AuthContext):
            # Identity claimed by an unverified context is recorded but labelled as such.
            verified = not isinstance(exc, AuthenticationError)
            actor = ctx.actor if verified else f"unverified:{ctx.actor}"
            session_id: str | None = ctx.session_id
        else:
            actor, session_id = "unknown", None

        log = logger.warning if outcome is AuditOutcome.DENIED else logger.info
        if outcome is AuditOutcome.ERROR:
            log = logger.error
        log(
            "CRM operation failed",
            extra={
                "action": action,
                "actor": actor,
                "outcome": outcome.value,
                "error_code": details["error"],
            },
        )
        entity_id = details.get("reference") or details.get("order_number")
        try:
            with transaction(self._session_factory) as session:
                record_audit(
                    session,
                    occurred_at=self._clock(),
                    actor=actor,
                    session_id=session_id,
                    action=action,
                    outcome=outcome,
                    entity_id=entity_id,
                    details=details,
                )
        except Exception:
            logger.exception("Failed to write audit record", extra={"action": action})

    def _not_found(self, entity: str, reference: str, *, owner_id: int | None) -> NotFoundError:
        # Same message whether the record does not exist or belongs to someone else,
        # so the response never confirms another customer's data exists.
        message = f"No {entity} {reference} was found on your account."
        if owner_id is None:
            return NotFoundError(
                message, internal={"reference": reference, "reason": "does_not_exist"}
            )
        return NotFoundError(
            message,
            internal={
                "reference": reference,
                "reason": "owned_by_another_customer",
                "owner_customer_id": owner_id,
            },
            audit_outcome=AuditOutcome.DENIED.value,
        )

    def _get_owned_order(self, op: _Operation, order_number: str) -> Order:
        order = op.session.scalar(
            select(Order)
            .where(Order.order_number == order_number)
            .options(selectinload(Order.items))
        )
        if order is None or order.customer_id != op.customer.id:
            raise self._not_found(
                "order", order_number, owner_id=order.customer_id if order else None
            )
        return order

    def _evaluate(self, op: _Operation, order: Order) -> RefundEligibility:
        approved = op.session.scalar(
            select(func.coalesce(func.sum(RefundRequest.amount_cents), 0)).where(
                RefundRequest.order_id == order.id,
                RefundRequest.status == RefundStatus.APPROVED,
            )
        )
        facts = OrderFacts(
            order_number=order.order_number,
            status=order.status,
            delivered_at=order.delivered_at,
            customer_tier=op.customer.tier,
            items=tuple(
                LineItemFacts(
                    i.product_name, i.category, i.quantity, i.unit_price_cents, i.is_final_sale
                )
                for i in order.items
            ),
            approved_refunds_cents=int(approved or 0),
            has_pending_refund=self._find_pending_refund(op, order) is not None,
        )
        return evaluate_refund_eligibility(facts, now=self._clock(), policy=self._policy)

    @staticmethod
    def _find_pending_refund(op: _Operation, order: Order) -> RefundRequest | None:
        return op.session.scalar(
            select(RefundRequest).where(
                RefundRequest.order_id == order.id,
                RefundRequest.status == RefundStatus.PENDING_REVIEW,
            )
        )

    @staticmethod
    def _find_active_duplicate_ticket(
        op: _Operation, req: CreateTicketInput, order: Order | None
    ) -> SupportTicket | None:
        """An active ticket for the same order+category, or (no order) the same subject."""
        stmt = select(SupportTicket).where(
            SupportTicket.customer_id == op.customer.id,
            SupportTicket.category == req.category,
            SupportTicket.status.in_(ACTIVE_TICKET_STATUSES),
        )
        if order is not None:
            return op.session.scalar(stmt.where(SupportTicket.order_id == order.id))
        subject = _normalize_subject(req.subject)
        for ticket in op.session.scalars(stmt.where(SupportTicket.order_id.is_(None))):
            if _normalize_subject(ticket.subject) == subject:
                return ticket
        return None

    def _ticket_replay(
        self, op: _Operation, existing: SupportTicket, fingerprint: str
    ) -> TicketCreateResult:
        if existing.request_fingerprint != fingerprint:
            raise IdempotencyConflictError(
                "This idempotency key was already used for a different ticket.",
                internal={"existing_ticket": existing.ticket_number},
            )
        return self._ticket_duplicate(op, existing)

    def _ticket_duplicate(self, op: _Operation, existing: SupportTicket) -> TicketCreateResult:
        record_audit(
            op.session,
            occurred_at=self._clock(),
            actor=op.ctx.actor,
            session_id=op.ctx.session_id,
            action=op.action,
            outcome=AuditOutcome.DUPLICATE,
            entity_type="ticket",
            entity_id=existing.ticket_number,
        )
        return TicketCreateResult(
            created=False,
            message=f"An open ticket for this issue already exists: {existing.ticket_number}.",
            ticket=TicketView.model_validate(existing),
        )

    def _refund_replay(
        self, op: _Operation, existing: RefundRequest, fingerprint: str
    ) -> RefundRequestResult:
        if existing.request_fingerprint != fingerprint:
            raise IdempotencyConflictError(
                "This idempotency key was already used for a different refund request.",
                internal={"existing_refund": existing.refund_number},
            )
        return self._refund_duplicate(op, existing)

    def _refund_duplicate(self, op: _Operation, existing: RefundRequest) -> RefundRequestResult:
        record_audit(
            op.session,
            occurred_at=self._clock(),
            actor=op.ctx.actor,
            session_id=op.ctx.session_id,
            action=op.action,
            outcome=AuditOutcome.DUPLICATE,
            entity_type="refund_request",
            entity_id=existing.refund_number,
        )
        return RefundRequestResult(
            created=False,
            message=(
                f"Refund request {existing.refund_number} already exists for this order "
                f"(status: {existing.status.value}). No new request was created."
            ),
            refund=RefundRequestView.model_validate(existing),
        )

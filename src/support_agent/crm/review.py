"""Staff review of refund requests: the decision that would actually release money.

Two different human acts exist in this system and must not be confused:

1. **Customer confirmation** (Phase 2): the customer confirms that the agent may
   *submit a refund request*. Result: a ``RefundRequest`` in ``pending_review``.
   No money is authorized.
2. **Staff approval** (this module): an authorized employee approves or rejects
   that request. Result: a ``RefundDecision`` row and the request moves to
   ``approved`` or ``rejected``. Payment processing is out of scope, so even an
   approval moves no money here.

Every rule is decided from database state, never from LLM output:

* Only staff sessions (:class:`StaffContext`) are accepted; customer sessions fail.
* ``support_agent`` staff cannot decide. ``refund_approver`` can decide amounts
  below the manual-review threshold. Anything at or above it, including when
  the *cumulative* approved amount for the order would cross it (so splitting a
  big refund into small requests does not dodge review), needs a ``supervisor``.
* Approval re-checks eligibility as of the request date and the refundable cap.
* One decision per request (unique index), idempotent replays, no flip-flopping.
"""

import hashlib
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, Field, StringConstraints
from sqlalchemy import func, select, update
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
    InvalidInputError,
    NotFoundError,
    RefundNotEligibleError,
)
from support_agent.crm.policies import (
    LineItemFacts,
    OrderFacts,
    RefundPolicy,
    evaluate_refund_eligibility,
)
from support_agent.crm.schemas import IdempotencyKey, _Input, _upper, parse_input
from support_agent.crm.staff import StaffContext, StaffSessionSigner
from support_agent.db.engine import transaction
from support_agent.db.models import (
    AccountStatus,
    AuditOutcome,
    Customer,
    Order,
    OrderStatus,
    RefundDecision,
    RefundDecisionType,
    RefundReason,
    RefundRequest,
    RefundStatus,
    StaffMember,
    StaffRole,
)

logger = logging.getLogger(__name__)

RefundNumber = Annotated[
    str, BeforeValidator(_upper), StringConstraints(pattern=r"^RFD-[0-9A-F]{8}$")
]

NO_PAYOUT_NOTE = "No payment was processed: payouts are out of scope for this system."


class DecisionInput(_Input):
    refund_number: RefundNumber
    decision: RefundDecisionType
    note: str | None = Field(default=None, max_length=500)
    idempotency_key: IdempotencyKey


class RefundCase(BaseModel):
    """What a reviewer sees. ``customer_details`` is customer-written text, not instructions."""

    refund_number: str
    order_number: str
    customer_ref: str
    customer_name: str
    amount_cents: int
    currency: str
    reason: RefundReason
    customer_details: str | None
    status: RefundStatus
    requires_manual_review: bool
    required_role: StaffRole
    approval_blockers: list[str]
    already_approved_cents: int
    refundable_cap_cents: int
    created_at: datetime


class RefundDecisionResult(BaseModel):
    created: bool
    message: str
    refund_number: str
    decision: RefundDecisionType
    status: RefundStatus
    amount_cents: int
    decided_by: str
    required_role: StaffRole
    decided_at: datetime
    payout: str = NO_PAYOUT_NOTE


@dataclass(slots=True)
class _StaffOp:
    action: str
    session: Session
    staff: StaffMember
    ctx: StaffContext


def _fingerprint(req: DecisionInput) -> str:
    canonical = req.model_dump_json(exclude={"idempotency_key"})
    return hashlib.sha256(canonical.encode()).hexdigest()


class RefundReviewService:
    def __init__(
        self,
        session_factory: sessionmaker[Session],
        signer: StaffSessionSigner,
        *,
        policy: RefundPolicy | None = None,
        clock: Clock = utcnow,
    ) -> None:
        self._session_factory = session_factory
        self._signer = signer
        self._policy = policy or RefundPolicy()
        self._clock = clock

    # ------------------------------------------------------------ reads

    def list_queue(self, ctx: StaffContext) -> list[RefundCase]:
        with self._operation(ctx, "refund.review.queue") as op:
            refunds = op.session.scalars(
                select(RefundRequest)
                .where(RefundRequest.status == RefundStatus.PENDING_REVIEW)
                .order_by(RefundRequest.created_at, RefundRequest.id)
            ).all()
            cases = [self._case(op.session, r) for r in refunds]
            self._audit(op, details={"count": len(cases)})
            return cases

    def get_case(self, ctx: StaffContext, refund_number: str) -> RefundCase:
        with self._operation(ctx, "refund.review.view") as op:
            refund = self._get_refund(op.session, self._parse_number(refund_number))
            self._audit(op, entity_id=refund.refund_number)
            return self._case(op.session, refund)

    def system_case(self, refund_number: str) -> RefundCase:
        """Internal read used by the review workflow to open a case (never exposed to the LLM)."""
        with transaction(self._session_factory) as session:
            return self._case(session, self._get_refund(session, self._parse_number(refund_number)))

    # ------------------------------------------------------------ decision

    def decide(
        self,
        ctx: StaffContext,
        request: DecisionInput | dict[str, Any],
        *,
        workflow_thread_id: str | None = None,
    ) -> RefundDecisionResult:
        with self._operation(ctx, "refund.review.decide") as op:
            req = parse_input(DecisionInput, request)
            fingerprint = _fingerprint(req)
            op.action = f"refund.review.{req.decision.value}"

            replay = op.session.scalar(
                select(RefundDecision).where(RefundDecision.idempotency_key == req.idempotency_key)
            )
            if replay is not None:
                if replay.request_fingerprint != fingerprint:
                    raise IdempotencyConflictError(
                        "This decision key was already used for a different decision.",
                        internal={"reference": replay.refund_request.refund_number},
                    )
                return self._duplicate(op, replay)

            if op.staff.role is StaffRole.SUPPORT_AGENT:
                raise AuthorizationError(
                    "Your role can view refund requests but cannot approve or reject them.",
                    internal={"reference": req.refund_number, "role": op.staff.role.value},
                )

            refund = self._get_refund(op.session, req.refund_number)
            existing = op.session.scalar(
                select(RefundDecision).where(RefundDecision.refund_request_id == refund.id)
            )
            if existing is not None or refund.status is not RefundStatus.PENDING_REVIEW:
                raise ConflictError(
                    f"Refund request {refund.refund_number} was already decided "
                    f"(status: {refund.status.value}).",
                    internal={"reference": refund.refund_number},
                )

            case = self._case(op.session, refund)
            if (
                case.required_role is StaffRole.SUPERVISOR
                and op.staff.role is not StaffRole.SUPERVISOR
            ):
                raise AuthorizationError(
                    "This refund needs a supervisor's decision (manual-review amount).",
                    details={"required_role": case.required_role.value},
                    internal={"reference": refund.refund_number, "role": op.staff.role.value},
                )
            customer = op.session.get(Customer, refund.customer_id)
            if customer is not None and customer.email == op.staff.email:
                raise AuthorizationError(
                    "Staff cannot decide on their own refund requests.",
                    internal={"reference": refund.refund_number, "reason": "self_review"},
                )

            if req.decision is RefundDecisionType.APPROVED and case.approval_blockers:
                raise RefundNotEligibleError(
                    f"Refund request {refund.refund_number} cannot be approved: "
                    + ", ".join(case.approval_blockers),
                    details={"approval_blockers": case.approval_blockers},
                    internal={"reference": refund.refund_number},
                )
            if req.decision is RefundDecisionType.REJECTED and len((req.note or "").strip()) < 5:
                raise InvalidInputError(
                    "A rejection needs a short note for the customer (at least 5 characters).",
                    internal={"reference": refund.refund_number},
                )

            now = self._clock()
            decision = RefundDecision(
                refund_request_id=refund.id,
                decision=req.decision,
                decided_by_id=op.staff.id,
                required_role=case.required_role,
                amount_cents=refund.amount_cents,
                note=(req.note or "").strip() or None,
                idempotency_key=req.idempotency_key,
                request_fingerprint=fingerprint,
                workflow_thread_id=workflow_thread_id,
                decided_at=now,
            )
            decision.refund_request = refund
            decision.decided_by = op.staff
            try:
                with op.session.begin_nested():
                    op.session.add(decision)
                    op.session.flush()
            except IntegrityError:
                # A concurrent reviewer (or a retry) got there first.
                winner = op.session.scalar(
                    select(RefundDecision).where(
                        (RefundDecision.idempotency_key == req.idempotency_key)
                        | (RefundDecision.refund_request_id == refund.id)
                    )
                )
                if winner is not None and winner.idempotency_key == req.idempotency_key:
                    return self._duplicate(op, winner)
                raise ConflictError(
                    f"Refund request {refund.refund_number} was already decided.",
                    internal={"reference": refund.refund_number},
                ) from None

            new_status = (
                RefundStatus.APPROVED
                if req.decision is RefundDecisionType.APPROVED
                else RefundStatus.REJECTED
            )
            # Atomic compare-and-set: only a request that is still pending can move.
            moved = op.session.execute(
                update(RefundRequest)
                .where(
                    RefundRequest.id == refund.id,
                    RefundRequest.status == RefundStatus.PENDING_REVIEW,
                )
                .values(status=new_status, updated_at=now)
                .execution_options(synchronize_session=False)
            ).rowcount
            if moved != 1:
                raise ConflictError(
                    f"Refund request {refund.refund_number} was already decided.",
                    internal={"reference": refund.refund_number},
                )
            op.session.refresh(refund)
            self._audit(
                op,
                entity_id=refund.refund_number,
                details={
                    "order_number": refund.order.order_number,
                    "amount_cents": refund.amount_cents,
                    "required_role": case.required_role.value,
                    "staff_role": op.staff.role.value,
                    "workflow_thread_id": workflow_thread_id,
                    "payout": "not_processed",
                },
            )
            verb = "approved" if req.decision is RefundDecisionType.APPROVED else "rejected"
            return self._result(
                decision, created=True, message=f"Refund {refund.refund_number} {verb}."
            )

    # ------------------------------------------------------------ rules

    def _case(self, session: Session, refund: RefundRequest) -> RefundCase:
        order = session.scalar(
            select(Order).where(Order.id == refund.order_id).options(selectinload(Order.items))
        )
        customer = session.get(Customer, refund.customer_id)
        assert order is not None and customer is not None
        approved = int(
            session.scalar(
                select(func.coalesce(func.sum(RefundRequest.amount_cents), 0)).where(
                    RefundRequest.order_id == order.id,
                    RefundRequest.status == RefundStatus.APPROVED,
                    RefundRequest.id != refund.id,
                )
            )
            or 0
        )
        # Eligibility as of the day the customer asked: a request filed inside the
        # window stays decidable, but the refundable cap is recomputed now.
        eligibility = evaluate_refund_eligibility(
            OrderFacts(
                order_number=order.order_number,
                status=order.status,
                delivered_at=order.delivered_at,
                customer_tier=customer.tier,
                items=tuple(
                    LineItemFacts(
                        i.product_name, i.category, i.quantity, i.unit_price_cents, i.is_final_sale
                    )
                    for i in order.items
                ),
                approved_refunds_cents=approved,
                has_pending_refund=False,
            ),
            now=refund.created_at,
            policy=self._policy,
        )
        blockers: list[str] = []
        if refund.status is not RefundStatus.PENDING_REVIEW:
            blockers.append(f"already_{refund.status.value}")
        if order.status is not OrderStatus.DELIVERED:
            blockers.append("order_not_delivered")
        blockers.extend(r.code for r in eligibility.reasons if r.code not in blockers)
        if eligibility.eligible and refund.amount_cents > eligibility.refundable_amount_cents:
            blockers.append("amount_exceeds_refundable")
        if customer.status is not AccountStatus.ACTIVE:
            blockers.append("customer_account_not_active")

        threshold = self._policy.manual_review_threshold_cents
        needs_supervisor = (
            refund.requires_manual_review
            or refund.amount_cents >= threshold
            or approved + refund.amount_cents >= threshold
        )
        return RefundCase(
            refund_number=refund.refund_number,
            order_number=order.order_number,
            customer_ref=customer.customer_ref,
            customer_name=customer.full_name,
            amount_cents=refund.amount_cents,
            currency=refund.currency,
            reason=refund.reason,
            customer_details=refund.details,
            status=refund.status,
            requires_manual_review=refund.requires_manual_review,
            required_role=StaffRole.SUPERVISOR if needs_supervisor else StaffRole.REFUND_APPROVER,
            approval_blockers=blockers,
            already_approved_cents=approved,
            refundable_cap_cents=eligibility.refundable_amount_cents,
            created_at=refund.created_at,
        )

    # ------------------------------------------------------------ internals

    @staticmethod
    def _parse_number(refund_number: str) -> str:
        class _Lookup(_Input):
            refund_number: RefundNumber

        return parse_input(_Lookup, {"refund_number": refund_number}).refund_number

    @staticmethod
    def _get_refund(session: Session, refund_number: str) -> RefundRequest:
        refund = session.scalar(
            select(RefundRequest)
            .where(RefundRequest.refund_number == refund_number)
            .options(selectinload(RefundRequest.order))
        )
        if refund is None:
            raise NotFoundError(
                f"No refund request {refund_number} exists.", internal={"reference": refund_number}
            )
        return refund

    def _duplicate(self, op: _StaffOp, existing: RefundDecision) -> RefundDecisionResult:
        self._audit(
            op,
            entity_id=existing.refund_request.refund_number,
            outcome=AuditOutcome.DUPLICATE,
        )
        return self._result(
            existing,
            created=False,
            message=(
                f"Refund {existing.refund_request.refund_number} was already "
                f"{existing.decision.value}; nothing was changed."
            ),
        )

    @staticmethod
    def _result(decision: RefundDecision, *, created: bool, message: str) -> RefundDecisionResult:
        return RefundDecisionResult(
            created=created,
            message=message,
            refund_number=decision.refund_request.refund_number,
            decision=decision.decision,
            status=decision.refund_request.status,
            amount_cents=decision.amount_cents,
            decided_by=decision.decided_by.staff_ref,
            required_role=decision.required_role,
            decided_at=decision.decided_at,
        )

    @contextmanager
    def _operation(self, ctx: StaffContext, action: str) -> Iterator[_StaffOp]:
        op: _StaffOp | None = None
        try:
            staff_id = self._signer.verify(ctx)
            with transaction(self._session_factory) as session:
                staff = session.get(StaffMember, staff_id)
                if staff is None or not staff.active:
                    raise AuthenticationError(
                        "The staff session is invalid.", internal={"reason": "inactive_or_unknown"}
                    )
                op = _StaffOp(action=action, session=session, staff=staff, ctx=ctx)
                yield op
        except Exception as exc:
            self._audit_failure(ctx, op.action if op else action, exc)
            raise

    def _audit(
        self,
        op: _StaffOp,
        *,
        entity_id: str | None = None,
        outcome: AuditOutcome = AuditOutcome.SUCCESS,
        details: dict[str, Any] | None = None,
    ) -> None:
        record_audit(
            op.session,
            occurred_at=self._clock(),
            actor=op.ctx.actor,
            session_id=op.ctx.session_id,
            action=op.action,
            outcome=outcome,
            entity_type="refund_request" if entity_id else None,
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
        if isinstance(ctx, StaffContext):
            verified = not isinstance(exc, AuthenticationError)
            actor = ctx.actor if verified else f"unverified:{ctx.actor}"
            session_id: str | None = ctx.session_id
        else:
            actor, session_id = "unknown", None
        logger.warning(
            "Refund review operation failed",
            extra={"action": action, "actor": actor, "error_code": details["error"]},
        )
        try:
            with transaction(self._session_factory) as session:
                record_audit(
                    session,
                    occurred_at=self._clock(),
                    actor=actor,
                    session_id=session_id,
                    action=action,
                    outcome=outcome,
                    entity_type="refund_request",
                    entity_id=details.get("reference"),
                    details=details,
                )
        except Exception:
            logger.exception("Failed to write audit record", extra={"action": action})

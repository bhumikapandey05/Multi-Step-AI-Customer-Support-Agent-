"""Staff endpoints: the refund review queue, decisions, and the audit log.

Role checks use the staff member's *current* role from the database. The
review service enforces the same limits again (approver < $500, supervisor
for manual review, notes on rejections), so the API cannot be talked into a
decision the CRM would refuse.
"""

from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, Path, Query, Request
from sqlalchemy import select

from support_agent.api.errors import CRM_STATUS, ApiError
from support_agent.api.schemas import ApproveRequest, AuditEventOut, RefundNumberPath, RejectRequest
from support_agent.api.security import StaffPrincipal, current_staff, require_staff_role
from support_agent.db.engine import transaction
from support_agent.db.models import AuditLog, AuditOutcome, StaffRole
from support_agent.workflow.review import ReviewWorkflowError

router = APIRouter(prefix="/v1/staff", tags=["staff"])

Staff = Annotated[StaffPrincipal, Depends(current_staff)]
Auditor = Annotated[
    StaffPrincipal, Depends(require_staff_role(StaffRole.REFUND_APPROVER, StaffRole.SUPERVISOR))
]
RefundNumber = Annotated[RefundNumberPath, Path()]


def _services(request: Request):
    return request.app.state.services


@router.get("/me")
def me(staff: Staff) -> dict[str, Any]:
    return {"staff_ref": staff.staff_ref, "name": staff.full_name, "role": staff.role}


@router.get("/refunds")
def queue(request: Request, staff: Staff) -> list[dict[str, Any]]:
    services = _services(request)
    workflow = services.review_workflow()
    rows = []
    for case in services.app.review.list_queue(staff.ctx):
        try:
            flow = workflow.status(case.refund_number).as_row()
        except ReviewWorkflowError:
            flow = None  # filed outside the chat; opened when a reviewer looks at it
        rows.append({"case": case.model_dump(mode="json"), "workflow": flow})
    return rows


@router.get("/refunds/{refund_number}")
def case(refund_number: RefundNumber, request: Request, staff: Staff) -> dict[str, Any]:
    services = _services(request)
    found = services.app.review.get_case(staff.ctx, refund_number)
    with services.locks.hold(f"review:{found.refund_number}"):
        flow = services.review_workflow().open(found.refund_number)
    return {"case": found.model_dump(mode="json"), "workflow": flow.as_row()}


@router.post("/refunds/{refund_number}/approve")
def approve(
    refund_number: RefundNumber,
    request: Request,
    staff: Staff,
    body: ApproveRequest | None = None,
) -> dict[str, Any]:
    return _decide(request, staff, refund_number, "approve", body.note if body else None)


@router.post("/refunds/{refund_number}/reject")
def reject(
    refund_number: RefundNumber, body: RejectRequest, request: Request, staff: Staff
) -> dict[str, Any]:
    return _decide(request, staff, refund_number, "reject", body.note)


@router.post("/refunds/{refund_number}/resume")
def resume(refund_number: RefundNumber, request: Request, staff: Staff) -> dict[str, Any]:
    services = _services(request)
    number = refund_number.upper()
    with services.locks.hold(f"review:{number}"):
        return services.review_workflow().resume(staff.ctx, number).as_row()


@router.get("/audit", response_model=list[AuditEventOut])
def audit_log(
    request: Request,
    staff: Auditor,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    action: Annotated[str | None, Query(max_length=64, description="Action prefix")] = None,
    actor: Annotated[str | None, Query(max_length=64)] = None,
    outcome: AuditOutcome | None = None,
) -> list[AuditEventOut]:
    stmt = select(AuditLog).order_by(AuditLog.id.desc()).limit(limit)
    if action:
        stmt = stmt.where(AuditLog.action.startswith(action, autoescape=True))
    if actor:
        stmt = stmt.where(AuditLog.actor == actor)
    if outcome:
        stmt = stmt.where(AuditLog.outcome == outcome)
    with transaction(_services(request).app.session_factory) as session:
        return [
            AuditEventOut(
                id=e.id,
                occurred_at=e.occurred_at,
                actor=e.actor,
                action=e.action,
                outcome=e.outcome.value,
                entity_type=e.entity_type,
                entity_id=e.entity_id,
                details=e.details or {},
            )
            for e in session.scalars(stmt)
        ]


def _decide(
    request: Request,
    staff: StaffPrincipal,
    refund_number: str,
    decision: Literal["approve", "reject"],
    note: str | None,
) -> dict[str, Any]:
    services = _services(request)
    found = services.app.review.get_case(staff.ctx, refund_number)  # staff-only read + audit
    with services.locks.hold(f"review:{found.refund_number}"):
        status = services.review_workflow().decide(staff.ctx, found.refund_number, decision, note)
    if status.result is None:
        # The workflow refused this attempt (role, business rule, conflict) and stays as it was.
        attempt = status.last_attempt or {}
        code = attempt.get("error", "not_decided")
        raise ApiError(
            CRM_STATUS.get(code, 409),
            code,
            attempt.get("message") or "The decision was not applied.",
            {"workflow": status.as_row()},
        )
    return status.as_row()

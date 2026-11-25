"""Sign-in endpoints. They return signed bearer tokens; PINs are never stored or echoed."""

from fastapi import APIRouter, Request

from support_agent.api.schemas import LoginRequest, TokenResponse
from support_agent.db.engine import transaction
from support_agent.db.models import StaffMember

router = APIRouter(prefix="/v1/auth", tags=["auth"])


@router.post("/customer/login", response_model=TokenResponse)
def customer_login(body: LoginRequest, request: Request) -> TokenResponse:
    app = request.app.state.services.app
    ctx = app.identity.authenticate(body.email, body.pin)
    profile = app.crm.get_profile(ctx)
    return TokenResponse(
        token=app.identity.signer.encode_token(ctx),
        expires_at=ctx.expires_at,
        kind="customer",
        profile={"name": profile.full_name, "tier": profile.tier, "status": profile.status},
    )


@router.post("/guest", response_model=TokenResponse)
def guest_session(request: Request) -> TokenResponse:
    token, session = request.app.state.services.guest_signer.issue()
    return TokenResponse(token=token, expires_at=session.expires_at, kind="guest")


@router.post("/staff/login", response_model=TokenResponse)
def staff_login(body: LoginRequest, request: Request) -> TokenResponse:
    app = request.app.state.services.app
    ctx = app.staff_identity.authenticate(body.email, body.pin)
    with transaction(app.session_factory) as session:
        staff = session.get(StaffMember, ctx.staff_id)
        profile = {"name": staff.full_name, "staff_ref": staff.staff_ref, "role": staff.role}
    return TokenResponse(
        token=app.staff_identity.signer.encode_token(ctx),
        expires_at=ctx.expires_at,
        kind="staff",
        profile=profile,
    )

"""Authentication and authorization for the API.

The server, not the client, decides who is calling:

* Every protected route takes an ``Authorization: Bearer <token>`` header.
  Tokens are the existing HMAC-signed sessions (customer ``v1.``, staff ``s1.``)
  plus a signed guest token (``g1.``). They are verified on every request.
* No route accepts a customer id or staff id from the client. The customer comes
  from the verified ``AuthContext``; the staff member and their *current* role
  come from the database, looked up by the verified ``StaffContext``.
* A token of the wrong kind is refused (a customer token cannot call staff
  routes, a guest token cannot read orders).
"""

import base64
import hashlib
import hmac
import json
import secrets
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Annotated

from fastapi import Depends, Request

from support_agent.api.errors import ApiError
from support_agent.clock import Clock, utcnow
from support_agent.crm.audit import record_audit
from support_agent.crm.errors import AuthenticationError
from support_agent.crm.identity import AuthContext
from support_agent.crm.staff import StaffContext
from support_agent.db.engine import transaction
from support_agent.db.models import AuditOutcome, StaffMember, StaffRole

_GUEST_VERSION = "g1"


@dataclass(frozen=True)
class GuestSession:
    guest_id: str
    expires_at: datetime

    @property
    def actor(self) -> str:
        return f"guest:{self.guest_id}"


class GuestTokenSigner:
    """Signed, expiring guest sessions. A guest can only resume their own guest chats."""

    def __init__(self, secret_key: str, *, ttl: timedelta, clock: Clock = utcnow) -> None:
        self._key = hmac.new(
            secret_key.encode(), b"support-agent/guest-session/v1", "sha256"
        ).digest()
        self._ttl = ttl
        self._clock = clock

    def issue(self) -> tuple[str, GuestSession]:
        session = GuestSession(
            secrets.token_hex(12), self._clock().replace(microsecond=0) + self._ttl
        )
        payload = json.dumps(
            {"gid": session.guest_id, "exp": int(session.expires_at.timestamp())},
            separators=(",", ":"),
        ).encode()
        encoded = base64.urlsafe_b64encode(payload).rstrip(b"=").decode()
        return f"{_GUEST_VERSION}.{encoded}.{self._sign(encoded)}", session

    def decode(self, token: str) -> GuestSession:
        try:
            version, encoded, signature = token.split(".")
            if version != _GUEST_VERSION or not hmac.compare_digest(self._sign(encoded), signature):
                raise ValueError("bad signature")
            data = json.loads(base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)))
            session = GuestSession(str(data["gid"]), datetime.fromtimestamp(int(data["exp"]), UTC))
        except (ValueError, KeyError, TypeError) as exc:
            raise AuthenticationError("The guest session is invalid.") from exc
        if self._clock() >= session.expires_at:
            raise AuthenticationError("The guest session has expired. Start a new one.")
        return session

    def _sign(self, encoded: str) -> str:
        return hmac.new(self._key, encoded.encode(), hashlib.sha256).hexdigest()


# --------------------------------------------------------------------------- principals


@dataclass(frozen=True)
class ChatPrincipal:
    """Whoever is chatting: a verified customer, or a guest session."""

    customer: AuthContext | None = None
    guest: GuestSession | None = None

    @property
    def actor(self) -> str:
        if self.customer is not None:
            return self.customer.actor
        assert self.guest is not None
        return self.guest.actor


@dataclass(frozen=True)
class StaffPrincipal:
    ctx: StaffContext
    staff_ref: str
    full_name: str
    role: StaffRole  # read from the database for this request, not taken from the token


def _token(request: Request) -> str:
    header = request.headers.get("authorization", "")
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise ApiError(401, "authentication_required", "Sign in and send a bearer token.")
    return token.strip()


def _services(request: Request):
    return request.app.state.services


def current_customer(request: Request) -> AuthContext:
    token = _token(request)
    if not token.startswith("v1."):
        raise ApiError(403, "not_authorized", "This endpoint is for signed-in customers.")
    return _services(request).app.identity.signer.decode_token(token)


def current_chat_principal(request: Request) -> ChatPrincipal:
    token = _token(request)
    services = _services(request)
    if token.startswith(f"{_GUEST_VERSION}."):
        return ChatPrincipal(guest=services.guest_signer.decode(token))
    if token.startswith("v1."):
        return ChatPrincipal(customer=services.app.identity.signer.decode_token(token))
    raise ApiError(403, "not_authorized", "Staff sessions cannot use the customer chat.")


def current_staff(request: Request) -> StaffPrincipal:
    token = _token(request)
    if not token.startswith("s1."):
        raise ApiError(403, "not_authorized", "This endpoint is for staff members.")
    services = _services(request)
    ctx = services.app.staff_identity.signer.decode_token(token)
    with transaction(services.app.session_factory) as session:
        staff = session.get(StaffMember, ctx.staff_id)
        if staff is None or not staff.active:
            raise AuthenticationError("The staff session is invalid.")
        return StaffPrincipal(ctx, staff.staff_ref, staff.full_name, staff.role)


def require_staff_role(*roles: StaffRole) -> Callable[..., StaffPrincipal]:
    """Dependency factory: the caller's *current* database role must be one of ``roles``."""

    def check(
        request: Request, staff: Annotated[StaffPrincipal, Depends(current_staff)]
    ) -> StaffPrincipal:
        if staff.role in roles:
            return staff
        services = _services(request)
        with transaction(services.app.session_factory) as session:
            record_audit(
                session,
                actor=staff.ctx.actor,
                session_id=staff.ctx.session_id,
                action=f"api.{request.url.path.strip('/').replace('/', '.')}",
                outcome=AuditOutcome.DENIED,
                details={"role": staff.role.value, "required": [r.value for r in roles]},
            )
        raise ApiError(
            403,
            "not_authorized",
            "Your role does not allow this.",
            {"required_roles": [r.value for r in roles]},
        )

    return check

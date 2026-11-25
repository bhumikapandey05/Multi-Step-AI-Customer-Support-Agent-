"""Staff identity: who may approve or reject an actual refund.

This is deliberately separate from customer identity:

* Staff sign in with their own email + PIN (same PBKDF2 hashing and lockout as
  customers) and receive a :class:`StaffContext`, never an ``AuthContext``.
* Staff contexts are signed with a key derived from the app secret for this
  purpose only, so a customer session signature can never verify as a staff one.
* The signed role is only a hint. Every decision re-reads the staff member's
  role and ``active`` flag from the database.
"""

import base64
import binascii
import hashlib
import hmac
import json
import logging
import secrets
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from support_agent.clock import Clock, utcnow
from support_agent.crm.audit import record_audit
from support_agent.crm.errors import AuthenticationError
from support_agent.crm.security import PIN_PATTERN, dummy_pin_hash, verify_pin
from support_agent.db.engine import transaction
from support_agent.db.models import AuditOutcome, StaffMember, StaffRole
from support_agent.logging_config import mask_email

logger = logging.getLogger(__name__)

_TOKEN_VERSION = "s1"
_LOGIN_FAILED_MESSAGE = "The staff email or PIN is incorrect, or the account is locked."


@dataclass(frozen=True, slots=True)
class StaffContext:
    """A verified staff session. Construct via :class:`StaffIdentityService`."""

    staff_id: int
    role: StaffRole
    session_id: str
    issued_at: datetime
    expires_at: datetime
    signature: str = field(repr=False, compare=False)

    @property
    def actor(self) -> str:
        return f"staff:{self.staff_id}"


class StaffSessionSigner:
    def __init__(self, secret_key: str, *, ttl: timedelta, clock: Clock = utcnow) -> None:
        if len(secret_key) < 32:
            raise ValueError("secret_key must be at least 32 characters")
        # Domain separation: never the same key that signs customer sessions.
        self._key = hmac.new(
            secret_key.encode(), b"support-agent/staff-session/v1", "sha256"
        ).digest()
        self._ttl = ttl
        self._clock = clock

    def issue(self, staff_id: int, role: StaffRole) -> StaffContext:
        issued_at = self._clock().replace(microsecond=0)
        expires_at = issued_at + self._ttl
        session_id = secrets.token_hex(12)
        signature = self._sign(staff_id, role, session_id, issued_at, expires_at)
        return StaffContext(staff_id, role, session_id, issued_at, expires_at, signature)

    def verify(self, ctx: object) -> int:
        """Return the verified staff id, or raise :class:`AuthenticationError`."""
        if not isinstance(ctx, StaffContext):
            raise AuthenticationError("A verified staff session is required.")
        expected = self._sign(ctx.staff_id, ctx.role, ctx.session_id, ctx.issued_at, ctx.expires_at)
        if not hmac.compare_digest(expected, ctx.signature):
            raise AuthenticationError(
                "The staff session is invalid.", internal={"reason": "bad_signature"}
            )
        if self._clock() >= ctx.expires_at:
            raise AuthenticationError(
                "The staff session has expired. Please sign in again.",
                internal={"reason": "expired"},
            )
        return ctx.staff_id

    def encode_token(self, ctx: StaffContext) -> str:
        """Opaque bearer token for the API. Verified again on every request."""
        self.verify(ctx)
        body = {
            "stf": ctx.staff_id,
            "role": StaffRole(ctx.role).value,
            "sid": ctx.session_id,
            "iat": int(ctx.issued_at.timestamp()),
            "exp": int(ctx.expires_at.timestamp()),
        }
        payload = base64.urlsafe_b64encode(json.dumps(body).encode()).rstrip(b"=").decode()
        return f"{_TOKEN_VERSION}.{payload}.{ctx.signature}"

    def decode_token(self, token: str) -> StaffContext:
        try:
            version, payload, signature = token.split(".")
            if version != _TOKEN_VERSION:
                raise ValueError("unsupported token version")
            data = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
            ctx = StaffContext(
                staff_id=int(data["stf"]),
                role=StaffRole(data["role"]),
                session_id=str(data["sid"]),
                issued_at=datetime.fromtimestamp(int(data["iat"]), UTC),
                expires_at=datetime.fromtimestamp(int(data["exp"]), UTC),
                signature=signature,
            )
        except (ValueError, KeyError, TypeError, binascii.Error) as exc:
            raise AuthenticationError(
                "The staff session is invalid.", internal={"reason": "malformed_token"}
            ) from exc
        self.verify(ctx)
        return ctx

    def _sign(
        self,
        staff_id: int,
        role: StaffRole,
        session_id: str,
        issued_at: datetime,
        expires_at: datetime,
    ) -> str:
        body = {
            "kind": "staff",
            "sid": session_id,
            "stf": staff_id,
            "role": StaffRole(role).value,
            "iat": int(issued_at.timestamp()),
            "exp": int(expires_at.timestamp()),
        }
        payload = json.dumps(body, separators=(",", ":"), sort_keys=True).encode()
        return hmac.new(self._key, payload, hashlib.sha256).hexdigest()


class StaffIdentityService:
    """Verifies staff by email + PIN, with lockout after repeated failures."""

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        signer: StaffSessionSigner,
        *,
        clock: Clock = utcnow,
        max_failed_attempts: int = 5,
        lockout: timedelta = timedelta(minutes=15),
    ) -> None:
        self._session_factory = session_factory
        self._signer = signer
        self._clock = clock
        self._max_failed_attempts = max_failed_attempts
        self._lockout = lockout

    @property
    def signer(self) -> StaffSessionSigner:
        return self._signer

    def authenticate(self, email: str, pin: str) -> StaffContext:
        email_norm = email.strip().lower()
        now = self._clock()
        ctx: StaffContext | None = None

        with transaction(self._session_factory) as session:
            staff = session.scalar(select(StaffMember).where(StaffMember.email == email_norm))
            reason: str | None = None

            if staff is None:
                verify_pin(pin, dummy_pin_hash())
                reason = "unknown_email"
            elif staff.locked_until is not None and staff.locked_until > now:
                reason = "locked"
            elif not PIN_PATTERN.fullmatch(pin) or not verify_pin(pin, staff.pin_hash):
                reason = "wrong_pin"
                staff.failed_login_attempts += 1
                if staff.failed_login_attempts >= self._max_failed_attempts:
                    staff.locked_until = now + self._lockout
                    staff.failed_login_attempts = 0
                    reason = "wrong_pin_now_locked"
            elif not staff.active:
                reason = "inactive"
            else:
                staff.failed_login_attempts = 0
                staff.locked_until = None
                ctx = self._signer.issue(staff.id, staff.role)

            record_audit(
                session,
                occurred_at=now,
                actor=ctx.actor if ctx else "anonymous",
                session_id=ctx.session_id if ctx else None,
                action="staff.login",
                outcome=AuditOutcome.SUCCESS if ctx else AuditOutcome.DENIED,
                entity_type="staff",
                entity_id=staff.staff_ref if staff else None,
                details={"email": mask_email(email_norm), **({"reason": reason} if reason else {})},
            )

        if ctx is None:
            logger.warning(
                "Staff login failed", extra={"email": mask_email(email_norm), "reason": reason}
            )
            raise AuthenticationError(_LOGIN_FAILED_MESSAGE)
        return ctx

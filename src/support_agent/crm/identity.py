"""Customer identity: verified login and signed session contexts.

Trust model
-----------
The CRM never accepts a bare ``customer_id`` as proof of identity. A caller
must present an :class:`AuthContext`, which is only produced by
:meth:`IdentityService.authenticate` (email + PIN) or by decoding a session
token. Every context carries an HMAC signature over its fields, and the CRM
re-verifies that signature on every call, so an agent (or a prompt-injected
LLM) that edits ``customer_id`` produces a context that is rejected.
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
from support_agent.db.models import AuditOutcome, Customer
from support_agent.logging_config import mask_email

logger = logging.getLogger(__name__)

_TOKEN_VERSION = "v1"
_LOGIN_FAILED_MESSAGE = "The email or PIN is incorrect, or the account is temporarily locked."


@dataclass(frozen=True, slots=True)
class AuthContext:
    """A verified customer session. Construct via :class:`IdentityService`, never by hand."""

    customer_id: int
    session_id: str
    issued_at: datetime
    expires_at: datetime
    signature: str = field(repr=False, compare=False)

    @property
    def actor(self) -> str:
        return f"customer:{self.customer_id}"


def _b64encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _b64decode(data: str) -> bytes:
    return base64.urlsafe_b64decode(data + "=" * (-len(data) % 4))


class SessionSigner:
    """Signs and verifies :class:`AuthContext` objects with HMAC-SHA256."""

    def __init__(self, secret_key: str, *, ttl: timedelta, clock: Clock = utcnow) -> None:
        if len(secret_key) < 32:
            raise ValueError("secret_key must be at least 32 characters")
        self._key = secret_key.encode()
        self._ttl = ttl
        self._clock = clock

    def issue(self, customer_id: int) -> AuthContext:
        issued_at = self._clock().replace(microsecond=0)
        expires_at = issued_at + self._ttl
        session_id = secrets.token_hex(12)
        signature = self._sign(customer_id, session_id, issued_at, expires_at)
        return AuthContext(customer_id, session_id, issued_at, expires_at, signature)

    def verify(self, ctx: object) -> int:
        """Return the verified customer id, or raise :class:`AuthenticationError`."""
        if not isinstance(ctx, AuthContext):
            raise AuthenticationError("A verified customer session is required.")
        expected = self._sign(ctx.customer_id, ctx.session_id, ctx.issued_at, ctx.expires_at)
        if not hmac.compare_digest(expected, ctx.signature):
            raise AuthenticationError(
                "The customer session is invalid.", internal={"reason": "bad_signature"}
            )
        if self._clock() >= ctx.expires_at:
            raise AuthenticationError(
                "The customer session has expired. Please verify again.",
                internal={"reason": "expired"},
            )
        return ctx.customer_id

    def encode_token(self, ctx: AuthContext) -> str:
        """Serialize a context to an opaque token (for HTTP/UI layers in later phases)."""
        self.verify(ctx)
        payload = _b64encode(
            self._payload(ctx.customer_id, ctx.session_id, ctx.issued_at, ctx.expires_at)
        )
        return f"{_TOKEN_VERSION}.{payload}.{ctx.signature}"

    def decode_token(self, token: str) -> AuthContext:
        try:
            version, payload_b64, signature = token.split(".")
            if version != _TOKEN_VERSION:
                raise ValueError("unsupported token version")
            data = json.loads(_b64decode(payload_b64))
            ctx = AuthContext(
                customer_id=int(data["cid"]),
                session_id=str(data["sid"]),
                issued_at=datetime.fromtimestamp(int(data["iat"]), UTC),
                expires_at=datetime.fromtimestamp(int(data["exp"]), UTC),
                signature=signature,
            )
        except (ValueError, KeyError, TypeError, binascii.Error) as exc:
            raise AuthenticationError(
                "The customer session is invalid.", internal={"reason": "malformed_token"}
            ) from exc
        self.verify(ctx)
        return ctx

    @staticmethod
    def _payload(
        customer_id: int, session_id: str, issued_at: datetime, expires_at: datetime
    ) -> bytes:
        body = {
            "cid": customer_id,
            "sid": session_id,
            "iat": int(issued_at.timestamp()),
            "exp": int(expires_at.timestamp()),
        }
        return json.dumps(body, separators=(",", ":"), sort_keys=True).encode()

    def _sign(
        self, customer_id: int, session_id: str, issued_at: datetime, expires_at: datetime
    ) -> str:
        payload = self._payload(customer_id, session_id, issued_at, expires_at)
        return _b64encode(hmac.new(self._key, payload, hashlib.sha256).digest())


class IdentityService:
    """Verifies customers by email + PIN, with lockout after repeated failures."""

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        signer: SessionSigner,
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
    def signer(self) -> SessionSigner:
        return self._signer

    def authenticate(self, email: str, pin: str) -> AuthContext:
        email_norm = email.strip().lower()
        now = self._clock()
        ctx: AuthContext | None = None

        # Failure state (attempt counter, lockout, audit row) must be committed,
        # so the error is raised only after the transaction closes.
        with transaction(self._session_factory) as session:
            customer = session.scalar(select(Customer).where(Customer.email == email_norm))
            reason: str | None = None

            if customer is None:
                verify_pin(pin, dummy_pin_hash())
                reason = "unknown_email"
            elif customer.locked_until is not None and customer.locked_until > now:
                reason = "locked"
            elif not PIN_PATTERN.fullmatch(pin) or not verify_pin(pin, customer.pin_hash):
                reason = "wrong_pin"
                customer.failed_login_attempts += 1
                if customer.failed_login_attempts >= self._max_failed_attempts:
                    customer.locked_until = now + self._lockout
                    customer.failed_login_attempts = 0
                    reason = "wrong_pin_now_locked"
            else:
                customer.failed_login_attempts = 0
                customer.locked_until = None
                ctx = self._signer.issue(customer.id)

            record_audit(
                session,
                occurred_at=now,
                actor=ctx.actor if ctx else "anonymous",
                session_id=ctx.session_id if ctx else None,
                action="auth.login",
                outcome=AuditOutcome.SUCCESS if ctx else AuditOutcome.DENIED,
                entity_type="customer",
                entity_id=customer.customer_ref if customer else None,
                details={"email": mask_email(email_norm), **({"reason": reason} if reason else {})},
            )

        if ctx is None:
            logger.warning(
                "Login failed", extra={"email": mask_email(email_norm), "reason": reason}
            )
            raise AuthenticationError(_LOGIN_FAILED_MESSAGE)
        logger.info("Login succeeded", extra={"actor": ctx.actor, "session_id": ctx.session_id})
        return ctx

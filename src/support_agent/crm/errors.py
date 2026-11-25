"""Typed CRM errors.

Each error has a stable machine-readable ``code`` that the agent layer can map
to a response. ``details`` are safe to show to the caller; ``internal`` is
written to the audit log only (e.g. "this order belongs to someone else").
"""

from typing import Any


class CRMError(Exception):
    code = "crm_error"
    audit_outcome = "rejected"

    def __init__(
        self,
        message: str,
        *,
        details: dict[str, Any] | None = None,
        internal: dict[str, Any] | None = None,
        audit_outcome: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.internal = internal or {}
        if audit_outcome is not None:
            self.audit_outcome = audit_outcome

    def to_dict(self) -> dict[str, Any]:
        """Caller-safe representation (never includes ``internal``)."""
        return {"error": self.code, "message": self.message, "details": self.details}


class InvalidInputError(CRMError):
    code = "invalid_input"


class AuthenticationError(CRMError):
    code = "authentication_failed"
    audit_outcome = "denied"


class AuthorizationError(CRMError):
    code = "not_authorized"
    audit_outcome = "denied"


class NotFoundError(CRMError):
    code = "not_found"


class IdempotencyConflictError(CRMError):
    code = "idempotency_conflict"


class RefundNotEligibleError(CRMError):
    code = "refund_not_eligible"


class ConflictError(CRMError):
    code = "conflict"

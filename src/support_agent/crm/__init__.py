"""CRM domain: identity, validated schemas, refund policy, and the CRM tool service."""

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
from support_agent.crm.identity import AuthContext, IdentityService, SessionSigner
from support_agent.crm.policies import RefundPolicy
from support_agent.crm.service import CRMService

__all__ = [
    "AuthContext",
    "AuthenticationError",
    "AuthorizationError",
    "CRMError",
    "CRMService",
    "ConflictError",
    "IdempotencyConflictError",
    "IdentityService",
    "InvalidInputError",
    "NotFoundError",
    "RefundNotEligibleError",
    "RefundPolicy",
    "SessionSigner",
]

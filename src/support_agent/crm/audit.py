"""Audit log helpers."""

from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from support_agent.clock import utcnow
from support_agent.db.models import AuditLog, AuditOutcome


def record_audit(
    session: Session,
    *,
    actor: str,
    action: str,
    outcome: AuditOutcome,
    entity_type: str | None = None,
    entity_id: str | None = None,
    session_id: str | None = None,
    details: dict[str, Any] | None = None,
    occurred_at: datetime | None = None,
) -> AuditLog:
    """Add an audit row to ``session``. It commits (or rolls back) with the caller's transaction."""
    entry = AuditLog(
        occurred_at=occurred_at or utcnow(),
        actor=actor,
        session_id=session_id,
        action=action,
        outcome=outcome,
        entity_type=entity_type,
        entity_id=entity_id,
        details=details or {},
    )
    session.add(entry)
    return entry


def list_recent_audit_events(session: Session, *, limit: int = 20) -> list[AuditLog]:
    stmt = select(AuditLog).order_by(AuditLog.id.desc()).limit(limit)
    return list(session.scalars(stmt))

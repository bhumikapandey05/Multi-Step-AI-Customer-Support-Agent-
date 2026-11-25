"""Composition root: wires settings, database, identity, and CRM service together.

Later phases (FastAPI, Streamlit) build on this same object
instead of constructing dependencies themselves.
"""

import logging
from dataclasses import dataclass
from datetime import timedelta

from sqlalchemy import Engine, inspect
from sqlalchemy.orm import Session, sessionmaker

from support_agent.clock import Clock, utcnow
from support_agent.config import Settings
from support_agent.crm.identity import IdentityService, SessionSigner
from support_agent.crm.review import RefundReviewService
from support_agent.crm.service import CRMService
from support_agent.crm.staff import StaffIdentityService, StaffSessionSigner
from support_agent.db.engine import create_db_engine, create_session_factory

logger = logging.getLogger(__name__)


@dataclass
class Application:
    settings: Settings
    engine: Engine
    session_factory: sessionmaker[Session]
    identity: IdentityService
    crm: CRMService
    staff_identity: StaffIdentityService
    review: RefundReviewService

    def schema_exists(self) -> bool:
        return inspect(self.engine).has_table("customers")

    def close(self) -> None:
        self.engine.dispose()


def build_application(settings: Settings, *, clock: Clock = utcnow) -> Application:
    if settings.uses_dev_secret:
        logger.info(
            "Using the built-in development secret key; set "
            "SUPPORT_AGENT_AUTH_SECRET_KEY outside local development"
        )
    engine = create_db_engine(settings.database_url)
    session_factory = create_session_factory(engine)
    signer = SessionSigner(
        settings.auth_secret_key.get_secret_value(),
        ttl=timedelta(minutes=settings.session_ttl_minutes),
        clock=clock,
    )
    staff_signer = StaffSessionSigner(
        settings.auth_secret_key.get_secret_value(),
        ttl=timedelta(minutes=settings.staff_session_ttl_minutes),
        clock=clock,
    )
    return Application(
        settings=settings,
        engine=engine,
        session_factory=session_factory,
        identity=IdentityService(session_factory, signer, clock=clock),
        crm=CRMService(session_factory, signer, clock=clock),
        staff_identity=StaffIdentityService(session_factory, staff_signer, clock=clock),
        review=RefundReviewService(session_factory, staff_signer, clock=clock),
    )

"""Shared fixtures. Every test gets its own temporary SQLite file and a frozen clock."""

from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session, sessionmaker

from support_agent.crm.identity import AuthContext, IdentityService, SessionSigner
from support_agent.crm.policies import RefundPolicy
from support_agent.crm.service import CRMService
from support_agent.db.engine import create_db_engine, create_session_factory, init_db, transaction
from support_agent.db.models import AuditLog
from support_agent.db.seed import DEMO_PINS, seed_database

FIXED_NOW = datetime(2026, 6, 15, 12, 0, tzinfo=UTC)
TEST_SECRET = "test-secret-key-that-is-definitely-long-enough"
FAST_PIN_ITERATIONS = 1_000  # keeps tests fast; production uses 600k

ALICE = "alice.nguyen@example.com"
BRIAN = "brian.okafor@example.com"
CARMEN = "carmen.diaz@example.com"
DANIEL = "daniel.kim@example.com"
EMMA = "emma.rossi@example.com"
FARAH = "farah.haddad@example.com"


class FakeClock:
    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now

    def advance(self, delta: timedelta) -> None:
        self.now += delta


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock(FIXED_NOW)


@pytest.fixture
def engine(tmp_path) -> Iterator[Engine]:
    engine = create_db_engine(f"sqlite:///{tmp_path / 'test.db'}")
    init_db(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def session_factory(engine: Engine) -> sessionmaker[Session]:
    return create_session_factory(engine)


@pytest.fixture
def seeded(session_factory: sessionmaker[Session]) -> None:
    with transaction(session_factory) as session:
        seed_database(session, now=FIXED_NOW, pin_iterations=FAST_PIN_ITERATIONS)


@pytest.fixture
def signer(clock: FakeClock) -> SessionSigner:
    return SessionSigner(TEST_SECRET, ttl=timedelta(minutes=30), clock=clock)


@pytest.fixture
def identity(session_factory, signer, clock, seeded) -> IdentityService:
    return IdentityService(session_factory, signer, clock=clock)


@pytest.fixture
def crm(session_factory, signer, clock, seeded) -> CRMService:
    return CRMService(session_factory, signer, policy=RefundPolicy(), clock=clock)


@pytest.fixture
def login(identity: IdentityService) -> Callable[[str], AuthContext]:
    def _login(email: str) -> AuthContext:
        return identity.authenticate(email, DEMO_PINS[email])

    return _login


@pytest.fixture
def audit_rows(session_factory) -> Callable[..., list[AuditLog]]:
    def _rows(action: str | None = None) -> list[AuditLog]:
        with transaction(session_factory) as session:
            stmt = select(AuditLog).order_by(AuditLog.id)
            if action:
                stmt = stmt.where(AuditLog.action == action)
            return list(session.scalars(stmt))

    return _rows

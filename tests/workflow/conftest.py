"""Workflow fixtures: a real SQLite checkpoint file, simulated restarts, staff logins."""

from datetime import timedelta
from typing import Any

import pytest
from sqlalchemy import func, select

from support_agent.agent.agent import AgentConfig, DatabaseApprovalAuditor, SupportAgent
from support_agent.crm.review import RefundReviewService
from support_agent.crm.staff import StaffIdentityService, StaffSessionSigner
from support_agent.db.engine import transaction
from support_agent.db.seed import DEMO_STAFF_PINS
from support_agent.workflow.checkpoint import connect_checkpointer
from support_agent.workflow.retry import RetryPolicy
from support_agent.workflow.review import RefundReviewWorkflow
from tests.conftest import TEST_SECRET

APPROVER = "riley.approver@brightcart.example"
SUPERVISOR = "morgan.supervisor@brightcart.example"
AGENT = "sam.agent@brightcart.example"

REFUND = {"order_number": "ORD-10001", "reason": "defective", "details": "Left side crackles"}


class SimulatedCrash(BaseException):
    """Stands in for the process dying (kill -9, power loss). Not caught by `except Exception`."""


def count(session_factory, model, *where) -> int:
    with transaction(session_factory) as s:
        return s.scalar(select(func.count()).select_from(model).where(*where))


class Checkpoints:
    """A checkpoint file that can be closed and reopened, like a process restart."""

    def __init__(self, path) -> None:
        self.path = path
        self.saver = connect_checkpointer(path)

    def restart(self):
        self.saver.conn.close()
        self.saver = connect_checkpointer(self.path)
        return self.saver

    def close(self) -> None:
        self.saver.conn.close()


@pytest.fixture
def checkpoints(tmp_path):
    cp = Checkpoints(tmp_path / "checkpoints.db")
    yield cp
    cp.close()


@pytest.fixture
def sleeps() -> list[float]:
    return []


@pytest.fixture
def retry(sleeps) -> RetryPolicy:
    return RetryPolicy(max_retries=2, base_delay_seconds=0.01, sleep=sleeps.append)


@pytest.fixture
def staff_signer(clock) -> StaffSessionSigner:
    return StaffSessionSigner(TEST_SECRET, ttl=timedelta(minutes=60), clock=clock)


@pytest.fixture
def staff_login(session_factory, staff_signer, clock, seeded):
    identity = StaffIdentityService(session_factory, staff_signer, clock=clock)
    return lambda email: identity.authenticate(email, DEMO_STAFF_PINS[email])


@pytest.fixture
def review_service(session_factory, staff_signer, clock, seeded) -> RefundReviewService:
    return RefundReviewService(session_factory, staff_signer, clock=clock)


@pytest.fixture
def make_review(review_service, checkpoints, retry, clock):
    def _make(saver=None) -> RefundReviewWorkflow:
        return RefundReviewWorkflow(
            review_service, saver or checkpoints.saver, retry=retry, clock=clock
        )

    return _make


@pytest.fixture
def make_wf_agent(crm, login, session_factory, clock, checkpoints, retry, make_review):
    """An agent on the SQLite checkpointer. Each call signs in again, like a new process."""

    def _make(
        llm, email: str | None, *, thread_id: str | None = None, saver=None, **config: Any
    ) -> SupportAgent:
        saver = saver or checkpoints.saver
        review = make_review(saver)
        return SupportAgent(
            llm,
            crm,
            ctx=login(email) if email else None,
            config=AgentConfig(**config),
            auditor=DatabaseApprovalAuditor(session_factory, clock=clock),
            clock=clock,
            checkpointer=saver,
            thread_id=thread_id,
            retry=retry,
            on_refund_requested=review.open,
        )

    return _make

"""Isolated, reproducible environments for evaluation scenarios.

Each scenario gets a brand-new temporary directory with its own seeded CRM
database and LangGraph checkpoint file, a fixed clock, and an eval-only signing
key. Nothing is read from or written to the normal demo database: the sandbox
builds its settings explicitly (ignoring ``SUPPORT_AGENT_*`` and ``.env``) and
refuses to start if any of its files would coincide with a protected path.
"""

import shutil
import tempfile
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver
from sqlalchemy import func, select
from sqlalchemy.engine import make_url

from support_agent.agent.agent import SupportAgent
from support_agent.agent.factory import build_agent, build_review_workflow
from support_agent.agent.llm import LLMClient
from support_agent.app import Application, build_application
from support_agent.config import Settings
from support_agent.crm.identity import AuthContext
from support_agent.crm.staff import StaffContext
from support_agent.db.engine import init_db, transaction
from support_agent.db.models import (
    ApprovalDecision,
    AuditLog,
    Customer,
    Order,
    OrderItem,
    RefundDecision,
    RefundRequest,
    SupportTicket,
)
from support_agent.db.seed import DEMO_PINS, DEMO_STAFF_PINS, seed_database
from support_agent.workflow.checkpoint import connect_checkpointer
from support_agent.workflow.review import RefundReviewWorkflow

EVAL_NOW = datetime(2026, 6, 15, 12, 0, tzinfo=UTC)
EVAL_SECRET = "evaluation-only-secret-key-never-used-for-real-sessions"

TABLES = {
    "refund_requests": RefundRequest,
    "tickets": SupportTicket,
    "refund_decisions": RefundDecision,
    "approval_decisions": ApprovalDecision,
    "audit_logs": AuditLog,
}


class UnsafeEvaluationTarget(RuntimeError):
    """The evaluation would touch a protected (real) database file."""


class EvalClock:
    def __init__(self, now: datetime = EVAL_NOW) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now

    def advance(self, minutes: int) -> None:
        self.now += timedelta(minutes=minutes)


def sqlite_path(database_url: str) -> Path | None:
    url = make_url(database_url)
    if url.get_backend_name() != "sqlite" or url.database in (None, "", ":memory:"):
        return None
    return Path(url.database)


def protected_paths(settings: Settings) -> list[Path]:
    """The real database and checkpoint files an evaluation must never touch."""
    paths = [settings.checkpoint_path]
    if (db := sqlite_path(settings.database_url)) is not None:
        paths.append(db)
    return [p.resolve() for p in paths]


class EvalSandbox:
    def __init__(
        self,
        *,
        protected: Iterable[Path] = (),
        require_ticket_approval: bool = True,
        workdir: Path | None = None,
        keep: bool = False,
    ) -> None:
        self._protected = [Path(p).resolve() for p in protected]
        self._require_ticket_approval = require_ticket_approval
        self._workdir = workdir
        self._keep = keep
        self.clock = EvalClock()
        self.dir: Path | None = None
        self.app: Application | None = None
        self._savers: list[SqliteSaver] = []

    # ------------------------------------------------------------ lifecycle

    def __enter__(self) -> "EvalSandbox":
        self.dir = Path(tempfile.mkdtemp(prefix="support-eval-", dir=self._workdir)).resolve()
        db_path = self.dir / "eval-crm.db"
        checkpoints = self.dir / "eval-checkpoints.db"
        for path in (db_path, checkpoints):
            if path in self._protected or any(p.parent == path.parent for p in self._protected):
                shutil.rmtree(self.dir, ignore_errors=True)
                raise UnsafeEvaluationTarget(f"Refusing to evaluate next to protected data: {path}")
        self.settings = Settings(
            _env_file=None,
            environment="test",
            database_url=f"sqlite:///{db_path}",
            checkpoint_db_path=str(checkpoints),
            auth_secret_key=EVAL_SECRET,
            pin_hash_iterations=1_000,
            log_level="ERROR",
            gemini_api_key=None,
            workflow_retry_base_delay_seconds=0,
            agent_require_ticket_approval=self._require_ticket_approval,
        )
        self.app = build_application(self.settings, clock=self.clock)
        init_db(self.app.engine)
        with transaction(self.app.session_factory) as session:
            seed_database(session, now=self.clock(), pin_iterations=1_000)
        self.checkpointer = self._open_saver()
        self.baseline = self.counts()
        self.max_audit_id = self._max_audit_id()
        with transaction(self.app.session_factory) as s:
            self.baseline_refund_statuses = {
                r.id: r.status.value for r in s.scalars(select(RefundRequest))
            }
        return self

    def __exit__(self, *exc: object) -> None:
        for saver in self._savers:
            saver.conn.close()
        if self.app is not None:
            self.app.close()
        if self.dir is not None and not self._keep:
            shutil.rmtree(self.dir, ignore_errors=True)

    def _open_saver(self) -> SqliteSaver:
        saver = connect_checkpointer(self.settings.checkpoint_path)
        self._savers.append(saver)
        return saver

    def restart(self) -> None:
        """Simulate a process restart: drop the checkpoint connection and open a new one."""
        self.checkpointer.conn.close()
        self._savers.remove(self.checkpointer)
        self.checkpointer = self._open_saver()

    def extra_checkpointer(self) -> SqliteSaver:
        """A second connection, as a second server process would have."""
        return self._open_saver()

    # ------------------------------------------------------------ actors

    def login(self, email: str) -> AuthContext:
        return self.app.identity.authenticate(email, DEMO_PINS[email])

    def staff_login(self, email: str) -> StaffContext:
        return self.app.staff_identity.authenticate(email, DEMO_STAFF_PINS[email])

    def agent(
        self,
        llm: LLMClient,
        ctx: AuthContext | None,
        *,
        thread_id: str | None = None,
        guest_id: str | None = None,
        checkpointer: SqliteSaver | None = None,
    ) -> SupportAgent:
        return build_agent(
            self.app,
            ctx,
            llm=llm,
            checkpointer=checkpointer or self.checkpointer,
            thread_id=thread_id,
            guest_id=guest_id,
            clock=self.clock,
        )

    def review(self) -> RefundReviewWorkflow:
        return build_review_workflow(self.app, self.checkpointer)

    # ------------------------------------------------------------ observation

    def counts(self) -> dict[str, int]:
        with transaction(self.app.session_factory) as s:
            return {
                name: s.scalar(select(func.count()).select_from(model))
                for name, model in TABLES.items()
            }

    def _max_audit_id(self) -> int:
        with transaction(self.app.session_factory) as s:
            return s.scalar(select(func.max(AuditLog.id))) or 0

    def new_rows(self, model: Any) -> list[Any]:
        """Rows created during the scenario (seeded rows have the lowest ids)."""
        with transaction(self.app.session_factory) as s:
            return list(s.scalars(select(model).order_by(model.id)))[self.baseline[_name(model)] :]

    def new_rows_refunds_with_orders(self) -> list[dict[str, Any]]:
        with transaction(self.app.session_factory) as s:
            rows = s.execute(
                select(RefundRequest, Order.order_number)
                .join(Order, Order.id == RefundRequest.order_id)
                .where(RefundRequest.id.not_in(self.baseline_refund_statuses))
                .order_by(RefundRequest.id)
            ).all()
            return [
                {
                    "refund_number": r.refund_number,
                    "order_number": number,
                    "status": r.status.value,
                    "requires_manual_review": r.requires_manual_review,
                    "amount_cents": r.amount_cents,
                }
                for r, number in rows
            ]

    def new_audit_rows(self) -> list[AuditLog]:
        with transaction(self.app.session_factory) as s:
            return list(
                s.scalars(
                    select(AuditLog).where(AuditLog.id > self.max_audit_id).order_by(AuditLog.id)
                )
            )

    def customer_id(self, email: str) -> int:
        with transaction(self.app.session_factory) as s:
            return s.scalar(select(Customer.id).where(Customer.email == email))

    def canaries(self) -> dict[int, dict[str, set[str]]]:
        """Identifiers that belong to exactly one customer, used to detect data leaks.

        ``ids`` (order/ticket/refund numbers) are only matched in data returned by
        tools, since a customer may type someone else's order number themselves.
        ``private`` (names, emails, product names) are matched in replies too.
        """
        out: dict[int, dict[str, set[str]]] = {}
        with transaction(self.app.session_factory) as s:
            for c in s.scalars(select(Customer)):
                out[c.id] = {"ids": set(), "private": {c.email, c.full_name}}
            for o in s.scalars(select(Order)):
                out[o.customer_id]["ids"].add(o.order_number)
            for item, owner in s.execute(
                select(OrderItem.product_name, Order.customer_id).join(Order)
            ):
                out[owner]["private"].add(item)
            for number, owner in s.execute(
                select(SupportTicket.ticket_number, SupportTicket.customer_id)
            ):
                out[owner]["ids"].add(number)
            for number, owner in s.execute(
                select(RefundRequest.refund_number, RefundRequest.customer_id)
            ):
                out[owner]["ids"].add(number)
        return out


def _name(model: Any) -> str:
    return next(name for name, m in TABLES.items() if m is model)

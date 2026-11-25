"""Runs scenarios against the real agent stack inside an :class:`EvalSandbox`.

The agent under test is built with the same factory the CLI and API use
(``build_agent``): real CRM, real LangGraph workflows, real checkpoints, real
approval auditing. Only the model is swapped (scripted offline, Gemini live).
"""

import json
import logging
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from sqlalchemy.exc import OperationalError

from support_agent.agent.agent import AgentError, AgentReply, SupportAgent
from support_agent.agent.llm import LLMClient
from support_agent.evals.checks import Check, evaluate
from support_agent.evals.llm import InstrumentedLLM, LLMCall, MockLLM
from support_agent.evals.sandbox import EvalSandbox
from support_agent.evals.scenario import Fault, Scenario
from support_agent.workflow.review import ReviewWorkflowError

logger = logging.getLogger(__name__)

Mode = Literal["offline", "live"]


class SimulatedCrash(BaseException):
    """Stands in for the process dying (not caught by ``except Exception``)."""


@dataclass
class ReplyRecord:
    text: str
    stop_reason: str
    executions: list[dict[str, Any]]


@dataclass
class TurnRecord:
    index: int
    actions: list[str]
    replies: list[ReplyRecord] = field(default_factory=list)
    latency_ms: float = 0.0
    crashed: bool = False
    input_rejected: bool = False
    race: list[str] = field(default_factory=list)
    staff: dict[str, Any] | None = None
    error: str | None = None


@dataclass
class ScenarioResult:
    id: str
    title: str
    category: str
    set: str
    mode: str
    passed: bool
    scored: bool
    checks: list[Check]
    turns: list[TurnRecord]
    llm_calls: list[LLMCall]
    duplicates: int = 0
    prevented_duplicates: int = 0
    skipped_reason: str | None = None
    framework_error: str | None = None

    @property
    def violations(self) -> list[Check]:
        return [c for c in self.checks if c.kind == "authorization" and not c.passed]

    @property
    def failed_checks(self) -> list[Check]:
        return [c for c in self.checks if not c.passed]

    @property
    def tool_selection_correct(self) -> bool | None:
        relevant = [c for c in self.checks if c.name in ("tools_called", "tools_not_called")]
        return all(c.passed for c in relevant) if relevant else None


class _Faults:
    """Arms a one-turn fault on a CRM write method of this sandbox's CRM instance."""

    def __init__(self, crm: Any) -> None:
        self._crm = crm
        self._armed: str | None = None

    def arm(self, fault: Fault) -> None:
        name = fault.target  # a CRMService method name
        original = getattr(self._crm, name)
        remaining = [fault.count]

        def faulty(*args: Any, **kwargs: Any) -> Any:
            if remaining[0] <= 0:
                return original(*args, **kwargs)
            remaining[0] -= 1
            if fault.kind == "transient_errors":
                raise OperationalError("eval fault", {}, Exception("database is locked"))
            original(*args, **kwargs)  # committed...
            raise SimulatedCrash  # ...then the process "dies" before the checkpoint

        setattr(self._crm, name, faulty)
        self._armed = name

    def disarm(self) -> None:
        if self._armed:
            delattr(self._crm, self._armed)  # back to the class method
            self._armed = None


def _record(reply: AgentReply) -> ReplyRecord:
    return ReplyRecord(
        text=reply.text,
        stop_reason=reply.stop_reason.value,
        executions=[
            {
                "tool": e.tool,
                "status": e.status.value,
                "approved": e.approved,
                "ok": e.result.get("ok"),
                "error": e.result.get("error"),
                "result_json": json.dumps(e.result, default=str, sort_keys=True),
            }
            for e in reply.tool_executions
        ],
    )


class _Run:
    def __init__(self, scenario: Scenario, sandbox: EvalSandbox, llm: InstrumentedLLM) -> None:
        self.s = scenario
        self.sb = sandbox
        self.llm = llm
        self.guest_id = None if scenario.customer else f"eval-{scenario.id}"
        self.agent = self._new_agent()
        self.faults = _Faults(sandbox.app.crm)

    def _new_agent(self, thread_id: str | None = None, checkpointer: Any = None) -> SupportAgent:
        ctx = self.sb.login(self.s.customer) if self.s.customer else None
        return self.sb.agent(
            self.llm, ctx, thread_id=thread_id, guest_id=self.guest_id, checkpointer=checkpointer
        )

    def turn(self, index: int) -> TurnRecord:
        t = self.s.turns[index]
        rec = TurnRecord(index, [k for k, v in t.model_dump(exclude_defaults=True).items() if v])
        started = time.perf_counter()
        try:
            if t.advance_minutes:
                self.sb.clock.advance(t.advance_minutes)
            if t.fault:
                self.faults.arm(t.fault)
            if t.restart:
                self.sb.restart()
                self.agent = self._new_agent(thread_id=self.agent.thread_id)
            reply: AgentReply | None = None
            if t.resume:
                reply = self.agent.resume()
                if reply is not None:
                    rec.replies.append(_record(reply))
            if t.say is not None or t.say_chars:
                try:
                    reply = self.agent.send(t.say if t.say is not None else "x" * t.say_chars)
                    rec.replies.append(_record(reply))
                except AgentError as exc:
                    rec.input_rejected, rec.error = True, str(exc)
            if t.race_approve:
                self._race(rec)
            else:
                self._confirm(t.confirm, rec)
            if t.staff:
                rec.staff = self._staff(t.staff)
        except SimulatedCrash:
            rec.crashed = True
        finally:
            self.faults.disarm()
            rec.latency_ms = round((time.perf_counter() - started) * 1000, 3)
        return rec

    def _confirm(self, answer: str, rec: TurnRecord) -> None:
        """Answer every pending confirmation the way the simulated customer would."""
        while answer != "leave" and (pending := self.agent.pending_approvals):
            action = pending[0].action_id
            reply = self.agent.approve(action) if answer == "approve" else self.agent.reject(action)
            rec.replies.append(_record(reply))

    def _race(self, rec: TurnRecord) -> None:
        pending = self.agent.pending_approvals
        if not pending:
            rec.race.append("nothing pending")
            return
        action = pending[0].action_id
        other = self._new_agent(self.agent.thread_id, checkpointer=self.sb.extra_checkpointer())
        barrier = threading.Barrier(2, timeout=10)
        for agent in (self.agent, other):
            auditor = agent._auditor
            original = auditor._record

            def synchronized(*a: Any, _orig: Callable = original, **k: Any) -> Any:
                barrier.wait()
                return _orig(*a, **k)

            auditor._record = synchronized
        lock = threading.Lock()

        def click(agent: SupportAgent) -> None:
            try:
                reply = agent.approve(action)
                outcome = f"ok:{reply.stop_reason.value}"
                with lock:
                    rec.replies.append(_record(reply))
            except AgentError as exc:
                outcome = f"refused:{type(exc).__name__}"
            except Exception as exc:
                outcome = f"error:{type(exc).__name__}"
            with lock:
                rec.race.append(outcome)

        threads = [threading.Thread(target=click, args=(a,)) for a in (self.agent, other)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

    def _staff(self, step: Any) -> dict[str, Any]:
        refunds = [
            r for r in self.sb.new_rows_refunds_with_orders() if r["order_number"] == step.order
        ]
        if not refunds:
            return {"refused": True, "error": "no_refund_created", "refund": None}
        number = refunds[-1]["refund_number"]
        try:
            status = self.sb.review().decide(
                self.sb.staff_login(step.email), number, step.decision, step.note
            )
        except ReviewWorkflowError as exc:
            return {
                "refund": number,
                "refused": True,
                "error": "workflow_state",
                "message": str(exc),
            }
        attempt = status.last_attempt or {}
        return {
            "refund": number,
            "refused": status.result is None,
            "error": None if status.result else attempt.get("error"),
            "refund_status": status.case.get("status"),
        }


def run_scenario(
    scenario: Scenario,
    *,
    mode: Mode,
    set_name: str,
    live_llm: LLMClient | None = None,
    protected: Iterable[Path] = (),
    keep: bool = False,
) -> ScenarioResult:
    if mode not in scenario.modes:
        return ScenarioResult(
            id=scenario.id,
            title=scenario.title,
            category=scenario.category,
            set=set_name,
            mode=mode,
            passed=False,
            scored=False,
            checks=[],
            turns=[],
            llm_calls=[],
            skipped_reason=f"scenario only runs in: {', '.join(scenario.modes)}",
        )
    mock = MockLLM(scenario.mock_steps) if mode == "offline" else None
    llm = InstrumentedLLM(mock if mock is not None else live_llm)  # type: ignore[arg-type]
    turns: list[TurnRecord] = []
    with EvalSandbox(
        protected=protected, require_ticket_approval=scenario.require_ticket_approval, keep=keep
    ) as sb:
        framework_error = None
        try:
            run = _Run(scenario, sb, llm)
            for i in range(len(scenario.turns)):
                turns.append(run.turn(i))
            pending_at_end = bool(run.agent.pending_approvals)
        except Exception as exc:
            logger.exception("Scenario crashed", extra={"scenario": scenario.id})
            framework_error = f"{type(exc).__name__}: {exc}"
            pending_at_end = False
        checks, duplicates, prevented = evaluate(
            scenario, sb, turns, llm, mock, pending_at_end=pending_at_end, mode=mode
        )
    infra = [c.error for c in llm.calls if c.error in _INFRA_ERRORS]
    scored = not (mode == "live" and infra)
    passed = scored and framework_error is None and all(c.passed for c in checks)
    return ScenarioResult(
        id=scenario.id,
        title=scenario.title,
        category=scenario.category,
        set=set_name,
        mode=mode,
        passed=passed,
        scored=scored,
        checks=checks,
        turns=turns,
        llm_calls=list(llm.calls),
        duplicates=duplicates,
        prevented_duplicates=prevented,
        skipped_reason=f"live model unavailable ({infra[0]})" if not scored else None,
        framework_error=framework_error,
    )


_INFRA_ERRORS = {"BudgetExhausted", "LLMUnavailableError", "LLMConfigurationError", "LLMError"}

"""Staff approval of an actual refund, as a checkpointed LangGraph workflow.

::

    START -> load_case --(not pending)--------------------------> END
                 |
                 v
           await_decision  <-- interrupt(): waits for an authorized staff member
                 |   ^         (persisted in SQLite; survives restarts)
                 v   | denied / not eligible / invalid -> stays pending
           apply_decision ---------------------------------------> END

One thread per refund request (``refund-review:RFD-XXXXXXXX``). The thread is
opened automatically when a customer's confirmed refund request is created, or
on demand by staff for requests filed another way.

Who decides is never part of the workflow input. The resume payload carries
only ``approve``/``reject`` and a note. The staff identity is the signed
:class:`StaffContext` held by this process for the current call, and
:class:`RefundReviewService` re-checks role, limits, and eligibility from the
database. The decision uses an idempotency key fixed per thread, so a retried
or resumed ``apply_decision`` replays the recorded decision instead of writing
a second one.
"""

import hashlib
import logging
import operator
from dataclasses import dataclass
from typing import Annotated, Any, Literal, TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from support_agent.clock import Clock, utcnow
from support_agent.crm.errors import ConflictError, CRMError
from support_agent.crm.review import RefundReviewService
from support_agent.crm.staff import StaffContext
from support_agent.workflow.checkpoint import list_thread_ids
from support_agent.workflow.retry import RetryPolicy

logger = logging.getLogger(__name__)

THREAD_PREFIX = "refund-review:"

LOAD_CASE = "load_case"
AWAIT_DECISION = "await_decision"
APPLY_DECISION = "apply_decision"

ReviewStage = Literal["awaiting_staff_decision", "deciding", "decided", "closed"]


class ReviewState(TypedDict, total=False):
    refund_number: str
    case: dict[str, Any]
    stage: ReviewStage
    decision_request: dict[str, Any] | None
    attempts: Annotated[list[dict[str, Any]], operator.add]  # refused attempts, for the record
    result: dict[str, Any] | None
    route: str


class ReviewWorkflowError(Exception):
    """The requested review operation is not possible in the workflow's current state."""


@dataclass(frozen=True)
class ReviewStatus:
    thread_id: str
    refund_number: str
    stage: str
    waiting_for_staff: bool
    needs_resume: bool
    case: dict[str, Any]
    result: dict[str, Any] | None
    attempts: list[dict[str, Any]]

    @property
    def last_attempt(self) -> dict[str, Any] | None:
        return self.attempts[-1] if self.attempts else None

    def as_row(self) -> dict[str, Any]:
        """Flat summary used by the CLI and the API."""
        if self.waiting_for_staff:
            state = "awaiting_staff_decision"
        elif self.needs_resume:
            state = "needs_resume"
        else:
            state = self.stage
        return {
            "thread_id": self.thread_id,
            "kind": "refund_review",
            "refund_number": self.refund_number,
            "status": state,
            "amount_cents": self.case.get("amount_cents"),
            "required_role": self.case.get("required_role"),
            "refund_status": self.case.get("status"),
            "decision": self.result,
            "refused_attempts": self.attempts,
        }


def review_thread_id(refund_number: str) -> str:
    return f"{THREAD_PREFIX}{refund_number.strip().upper()}"


class RefundReviewWorkflow:
    def __init__(
        self,
        review: RefundReviewService,
        checkpointer: BaseCheckpointSaver,
        *,
        retry: RetryPolicy | None = None,
        clock: Clock = utcnow,
    ) -> None:
        self._review = review
        self._checkpointer = checkpointer
        self._retry = retry or RetryPolicy()
        self._clock = clock
        self._staff: StaffContext | None = None  # set only for the duration of one call

        graph = StateGraph(ReviewState)
        graph.add_node(LOAD_CASE, self._load_case)
        graph.add_node(AWAIT_DECISION, self._await_decision)
        graph.add_node(APPLY_DECISION, self._apply_decision)
        graph.add_edge(START, LOAD_CASE)
        route = lambda s: s.get("route", END)  # noqa: E731
        graph.add_conditional_edges(LOAD_CASE, route, [AWAIT_DECISION, END])
        graph.add_conditional_edges(AWAIT_DECISION, route, [AWAIT_DECISION, APPLY_DECISION])
        graph.add_conditional_edges(APPLY_DECISION, route, [AWAIT_DECISION, END])
        self._graph = graph.compile(checkpointer=checkpointer)

    # ------------------------------------------------------------ public API

    def open(self, refund_number: str) -> ReviewStatus:
        """Start the review thread if it does not exist yet. Idempotent."""
        thread_id = review_thread_id(refund_number)
        if not self._snapshot(thread_id).values:
            self._graph.invoke(
                {"refund_number": refund_number.strip().upper()}, self._cfg(thread_id)
            )
            logger.info("Refund review opened", extra={"thread_id": thread_id})
        return self.status(refund_number)

    def status(self, refund_number: str) -> ReviewStatus:
        thread_id = review_thread_id(refund_number)
        snapshot = self._snapshot(thread_id)
        if not snapshot.values:
            raise ReviewWorkflowError(f"No review workflow exists for {refund_number}.")
        values = snapshot.values
        return ReviewStatus(
            thread_id=thread_id,
            refund_number=values["refund_number"],
            stage=values.get("stage", "closed"),
            waiting_for_staff=bool(snapshot.interrupts),
            needs_resume=bool(snapshot.next) and not snapshot.interrupts,
            case=values.get("case", {}),
            result=values.get("result"),
            attempts=list(values.get("attempts", [])),
        )

    def decide(
        self,
        staff: StaffContext,
        refund_number: str,
        decision: Literal["approve", "reject"],
        note: str | None = None,
    ) -> ReviewStatus:
        """Resume the paused review with a staff decision."""
        if decision not in ("approve", "reject"):
            raise ReviewWorkflowError("The decision must be 'approve' or 'reject'.")
        current = self.open(refund_number)
        if current.needs_resume:
            raise ReviewWorkflowError(
                f"A decision on {current.refund_number} is still being applied. "
                "Run `review resume` to finish it."
            )
        if not current.waiting_for_staff:
            raise ReviewWorkflowError(
                f"{current.refund_number} is not waiting for a decision (stage: {current.stage})."
            )
        return self._run(
            staff, current.thread_id, Command(resume={"decision": decision, "note": note})
        )

    def resume(self, staff: StaffContext, refund_number: str) -> ReviewStatus:
        """Finish a decision that was cut short (crash or persistent database outage)."""
        current = self.status(refund_number)
        if not current.needs_resume:
            raise ReviewWorkflowError(f"Nothing to resume for {current.refund_number}.")
        return self._run(staff, current.thread_id, None)

    def list_threads(self) -> list[ReviewStatus]:
        return [
            self.status(t.removeprefix(THREAD_PREFIX))
            for t in list_thread_ids(self._checkpointer, prefix=THREAD_PREFIX)
        ]

    # ------------------------------------------------------------ nodes

    def _load_case(self, state: ReviewState) -> dict[str, Any]:
        case = self._retry.run(
            "review:load_case", lambda: self._review.system_case(state["refund_number"])
        )
        data = case.model_dump(mode="json")
        if data["status"] != "pending_review":
            return {"case": data, "stage": "closed", "route": END}
        return {"case": data, "stage": "awaiting_staff_decision", "route": AWAIT_DECISION}

    def _await_decision(self, state: ReviewState) -> dict[str, Any]:
        # Nothing may happen before interrupt(): this node re-runs from the top on resume.
        case = state["case"]
        payload = interrupt(
            {
                "kind": "staff_refund_approval",
                "refund_number": state["refund_number"],
                "amount_cents": case["amount_cents"],
                "required_role": case["required_role"],
                "approval_blockers": case["approval_blockers"],
            }
        )
        staff = self._staff
        decision = payload.get("decision") if isinstance(payload, dict) else None
        if staff is None or decision not in ("approve", "reject"):
            logger.warning("Ignored an invalid staff decision", extra={"decision": decision})
            return {"route": AWAIT_DECISION}
        return {
            "stage": "deciding",
            "decision_request": {
                "decision": "approved" if decision == "approve" else "rejected",
                "note": payload.get("note"),
                "staff_id": staff.staff_id,
                "actor": staff.actor,
                "requested_at": self._clock().isoformat(),
            },
            "route": APPLY_DECISION,
        }

    def _apply_decision(self, state: ReviewState) -> dict[str, Any]:
        request = state["decision_request"]
        assert request is not None
        staff = self._staff
        if staff is None or staff.staff_id != request["staff_id"]:
            # Only the staff member who made the decision may complete it.
            raise ReviewWorkflowError(
                "Only the staff member who made this decision can resume applying it."
            )
        thread_id = review_thread_id(state["refund_number"])
        payload = {
            "refund_number": state["refund_number"],
            "decision": request["decision"],
            "note": request["note"],
            "idempotency_key": self._idempotency_key(thread_id),
        }
        try:
            # TransientFailure propagates: the thread stays at this node and resume()
            # repeats the call with the same idempotency key.
            result = self._retry.run(
                "review:decide",
                lambda: self._review.decide(staff, payload, workflow_thread_id=thread_id),
            )
        except CRMError as exc:
            attempt = {
                "actor": request["actor"],
                "decision": request["decision"],
                "error": exc.code,
                "message": exc.message,
                "at": self._clock().isoformat(),
            }
            if isinstance(exc, ConflictError):
                case = self._review.system_case(state["refund_number"]).model_dump(mode="json")
                return {"attempts": [attempt], "case": case, "stage": "closed", "route": END}
            return {
                "attempts": [attempt],
                "decision_request": None,
                "stage": "awaiting_staff_decision",
                "route": AWAIT_DECISION,
            }
        case = self._review.system_case(state["refund_number"]).model_dump(mode="json")
        return {
            "result": result.model_dump(mode="json"),
            "case": case,
            "stage": "decided",
            "route": END,
        }

    # ------------------------------------------------------------ helpers

    def _run(self, staff: StaffContext, thread_id: str, payload: Any) -> ReviewStatus:
        self._staff = staff
        try:
            self._graph.invoke(payload, self._cfg(thread_id))
        finally:
            self._staff = None
        return self.status(thread_id.removeprefix(THREAD_PREFIX))

    @staticmethod
    def _idempotency_key(thread_id: str) -> str:
        return f"review-{hashlib.sha256(thread_id.encode()).hexdigest()[:40]}"

    @staticmethod
    def _cfg(thread_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": thread_id}}

    def _snapshot(self, thread_id: str) -> Any:
        return self._graph.get_state(self._cfg(thread_id))

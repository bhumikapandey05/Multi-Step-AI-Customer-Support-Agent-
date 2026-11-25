"""The tool-calling agent, run as a checkpointed LangGraph workflow.

One customer turn works like this:

    user message -> LLM -> (tool calls -> Python executes -> results -> LLM)* -> reply

Python stays in control throughout:

* The LLM only *proposes* calls. Unknown tools, invalid arguments, and calls
  needing a login are turned into error results that the model can recover from.
* The verified :class:`AuthContext` is bound to the agent, never stored in the
  workflow state, and the CRM checks it on every call.
* Write tools pause the workflow at an ``interrupt()`` and return an
  :class:`AgentReply` with ``pending_approvals``. Only the application (the
  human at the CLI) can call :meth:`SupportAgent.approve` or
  :meth:`SupportAgent.reject`. The model has no tool that grants approval.
* LLM rounds and tool calls per turn are capped, transient CRM failures are
  retried a bounded number of times, and every failure path ends in a safe,
  fixed message.

Phase 3: the loop's state lives in a LangGraph checkpointer (SQLite in the
CLI), keyed by ``thread_id``. A new :class:`SupportAgent` with the same thread
id, created after a restart by the same, freshly signed-in customer, picks up
the conversation and any pending confirmation exactly where it stopped. The
graph topology is in :mod:`support_agent.workflow.conversation`; the node logic
is below.
"""

import copy
import hashlib
import json
import logging
import secrets
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any, Protocol

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END
from langgraph.types import Command, interrupt
from pydantic import BaseModel, ValidationError
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from support_agent.agent.llm import (
    LLMClient,
    LLMError,
    Message,
    ToolCall,
    message_from_dict,
    message_to_dict,
    tool_call_from_dict,
    tool_call_to_dict,
)
from support_agent.agent.prompts import system_prompt
from support_agent.agent.tools import AgentTool, default_tools
from support_agent.clock import Clock, utcnow
from support_agent.config import Settings
from support_agent.crm.audit import record_audit
from support_agent.crm.errors import AuthenticationError, CRMError
from support_agent.crm.identity import AuthContext
from support_agent.crm.service import CRMService
from support_agent.db.engine import transaction
from support_agent.db.models import ApprovalDecision, AuditOutcome
from support_agent.workflow.conversation import (
    AWAIT_CONFIRMATION,
    CALL_MODEL,
    DISPATCH_TOOLS,
    EXECUTE_ACTION,
    EXPIRE_SESSION,
    THREAD_PREFIX,
    ConversationState,
    build_conversation_graph,
)
from support_agent.workflow.retry import RetryPolicy, TransientFailure

logger = logging.getLogger(__name__)

LLM_ERROR_TEXT = (
    "Sorry, I'm having trouble reaching our assistant service right now. Please try again in "
    "a moment. Nothing on your account was changed."
)
ITERATION_LIMIT_TEXT = (
    "Sorry, I couldn't finish that request. Could you rephrase it or break it into smaller "
    "steps? I can also open a support ticket so a team member can help."
)
EMPTY_REPLY_TEXT = "Sorry, I didn't catch that. Could you tell me a bit more about what you need?"
SESSION_EXPIRED_TEXT = (
    "Your verified session has expired, so I can't access your account anymore. Please sign "
    "in again to continue."
)
RETRY_LATER_TEXT = (
    "Our systems are busy, so I couldn't finish the action you confirmed. It is saved and has "
    "not been done twice. It will be completed when this conversation is resumed."
)

_RECURSION_LIMIT = 200


class StopReason(StrEnum):
    COMPLETED = "completed"
    AWAITING_APPROVAL = "awaiting_approval"
    ITERATION_LIMIT = "iteration_limit"
    LLM_ERROR = "llm_error"
    SESSION_EXPIRED = "session_expired"
    RETRY_LATER = "retry_later"  # a confirmed action is saved but could not finish yet


class ExecutionStatus(StrEnum):
    SUCCESS = "success"
    ERROR = "error"
    AWAITING_APPROVAL = "awaiting_approval"
    DECLINED = "declined"
    CACHED = "cached"  # identical call already answered; served without re-running


class AgentError(Exception):
    """Misuse of the agent API by the application (not shown to the LLM)."""


class ThreadOwnershipError(AgentError):
    """The conversation thread belongs to a different customer (or a guest)."""


class ApprovalConflictError(AgentError):
    """Another session already recorded a different decision for this action."""


class ApprovalAlreadyDecided(Exception):
    """Raised by an auditor when a decision for the action is already recorded."""

    def __init__(self, decision: str) -> None:
        super().__init__(f"already {decision}")
        self.decision = decision


@dataclass(frozen=True)
class AgentConfig:
    max_llm_rounds: int = 6
    max_tool_calls_per_turn: int = 8
    approval_ttl: timedelta = timedelta(minutes=10)
    require_ticket_approval: bool = True
    max_user_message_chars: int = 4000

    @classmethod
    def from_settings(cls, settings: Settings) -> "AgentConfig":
        return cls(
            max_llm_rounds=settings.agent_max_llm_rounds,
            max_tool_calls_per_turn=settings.agent_max_tool_calls_per_turn,
            approval_ttl=timedelta(minutes=settings.agent_approval_ttl_minutes),
            require_ticket_approval=settings.agent_require_ticket_approval,
        )


@dataclass(frozen=True)
class ToolExecution:
    tool: str
    arguments: dict[str, Any]
    status: ExecutionStatus
    result: dict[str, Any]
    approved: bool = False


@dataclass(frozen=True)
class PendingApproval:
    action_id: str
    tool: str
    summary: str
    details: dict[str, Any]
    expires_at: datetime
    thread_id: str | None = None


@dataclass(frozen=True)
class AgentReply:
    text: str
    stop_reason: StopReason
    tool_executions: list[ToolExecution] = field(default_factory=list)
    pending_approvals: list[PendingApproval] = field(default_factory=list)


class ApprovalAuditor(Protocol):
    def record(self, ctx: AuthContext, approval: PendingApproval, decision: str) -> None: ...


class DatabaseApprovalAuditor:
    """Writes human approval decisions (and workflow security events) to the CRM audit log."""

    def __init__(self, session_factory: sessionmaker[Session], *, clock: Clock = utcnow) -> None:
        self._session_factory = session_factory
        self._clock = clock

    def record(self, ctx: AuthContext, approval: PendingApproval, decision: str) -> None:
        """Atomically claim the decision for this action, then audit it in the same transaction.

        Raises :class:`ApprovalAlreadyDecided` if any decision was recorded first.
        """
        try:
            self._record(ctx, approval, decision)
        except IntegrityError:
            with transaction(self._session_factory) as session:
                existing = session.scalar(
                    select(ApprovalDecision.decision).where(
                        ApprovalDecision.action_id == approval.action_id
                    )
                )
            if existing is None:
                raise
            raise ApprovalAlreadyDecided(existing) from None

    def _record(self, ctx: AuthContext, approval: PendingApproval, decision: str) -> None:
        with transaction(self._session_factory) as session:
            session.add(
                ApprovalDecision(
                    action_id=approval.action_id,
                    thread_id=approval.thread_id or "",
                    tool=approval.tool,
                    decision=decision,
                    actor=ctx.actor,
                    session_id=ctx.session_id,
                    decided_at=self._clock(),
                )
            )
            session.flush()  # the unique index on action_id decides the race here
            record_audit(
                session,
                occurred_at=self._clock(),
                actor=ctx.actor,
                session_id=ctx.session_id,
                action=f"agent.approval.{decision}",
                outcome=AuditOutcome.SUCCESS if decision == "granted" else AuditOutcome.REJECTED,
                entity_type="tool",
                entity_id=approval.tool,
                details={
                    "action_id": approval.action_id,
                    "thread_id": approval.thread_id,
                    "summary": approval.summary,
                    "arguments": approval.details,
                },
            )

    def record_event(
        self, *, actor: str, session_id: str | None, action: str, outcome: AuditOutcome, **details
    ) -> None:
        with transaction(self._session_factory) as session:
            record_audit(
                session,
                occurred_at=self._clock(),
                actor=actor,
                session_id=session_id,
                action=action,
                outcome=outcome,
                entity_type="workflow",
                entity_id=details.get("thread_id"),
                details=details,
            )


# --------------------------------------------------------------------------- helpers


def _error(code: str, message: str, **details: Any) -> dict[str, Any]:
    return {"ok": False, "error": code, "message": message, "details": details}


def _serialize(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, list | tuple):
        return [_serialize(v) for v in value]
    return value


def new_thread_id() -> str:
    return f"{THREAD_PREFIX}{secrets.token_hex(8)}"


def _new_turn() -> dict[str, Any]:
    return {"llm_rounds": 0, "tool_calls": 0, "read_cache": {}}


def _assistant(text: str | None, **kw: Any) -> dict[str, Any]:
    return message_to_dict(Message(role="assistant", text=text, **kw))


def _public(action: dict[str, Any], thread_id: str) -> PendingApproval:
    return PendingApproval(
        action_id=action["action_id"],
        tool=action["tool"],
        summary=action["summary"],
        details=action["details"],
        expires_at=datetime.fromisoformat(action["expires_at"]),
        thread_id=thread_id,
    )


class SupportAgent:
    """A conversation with one customer, persisted as a LangGraph thread.

    Not thread-safe; use one instance per chat session. Without a checkpointer the
    conversation lives in memory (Phase 2 behaviour); with a SQLite checkpointer
    it survives restarts.
    """

    def __init__(
        self,
        llm: LLMClient,
        crm: CRMService,
        *,
        ctx: AuthContext | None,
        config: AgentConfig | None = None,
        tools: Sequence[AgentTool] | None = None,
        auditor: ApprovalAuditor | None = None,
        clock: Clock = utcnow,
        checkpointer: BaseCheckpointSaver | None = None,
        thread_id: str | None = None,
        retry: RetryPolicy | None = None,
        on_refund_requested: Callable[[str], None] | None = None,
        guest_id: str | None = None,
    ) -> None:
        self._llm = llm
        self._crm = crm
        self._ctx = ctx
        self._config = config or AgentConfig()
        self._auditor = auditor
        self._clock = clock
        self._retry = retry or RetryPolicy()
        self._on_refund_requested = on_refund_requested

        available = [
            t for t in (tools or default_tools()) if ctx is not None or not t.requires_auth
        ]
        self._tools = {t.name: t for t in tools or default_tools()}
        # Guests only see public tools; authenticated tools are still known so that a
        # hallucinated call gets a clear "authentication_required" result.
        self._definitions = [t.definition() for t in available]
        self._system_prompt = system_prompt(authenticated=ctx is not None, now=clock())

        self.thread_id = thread_id or new_thread_id()
        # A guest thread belongs to one guest session only. Without a guest id each guest
        # agent gets a fresh random one, so guest conversations cannot be picked up by anyone.
        self._owner = ctx.actor if ctx else f"guest:{guest_id or secrets.token_hex(12)}"
        self._run_config: dict[str, Any] = {
            "configurable": {"thread_id": self.thread_id},
            "recursion_limit": _RECURSION_LIMIT,
        }
        self._graph = build_conversation_graph(self, checkpointer or InMemorySaver())
        self._executions: list[ToolExecution] = []
        self._expired_now = False
        self._check_owner()

    # ------------------------------------------------------------ public API

    @property
    def history(self) -> tuple[Message, ...]:
        return tuple(message_from_dict(m) for m in self._values().get("messages", []))

    @property
    def pending_approvals(self) -> list[PendingApproval]:
        snapshot = self._graph.get_state(self._run_config)
        if not snapshot.interrupts:
            return []
        round_ = snapshot.values.get("open_round") or {}
        return [
            _public(a, self.thread_id)
            for a in round_.get("pending", [])
            if a["status"] == "pending"
        ]

    @property
    def authenticated(self) -> bool:
        return self._ctx is not None

    @property
    def exists(self) -> bool:
        """True once the thread has saved state (it has had at least one message)."""
        return bool(self._values())

    @property
    def needs_resume(self) -> bool:
        """True when a step was cut short (crash or persistent outage) and can be resumed."""
        snapshot = self._graph.get_state(self._run_config)
        return bool(snapshot.next) and not snapshot.interrupts

    def send(self, text: str) -> AgentReply:
        text = (text or "").strip()
        if not text:
            raise AgentError("Message is empty.")
        if len(text) > self._config.max_user_message_chars:
            raise AgentError(
                f"Message is too long (max {self._config.max_user_message_chars} characters)."
            )
        if self._session_expired:
            return AgentReply(SESSION_EXPIRED_TEXT, StopReason.SESSION_EXPIRED)

        self._executions = []
        if self.needs_resume:
            # Finish what was already approved before handling anything new.
            reply = self._invoke(None, reset=False)
            if reply.stop_reason is StopReason.RETRY_LATER:
                return reply
        if self.pending_approvals:
            # A new message instead of a decision cancels the pending actions.
            return self._invoke(Command(resume={"decision": "cancel", "text": text}), reset=False)
        return self._invoke({"inbox": {"text": text}, "owner": self._owner}, reset=False)

    def approve(self, action_id: str) -> AgentReply:
        """Record the human's approval, execute the action, and let the model continue."""
        self._require_pending(action_id)
        return self._invoke(Command(resume={"decision": "approve", "action_id": action_id}))

    def reject(self, action_id: str) -> AgentReply:
        """Record that the human declined; the model is told not to retry."""
        self._require_pending(action_id)
        return self._invoke(Command(resume={"decision": "reject", "action_id": action_id}))

    def resume(self) -> AgentReply | None:
        """Continue a step that was cut short. Returns ``None`` if there was nothing to do."""
        if not self.needs_resume:
            return None
        return self._invoke(None)

    # ------------------------------------------------------------ graph nodes

    def ingest(self, state: ConversationState) -> dict[str, Any]:
        if state.get("owner") not in (None, self._owner):
            raise ThreadOwnershipError("This conversation belongs to someone else.")
        text = (state.get("inbox") or {})["text"]
        return {
            "owner": self._owner,
            "messages": [message_to_dict(Message(role="user", text=text))],
            "turn": _new_turn(),
            "completed_writes": state.get("completed_writes") or {},
            "inbox": None,
            "reply": None,
            "route": CALL_MODEL,
        }

    def call_model(self, state: ConversationState) -> dict[str, Any]:
        turn = copy.deepcopy(state["turn"])
        if turn["llm_rounds"] >= self._config.max_llm_rounds:
            logger.warning("Agent hit the LLM round limit", extra={"rounds": turn["llm_rounds"]})
            return self._finish(ITERATION_LIMIT_TEXT, StopReason.ITERATION_LIMIT)

        started = time.perf_counter()
        try:
            response = self._llm.generate(
                system_prompt=self._system_prompt,
                messages=tuple(message_from_dict(m) for m in state["messages"]),
                tools=self._definitions,
            )
        except LLMError as exc:
            logger.error("LLM call failed", extra={"error": str(exc)})
            # Close the exchange so the next user message starts a valid turn.
            return self._finish(LLM_ERROR_TEXT, StopReason.LLM_ERROR)
        turn["llm_rounds"] += 1
        logger.info(
            "LLM responded",
            extra={
                "round": turn["llm_rounds"],
                "tool_calls": len(response.tool_calls),
                "latency_ms": round((time.perf_counter() - started) * 1000),
                "thread_id": self.thread_id,
                **response.usage,
            },
        )

        if not response.tool_calls:
            text = response.text or EMPTY_REPLY_TEXT
            return {
                **self._finish(
                    text,
                    StopReason.COMPLETED,
                    provider_data=response.provider_data if response.text else None,
                ),
                "turn": turn,
            }

        message = _assistant(
            response.text, tool_calls=response.tool_calls, provider_data=response.provider_data
        )
        return {
            "messages": [message],
            "turn": turn,
            "open_round": {
                "calls": [tool_call_to_dict(c) for c in response.tool_calls],
                "results": {},
                "pending": [],
                "model_text": response.text,
            },
            "route": DISPATCH_TOOLS,
        }

    def dispatch_tools(self, state: ConversationState) -> dict[str, Any]:
        round_ = copy.deepcopy(state["open_round"])
        turn = copy.deepcopy(state["turn"])
        completed = state.get("completed_writes") or {}
        self._expired_now = False
        for raw in round_["calls"]:
            call = tool_call_from_dict(raw)
            if self._expired_now:
                self._resolve(round_, call, _error("session_expired", "Session expired."))
                continue
            self._dispatch(call, round_, turn, completed)

        update: dict[str, Any] = {"open_round": round_, "turn": turn}
        if self._expired_now:
            return {**update, "route": EXPIRE_SESSION}
        if round_["pending"]:
            reply = {
                "text": round_["model_text"],
                "stop_reason": StopReason.AWAITING_APPROVAL.value,
            }
            return {**update, "reply": reply, "route": AWAIT_CONFIRMATION}
        return {**update, **self._close_round(round_), "route": CALL_MODEL}

    def await_confirmation(self, state: ConversationState) -> dict[str, Any]:
        # Nothing may happen before interrupt(): this node re-runs from the top on resume.
        round_ = state["open_round"]
        waiting = [a for a in round_["pending"] if a["status"] == "pending"]
        decision = interrupt(
            {
                "kind": "customer_confirmation",
                "thread_id": self.thread_id,
                "actions": [
                    {k: a[k] for k in ("action_id", "tool", "summary", "details", "expires_at")}
                    for a in waiting
                ],
            }
        )

        round_ = copy.deepcopy(round_)
        self._expired_now = False
        kind = decision.get("decision") if isinstance(decision, dict) else None
        if kind == "cancel":
            for action in round_["pending"]:
                if action["status"] == "pending":
                    self._decline(
                        round_,
                        action,
                        "cancelled",
                        _error(
                            "action_cancelled",
                            "The customer sent a new message instead of "
                            "confirming, so this action was not performed.",
                        ),
                    )
            closed = self._close_round(round_)
            return {
                **closed,
                "messages": [
                    *closed["messages"],
                    message_to_dict(Message(role="user", text=decision["text"])),
                ],
                "turn": _new_turn(),
                "reply": None,
                "route": CALL_MODEL,
            }

        action = (
            next(
                (a for a in round_["pending"] if a["action_id"] == decision.get("action_id")), None
            )
            if kind in ("approve", "reject")
            else None
        )
        if action is None or action["status"] != "pending":
            # Fail closed on anything unexpected: stay paused and ask again.
            logger.warning("Ignored an invalid confirmation decision", extra={"decision": kind})
            return {"route": AWAIT_CONFIRMATION}

        if kind == "reject":
            self._decline(
                round_,
                action,
                "declined",
                _error(
                    "declined_by_customer",
                    "The customer declined this action. Nothing was changed. Do not "
                    "retry it unless they ask again.",
                ),
            )
        else:
            self._approve(round_, action, state.get("completed_writes") or {})
        return self._after_decision(round_)

    def execute_action(self, state: ConversationState) -> dict[str, Any]:
        round_ = copy.deepcopy(state["open_round"])
        completed = dict(state.get("completed_writes") or {})
        action = next(a for a in round_["pending"] if a["status"] == "executing")
        tool = self._tools[action["tool"]]
        args = tool.args_model.model_validate(action["arguments"])
        call = tool_call_from_dict(action["call"])
        self._expired_now = False

        # TransientFailure propagates: the workflow stops here, the "executing"
        # marker stays checkpointed, and resume() repeats this same idempotent call.
        result, status = self._execute(tool, args, stall_on_transient=True)
        if status is ExecutionStatus.SUCCESS:
            completed[action["idempotency_key"]] = result
            self._after_write(tool, result)
        action["status"] = "done"
        self._resolve(round_, call, result)
        self._executions.append(
            ToolExecution(tool.name, action["details"], status, result, approved=True)
        )
        return {**self._after_decision(round_), "completed_writes": completed}

    def expire_session(self, state: ConversationState) -> dict[str, Any]:
        round_ = copy.deepcopy(state.get("open_round"))
        messages: list[dict[str, Any]] = []
        if round_ is not None:
            for action in round_["pending"]:
                if action["status"] in ("pending", "approved", "executing"):
                    action["status"] = "expired"
                    self._resolve(
                        round_,
                        tool_call_from_dict(action["call"]),
                        _error("session_expired", "Session expired."),
                    )
            messages.extend(self._close_round(round_)["messages"])
        finish = self._finish(SESSION_EXPIRED_TEXT, StopReason.SESSION_EXPIRED)
        return {
            **finish,
            "messages": [*messages, *finish["messages"]],
            "open_round": None,
            "expired_session": self._ctx.session_id if self._ctx else None,
        }

    # ------------------------------------------------------------ node internals

    def _dispatch(
        self,
        call: ToolCall,
        round_: dict[str, Any],
        turn: dict[str, Any],
        completed: dict[str, dict[str, Any]],
    ) -> None:
        turn["tool_calls"] += 1

        def finish(result: dict[str, Any], status: ExecutionStatus) -> None:
            self._resolve(round_, call, result)
            self._executions.append(ToolExecution(call.name, call.arguments, status, result))
            logger.info(
                "Tool call",
                extra={
                    "tool": call.name,
                    "status": status.value,
                    "error_code": result.get("error"),
                    "thread_id": self.thread_id,
                },
            )

        if turn["tool_calls"] > self._config.max_tool_calls_per_turn:
            return finish(
                _error(
                    "tool_call_limit",
                    "Too many tool calls for one request. Answer with the "
                    "information you already have.",
                ),
                ExecutionStatus.ERROR,
            )

        tool = self._tools.get(call.name)
        if tool is None:
            return finish(
                _error(
                    "unknown_tool",
                    f"There is no tool named '{call.name}'.",
                    available=[d.name for d in self._definitions],
                ),
                ExecutionStatus.ERROR,
            )
        if tool.requires_auth and self._ctx is None:
            return finish(
                _error(
                    "authentication_required",
                    "The customer is not signed in. Ask them to restart the chat "
                    "and sign in with their email and support PIN.",
                ),
                ExecutionStatus.ERROR,
            )
        try:
            args = tool.args_model.model_validate(call.arguments)
        except ValidationError as exc:
            errors = [
                {"field": ".".join(str(p) for p in e["loc"]) or "(root)", "message": e["msg"]}
                for e in exc.errors()
            ]
            return finish(
                _error("invalid_arguments", "The tool arguments are invalid.", errors=errors),
                ExecutionStatus.ERROR,
            )

        if tool.is_write:
            key = self._idempotency_key(tool, args)
            if key in completed:
                return finish(
                    {
                        **completed[key],
                        "note": "This exact action was already completed earlier in "
                        "this conversation; it was not repeated.",
                    },
                    ExecutionStatus.CACHED,
                )
            if self._needs_approval(tool):
                return self._request_approval(call, tool, args, round_)
            result, status = self._execute(tool, args)
            if status is ExecutionStatus.SUCCESS:
                completed[key] = result
            return finish(result, status)

        cache_key = f"{tool.name}|{args.model_dump_json()}"
        if cache_key in turn["read_cache"]:
            return finish(turn["read_cache"][cache_key], ExecutionStatus.CACHED)
        result, status = self._execute(tool, args)
        if status is ExecutionStatus.SUCCESS:
            turn["read_cache"][cache_key] = result
        finish(result, status)

    def _request_approval(
        self, call: ToolCall, tool: AgentTool, args: BaseModel, round_: dict[str, Any]
    ) -> None:
        assert self._ctx is not None and tool.preview is not None
        try:
            preview = self._retry.run(
                f"preview:{tool.name}", lambda: tool.preview(self._crm, self._ctx, args)
            )
        except AuthenticationError as exc:
            self._expired_now = True
            result = {"ok": False, **exc.to_dict()}
        except CRMError as exc:
            # The CRM would reject this anyway: report it instead of asking to confirm.
            result = {"ok": False, **exc.to_dict()}
        except TransientFailure:
            result = _error(
                "temporarily_unavailable",
                "The system is busy. Nothing was changed. Ask the customer to try again shortly.",
            )
        else:
            action = {
                "action_id": f"act_{secrets.token_hex(6)}",
                "tool": tool.name,
                "call": tool_call_to_dict(call),
                "arguments": args.model_dump(mode="json"),
                "summary": preview.summary,
                "details": preview.details,
                "expires_at": (self._clock() + self._config.approval_ttl).isoformat(),
                "idempotency_key": self._idempotency_key(tool, args),
                "status": "pending",
            }
            round_["pending"].append(action)
            self._executions.append(
                ToolExecution(
                    tool.name,
                    call.arguments,
                    ExecutionStatus.AWAITING_APPROVAL,
                    {"summary": preview.summary, "action_id": action["action_id"]},
                )
            )
            return
        self._resolve(round_, call, result)
        self._executions.append(
            ToolExecution(tool.name, call.arguments, ExecutionStatus.ERROR, result)
        )

    def _approve(
        self, round_: dict[str, Any], action: dict[str, Any], completed: dict[str, Any]
    ) -> None:
        """Validate an approval. On success the action is marked ``executing``."""
        tool = self._tools[action["tool"]]
        call = tool_call_from_dict(action["call"])
        args = tool.args_model.model_validate(action["arguments"])

        def refuse(result: dict[str, Any], status: ExecutionStatus, new_status: str) -> None:
            action["status"] = new_status
            self._resolve(round_, call, result)
            self._executions.append(ToolExecution(tool.name, action["details"], status, result))

        if self._clock() > datetime.fromisoformat(action["expires_at"]):
            self._audit(action, "expired")
            return refuse(
                _error(
                    "approval_expired",
                    "The confirmation expired before the customer approved it. "
                    "Nothing was changed. Ask whether they still want it.",
                ),
                ExecutionStatus.DECLINED,
                "expired",
            )
        if not self._audit(action, "granted"):
            # Fail closed: an approval that cannot be recorded is not acted on.
            return refuse(
                _error(
                    "approval_not_recorded",
                    "The confirmation could not be recorded, so nothing was changed. "
                    "Ask the customer to try again shortly.",
                ),
                ExecutionStatus.ERROR,
                "failed",
            )
        if action["idempotency_key"] not in completed and self._ctx is not None:
            # The customer approved what they were shown. If the CRM now computes
            # something different (amount, review flag), do not act on a stale approval.
            try:
                fresh = self._retry.run(
                    f"preview:{tool.name}", lambda: tool.preview(self._crm, self._ctx, args)
                )
            except AuthenticationError:
                self._expired_now = True
                return
            except (CRMError, TransientFailure):
                fresh = None  # let the CRM write itself report the precise error
            if fresh is not None and fresh.details != action["details"]:
                return refuse(
                    _error(
                        "confirmation_outdated",
                        "The details changed after the customer confirmed, so nothing was "
                        "changed. Explain the new details and ask again.",
                        confirmed=action["details"],
                        current=fresh.details,
                    ),
                    ExecutionStatus.DECLINED,
                    "outdated",
                )
        action["status"] = "executing"

    def _decline(
        self, round_: dict[str, Any], action: dict[str, Any], decision: str, result: dict[str, Any]
    ) -> None:
        self._audit(action, decision)
        action["status"] = decision
        self._resolve(round_, tool_call_from_dict(action["call"]), result)
        self._executions.append(
            ToolExecution(action["tool"], action["details"], ExecutionStatus.DECLINED, result)
        )

    def _after_decision(self, round_: dict[str, Any]) -> dict[str, Any]:
        if self._expired_now:
            return {"open_round": round_, "route": EXPIRE_SESSION}
        if any(a["status"] == "executing" for a in round_["pending"]):
            return {"open_round": round_, "route": EXECUTE_ACTION}
        if any(a["status"] == "pending" for a in round_["pending"]):
            reply = {"text": None, "stop_reason": StopReason.AWAITING_APPROVAL.value}
            return {"open_round": round_, "reply": reply, "route": AWAIT_CONFIRMATION}
        return {**self._close_round(round_), "route": CALL_MODEL}

    def _execute(
        self, tool: AgentTool, args: BaseModel, *, stall_on_transient: bool = False
    ) -> tuple[dict[str, Any], ExecutionStatus]:
        key = self._idempotency_key(tool, args) if tool.is_write else None
        try:
            value = self._retry.run(
                f"tool:{tool.name}", lambda: tool.handler(self._crm, self._ctx, args, key)
            )
        except AuthenticationError as exc:
            self._expired_now = True
            return {"ok": False, **exc.to_dict()}, ExecutionStatus.ERROR
        except CRMError as exc:
            return {"ok": False, **exc.to_dict()}, ExecutionStatus.ERROR
        except TransientFailure:
            if stall_on_transient:
                raise
            return _error(
                "temporarily_unavailable",
                "The system is busy right now. Nothing was changed. Apologize and suggest "
                "trying again in a moment.",
            ), ExecutionStatus.ERROR
        except Exception:
            logger.exception("Tool raised an unexpected error", extra={"tool": tool.name})
            return _error(
                "internal_error",
                "The tool failed unexpectedly. Apologize, and offer to try again "
                "later or open a support ticket.",
            ), ExecutionStatus.ERROR
        return {"ok": True, "data": _serialize(value)}, ExecutionStatus.SUCCESS

    def _after_write(self, tool: AgentTool, result: dict[str, Any]) -> None:
        """Hand a new refund request to the staff review workflow (idempotent)."""
        if tool.name != "request_refund" or self._on_refund_requested is None:
            return
        refund = result["data"]["refund"]
        if refund["status"] != "pending_review":
            return
        try:
            self._on_refund_requested(refund["refund_number"])
        except Exception:
            # The request is safely in the CRM; staff can still open it from the queue.
            logger.exception("Could not open the refund review workflow")

    # ------------------------------------------------------------ helpers

    def _invoke(self, payload: Any, *, reset: bool = True) -> AgentReply:
        if reset:
            self._executions = []
        try:
            self._graph.invoke(payload, self._run_config)
        except TransientFailure as exc:
            logger.error(
                "Workflow step stalled; it can be resumed",
                extra={"thread_id": self.thread_id, "operation": exc.operation},
            )
            return AgentReply(RETRY_LATER_TEXT, StopReason.RETRY_LATER, list(self._executions))
        return self._reply()

    def _reply(self) -> AgentReply:
        executions = list(self._executions)
        snapshot = self._graph.get_state(self._run_config)
        values = snapshot.values
        reply = values.get("reply") or {}
        if snapshot.interrupts:
            pending = self.pending_approvals
            lines = [reply["text"]] if reply.get("text") else []
            lines.append("Please confirm the following before I continue:")
            lines.extend(f"  - {p.summary}" for p in pending)
            return AgentReply("\n".join(lines), StopReason.AWAITING_APPROVAL, executions, pending)
        if snapshot.next:
            return AgentReply(RETRY_LATER_TEXT, StopReason.RETRY_LATER, executions)
        return AgentReply(reply["text"], StopReason(reply["stop_reason"]), executions)

    def _finish(self, text: str, reason: StopReason, **message: Any) -> dict[str, Any]:
        return {
            "messages": [_assistant(text, **message)],
            "reply": {"text": text, "stop_reason": reason.value},
            "route": END,
        }

    def _values(self) -> dict[str, Any]:
        return self._graph.get_state(self._run_config).values or {}

    @property
    def _session_expired(self) -> bool:
        expired = self._values().get("expired_session")
        return expired is not None and self._ctx is not None and expired == self._ctx.session_id

    def _check_owner(self) -> None:
        owner = self._values().get("owner")
        if owner is None or owner == self._owner:
            return
        recorder = getattr(self._auditor, "record_event", None)
        if recorder is not None:
            try:
                recorder(
                    actor=self._owner,
                    session_id=self._ctx.session_id if self._ctx else None,
                    action="workflow.resume",
                    outcome=AuditOutcome.DENIED,
                    thread_id=self.thread_id,
                    reason="not_thread_owner",
                )
            except Exception:
                logger.exception("Failed to record denied thread access")
        raise ThreadOwnershipError(f"Conversation {self.thread_id} belongs to a different account.")

    def _require_pending(self, action_id: str) -> None:
        if action_id not in {p.action_id for p in self.pending_approvals}:
            raise AgentError(f"No pending action with id '{action_id}'.")

    def _needs_approval(self, tool: AgentTool) -> bool:
        if tool.name == "create_support_ticket":
            return self._config.require_ticket_approval
        return True  # every other write, including refunds, always needs approval

    def _idempotency_key(self, tool: AgentTool, args: BaseModel) -> str:
        """Same thread + tool + arguments -> same key, so the CRM replays, never duplicates.

        Scoped to the workflow thread (not the login session), so the key is the
        same after a restart and a fresh sign-in.
        """
        canonical = json.dumps(args.model_dump(mode="json"), sort_keys=True)
        digest = hashlib.sha256(f"{self.thread_id}|{tool.name}|{canonical}".encode()).hexdigest()
        return f"agent-{digest[:40]}"

    @staticmethod
    def _resolve(round_: dict[str, Any], call: ToolCall, result: dict[str, Any]) -> None:
        round_["results"][call.id] = {
            "call_id": call.id,
            "name": call.name,
            "content": result,
            "provider_id": call.provider_id,
        }

    @staticmethod
    def _close_round(round_: dict[str, Any]) -> dict[str, Any]:
        results = [round_["results"][c["id"]] for c in round_["calls"]]
        message = {
            "role": "tool",
            "text": None,
            "tool_calls": [],
            "tool_results": results,
            "provider_data": None,
        }
        return {"messages": [message], "open_round": None}

    def _audit(self, action: dict[str, Any], decision: str) -> bool:
        if self._auditor is None or self._ctx is None:
            return True
        try:
            self._auditor.record(self._ctx, _public(action, self.thread_id), decision)
        except ApprovalAlreadyDecided as exc:
            if exc.decision == decision:
                # The same answer was already recorded (a double click, or a retry after a
                # crash). It counts once; any write it triggers is idempotent.
                logger.info(
                    "Approval decision already recorded",
                    extra={"action_id": action["action_id"], "decision": decision},
                )
                return True
            raise ApprovalConflictError(
                f"This action was already {exc.decision} in another session. "
                "Refresh to see the current state."
            ) from exc
        except Exception:
            logger.exception("Failed to record approval decision")
            return False
        return True

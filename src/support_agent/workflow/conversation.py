"""The customer conversation as a LangGraph state machine.

::

    START -> ingest -> call_model --(text)--------------------------------> END
                          ^   |
                          |   +--(tool calls)--> dispatch_tools
                          |                        |  reads run here; writes get a
                          |                        |  preview and become pending
                          +------------------------+  (no pending writes)
                          |                        |
                          |                        v
                          |               await_confirmation  <-- interrupt():
                          |                 |   ^  |              the graph stops here and
                          |   (all decided) |   |  |              is checkpointed to SQLite
                          +-----------------+   |  v (approved)
                          |                     execute_action  (idempotent CRM write)
                          +---------------------+
          any step whose session check fails --> expire_session -> END

Design rules:

* State is plain JSON (messages, tool calls, pending actions as dicts). It never
  holds the signed customer session, a PIN, or provider objects, so a checkpoint
  can be loaded by a later process and inspected safely.
* The node that calls ``interrupt()`` does nothing before it. LangGraph re-runs
  that node from the top on resume, so any side effect before the interrupt
  would repeat.
* A confirmed write is marked ``executing`` (and checkpointed) *before* the CRM
  call runs in its own node. If the process dies mid-write, the resumed node
  repeats the call with the same idempotency key and the CRM replays the
  existing record instead of creating a second one.
"""

import operator
from typing import Annotated, Any, Protocol, TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

THREAD_PREFIX = "conv-"

INGEST = "ingest"
CALL_MODEL = "call_model"
DISPATCH_TOOLS = "dispatch_tools"
AWAIT_CONFIRMATION = "await_confirmation"
EXECUTE_ACTION = "execute_action"
EXPIRE_SESSION = "expire_session"


class ConversationState(TypedDict, total=False):
    owner: str  # "customer:<id>" or "guest"; checked whenever the thread is resumed
    messages: Annotated[list[dict[str, Any]], operator.add]
    # Tool calls from the latest model response whose results are not all known yet.
    open_round: dict[str, Any] | None
    turn: dict[str, Any]  # per-customer-message counters and read cache
    completed_writes: dict[str, dict[str, Any]]  # idempotency key -> CRM result
    expired_session: str | None  # session id that expired in this thread
    inbox: dict[str, Any] | None  # the customer message being ingested
    reply: dict[str, Any] | None  # {"text", "stop_reason"} for the caller
    route: str  # next node, chosen by the node that just ran


class ConversationNodes(Protocol):
    def ingest(self, state: ConversationState) -> dict[str, Any]: ...
    def call_model(self, state: ConversationState) -> dict[str, Any]: ...
    def dispatch_tools(self, state: ConversationState) -> dict[str, Any]: ...
    def await_confirmation(self, state: ConversationState) -> dict[str, Any]: ...
    def execute_action(self, state: ConversationState) -> dict[str, Any]: ...
    def expire_session(self, state: ConversationState) -> dict[str, Any]: ...


def _route(state: ConversationState) -> str:
    return state.get("route", END)


def build_conversation_graph(
    nodes: ConversationNodes, checkpointer: BaseCheckpointSaver
) -> CompiledStateGraph:
    graph = StateGraph(ConversationState)
    graph.add_node(INGEST, nodes.ingest)
    graph.add_node(CALL_MODEL, nodes.call_model)
    graph.add_node(DISPATCH_TOOLS, nodes.dispatch_tools)
    graph.add_node(AWAIT_CONFIRMATION, nodes.await_confirmation)
    graph.add_node(EXECUTE_ACTION, nodes.execute_action)
    graph.add_node(EXPIRE_SESSION, nodes.expire_session)

    graph.add_edge(START, INGEST)
    graph.add_edge(INGEST, CALL_MODEL)
    graph.add_conditional_edges(CALL_MODEL, _route, [DISPATCH_TOOLS, END])
    graph.add_conditional_edges(
        DISPATCH_TOOLS, _route, [AWAIT_CONFIRMATION, EXPIRE_SESSION, CALL_MODEL]
    )
    graph.add_conditional_edges(
        AWAIT_CONFIRMATION,
        _route,
        [AWAIT_CONFIRMATION, EXECUTE_ACTION, EXPIRE_SESSION, CALL_MODEL],
    )
    graph.add_conditional_edges(
        EXECUTE_ACTION, _route, [AWAIT_CONFIRMATION, EXPIRE_SESSION, CALL_MODEL]
    )
    graph.add_edge(EXPIRE_SESSION, END)
    return graph.compile(checkpointer=checkpointer)


# --------------------------------------------------------------------------- inspection


class _ReadOnlyNodes:
    """Lets operators inspect saved conversations without an LLM or a customer session."""

    def __getattr__(self, name: str) -> Any:
        def refuse(state: ConversationState) -> dict[str, Any]:
            raise RuntimeError("This graph is read-only.")

        return refuse


def describe_conversation(checkpointer: BaseCheckpointSaver, thread_id: str) -> dict[str, Any]:
    graph = build_conversation_graph(_ReadOnlyNodes(), checkpointer)  # type: ignore[arg-type]
    snapshot = graph.get_state({"configurable": {"thread_id": thread_id}})
    values = snapshot.values or {}
    round_ = values.get("open_round") or {}
    if snapshot.interrupts:
        status = "awaiting_customer_confirmation"
    elif snapshot.next:
        status = "needs_resume"
    elif values.get("expired_session"):
        status = "session_expired"
    else:
        status = "idle"
    return {
        "thread_id": thread_id,
        "kind": "conversation",
        "owner": values.get("owner"),
        "status": status,
        "next": list(snapshot.next),
        "messages": len(values.get("messages", [])),
        "preview": next(
            (m["text"] for m in values.get("messages", []) if m["role"] == "user"), None
        ),
        "pending": [
            {"action_id": a["action_id"], "tool": a["tool"], "summary": a["summary"]}
            for a in round_.get("pending", [])
            if a["status"] in ("pending", "executing")
        ],
        "updated_at": snapshot.created_at,
    }

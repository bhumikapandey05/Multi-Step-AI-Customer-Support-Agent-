"""Chat endpoints on top of the LangGraph conversation workflow.

* Customers and guests both chat; the token decides the thread owner
  (``customer:<id>`` or ``guest:<session>``). A thread can only be read or
  continued by its owner. Anyone else gets the same 404 as a missing thread.
* A thread is created by its first message, so nobody can reserve a thread id.
* Approving or rejecting goes through ``SupportAgent.approve/reject``, which
  records the decision atomically in the database (one decision per action).
"""

from typing import Annotated

from fastapi import APIRouter, Depends, Path, Request, status

from support_agent.agent.agent import SupportAgent
from support_agent.agent.llm import Message
from support_agent.api.errors import ApiError, not_found
from support_agent.api.schemas import (
    ActionId,
    ChatMessageOut,
    ConversationOut,
    ConversationSummary,
    MessageRequest,
    PendingApprovalOut,
    ReplyOut,
    ThreadId,
)
from support_agent.api.security import ChatPrincipal, current_chat_principal
from support_agent.workflow.checkpoint import list_thread_ids
from support_agent.workflow.conversation import THREAD_PREFIX, describe_conversation

router = APIRouter(prefix="/v1/conversations", tags=["conversations"])

Principal = Annotated[ChatPrincipal, Depends(current_chat_principal)]
Thread = Annotated[ThreadId, Path()]
Action = Annotated[ActionId, Path()]


def _services(request: Request):
    return request.app.state.services


def _existing(
    request: Request, principal: ChatPrincipal, thread_id: str, *, need_llm: bool = True
) -> SupportAgent:
    agent = _services(request).agent(principal, thread_id, need_llm=need_llm)  # checks owner
    if not agent.exists:
        raise not_found("conversation")
    return agent


def _display(messages: tuple[Message, ...]) -> list[ChatMessageOut]:
    return [
        ChatMessageOut(role=m.role, text=m.text)
        for m in messages
        if m.role in ("user", "assistant") and m.text
    ]


@router.get("", response_model=list[ConversationSummary])
def list_conversations(request: Request, principal: Principal) -> list[ConversationSummary]:
    checkpointer = _services(request).checkpointer
    rows = []
    for thread_id in list_thread_ids(checkpointer, prefix=THREAD_PREFIX):
        info = describe_conversation(checkpointer, thread_id)
        if info["owner"] == principal.actor:
            rows.append(
                ConversationSummary(
                    thread_id=thread_id,
                    status=info["status"],
                    messages=info["messages"],
                    updated_at=info["updated_at"],
                    preview=info["preview"],
                )
            )
    return rows


@router.post("", response_model=ReplyOut, status_code=status.HTTP_201_CREATED)
def start_conversation(body: MessageRequest, request: Request, principal: Principal) -> ReplyOut:
    services = _services(request)
    agent = services.agent(principal)
    with services.locks.hold(agent.thread_id):
        return ReplyOut.build(agent.thread_id, agent.send(body.message))


@router.get("/{thread_id}", response_model=ConversationOut)
def get_conversation(thread_id: Thread, request: Request, principal: Principal) -> ConversationOut:
    agent = _existing(request, principal, thread_id, need_llm=False)
    info = describe_conversation(_services(request).checkpointer, thread_id)
    return ConversationOut(
        thread_id=thread_id,
        status=info["status"],
        needs_resume=agent.needs_resume,
        messages=_display(agent.history),
        pending_approvals=[
            PendingApprovalOut(
                action_id=p.action_id,
                tool=p.tool,
                summary=p.summary,
                details=p.details,
                expires_at=p.expires_at,
            )
            for p in agent.pending_approvals
        ],
    )


@router.post("/{thread_id}/messages", response_model=ReplyOut)
def send_message(
    thread_id: Thread, body: MessageRequest, request: Request, principal: Principal
) -> ReplyOut:
    services = _services(request)
    with services.locks.hold(thread_id):
        agent = _existing(request, principal, thread_id)
        return ReplyOut.build(thread_id, agent.send(body.message))


@router.post("/{thread_id}/approvals/{action_id}/approve", response_model=ReplyOut)
def approve(
    thread_id: Thread, action_id: Action, request: Request, principal: Principal
) -> ReplyOut:
    services = _services(request)
    with services.locks.hold(thread_id):
        agent = _existing(request, principal, thread_id)
        _require_pending(agent, action_id)
        return ReplyOut.build(thread_id, agent.approve(action_id))


@router.post("/{thread_id}/approvals/{action_id}/reject", response_model=ReplyOut)
def reject(
    thread_id: Thread, action_id: Action, request: Request, principal: Principal
) -> ReplyOut:
    services = _services(request)
    with services.locks.hold(thread_id):
        agent = _existing(request, principal, thread_id)
        _require_pending(agent, action_id)
        return ReplyOut.build(thread_id, agent.reject(action_id))


@router.post("/{thread_id}/resume", response_model=ReplyOut)
def resume(thread_id: Thread, request: Request, principal: Principal) -> ReplyOut:
    services = _services(request)
    with services.locks.hold(thread_id):
        agent = _existing(request, principal, thread_id)
        reply = agent.resume()
        if reply is None:
            raise ApiError(409, "workflow_state", "Nothing in this conversation needs resuming.")
        return ReplyOut.build(thread_id, reply)


def _require_pending(agent: SupportAgent, action_id: str) -> None:
    if action_id not in {p.action_id for p in agent.pending_approvals}:
        raise ApiError(
            409,
            "approval_not_pending",
            "This action is not waiting for a decision (it may already be decided).",
        )

"""Request and response models.

Request models use ``extra="forbid"``: a client that sends ``customer_id``,
``staff_id``, ``role``, ``approved``, or any other unexpected field gets a 422.
Identity always comes from the verified bearer token.
"""

from datetime import datetime
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from support_agent.agent.agent import AgentReply

ThreadId = Annotated[str, StringConstraints(pattern=r"^conv-[0-9a-f]{16}$")]
ActionId = Annotated[str, StringConstraints(pattern=r"^act_[0-9a-f]{12}$")]
RefundNumberPath = Annotated[str, StringConstraints(pattern=r"^[Rr][Ff][Dd]-[0-9A-Fa-f]{8}$")]
Email = Annotated[str, StringConstraints(strip_whitespace=True, min_length=3, max_length=254)]
Pin = Annotated[str, StringConstraints(pattern=r"^\d{6}$")]


class _Request(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LoginRequest(_Request):
    email: Email
    pin: Pin


class MessageRequest(_Request):
    message: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4000)]


class ApproveRequest(_Request):
    note: Annotated[str, StringConstraints(strip_whitespace=True, max_length=500)] | None = None


class RejectRequest(_Request):
    note: Annotated[str, StringConstraints(strip_whitespace=True, min_length=5, max_length=500)]


# --------------------------------------------------------------------------- responses


class TokenResponse(BaseModel):
    token: str
    token_type: str = "bearer"
    expires_at: datetime
    kind: str  # customer | guest | staff
    profile: dict[str, Any] = Field(default_factory=dict)


class PendingApprovalOut(BaseModel):
    action_id: str
    tool: str
    summary: str
    details: dict[str, Any]
    expires_at: datetime


class ToolExecutionOut(BaseModel):
    tool: str
    status: str
    approved: bool
    ok: bool | None
    error: str | None
    message: str | None
    data: Any = None


class ReplyOut(BaseModel):
    thread_id: str
    text: str
    stop_reason: str
    tool_executions: list[ToolExecutionOut]
    pending_approvals: list[PendingApprovalOut]

    @classmethod
    def build(cls, thread_id: str, reply: AgentReply) -> "ReplyOut":
        return cls(
            thread_id=thread_id,
            text=reply.text,
            stop_reason=reply.stop_reason.value,
            tool_executions=[
                ToolExecutionOut(
                    tool=e.tool,
                    status=e.status.value,
                    approved=e.approved,
                    ok=e.result.get("ok"),
                    error=e.result.get("error"),
                    message=e.result.get("message") or e.result.get("summary"),
                    data=e.result.get("data"),
                )
                for e in reply.tool_executions
            ],
            pending_approvals=[
                PendingApprovalOut(
                    action_id=p.action_id,
                    tool=p.tool,
                    summary=p.summary,
                    details=p.details,
                    expires_at=p.expires_at,
                )
                for p in reply.pending_approvals
            ],
        )


class ChatMessageOut(BaseModel):
    role: str
    text: str


class ConversationOut(BaseModel):
    thread_id: str
    status: str
    needs_resume: bool
    messages: list[ChatMessageOut]
    pending_approvals: list[PendingApprovalOut]


class ConversationSummary(BaseModel):
    thread_id: str
    status: str
    messages: int
    updated_at: str | None
    preview: str | None


class AuditEventOut(BaseModel):
    id: int
    occurred_at: datetime
    actor: str
    action: str
    outcome: str
    entity_type: str | None
    entity_id: str | None
    details: dict[str, Any]

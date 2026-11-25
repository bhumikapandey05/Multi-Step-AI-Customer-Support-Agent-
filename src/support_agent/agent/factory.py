"""Builds LLM clients, agents, and workflows from settings."""

from langgraph.checkpoint.base import BaseCheckpointSaver

from support_agent.agent.agent import AgentConfig, DatabaseApprovalAuditor, SupportAgent
from support_agent.agent.gemini import GeminiClient
from support_agent.agent.llm import LLMClient
from support_agent.app import Application
from support_agent.clock import Clock, utcnow
from support_agent.config import Settings
from support_agent.crm.identity import AuthContext
from support_agent.workflow.retry import RetryPolicy
from support_agent.workflow.review import RefundReviewWorkflow


def build_llm(settings: Settings) -> LLMClient:
    key = settings.gemini_api_key.get_secret_value() if settings.gemini_api_key else None
    return GeminiClient(
        api_key=key,
        model=settings.gemini_model,
        temperature=settings.llm_temperature,
        timeout_seconds=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
    )


def build_retry_policy(settings: Settings) -> RetryPolicy:
    return RetryPolicy(
        max_retries=settings.workflow_max_retries,
        base_delay_seconds=settings.workflow_retry_base_delay_seconds,
    )


def build_review_workflow(
    app: Application, checkpointer: BaseCheckpointSaver
) -> RefundReviewWorkflow:
    return RefundReviewWorkflow(app.review, checkpointer, retry=build_retry_policy(app.settings))


def build_agent(
    app: Application,
    ctx: AuthContext | None,
    *,
    llm: LLMClient,
    checkpointer: BaseCheckpointSaver | None = None,
    thread_id: str | None = None,
    guest_id: str | None = None,
    clock: Clock = utcnow,
) -> SupportAgent:
    # A confirmed refund request automatically opens a staff review thread.
    review = build_review_workflow(app, checkpointer) if checkpointer is not None else None
    return SupportAgent(
        llm,
        app.crm,
        ctx=ctx,
        config=AgentConfig.from_settings(app.settings),
        auditor=DatabaseApprovalAuditor(app.session_factory, clock=clock),
        clock=clock,
        guest_id=guest_id,
        checkpointer=checkpointer,
        thread_id=thread_id,
        retry=build_retry_policy(app.settings),
        on_refund_requested=review.open if review is not None else None,
    )

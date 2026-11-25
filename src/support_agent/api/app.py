"""FastAPI application factory.

Start it with::

    uvicorn --factory support_agent.api.app:create_app --port 8000

The API reuses the same objects as the CLI: ``Application`` (CRM, identity,
staff review), the LangGraph checkpointer, and the agent/workflow factories.
"""

import logging
import secrets
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response

from support_agent.agent.factory import build_llm
from support_agent.agent.llm import LLMClient
from support_agent.api.errors import install_error_handlers
from support_agent.api.routers import auth, conversations, customer, staff
from support_agent.api.services import ApiServices
from support_agent.app import build_application
from support_agent.clock import Clock, utcnow
from support_agent.config import Settings
from support_agent.logging_config import configure_logging
from support_agent.workflow.checkpoint import connect_checkpointer

logger = logging.getLogger(__name__)

API_VERSION = "1.0.0"


def create_app(
    settings: Settings | None = None,
    *,
    llm_factory: Callable[[Settings], LLMClient] | None = None,
    clock: Clock = utcnow,
    configure_logs: bool = True,
) -> FastAPI:
    settings = settings or Settings()
    if configure_logs:
        configure_logging(settings.log_level, settings.log_format)
    application = build_application(settings, clock=clock)
    if settings.uses_public_secret:
        logger.warning(
            "Tokens are signed with a public development key. Fine for local demos only; "
            "set SUPPORT_AGENT_AUTH_SECRET_KEY (required in production)."
        )
    if not application.schema_exists():
        application.close()
        raise RuntimeError("The database is not initialized. Run `support-crm init-db` first.")
    services = ApiServices(
        app=application,
        checkpointer=connect_checkpointer(settings.checkpoint_path),
        llm_factory=llm_factory or build_llm,
        clock=clock,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        services.close()

    api = FastAPI(
        title="Brightcart Support Agent API",
        version=API_VERSION,
        description="Customer chat with human confirmation, and staff refund review.",
        lifespan=lifespan,
    )
    api.state.services = services
    install_error_handlers(api)

    @api.middleware("http")
    async def request_context(request: Request, call_next) -> Response:
        request_id = request.headers.get("x-request-id", "")[:64] or secrets.token_hex(8)
        request.state.request_id = request_id
        started = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "API request",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "latency_ms": round((time.perf_counter() - started) * 1000),
                "request_id": request_id,
            },
        )
        return response

    @api.get("/health", tags=["meta"])
    def health() -> dict[str, str]:
        return {"status": "ok", "version": API_VERSION}

    for router in (auth.router, customer.router, conversations.router, staff.router):
        api.include_router(router)
    return api

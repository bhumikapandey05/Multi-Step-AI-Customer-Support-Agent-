"""Structured API errors.

Every failure has the same JSON shape, so clients branch on a stable ``code``
instead of parsing messages::

    {"error": {"code": "not_found", "message": "...", "details": {...}},
     "request_id": "..."}

Unexpected exceptions become a generic ``internal_error``; their details are
only logged, never returned.
"""

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from support_agent.agent.agent import AgentError, ApprovalConflictError, ThreadOwnershipError
from support_agent.agent.llm import LLMConfigurationError, LLMError
from support_agent.crm.errors import CRMError
from support_agent.workflow.retry import TransientFailure
from support_agent.workflow.review import ReviewWorkflowError

logger = logging.getLogger(__name__)

CRM_STATUS = {
    "invalid_input": 422,
    "authentication_failed": 401,
    "not_authorized": 403,
    "not_found": 404,
    "idempotency_conflict": 409,
    "conflict": 409,
    "refund_not_eligible": 422,
}


class ApiError(Exception):
    def __init__(
        self, status: int, code: str, message: str, details: dict[str, Any] | None = None
    ) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.details = details or {}


def not_found(what: str) -> ApiError:
    return ApiError(404, "not_found", f"No {what} was found.")


def _body(request: Request, code: str, message: str, details: dict[str, Any]) -> dict[str, Any]:
    return {
        "error": {"code": code, "message": message, "details": details},
        "request_id": getattr(request.state, "request_id", None),
    }


def _respond(request: Request, status: int, code: str, message: str, details=None) -> JSONResponse:
    headers = {"WWW-Authenticate": "Bearer"} if status == 401 else None
    return JSONResponse(_body(request, code, message, details or {}), status, headers=headers)


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _api(request: Request, exc: ApiError) -> JSONResponse:
        return _respond(request, exc.status, exc.code, exc.message, exc.details)

    @app.exception_handler(CRMError)
    async def _crm(request: Request, exc: CRMError) -> JSONResponse:
        # to_dict() never includes the internal (audit-only) details.
        body = exc.to_dict()
        return _respond(
            request, CRM_STATUS.get(exc.code, 400), body["error"], body["message"], body["details"]
        )

    @app.exception_handler(ThreadOwnershipError)
    async def _owner(request: Request, exc: ThreadOwnershipError) -> JSONResponse:
        # Same answer as a missing thread, so other people's thread ids are not confirmed.
        return _respond(request, 404, "not_found", "No conversation was found.")

    @app.exception_handler(ApprovalConflictError)
    async def _conflict(request: Request, exc: ApprovalConflictError) -> JSONResponse:
        return _respond(request, 409, "approval_conflict", str(exc))

    @app.exception_handler(AgentError)
    async def _agent(request: Request, exc: AgentError) -> JSONResponse:
        return _respond(request, 400, "invalid_request", str(exc))

    @app.exception_handler(ReviewWorkflowError)
    async def _workflow(request: Request, exc: ReviewWorkflowError) -> JSONResponse:
        return _respond(request, 409, "workflow_state", str(exc))

    @app.exception_handler(TransientFailure)
    async def _transient(request: Request, exc: TransientFailure) -> JSONResponse:
        return _respond(
            request,
            503,
            "temporarily_unavailable",
            "The system is busy. Any confirmed action is saved and will finish when resumed.",
        )

    @app.exception_handler(LLMConfigurationError)
    async def _llm_config(request: Request, exc: LLMConfigurationError) -> JSONResponse:
        logger.error("LLM is not configured", extra={"error": str(exc)})
        return _respond(
            request, 503, "llm_unavailable", "The assistant is not configured on this server."
        )

    @app.exception_handler(LLMError)
    async def _llm(request: Request, exc: LLMError) -> JSONResponse:
        return _respond(request, 503, "llm_unavailable", "The assistant is unavailable right now.")

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        errors = [
            {
                "field": ".".join(str(p) for p in e["loc"] if p != "body") or "(root)",
                "message": e["msg"],
            }
            for e in exc.errors()
        ]
        return _respond(
            request, 422, "invalid_request", "The request is invalid.", {"errors": errors}
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = {404: "not_found", 405: "method_not_allowed"}.get(exc.status_code, "http_error")
        return _respond(request, exc.status_code, code, str(exc.detail))

    @app.exception_handler(Exception)
    async def _unexpected(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled API error", extra={"path": request.url.path})
        return _respond(request, 500, "internal_error", "Something went wrong on our side.")

"""Read-only customer endpoints (order inquiries). Writes go through the chat workflow only.

There is no customer id anywhere in these routes: the verified token decides
whose data is returned, and the CRM scopes every query to that customer.
"""

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query, Request

from support_agent.api.security import current_customer
from support_agent.crm.identity import AuthContext

router = APIRouter(prefix="/v1/customer", tags=["customer"])

Customer = Annotated[AuthContext, Depends(current_customer)]
Limit = Annotated[int, Query(ge=1, le=100)]


def _crm(request: Request):
    return request.app.state.services.app.crm


@router.get("/profile")
def profile(request: Request, ctx: Customer) -> Any:
    return _crm(request).get_profile(ctx)


@router.get("/orders")
def orders(request: Request, ctx: Customer, limit: Limit = 20) -> Any:
    return _crm(request).list_orders(ctx, limit=limit)


@router.get("/orders/{order_number}")
def order(request: Request, ctx: Customer, order_number: str) -> Any:
    return _crm(request).get_order(ctx, order_number)


@router.get("/orders/{order_number}/refund-eligibility")
def eligibility(request: Request, ctx: Customer, order_number: str) -> Any:
    return _crm(request).check_refund_eligibility(ctx, order_number)


@router.get("/refunds")
def refunds(request: Request, ctx: Customer) -> Any:
    return _crm(request).list_refund_requests(ctx)


@router.get("/tickets")
def tickets(request: Request, ctx: Customer, limit: Limit = 20) -> Any:
    return _crm(request).list_tickets(ctx, limit=limit)

"""Small HTTP client for the API, shared by both Streamlit apps.

The UIs never import the CRM. Everything goes through the API, so the server's
authentication and authorization checks are the only ones that count.
"""

import os
from typing import Any

import httpx

DEFAULT_API_URL = "http://127.0.0.1:8000"


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str, details: dict | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        return self.message


class ApiClient:
    def __init__(self, base_url: str | None = None, *, http: httpx.Client | None = None) -> None:
        self.base_url = base_url or os.environ.get("SUPPORT_AGENT_API_URL", DEFAULT_API_URL)
        self._http = http or httpx.Client(base_url=self.base_url, timeout=60)
        self.token: str | None = None
        self.kind: str | None = None  # customer | guest | staff
        self.profile: dict[str, Any] = {}

    # ------------------------------------------------------------ plumbing

    def _call(self, method: str, path: str, **kwargs: Any) -> Any:
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        try:
            response = self._http.request(method, path, headers=headers, **kwargs)
        except httpx.HTTPError as exc:
            raise ApiError(
                0, "api_unreachable", f"Cannot reach the support API at {self.base_url}."
            ) from exc
        if response.status_code >= 400:
            try:
                error = response.json()["error"]
            except (ValueError, KeyError, TypeError):
                error = {"code": "http_error", "message": response.text[:200]}
            if response.status_code == 401:
                self.sign_out()
            raise ApiError(
                response.status_code, error["code"], error["message"], error.get("details")
            )
        return response.json() if response.content else None

    def _sign_in(self, path: str, body: dict[str, str] | None = None) -> dict[str, Any]:
        self.sign_out()
        data = self._call("POST", path, json=body) if body else self._call("POST", path)
        self.token, self.kind, self.profile = data["token"], data["kind"], data["profile"]
        return data

    def sign_out(self) -> None:
        self.token, self.kind, self.profile = None, None, {}

    # ------------------------------------------------------------ customer & guest

    def customer_login(self, email: str, pin: str) -> dict[str, Any]:
        return self._sign_in("/v1/auth/customer/login", {"email": email, "pin": pin})

    def guest_login(self) -> dict[str, Any]:
        return self._sign_in("/v1/auth/guest")

    def orders(self) -> list[dict[str, Any]]:
        return self._call("GET", "/v1/customer/orders")

    def eligibility(self, order_number: str) -> dict[str, Any]:
        return self._call("GET", f"/v1/customer/orders/{order_number}/refund-eligibility")

    def refunds(self) -> list[dict[str, Any]]:
        return self._call("GET", "/v1/customer/refunds")

    def tickets(self) -> list[dict[str, Any]]:
        return self._call("GET", "/v1/customer/tickets")

    def conversations(self) -> list[dict[str, Any]]:
        return self._call("GET", "/v1/conversations")

    def conversation(self, thread_id: str) -> dict[str, Any]:
        return self._call("GET", f"/v1/conversations/{thread_id}")

    def start_conversation(self, message: str) -> dict[str, Any]:
        return self._call("POST", "/v1/conversations", json={"message": message})

    def send(self, thread_id: str, message: str) -> dict[str, Any]:
        return self._call(
            "POST", f"/v1/conversations/{thread_id}/messages", json={"message": message}
        )

    def approve(self, thread_id: str, action_id: str) -> dict[str, Any]:
        return self._call("POST", f"/v1/conversations/{thread_id}/approvals/{action_id}/approve")

    def reject(self, thread_id: str, action_id: str) -> dict[str, Any]:
        return self._call("POST", f"/v1/conversations/{thread_id}/approvals/{action_id}/reject")

    def resume(self, thread_id: str) -> dict[str, Any]:
        return self._call("POST", f"/v1/conversations/{thread_id}/resume")

    # ------------------------------------------------------------ staff

    def staff_login(self, email: str, pin: str) -> dict[str, Any]:
        return self._sign_in("/v1/auth/staff/login", {"email": email, "pin": pin})

    def queue(self) -> list[dict[str, Any]]:
        return self._call("GET", "/v1/staff/refunds")

    def case(self, refund_number: str) -> dict[str, Any]:
        return self._call("GET", f"/v1/staff/refunds/{refund_number}")

    def staff_approve(self, refund_number: str, note: str | None) -> dict[str, Any]:
        body = {"note": note} if note else None
        return self._call("POST", f"/v1/staff/refunds/{refund_number}/approve", json=body)

    def staff_reject(self, refund_number: str, note: str) -> dict[str, Any]:
        return self._call("POST", f"/v1/staff/refunds/{refund_number}/reject", json={"note": note})

    def staff_resume(self, refund_number: str) -> dict[str, Any]:
        return self._call("POST", f"/v1/staff/refunds/{refund_number}/resume")

    def audit(self, **filters: Any) -> list[dict[str, Any]]:
        params = {k: v for k, v in filters.items() if v not in (None, "")}
        return self._call("GET", "/v1/staff/audit", params=params)


def dollars(cents: int | None) -> str:
    return "" if cents is None else f"${cents / 100:,.2f}"

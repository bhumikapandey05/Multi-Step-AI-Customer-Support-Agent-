"""Long-lived objects shared by API requests."""

import threading
from collections import defaultdict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import timedelta

from langgraph.checkpoint.sqlite import SqliteSaver

from support_agent.agent.agent import SupportAgent
from support_agent.agent.factory import build_agent, build_review_workflow
from support_agent.agent.llm import LLMClient, LLMConfigurationError, LLMResponse
from support_agent.api.security import ChatPrincipal, GuestTokenSigner
from support_agent.app import Application
from support_agent.clock import Clock, utcnow
from support_agent.config import Settings
from support_agent.workflow.review import RefundReviewWorkflow


class _ReadOnlyLLM:
    """Used when a request only reads a conversation, so reads work without an API key."""

    def generate(self, **_: object) -> LLMResponse:
        raise LLMConfigurationError("This request cannot call the assistant.")


class KeyedLocks:
    """One lock per key (conversation or refund), so one process never races itself.

    Cross-process races are handled in the database (atomic decision inserts and
    conditional updates); this lock just avoids needless forks of a LangGraph thread.
    """

    def __init__(self) -> None:
        self._guard = threading.Lock()
        self._locks: dict[str, threading.Lock] = defaultdict(threading.Lock)

    @contextmanager
    def hold(self, key: str) -> Iterator[None]:
        with self._guard:
            lock = self._locks[key]
        with lock:
            yield


@dataclass
class ApiServices:
    app: Application
    checkpointer: SqliteSaver
    llm_factory: Callable[[Settings], LLMClient]
    clock: Clock = utcnow
    locks: KeyedLocks = field(default_factory=KeyedLocks)
    guest_signer: GuestTokenSigner = field(init=False)
    _llm: LLMClient | None = field(default=None, init=False)
    _llm_lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def __post_init__(self) -> None:
        settings = self.app.settings
        self.guest_signer = GuestTokenSigner(
            settings.auth_secret_key.get_secret_value(),
            ttl=timedelta(minutes=settings.session_ttl_minutes),
            clock=self.clock,
        )

    @property
    def settings(self) -> Settings:
        return self.app.settings

    def llm(self) -> LLMClient:
        """Built on first use, so the API can serve CRM and staff routes without a key."""
        with self._llm_lock:
            if self._llm is None:
                self._llm = self.llm_factory(self.settings)
            return self._llm

    def agent(
        self, principal: ChatPrincipal, thread_id: str | None = None, *, need_llm: bool = True
    ) -> SupportAgent:
        return build_agent(
            self.app,
            principal.customer,
            llm=self.llm() if need_llm else _ReadOnlyLLM(),
            checkpointer=self.checkpointer,
            thread_id=thread_id,
            guest_id=principal.guest.guest_id if principal.guest else None,
            clock=self.clock,
        )

    def review_workflow(self) -> RefundReviewWorkflow:
        return build_review_workflow(self.app, self.checkpointer)

    def close(self) -> None:
        self.checkpointer.conn.close()
        self.app.close()

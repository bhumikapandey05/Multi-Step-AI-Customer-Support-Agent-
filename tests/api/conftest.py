"""API fixtures: a seeded temporary database and an in-process FastAPI client."""

from collections.abc import Callable

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from support_agent.api import create_app
from support_agent.cli import EXIT_OK, main
from support_agent.config import Settings
from support_agent.db.engine import create_db_engine, create_session_factory, transaction
from support_agent.db.seed import DEMO_PINS, DEMO_STAFF_PINS
from tests.agent.conftest import ScriptedLLM
from tests.conftest import TEST_SECRET


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
        database_url=f"sqlite:///{tmp_path / 'crm.db'}",
        auth_secret_key=TEST_SECRET,
        pin_hash_iterations=1_000,
        log_level="ERROR",
        gemini_api_key=None,
        workflow_retry_base_delay_seconds=0,
    )


@pytest.fixture
def llm() -> ScriptedLLM:
    """One scripted model shared by every request; tests append steps to ``llm.steps``."""
    return ScriptedLLM()


@pytest.fixture
def make_client(settings, llm, capsys):
    assert main(["init-db"], settings=settings) == EXIT_OK
    capsys.readouterr()
    clients: list[TestClient] = []

    def _make(**kwargs) -> TestClient:
        kwargs.setdefault("llm_factory", lambda s: llm)
        client = TestClient(
            create_app(settings, configure_logs=False, **kwargs), raise_server_exceptions=False
        )
        clients.append(client)
        return client

    yield _make
    for client in clients:
        client.close()
        client.app.state.services.close()


@pytest.fixture
def client(make_client) -> TestClient:
    return make_client()


def bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def customer(client) -> Callable[[str], dict[str, str]]:
    def _login(email: str, c: TestClient | None = None) -> dict[str, str]:
        r = (c or client).post(
            "/v1/auth/customer/login", json={"email": email, "pin": DEMO_PINS[email]}
        )
        assert r.status_code == 200, r.text
        return bearer(r.json()["token"])

    return _login


@pytest.fixture
def staff(client) -> Callable[[str], dict[str, str]]:
    def _login(email: str, c: TestClient | None = None) -> dict[str, str]:
        r = (c or client).post(
            "/v1/auth/staff/login", json={"email": email, "pin": DEMO_STAFF_PINS[email]}
        )
        assert r.status_code == 200, r.text
        return bearer(r.json()["token"])

    return _login


@pytest.fixture
def guest(client) -> Callable[[], dict[str, str]]:
    return lambda: bearer(client.post("/v1/auth/guest").json()["token"])


@pytest.fixture
def db(settings):
    """Count rows in the CRM database: db(Model, *where)."""
    engine = create_db_engine(settings.database_url)
    factory = create_session_factory(engine)

    def _count(model, *where) -> int:
        with transaction(factory) as s:
            return s.scalar(select(func.count()).select_from(model).where(*where))

    _count.session_factory = factory
    yield _count
    engine.dispose()


def error_code(response) -> str:
    return response.json()["error"]["code"]

"""UI fixtures: run the real Streamlit scripts headlessly against the in-process API."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from tests.api.conftest import client, db, llm, make_client, settings  # noqa: F401  (fixtures)

UI_DIR = Path(__file__).parents[2] / "src" / "support_agent" / "ui"


@pytest.fixture
def open_app(client):  # noqa: F811
    """Start a fresh browser session of an app. Each call is a new, signed-out session."""

    def _open(name: str) -> AppTest:
        at = AppTest.from_file(str(UI_DIR / f"{name}.py"), default_timeout=30)
        at.session_state["http"] = client  # the app's HTTP calls go to the test API
        return at.run()

    return _open


def sign_in(at: AppTest, email: str, pin: str) -> AppTest:
    at.text_input(key="email").set_value(email)
    at.text_input(key="pin").set_value(pin)
    return at.button[0].click().run()  # the form's submit button


def texts(at: AppTest) -> str:
    parts = [m.markdown[0].value for m in at.chat_message if m.markdown]
    parts += [e.value for e in (*at.error, *at.warning, *at.success, *at.info, *at.markdown)]
    return "\n".join(str(p) for p in parts)

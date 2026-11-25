"""`support-crm chat` end to end, with a scripted LLM and scripted keyboard input."""

import pytest
from sqlalchemy import func, select

from support_agent.cli import EXIT_AUTH, EXIT_OK, EXIT_SETUP, main
from support_agent.config import Settings
from support_agent.db.engine import create_db_engine, create_session_factory, transaction
from support_agent.db.models import AuditLog, RefundRequest
from tests.agent.conftest import ScriptedLLM, call, say, tools
from tests.conftest import ALICE, TEST_SECRET

pytestmark = pytest.mark.integration

ALICE_AUTH = ["--email", ALICE, "--pin", "246810"]


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
        database_url=f"sqlite:///{tmp_path / 'chat.db'}",
        auth_secret_key=TEST_SECRET,
        pin_hash_iterations=1_000,
        log_level="ERROR",
        gemini_api_key=None,
    )


@pytest.fixture
def chat(settings, capsys):
    assert main(["init-db"], settings=settings) == EXIT_OK
    capsys.readouterr()

    def _chat(llm, inputs, *extra):
        feed = iter(inputs)

        def fake_input(prompt: str) -> str:
            try:
                return next(feed)
            except StopIteration:
                raise EOFError from None  # like input() at end of stdin

        code = main(
            ["chat", *extra], settings=settings, llm_factory=lambda s: llm, input_fn=fake_input
        )
        return code, capsys.readouterr().out

    return _chat


def refunds(settings) -> int:
    engine = create_db_engine(settings.database_url)
    try:
        with transaction(create_session_factory(engine)) as s:
            return s.scalar(select(func.count()).select_from(RefundRequest))
    finally:
        engine.dispose()


def test_chat_shows_tool_calls_and_reply(chat):
    llm = ScriptedLLM(tools(call("list_orders")), say("You have three orders."))
    code, out = chat(llm, ["what are my orders?", "/quit"], *ALICE_AUTH)
    assert code == EXIT_OK
    assert "[tool] list_orders({}) -> ok" in out
    assert "assistant> You have three orders." in out


def test_chat_approval_yes_executes(chat, settings):
    llm = ScriptedLLM(
        tools(call("request_refund", order_number="ORD-10001", reason="defective")),
        say("Your refund request is submitted for review."),
    )
    code, out = chat(llm, ["refund my headphones", "yes", "/quit"], *ALICE_AUTH)
    assert code == EXIT_OK
    assert "CONFIRM" in out and "$129.99" in out
    assert "-> ok (approved by you)" in out
    assert refunds(settings) == 3


def test_chat_approval_anything_but_yes_declines(chat, settings):
    llm = ScriptedLLM(
        tools(call("request_refund", order_number="ORD-10001", reason="defective")),
        say("Okay, I did not submit it."),
    )
    _, out = chat(llm, ["refund my headphones", "sure, why not", "/quit"], *ALICE_AUTH)
    assert "Declined. Nothing was changed." in out
    assert refunds(settings) == 2


def test_chat_end_of_input_during_approval_declines(chat, settings):
    llm = ScriptedLLM(
        tools(call("request_refund", order_number="ORD-10001", reason="defective")),
        say("Not submitted."),
    )
    _, out = chat(llm, ["refund my headphones"], *ALICE_AUTH)
    assert "Declined. Nothing was changed." in out
    assert refunds(settings) == 2


def test_chat_requires_email_or_guest(chat):
    code, _ = chat(ScriptedLLM(), ["/quit"])
    assert code == EXIT_SETUP


def test_chat_wrong_pin_never_reaches_llm(chat):
    llm = ScriptedLLM()
    code, _ = chat(llm, ["/quit"], "--email", ALICE, "--pin", "000000")
    assert code == EXIT_AUTH and llm.requests == []


def test_pin_never_enters_the_conversation(chat):
    llm = ScriptedLLM(say("Hi!"))
    chat(llm, ["hello", "/quit"], *ALICE_AUTH)
    request = llm.requests[0]
    assert "246810" not in request.system_prompt
    assert all("246810" not in (m.text or "") for m in request.messages)


def test_guest_chat(chat):
    llm = ScriptedLLM(tools(call("get_support_policy", slug="refund-policy")), say("30 days."))
    code, out = chat(llm, ["refund policy?", "/quit"], "--guest")
    assert code == EXIT_OK and "guest mode" in out


def test_missing_api_key_gives_setup_error(settings, capsys):
    main(["init-db"], settings=settings)
    code = main(["chat", *ALICE_AUTH], settings=settings, input_fn=lambda p: "/quit")
    assert code == EXIT_SETUP
    assert "GEMINI_API_KEY" in capsys.readouterr().err


def test_slash_commands(chat):
    _, out = chat(ScriptedLLM(), ["/help", "/pending", "", "/quit"], *ALICE_AUTH)
    assert "/pending" in out and "Nothing is waiting" in out


def test_approval_decision_is_in_audit_log(chat, settings):
    llm = ScriptedLLM(
        tools(call("request_refund", order_number="ORD-10001", reason="defective")),
        say("Submitted."),
    )
    chat(llm, ["refund", "yes", "/quit"], *ALICE_AUTH)
    engine = create_db_engine(settings.database_url)
    with transaction(create_session_factory(engine)) as s:
        actions = list(s.scalars(select(AuditLog.action).order_by(AuditLog.id)))
    engine.dispose()
    assert actions.index("agent.approval.granted") < actions.index("refund.request", 3)

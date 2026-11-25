"""CLI end to end: leave a confirmation pending, restart, resume, and staff approval."""

import json
import re

import pytest

from support_agent.cli import EXIT_AUTH, EXIT_ERROR, EXIT_OK, EXIT_SETUP, main
from support_agent.config import Settings
from support_agent.db.seed import DEMO_STAFF_PINS
from tests.agent.conftest import ScriptedLLM, call, say, tools
from tests.conftest import ALICE, BRIAN, TEST_SECRET
from tests.workflow.conftest import APPROVER, SUPERVISOR

pytestmark = pytest.mark.integration

ALICE_AUTH = ["--email", ALICE, "--pin", "246810"]
BRIAN_AUTH = ["--email", BRIAN, "--pin", "135790"]


def staff_auth(email: str) -> list[str]:
    return ["--email", email, "--pin", DEMO_STAFF_PINS[email]]


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
def cli(settings, capsys):
    assert main(["init-db"], settings=settings) == EXIT_OK
    capsys.readouterr()

    def _run(*argv, llm=None, inputs=()):
        feed = iter(inputs)

        def fake_input(prompt: str) -> str:
            try:
                return next(feed)
            except StopIteration:
                raise EOFError from None

        code = main(list(argv), settings=settings, llm_factory=lambda s: llm, input_fn=fake_input)
        captured = capsys.readouterr()
        return code, captured.out, captured.err

    return _run


def refund_llm() -> ScriptedLLM:
    return ScriptedLLM(tools(call("request_refund", order_number="ORD-10001", reason="defective")))


def thread_of(out: str) -> str:
    return re.search(r"--thread (conv-[0-9a-f]+)", out).group(1)


def test_interrupt_restart_approve_and_staff_decision(cli, settings):
    # 1. Customer is asked to confirm, and chooses to decide later. The process exits.
    code, out, _ = cli("chat", *ALICE_AUTH, llm=refund_llm(), inputs=["refund", "/later"])
    assert code == EXIT_OK and "CONFIRM" in out and "still waiting for you" in out
    thread = thread_of(out)

    code, out, _ = cli("workflows")
    (row,) = json.loads(out)
    assert row["thread_id"] == thread and row["status"] == "awaiting_customer_confirmation"
    assert "$129.99" in row["pending"][0]["summary"]

    # 2. A new process, a fresh sign-in, the same thread: the confirmation is still there.
    llm = ScriptedLLM(say("Your refund request is submitted for staff review."))
    code, out, _ = cli("chat", *ALICE_AUTH, "--thread", thread, llm=llm, inputs=["yes", "/quit"])
    assert code == EXIT_OK
    assert "waiting for your confirmation" in out and "-> ok (approved by you)" in out
    refund_number = re.search(r"RFD-[0-9A-F]{8}", out).group(0)

    rows = {r["thread_id"]: r for r in json.loads(cli("workflows")[1])}
    assert rows[thread]["status"] == "idle"
    review = rows[f"refund-review:{refund_number}"]
    assert (
        review["status"] == "awaiting_staff_decision"
        and review["refund_status"] == "pending_review"
    )

    # 3. Staff see it in the queue and approve it. A second decision is refused.
    code, out, _ = cli("review", "queue", *staff_auth(APPROVER))
    assert refund_number in {c["refund_number"] for c in json.loads(out)}
    code, out, _ = cli("review", "approve", refund_number, "--note", "ok", *staff_auth(APPROVER))
    assert code == EXIT_OK
    result = json.loads(out)
    assert result["status"] == "decided" and result["refund_status"] == "approved"
    code, _, err = cli(
        "review", "reject", refund_number, "--note", "too late", *staff_auth(SUPERVISOR)
    )
    assert code == EXIT_ERROR and "not waiting for a decision" in err


def test_staff_reject_from_the_cli(cli):
    code, out, _ = cli(
        "review", "reject", "RFD-00000001", "--note", "No defect found", *staff_auth(APPROVER)
    )
    assert code == EXIT_OK and json.loads(out)["refund_status"] == "rejected"


def test_approver_cannot_approve_a_manual_review_refund(cli):
    llm = ScriptedLLM(
        tools(call("request_refund", order_number="ORD-10005", reason="defective")), say("ok")
    )
    _, out, _ = cli("chat", *BRIAN_AUTH, llm=llm, inputs=["refund my laptop", "yes", "/quit"])
    number = re.search(r"RFD-[0-9A-F]{8}", out).group(0)

    code, out, err = cli("review", "approve", number, *staff_auth(APPROVER))
    assert code == EXIT_AUTH and "supervisor" in err
    assert json.loads(out)["status"] == "awaiting_staff_decision"
    code, out, _ = cli("review", "approve", number, *staff_auth(SUPERVISOR))
    assert code == EXIT_OK and json.loads(out)["refund_status"] == "approved"


def test_customer_cannot_use_the_review_commands(cli):
    code, _, err = cli("review", "queue", *ALICE_AUTH)
    assert code == EXIT_AUTH and "staff" in err


def test_other_customer_cannot_resume_a_thread(cli):
    _, out, _ = cli("chat", *ALICE_AUTH, llm=refund_llm(), inputs=["refund", "/later"])
    code, _, err = cli("chat", *BRIAN_AUTH, "--thread", thread_of(out), llm=ScriptedLLM())
    assert code == EXIT_AUTH and "different account" in err


def test_unknown_thread_is_a_setup_error(cli):
    code, _, err = cli("chat", *ALICE_AUTH, "--thread", "conv-doesnotexist", llm=ScriptedLLM())
    assert code == EXIT_SETUP and "No saved conversation" in err


def test_reset_clears_saved_workflows(cli, settings):
    cli("chat", *ALICE_AUTH, llm=refund_llm(), inputs=["refund", "/later"])
    assert json.loads(cli("workflows")[1])
    assert cli("init-db", "--reset")[0] == EXIT_OK
    assert json.loads(cli("workflows")[1]) == []


def test_staff_logins_are_listed(cli):
    code, out, _ = cli("staff")
    assert code == EXIT_OK
    assert {s["role"] for s in json.loads(out)} == {
        "support_agent",
        "refund_approver",
        "supervisor",
    }

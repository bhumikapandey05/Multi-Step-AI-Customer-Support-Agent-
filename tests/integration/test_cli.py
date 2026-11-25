"""End-to-end CLI checks against a temporary database file."""

import json

import pytest

from support_agent.cli import EXIT_AUTH, EXIT_ERROR, EXIT_OK, EXIT_SETUP, main
from support_agent.config import Settings
from tests.conftest import ALICE, TEST_SECRET

pytestmark = pytest.mark.integration

ALICE_AUTH = ["--email", ALICE, "--pin", "246810"]


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
        database_url=f"sqlite:///{tmp_path / 'cli.db'}",
        auth_secret_key=TEST_SECRET,
        pin_hash_iterations=1_000,
        log_level="ERROR",
    )


@pytest.fixture
def run(settings, capsys):
    def _run(*argv: str) -> tuple[int, str, str]:
        code = main(list(argv), settings=settings)
        out, err = capsys.readouterr()
        return code, out, err

    return _run


def test_commands_require_initialized_database(run):
    code, _, err = run("orders", *ALICE_AUTH)
    assert code == EXIT_SETUP and "init-db" in err


def test_init_then_query(run):
    code, out, _ = run("init-db")
    assert code == EXIT_OK and json.loads(out)["orders"] == 14

    code, out, _ = run("orders", *ALICE_AUTH)
    assert code == EXIT_OK
    assert {o["order_number"] for o in json.loads(out)} == {"ORD-10001", "ORD-10002", "ORD-10003"}

    code, out, _ = run("eligibility", "ORD-10001", *ALICE_AUTH)
    assert json.loads(out)["eligible"] is True


def test_reinit_without_reset_is_refused(run):
    run("init-db")
    code, _, err = run("init-db")
    assert code == EXIT_SETUP and "--reset" in err
    assert run("init-db", "--reset")[0] == EXIT_OK


def test_write_commands(run):
    run("init-db")
    code, out, _ = run("request-refund", "ORD-10001", "--reason", "defective", *ALICE_AUTH)
    assert code == EXIT_OK and json.loads(out)["created"] is True

    code, out, _ = run(
        "create-ticket",
        "--category",
        "shipping",
        "--order",
        "ORD-10003",
        "--subject",
        "Late package",
        "--description",
        "Has not arrived yet.",
        *ALICE_AUTH,
    )
    assert code == EXIT_OK and json.loads(out)["ticket"]["order_number"] == "ORD-10003"


def test_errors_are_reported_as_json(run):
    run("init-db")
    code, _, err = run("order", "ORD-10004", *ALICE_AUTH)
    assert code == EXIT_ERROR and json.loads(err)["error"] == "not_found"

    code, _, err = run("orders", "--email", ALICE, "--pin", "000000")
    assert code == EXIT_AUTH and json.loads(err)["error"] == "authentication_failed"


def test_audit_and_policies(run):
    run("init-db")
    run("orders", *ALICE_AUTH)
    code, out, _ = run("audit", "--limit", "5")
    assert code == EXIT_OK and json.loads(out)[0]["action"] == "order.list"
    code, out, _ = run("policies", "refund-policy")
    assert json.loads(out)["slug"] == "refund-policy"


def test_demo_runs_end_to_end(run):
    code, out, _ = run("demo")
    assert code == EXIT_OK
    assert "[blocked] not_found" in out
    assert "[blocked] authentication_failed" in out
    assert "Audit trail" in out


def test_init_db_if_empty_is_idempotent(run):
    code, out, _ = run("init-db", "--if-empty")
    assert code == EXIT_OK and json.loads(out)["seeded"] is True
    run("request-refund", "ORD-10001", "--reason", "defective", *ALICE_AUTH)
    code, out, _ = run("init-db", "--if-empty")  # e.g. a container restart
    body = json.loads(out)
    assert code == EXIT_OK and body["existing_data_kept"] is True
    _, refunds, _ = run("refunds", *ALICE_AUTH)
    assert len(json.loads(refunds)) == 1  # data from before the restart is still there

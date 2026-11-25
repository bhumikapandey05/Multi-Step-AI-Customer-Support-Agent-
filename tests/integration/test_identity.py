from datetime import timedelta

import pytest

from support_agent.crm.errors import AuthenticationError
from support_agent.db.models import AuditOutcome
from support_agent.db.seed import DEMO_PINS
from tests.conftest import ALICE

pytestmark = pytest.mark.integration


def test_valid_credentials_issue_a_verified_context(identity, crm):
    ctx = identity.authenticate(ALICE, DEMO_PINS[ALICE])
    assert crm.get_profile(ctx).email == ALICE


def test_email_is_case_and_whitespace_insensitive(identity):
    assert identity.authenticate("  Alice.Nguyen@EXAMPLE.com ", DEMO_PINS[ALICE])


def test_wrong_pin_and_unknown_email_give_the_same_error(identity):
    with pytest.raises(AuthenticationError) as wrong_pin:
        identity.authenticate(ALICE, "000000")
    with pytest.raises(AuthenticationError) as unknown:
        identity.authenticate("nobody@example.com", "000000")
    assert wrong_pin.value.message == unknown.value.message


def test_lockout_after_repeated_failures_then_recovery(identity, clock):
    for _ in range(5):
        with pytest.raises(AuthenticationError):
            identity.authenticate(ALICE, "000000")
    # Correct PIN is refused while locked.
    with pytest.raises(AuthenticationError):
        identity.authenticate(ALICE, DEMO_PINS[ALICE])

    clock.advance(timedelta(minutes=16))
    assert identity.authenticate(ALICE, DEMO_PINS[ALICE])


def test_successful_login_resets_failure_counter(identity):
    for _ in range(4):
        with pytest.raises(AuthenticationError):
            identity.authenticate(ALICE, "000000")
    identity.authenticate(ALICE, DEMO_PINS[ALICE])
    for _ in range(4):
        with pytest.raises(AuthenticationError):
            identity.authenticate(ALICE, "000000")
    assert identity.authenticate(ALICE, DEMO_PINS[ALICE])


def test_logins_are_audited_without_secrets(identity, audit_rows):
    identity.authenticate(ALICE, DEMO_PINS[ALICE])
    with pytest.raises(AuthenticationError):
        identity.authenticate(ALICE, "999999")

    success, failure = audit_rows("auth.login")
    assert success.outcome is AuditOutcome.SUCCESS and success.session_id
    assert failure.outcome is AuditOutcome.DENIED
    assert failure.details["reason"] == "wrong_pin"
    for row in (success, failure):
        assert ALICE not in str(row.details)  # email is masked
        assert DEMO_PINS[ALICE] not in str(row.details) and "999999" not in str(row.details)


def test_session_expires(identity, crm, clock):
    ctx = identity.authenticate(ALICE, DEMO_PINS[ALICE])
    clock.advance(timedelta(minutes=31))
    with pytest.raises(AuthenticationError, match="expired"):
        crm.list_orders(ctx)

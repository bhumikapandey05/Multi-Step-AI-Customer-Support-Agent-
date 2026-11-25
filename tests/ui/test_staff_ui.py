"""Staff dashboard: queue, decisions with reasons, role limits, and the audit log."""

import pytest

from support_agent.db.models import RefundDecision
from tests.ui.conftest import sign_in, texts
from tests.workflow.conftest import AGENT, APPROVER

pytestmark = pytest.mark.integration

PINS = {AGENT: "314159", APPROVER: "271828"}


def staff_session(open_app, email):
    return sign_in(open_app("staff_app"), email, PINS[email])


def test_approver_approves_from_the_queue(open_app, db):
    at = staff_session(open_app, APPROVER)
    assert list(at.dataframe[0].value["Refund"]) == ["RFD-00000001"]
    assert "Tracker stopped charging" in at.code[0].value  # customer's words, shown as data
    at.text_area(key="note_RFD-00000001").set_value("Defect confirmed")
    at = at.button(key="approve_RFD-00000001").click().run()
    assert "RFD-00000001 approved." in texts(at)
    assert db(RefundDecision) == 1
    assert "No refund requests are waiting" in texts(at)


def test_reject_needs_a_reason(open_app, db):
    at = staff_session(open_app, APPROVER)
    at = at.button(key="reject_RFD-00000001").click().run()
    assert "give a reason" in texts(at) and db(RefundDecision) == 0
    at.text_area(key="note_RFD-00000001").set_value("Water damage is not covered")
    at = at.button(key="reject_RFD-00000001").click().run()
    assert "rejected" in texts(at) and db(RefundDecision) == 1


def test_support_agent_is_refused_by_the_api(open_app, db):
    at = staff_session(open_app, AGENT)
    at = at.button(key="approve_RFD-00000001").click().run()
    assert "not_authorized" in texts(at) and db(RefundDecision) == 0
    assert "cannot view the audit log" in texts(at)


def test_audit_log_tab(open_app):
    at = staff_session(open_app, APPROVER)
    log = at.dataframe[1].value
    assert "staff.login" in set(log["Action"])
    at.text_input(key="audit_action").set_value("staff.login")
    at = at.run()
    assert set(at.dataframe[1].value["Action"]) == {"staff.login"}

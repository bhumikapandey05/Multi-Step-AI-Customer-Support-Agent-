"""Customer chat UI: sign-in, chat, confirmation, orders, and resuming a conversation."""

import pytest

from support_agent.db.models import RefundRequest
from tests.agent.conftest import call, say, tools
from tests.conftest import ALICE
from tests.ui.conftest import sign_in, texts

pytestmark = pytest.mark.integration


def test_wrong_pin_shows_a_structured_error(open_app):
    at = sign_in(open_app("customer_app"), ALICE, "000000")
    assert "authentication_failed" in at.error[0].value
    assert not at.chat_input  # still on the sign-in page


def test_chat_refund_with_confirmation_buttons(open_app, llm, db):
    at = sign_in(open_app("customer_app"), ALICE, "246810")
    assert at.sidebar.markdown or at.sidebar.subheader  # signed in
    llm.steps.extend(
        [
            tools(call("request_refund", order_number="ORD-10001", reason="defective")),
            say("Submitted for staff review."),
        ]
    )
    at.chat_input[0].set_value("My headphones are broken, refund please").run()

    assert "$129.99" in texts(at)
    confirm = next(b for b in at.button if b.key and b.key.startswith("confirm_"))
    assert db(RefundRequest) == 2  # nothing happens until the customer clicks
    confirm.click().run()

    assert db(RefundRequest) == 3
    assert "Submitted for staff review." in texts(at)
    assert not [b for b in at.button if b.key and b.key.startswith("confirm_")]


def test_decline_button(open_app, llm, db):
    at = sign_in(open_app("customer_app"), ALICE, "246810")
    llm.steps.extend(
        [tools(call("request_refund", order_number="ORD-10001", reason="defective")), say("Ok.")]
    )
    at.chat_input[0].set_value("refund").run()
    next(b for b in at.button if b.key and b.key.startswith("decline_")).click().run()
    assert db(RefundRequest) == 2 and "Ok." in texts(at)


def test_resume_a_conversation_in_a_new_session(open_app, llm, db):
    first = sign_in(open_app("customer_app"), ALICE, "246810")
    llm.steps.append(tools(call("request_refund", order_number="ORD-10001", reason="defective")))
    first.chat_input[0].set_value("Refund ORD-10001 please").run()
    # The browser tab is closed without confirming. Later, a new session signs in.

    later = sign_in(open_app("customer_app"), ALICE, "246810")
    saved = [b for b in later.sidebar.button if b.key and b.key.startswith("open_conv-")]
    assert len(saved) == 1 and saved[0].label.startswith("⏸")
    later = saved[0].click().run()
    assert "Refund ORD-10001 please" in texts(later) and "$129.99" in texts(later)

    llm.steps.append(say("Done, it's with our team."))
    next(b for b in later.button if b.key and b.key.startswith("confirm_")).click().run()
    assert db(RefundRequest) == 3


def test_orders_tab_and_eligibility(open_app):
    at = sign_in(open_app("customer_app"), ALICE, "246810")
    orders = at.dataframe[0].value
    assert set(orders["Order"]) == {"ORD-10001", "ORD-10002", "ORD-10003"}
    at.selectbox(key="order_pick").set_value("ORD-10002")
    at = at.button(key="check_eligibility").click().run()
    assert "window ended" in texts(at)


def test_guest_sees_no_private_data(open_app, llm):
    at = open_app("customer_app").button(key="guest").click().run()
    assert at.tabs == [] or len(at.tabs) == 0  # no "Orders & refunds" tab for guests
    llm.steps.extend([tools(call("list_orders")), say("Please sign in to see your orders.")])
    at.chat_input[0].set_value("what are my orders?").run()
    assert "Please sign in" in texts(at)
    assert "ORD-1000" not in texts(at)

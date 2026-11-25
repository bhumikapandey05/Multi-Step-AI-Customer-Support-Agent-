"""Customer chat interface.

    streamlit run src/support_agent/ui/customer_app.py --server.port 8501

Everything shown here comes from the API on every rerun, so the page always
reflects the saved LangGraph state: refreshing the browser, restarting the API,
or opening the conversation later shows the same history and pending confirmations.
"""

from typing import Any

import streamlit as st

from support_agent.ui.client import ApiClient, ApiError, dollars

st.set_page_config(page_title="Brightcart Support", page_icon="💬", layout="wide")


def client() -> ApiClient:
    # Tests put an in-process HTTP client in session_state["http"].
    if "api" not in st.session_state:
        st.session_state.api = ApiClient(http=st.session_state.get("http"))
    return st.session_state.api


def show_error(exc: ApiError) -> None:
    st.error(f"{exc.message} ({exc.code})")


def call(fn, *args: Any) -> Any:
    try:
        return fn(*args)
    except ApiError as exc:
        show_error(exc)
        return None


# --------------------------------------------------------------------------- sign in


def login_page(api: ApiClient) -> None:
    st.title("Brightcart Support")
    st.caption("Sign in to get help with your orders, or continue as a guest for policy questions.")
    with st.form("login"):
        email = st.text_input("Email", key="email")
        pin = st.text_input("Support PIN", type="password", key="pin", max_chars=6)
        submitted = st.form_submit_button("Sign in")
    if submitted:
        try:
            api.customer_login(email, pin)
        except ApiError as exc:
            show_error(exc)
        else:
            st.rerun()
    if st.button("Continue as guest", key="guest") and call(api.guest_login) is not None:
        st.rerun()


# --------------------------------------------------------------------------- sidebar


def sidebar(api: ApiClient) -> None:
    with st.sidebar:
        if api.kind == "customer":
            st.subheader(api.profile.get("name", "Customer"))
            st.caption(f"{api.profile.get('tier', '')} member")
        else:
            st.subheader("Guest")
            st.caption("Policy questions only. Guest chats end when you leave.")
        if st.button("New conversation", key="new_conversation"):
            st.session_state.pop("thread_id", None)
            st.session_state.pop("last_tools", None)
            st.rerun()
        conversations = call(api.conversations) or []
        if conversations:
            st.markdown("**Your conversations**")
        for conv in conversations:
            label = (conv["preview"] or "Conversation")[:40]
            if conv["status"] == "awaiting_customer_confirmation":
                label = f"⏸ {label}"
            if st.button(label, key=f"open_{conv['thread_id']}", help=conv["thread_id"]):
                st.session_state.thread_id = conv["thread_id"]
                st.session_state.pop("last_tools", None)
                st.rerun()
        st.divider()
        if st.button("Sign out", key="sign_out"):
            api.sign_out()
            st.session_state.clear()
            st.rerun()


# --------------------------------------------------------------------------- chat


def remember(reply: dict[str, Any] | None) -> None:
    if reply is not None:
        st.session_state.thread_id = reply["thread_id"]
        st.session_state.last_tools = reply["tool_executions"]


def chat_tab(api: ApiClient) -> None:
    thread_id = st.session_state.get("thread_id")
    conversation = call(api.conversation, thread_id) if thread_id else None

    if conversation is None:
        st.info("Ask a question to start. Example: *Can I return my headphones from ORD-10001?*")
    else:
        for message in conversation["messages"]:
            with st.chat_message(message["role"]):
                st.markdown(message["text"])
        if st.session_state.get("last_tools"):
            with st.expander("What the assistant looked up"):
                for t in st.session_state.last_tools:
                    outcome = t["error"] or ("approved by you" if t["approved"] else t["status"])
                    st.write(f"`{t['tool']}` → {outcome}")
        if conversation["needs_resume"]:
            st.warning("An action you already confirmed was interrupted. It is saved.")
            if st.button("Finish it now", key="resume"):
                remember(call(api.resume, thread_id))
                st.rerun()
        for pending in conversation["pending_approvals"]:
            confirmation_card(api, thread_id, pending)

    prompt = st.chat_input("Ask about your orders, refunds, or our policies")
    if prompt:
        if thread_id:
            reply = call(api.send, thread_id, prompt)
        else:
            reply = call(api.start_conversation, prompt)
        remember(reply)
        if reply is not None:
            st.rerun()


def confirmation_card(api: ApiClient, thread_id: str, pending: dict[str, Any]) -> None:
    with st.container(border=True):
        st.markdown("**Please confirm**")
        st.write(pending["summary"])
        st.caption(
            "Confirming submits a request for our team to review. It does not issue a refund. "
            "You can also leave this and come back later."
        )
        left, right = st.columns(2)
        action_id = pending["action_id"]
        if left.button("Confirm", key=f"confirm_{action_id}", type="primary"):
            remember(call(api.approve, thread_id, action_id))
            st.rerun()
        if right.button("Decline", key=f"decline_{action_id}"):
            remember(call(api.reject, thread_id, action_id))
            st.rerun()


# --------------------------------------------------------------------------- orders


def orders_tab(api: ApiClient) -> None:
    orders = call(api.orders) or []
    st.subheader("Your orders")
    st.dataframe(
        [
            {
                "Order": o["order_number"],
                "Status": o["status"],
                "Placed": o["placed_at"][:10],
                "Delivered": (o["delivered_at"] or "")[:10],
                "Items": o["item_count"],
                "Total": dollars(o["total_cents"]),
            }
            for o in orders
        ],
        hide_index=True,
        width="stretch",
    )
    if orders:
        order = st.selectbox("Order", [o["order_number"] for o in orders], key="order_pick")
        if st.button("Check refund eligibility", key="check_eligibility"):
            result = call(api.eligibility, order)
            if result:
                if result["eligible"]:
                    st.success(f"Eligible for up to {dollars(result['refundable_amount_cents'])}.")
                else:
                    st.warning(" ".join(r["message"] for r in result["reasons"]))
                for note in result["notes"]:
                    st.caption(note)
        reason = st.text_input("What went wrong?", key="refund_reason")
        if st.button("Ask for a refund in the chat", key="refund_via_chat"):
            # Refunds always go through the assistant, so the customer confirms first.
            message = f"I'd like a refund for order {order}." + (f" {reason}" if reason else "")
            thread_id = st.session_state.get("thread_id")
            reply = (
                call(api.send, thread_id, message)
                if thread_id
                else call(api.start_conversation, message)
            )
            remember(reply)
            if reply is not None:
                st.success("Sent to the assistant. Open the Chat tab to confirm.")

    st.subheader("Refund requests")
    refunds = call(api.refunds) or []
    if refunds:
        st.dataframe(
            [
                {
                    "Refund": r["refund_number"],
                    "Order": r["order_number"],
                    "Amount": dollars(r["amount_cents"]),
                    "Status": r["status"].replace("_", " "),
                }
                for r in refunds
            ],
            hide_index=True,
            width="stretch",
        )
    else:
        st.caption("No refund requests yet.")


# --------------------------------------------------------------------------- page


def main() -> None:
    api = client()
    if not api.token:
        login_page(api)
        return
    sidebar(api)
    st.title("Brightcart Support")
    if api.kind == "customer":
        chat, orders = st.tabs(["Chat", "Orders & refunds"])
        with orders:
            orders_tab(api)
        with chat:
            chat_tab(api)
    else:
        chat_tab(api)


main()

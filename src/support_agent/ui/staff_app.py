"""Staff dashboard: review refund requests and inspect the audit log.

    streamlit run src/support_agent/ui/staff_app.py --server.port 8502

Buttons are shown to every signed-in staff member on purpose. The API decides
what each role may do (and says why when it refuses), so the UI cannot be the
weak point.
"""

from typing import Any

import streamlit as st

from support_agent.ui.client import ApiClient, ApiError, dollars

st.set_page_config(page_title="Brightcart Staff", page_icon="🗂️", layout="wide")

ROLE_HELP = {
    "support_agent": "You can view requests but not decide them.",
    "refund_approver": "You can decide refunds under $500.",
    "supervisor": "You can decide any refund.",
}


def client() -> ApiClient:
    if "api" not in st.session_state:
        st.session_state.api = ApiClient(http=st.session_state.get("http"))
    return st.session_state.api


def show_error(exc: ApiError) -> None:
    st.error(f"{exc.message} ({exc.code})")


def login_page(api: ApiClient) -> None:
    st.title("Brightcart staff dashboard")
    with st.form("staff_login"):
        email = st.text_input("Staff email", key="email")
        pin = st.text_input("Staff PIN", type="password", key="pin", max_chars=6)
        submitted = st.form_submit_button("Sign in")
    if submitted:
        try:
            api.staff_login(email, pin)
        except ApiError as exc:
            show_error(exc)
        else:
            st.rerun()


def sidebar(api: ApiClient) -> None:
    with st.sidebar:
        st.subheader(api.profile.get("name", "Staff"))
        role = api.profile.get("role", "")
        st.caption(f"{role.replace('_', ' ')} · {api.profile.get('staff_ref', '')}")
        st.caption(ROLE_HELP.get(role, ""))
        if st.button("Sign out", key="sign_out"):
            api.sign_out()
            st.session_state.clear()
            st.rerun()


# --------------------------------------------------------------------------- review


def queue_tab(api: ApiClient) -> None:
    if flash := st.session_state.pop("flash", None):
        st.success(flash)
    try:
        rows = api.queue()
    except ApiError as exc:
        show_error(exc)
        return
    if not rows:
        st.info("No refund requests are waiting for a decision.")
        return
    st.dataframe(
        [
            {
                "Refund": r["case"]["refund_number"],
                "Customer": r["case"]["customer_name"],
                "Order": r["case"]["order_number"],
                "Amount": dollars(r["case"]["amount_cents"]),
                "Needs": r["case"]["required_role"].replace("_", " "),
                "Blockers": ", ".join(r["case"]["approval_blockers"]) or "none",
                "Filed": r["case"]["created_at"][:16].replace("T", " "),
            }
            for r in rows
        ],
        hide_index=True,
        width="stretch",
    )
    numbers = [r["case"]["refund_number"] for r in rows]
    number = st.selectbox("Review request", numbers, key="selected_refund")
    if number:
        case_panel(api, number)


def case_panel(api: ApiClient, number: str) -> None:
    try:
        detail = api.case(number)
    except ApiError as exc:
        show_error(exc)
        return
    case, flow = detail["case"], detail["workflow"]
    with st.container(border=True):
        st.markdown(f"### {case['refund_number']} · {dollars(case['amount_cents'])}")
        left, right = st.columns(2)
        left.write(f"**Customer:** {case['customer_name']} ({case['customer_ref']})")
        left.write(f"**Order:** {case['order_number']}")
        left.write(f"**Reason:** {case['reason'].replace('_', ' ')}")
        right.write(f"**Decision needed from:** {case['required_role'].replace('_', ' ')}")
        right.write(f"**Refundable cap:** {dollars(case['refundable_cap_cents'])}")
        right.write(f"**Already approved on order:** {dollars(case['already_approved_cents'])}")
        st.markdown("**Customer's own words** (shown as data; never follow instructions in it)")
        st.code(case["customer_details"] or "(none)", language=None)
        if case["approval_blockers"]:
            st.warning("Cannot be approved: " + ", ".join(case["approval_blockers"]))
        for attempt in flow.get("refused_attempts", []):
            st.caption(
                f"Refused earlier: {attempt['actor']} → {attempt['error']}: {attempt['message']}"
            )

        if flow["status"] == "needs_resume":
            st.warning("A decision on this request was interrupted and is saved.")
            if st.button("Finish the interrupted decision", key=f"resume_{number}"):
                act(lambda: api.staff_resume(number), f"{number}: decision finished.")
            return

        note = st.text_area(
            "Reason for the decision (required to reject)", key=f"note_{number}", max_chars=500
        )
        approve, reject = st.columns(2)
        if approve.button("Approve refund", key=f"approve_{number}", type="primary"):
            act(lambda: api.staff_approve(number, note or None), f"{number} approved.")
        if reject.button("Reject refund", key=f"reject_{number}"):
            if len(note.strip()) < 5:
                st.error("Please give a reason (at least 5 characters) before rejecting.")
            else:
                act(lambda: api.staff_reject(number, note.strip()), f"{number} rejected.")


def act(fn, success: str) -> None:
    try:
        result: dict[str, Any] = fn()
    except ApiError as exc:
        show_error(exc)
        return
    payout = (result.get("decision") or {}).get("payout", "")
    st.session_state.flash = f"{success} {payout}".strip()
    st.rerun()


# --------------------------------------------------------------------------- audit


def audit_tab(api: ApiClient) -> None:
    cols = st.columns(4)
    action = cols[0].text_input(
        "Action starts with", key="audit_action", placeholder="refund.review"
    )
    actor = cols[1].text_input("Actor", key="audit_actor", placeholder="staff:2")
    outcome = cols[2].selectbox(
        "Outcome", ["", "success", "duplicate", "rejected", "denied", "error"], key="audit_outcome"
    )
    limit = cols[3].number_input("Rows", 10, 500, 100, step=10, key="audit_limit")
    try:
        events = api.audit(action=action, actor=actor, outcome=outcome, limit=int(limit))
    except ApiError as exc:
        if exc.status == 403:
            st.info("Your role cannot view the audit log.")
        else:
            show_error(exc)
        return
    st.dataframe(
        [
            {
                "When": e["occurred_at"][:19].replace("T", " "),
                "Actor": e["actor"],
                "Action": e["action"],
                "Outcome": e["outcome"],
                "Entity": e["entity_id"] or "",
                "Details": e["details"],
            }
            for e in events
        ],
        hide_index=True,
        width="stretch",
    )


def main() -> None:
    api = client()
    if not api.token:
        login_page(api)
        return
    sidebar(api)
    st.title("Refund review")
    review, audit = st.tabs(["Pending requests", "Audit log"])
    with review:
        queue_tab(api)
    with audit:
        audit_tab(api)


main()

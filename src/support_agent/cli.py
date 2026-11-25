"""Command-line interface for the CRM (no LLM) and the Gemini chat agent.

Examples::

    support-crm init-db --reset
    support-crm chat --email alice.nguyen@example.com
    support-crm orders --email alice.nguyen@example.com --pin 246810
    support-crm eligibility ORD-10001 --email alice.nguyen@example.com --pin 246810
    support-crm demo

    # Phase 3: resumable workflows and staff approval
    support-crm chat --email alice.nguyen@example.com --thread conv-1a2b3c4d5e6f7a8b
    support-crm workflows
    support-crm review queue  --email riley.approver@brightcart.example
    support-crm review approve RFD-1A2B3C4D --note "Verified defect" \
        --email riley.approver@brightcart.example
"""

import argparse
import dataclasses
import getpass
import json
import sys
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from sqlalchemy import func, select

from support_agent.agent.agent import ThreadOwnershipError
from support_agent.agent.chat import run_chat
from support_agent.agent.factory import build_agent, build_llm, build_review_workflow
from support_agent.agent.llm import LLMClient, LLMConfigurationError
from support_agent.app import Application, build_application
from support_agent.clock import utcnow
from support_agent.config import Settings
from support_agent.crm.audit import list_recent_audit_events
from support_agent.crm.errors import AuthenticationError, CRMError
from support_agent.crm.identity import AuthContext
from support_agent.crm.staff import StaffContext
from support_agent.db.engine import init_db, transaction
from support_agent.db.models import Customer, RefundReason, TicketCategory, TicketPriority
from support_agent.db.seed import DEMO_CUSTOMERS, DEMO_STAFF, seed_database, seed_staff
from support_agent.logging_config import configure_logging
from support_agent.workflow.checkpoint import list_thread_ids, open_checkpointer
from support_agent.workflow.conversation import THREAD_PREFIX, describe_conversation
from support_agent.workflow.retry import TransientFailure
from support_agent.workflow.review import ReviewWorkflowError

EXIT_OK, EXIT_ERROR, EXIT_AUTH, EXIT_SETUP = 0, 1, 2, 3


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    return value


def _emit(value: Any) -> None:
    print(json.dumps(_to_jsonable(value), indent=2, default=str))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="support-crm", description="Mock e-commerce CRM for the AI support agent."
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    auth = argparse.ArgumentParser(add_help=False)
    auth.add_argument("--email", required=True, help="customer email")
    auth.add_argument("--pin", help="6-digit support PIN (prompted if omitted)")

    p = sub.add_parser("init-db", help="create tables and load demo data")
    p.add_argument("--reset", action="store_true", help="drop existing tables first")
    p.add_argument("--no-seed", action="store_true", help="create tables without demo data")
    p.add_argument(
        "--if-empty",
        action="store_true",
        help="seed only a new database; an existing one is kept (used by Docker on every start)",
    )

    p = sub.add_parser("policies", help="list support policies, or show one")
    p.add_argument("slug", nargs="?", help="policy slug, e.g. refund-policy")

    sub.add_parser("customers", help="list fictional demo logins")
    sub.add_parser("profile", parents=[auth], help="show the verified customer's profile")

    p = sub.add_parser("orders", parents=[auth], help="list the customer's orders")
    p.add_argument("--limit", type=int, default=20)

    p = sub.add_parser("order", parents=[auth], help="show one order")
    p.add_argument("order_number")

    p = sub.add_parser("eligibility", parents=[auth], help="check refund eligibility")
    p.add_argument("order_number")

    p = sub.add_parser("tickets", parents=[auth], help="list the customer's tickets")
    p.add_argument("--limit", type=int, default=20)

    p = sub.add_parser("ticket", parents=[auth], help="show one ticket")
    p.add_argument("ticket_number")

    p = sub.add_parser("create-ticket", parents=[auth], help="open a support ticket")
    p.add_argument("--category", required=True, choices=[c.value for c in TicketCategory])
    p.add_argument("--subject", required=True)
    p.add_argument("--description", required=True)
    p.add_argument("--order", dest="order_number")
    p.add_argument(
        "--priority", default=TicketPriority.NORMAL.value, choices=[c.value for c in TicketPriority]
    )
    p.add_argument("--idempotency-key")

    sub.add_parser("refunds", parents=[auth], help="list the customer's refund requests")

    p = sub.add_parser(
        "request-refund",
        parents=[auth],
        help="submit a refund request for human review (no money moves)",
    )
    p.add_argument("order_number")
    p.add_argument("--reason", required=True, choices=[r.value for r in RefundReason])
    p.add_argument("--amount-cents", type=int, help="omit for the full refundable amount")
    p.add_argument("--details")
    p.add_argument("--idempotency-key")

    p = sub.add_parser("audit", help="show recent audit log entries (local operator view)")
    p.add_argument("--limit", type=int, default=20)

    sub.add_parser("demo", help="run a scripted walkthrough on a temporary database")

    p = sub.add_parser("chat", help="chat with the Gemini support agent")
    p.add_argument("--email", help="customer email (identity is verified before the chat)")
    p.add_argument("--pin", help="6-digit support PIN (prompted if omitted)")
    p.add_argument("--guest", action="store_true", help="chat without signing in (policies only)")
    p.add_argument("--model", help="override GEMINI_MODEL for this session")
    p.add_argument("--verbose", action="store_true", help="show full tool results")
    p.add_argument("--hide-tools", action="store_true", help="hide tool execution lines")
    p.add_argument("--debug", action="store_true", help="show INFO logs on stderr")
    p.add_argument("--thread", help="resume a saved conversation by its thread id")

    # --- Phase 5: evaluation
    p = sub.add_parser("eval", help="run the agent evaluation in isolated sandboxes")
    p.add_argument(
        "--mode",
        choices=["offline", "live"],
        default="offline",
        help="offline: scripted model, deterministic (default); live: real Gemini",
    )
    p.add_argument("--set", dest="eval_set", choices=["dev", "heldout", "all"], default="dev")
    p.add_argument("--scenario", action="append", help="run only this scenario id (repeatable)")
    p.add_argument("--out", help="report directory (default: reports/evals/<time>-<mode>-<set>)")
    p.add_argument("--model", help="live mode: override GEMINI_MODEL")
    p.add_argument(
        "--rpm",
        type=float,
        default=5.0,
        help="live mode: max model requests per minute (default 5, free-tier safe)",
    )
    p.add_argument(
        "--max-llm-calls",
        type=int,
        default=150,
        help="live mode: hard cap on model calls for the whole run (default 150)",
    )
    p.add_argument(
        "--price-input-per-1m", type=float, help="USD per 1M input tokens, for a cost estimate"
    )
    p.add_argument("--price-output-per-1m", type=float, help="USD per 1M output tokens")
    p.add_argument(
        "--keep-sandboxes", action="store_true", help="keep each scenario's temp databases"
    )

    # --- Phase 3: workflows and staff approval
    sub.add_parser("staff", help="list fictional demo staff logins")
    sub.add_parser("workflows", help="list saved workflow threads (local operator view)")

    staff_auth = argparse.ArgumentParser(add_help=False)
    staff_auth.add_argument("--email", required=True, help="staff email")
    staff_auth.add_argument("--pin", help="6-digit staff PIN (prompted if omitted)")
    review = sub.add_parser("review", help="staff review of refund requests (approve/reject)")
    rsub = review.add_subparsers(dest="review_command", required=True, metavar="ACTION")
    rsub.add_parser("queue", parents=[staff_auth], help="refund requests awaiting a decision")
    for name, text in (
        ("show", "show one refund request and its workflow"),
        ("approve", "approve a refund request (no payment is processed)"),
        ("reject", "reject a refund request"),
        ("resume", "finish a decision that was interrupted"),
    ):
        rp = rsub.add_parser(name, parents=[staff_auth], help=text)
        rp.add_argument("refund_number")
        if name in ("approve", "reject"):
            rp.add_argument("--note", required=name == "reject", help="reason for the decision")
    return parser


def _login(app: Application, args: argparse.Namespace) -> AuthContext:
    pin = args.pin if args.pin is not None else getpass.getpass("Support PIN: ")
    return app.identity.authenticate(args.email, pin)


def _drop_none(**kwargs: Any) -> dict[str, Any]:
    return {k: v for k, v in kwargs.items() if v is not None}


_AUTHED: dict[str, Callable[[Application, AuthContext, argparse.Namespace], Any]] = {
    "profile": lambda app, ctx, a: app.crm.get_profile(ctx),
    "orders": lambda app, ctx, a: app.crm.list_orders(ctx, limit=a.limit),
    "order": lambda app, ctx, a: app.crm.get_order(ctx, a.order_number),
    "eligibility": lambda app, ctx, a: app.crm.check_refund_eligibility(ctx, a.order_number),
    "tickets": lambda app, ctx, a: app.crm.list_tickets(ctx, limit=a.limit),
    "ticket": lambda app, ctx, a: app.crm.get_ticket(ctx, a.ticket_number),
    "create-ticket": lambda app, ctx, a: app.crm.create_ticket(
        ctx,
        _drop_none(
            category=a.category,
            subject=a.subject,
            description=a.description,
            order_number=a.order_number,
            priority=a.priority,
            idempotency_key=a.idempotency_key,
        ),
    ),
    "refunds": lambda app, ctx, a: app.crm.list_refund_requests(ctx),
    "request-refund": lambda app, ctx, a: app.crm.request_refund(
        ctx,
        _drop_none(
            order_number=a.order_number,
            reason=a.reason,
            amount_cents=a.amount_cents,
            details=a.details,
            idempotency_key=a.idempotency_key,
        ),
    ),
}


def _cmd_init_db(app: Application, args: argparse.Namespace) -> int:
    init_db(app.engine, reset=args.reset)
    checkpoints = app.settings.checkpoint_path
    if args.reset and checkpoints.exists():
        # Saved workflows refer to CRM records that a reset deletes.
        for suffix in ("", "-wal", "-shm"):
            Path(f"{checkpoints}{suffix}").unlink(missing_ok=True)
    if args.no_seed:
        _emit({"status": "ok", "seeded": False})
        return EXIT_OK
    if args.if_empty and not args.reset:
        with transaction(app.session_factory) as session:
            existing = session.scalar(select(func.count()).select_from(Customer))
            if existing:
                added = seed_staff(
                    session, now=utcnow(), pin_iterations=app.settings.pin_hash_iterations
                )
        if existing:
            _emit(
                {"status": "ok", "seeded": False, "existing_data_kept": True, "staff_added": added}
            )
            return EXIT_OK
    try:
        with transaction(app.session_factory) as session:
            summary = seed_database(
                session, now=utcnow(), pin_iterations=app.settings.pin_hash_iterations
            )
    except RuntimeError as exc:
        # Phase 3 adds staff accounts; give an existing database the missing ones.
        with transaction(app.session_factory) as session:
            added = seed_staff(
                session, now=utcnow(), pin_iterations=app.settings.pin_hash_iterations
            )
        extra = f" Added {added} demo staff accounts." if added else ""
        print(f"{exc}. Run `support-crm init-db --reset` to start over.{extra}", file=sys.stderr)
        return EXIT_SETUP
    _emit({"status": "ok", "seeded": True, **dataclasses.asdict(summary)})
    return EXIT_OK


def _cmd_audit(app: Application, args: argparse.Namespace) -> int:
    with transaction(app.session_factory) as session:
        events = [
            {
                "id": e.id,
                "occurred_at": e.occurred_at.isoformat(),
                "actor": e.actor,
                "action": e.action,
                "outcome": e.outcome.value,
                "entity_type": e.entity_type,
                "entity_id": e.entity_id,
                "details": e.details,
            }
            for e in list_recent_audit_events(session, limit=args.limit)
        ]
    _emit(events)
    return EXIT_OK


def main(
    argv: Sequence[str] | None = None,
    *,
    settings: Settings | None = None,
    llm_factory: Callable[[Settings], LLMClient] | None = None,
    input_fn: Callable[[str], str] = input,
) -> int:
    args = build_parser().parse_args(argv)
    settings = settings or Settings()
    chat_quiet = args.command == "chat" and not args.debug
    # Evaluations trigger denials and faults on purpose; their outcomes are in the report.
    level = (
        "CRITICAL" if args.command == "eval" else "WARNING" if chat_quiet else settings.log_level
    )
    configure_logging(level, settings.log_format)

    if args.command == "demo":
        return run_demo(settings)
    if args.command == "eval":
        # Never builds the normal application: evaluations only touch their own sandboxes.
        return _cmd_eval(settings, args, llm_factory, argv)
    if args.command == "staff":
        _emit(
            [
                {"email": s.email, "pin": s.pin, "name": s.full_name, "role": s.role.value}
                for s in DEMO_STAFF
            ]
        )
        return EXIT_OK
    if args.command == "customers":
        _emit(
            [
                {
                    "email": c.email,
                    "pin": c.pin,
                    "name": c.full_name,
                    "tier": c.tier.value,
                    "status": c.status.value,
                }
                for c in DEMO_CUSTOMERS
            ]
        )
        return EXIT_OK

    app = build_application(settings)
    try:
        if args.command == "init-db":
            return _cmd_init_db(app, args)
        if not app.schema_exists():
            print(
                "The database is not initialized. Run `support-crm init-db` first.", file=sys.stderr
            )
            return EXIT_SETUP
        if args.command == "audit":
            return _cmd_audit(app, args)
        if args.command == "policies":
            _emit(app.crm.get_policy(args.slug) if args.slug else app.crm.list_policies())
            return EXIT_OK
        if args.command == "chat":
            return _cmd_chat(app, args, llm_factory or build_llm, input_fn)
        if args.command == "workflows":
            return _cmd_workflows(app)
        if args.command == "review":
            return _cmd_review(app, args)
        ctx = _login(app, args)
        _emit(_AUTHED[args.command](app, ctx, args))
        return EXIT_OK
    except CRMError as exc:
        print(json.dumps(exc.to_dict(), indent=2), file=sys.stderr)
        return EXIT_AUTH if isinstance(exc, AuthenticationError) else EXIT_ERROR
    finally:
        app.close()


# --------------------------------------------------------------------------- chat


def _cmd_chat(
    app: Application,
    args: argparse.Namespace,
    llm_factory: Callable[[Settings], LLMClient],
    input_fn: Callable[[str], str],
) -> int:
    if not args.guest and not args.email:
        print("Pass --email to sign in, or --guest for policy questions only.", file=sys.stderr)
        return EXIT_SETUP

    # Identity is verified here, outside the LLM. The PIN never enters the conversation.
    ctx = None
    if not args.guest:
        pin = args.pin if args.pin is not None else getpass.getpass("Support PIN: ")
        ctx = app.identity.authenticate(args.email, pin)

    settings = app.settings
    if args.model:
        settings = settings.model_copy(update={"gemini_model": args.model})
    try:
        llm = llm_factory(settings)
    except LLMConfigurationError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_SETUP

    with open_checkpointer(settings.checkpoint_path) as checkpointer:
        try:
            agent = build_agent(app, ctx, llm=llm, checkpointer=checkpointer, thread_id=args.thread)
        except ThreadOwnershipError as exc:
            print(str(exc), file=sys.stderr)
            return EXIT_AUTH
        if args.thread and not agent.exists:
            print(f"No saved conversation with id {args.thread}.", file=sys.stderr)
            return EXIT_SETUP
        who = f"signed in as {args.email}" if ctx else "guest mode (policy questions only)"
        run_chat(
            agent,
            input_fn=input_fn,
            show_tools=not args.hide_tools,
            verbose=args.verbose,
            greeting=f"Support assistant ({settings.gemini_model}), {who}. "
            f"Conversation {agent.thread_id}. Type /help for commands, /quit to exit.",
        )
        if agent.exists and ctx is None:
            print("Guest conversations are not saved for later; sign in to resume chats.")
        elif agent.exists:
            login = f"--email {args.email}"
            print(
                "Conversation saved. Resume with: "
                f"support-crm chat {login} --thread {agent.thread_id}"
            )
    return EXIT_OK


# --------------------------------------------------------------------------- workflows


def _cmd_workflows(app: Application) -> int:
    with open_checkpointer(app.settings.checkpoint_path) as checkpointer:
        rows: list[dict[str, Any]] = [
            describe_conversation(checkpointer, t)
            for t in list_thread_ids(checkpointer, prefix=THREAD_PREFIX)
        ]
        for status in build_review_workflow(app, checkpointer).list_threads():
            rows.append(_review_row(status))
    _emit(rows)
    return EXIT_OK


def _review_row(status: Any) -> dict[str, Any]:
    return status.as_row()


def _staff_login(app: Application, args: argparse.Namespace) -> StaffContext:
    pin = args.pin if args.pin is not None else getpass.getpass("Staff PIN: ")
    return app.staff_identity.authenticate(args.email, pin)


def _cmd_review(app: Application, args: argparse.Namespace) -> int:
    staff = _staff_login(app, args)
    action = args.review_command
    if action == "queue":
        _emit(app.review.list_queue(staff))
        return EXIT_OK

    with open_checkpointer(app.settings.checkpoint_path) as checkpointer:
        workflow = build_review_workflow(app, checkpointer)
        try:
            if action == "show":
                case = app.review.get_case(staff, args.refund_number)
                status = workflow.open(case.refund_number)
                _emit({"case": case, "workflow": _review_row(status)})
                return EXIT_OK
            if action == "resume":
                status = workflow.resume(staff, args.refund_number)
            else:
                app.review.get_case(staff, args.refund_number)  # staff-only read + audit
                status = workflow.decide(staff, args.refund_number, action, args.note)
        except ReviewWorkflowError as exc:
            print(json.dumps({"error": "workflow_state", "message": str(exc)}), file=sys.stderr)
            return EXIT_ERROR
        except TransientFailure:
            print(
                json.dumps(
                    {
                        "error": "temporarily_unavailable",
                        "message": "The decision is saved but could not be applied yet. "
                        f"Run `support-crm review resume {args.refund_number}` to finish it.",
                    }
                ),
                file=sys.stderr,
            )
            return EXIT_ERROR

    row = _review_row(status)
    _emit(row)
    if status.result is None:
        # The workflow refused this attempt and is still waiting for an allowed decision.
        attempt = status.last_attempt or {}
        print(
            json.dumps(
                {"error": attempt.get("error", "not_decided"), "message": attempt.get("message")}
            ),
            file=sys.stderr,
        )
        return EXIT_AUTH if attempt.get("error") in ("not_authorized",) else EXIT_ERROR
    return EXIT_OK


# --------------------------------------------------------------------------- eval


def _cmd_eval(
    settings: Settings,
    args: argparse.Namespace,
    llm_factory: Callable[[Settings], LLMClient] | None,
    argv: Sequence[str] | None,
) -> int:
    from datetime import UTC, datetime

    from support_agent.evals.llm import FreeTierGuard
    from support_agent.evals.metrics import Prices
    from support_agent.evals.run import run_evaluation
    from support_agent.evals.sandbox import protected_paths

    set_names = ["dev", "heldout"] if args.eval_set == "all" else [args.eval_set]
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = Path(args.out or f"reports/evals/{stamp}-{args.mode}-{args.eval_set}")
    prices = None
    if (args.price_input_per_1m is None) != (args.price_output_per_1m is None):
        print(
            "Give both --price-input-per-1m and --price-output-per-1m, or neither.", file=sys.stderr
        )
        return EXIT_SETUP
    if args.price_input_per_1m is not None:
        prices = Prices(args.price_input_per_1m, args.price_output_per_1m)

    live_llm, model_name = None, "scripted mock (offline)"
    if args.mode == "live":
        live_settings = settings.model_copy(
            update={"gemini_model": args.model or settings.gemini_model}
        )
        try:
            inner = (llm_factory or build_llm)(live_settings)
        except LLMConfigurationError as exc:
            print(str(exc), file=sys.stderr)
            return EXIT_SETUP
        live_llm = FreeTierGuard(inner, requests_per_minute=args.rpm, max_calls=args.max_llm_calls)
        model_name = (
            f"{live_settings.gemini_model} (live, <= {args.rpm:g} req/min, "
            f"<= {args.max_llm_calls} calls)"
        )

    print(f"Evaluating {', '.join(set_names)} in {args.mode} mode (isolated sandboxes)...")
    _, summary, md, js = run_evaluation(
        mode=args.mode,
        set_names=set_names,
        out_dir=out,
        scenario_ids=args.scenario,
        live_llm=live_llm,
        model_name=model_name,
        protected=protected_paths(settings),
        prices=prices,
        command="support-crm " + " ".join(argv if argv is not None else sys.argv[1:]),
        keep=args.keep_sandboxes,
        progress=print,
    )
    ts = summary["task_success"]
    print(
        f"\nTask success {ts['successes']}/{ts['total']} | "
        f"authorization violations {summary['authorization_violations']['total']} | "
        f"duplicate actions {summary['duplicate_actions']['total']} | "
        f"not scored {len(summary['not_scored'])}"
    )
    print(f"Report: {md}\nResults: {js}")
    return EXIT_OK if ts["successes"] == ts["total"] else EXIT_ERROR


# --------------------------------------------------------------------------- demo


def run_demo(settings: Settings) -> int:
    """Walk through the main CRM scenarios against a throwaway database."""
    with tempfile.TemporaryDirectory() as tmp:
        demo_settings = settings.model_copy(
            update={"database_url": f"sqlite:///{Path(tmp) / 'demo.db'}"}
        )
        app = build_application(demo_settings)
        try:
            return _run_demo_steps(app)
        finally:
            app.close()


def _step(title: str, fn: Callable[[], Any]) -> Any:
    print(f"\n=== {title} ===")
    try:
        result = fn()
    except CRMError as exc:
        print(f"[blocked] {exc.code}: {exc.message}")
        for err in exc.details.get("errors", []):
            print(f"          - {err['field']}: {err['message']}")
        return None
    _emit(result)
    return result


def _run_demo_steps(app: Application) -> int:
    init_db(app.engine)
    with transaction(app.session_factory) as session:
        seed_database(session, now=utcnow(), pin_iterations=app.settings.pin_hash_iterations)
    crm, identity = app.crm, app.identity

    _step("Public policy lookup", lambda: crm.get_policy("refund-policy"))
    _step(
        "Login with a wrong PIN",
        lambda: identity.authenticate("alice.nguyen@example.com", "000000"),
    )
    alice = identity.authenticate("alice.nguyen@example.com", "246810")
    print(f"\nAlice verified: session {alice.session_id} (expires {alice.expires_at:%H:%M} UTC)")

    _step("Alice lists her orders", lambda: crm.list_orders(alice))
    _step(
        "Alice checks refund eligibility for ORD-10001",
        lambda: crm.check_refund_eligibility(alice, "ORD-10001"),
    )
    refund = {
        "order_number": "ORD-10001",
        "reason": "defective",
        "details": "Left earcup crackles.",
        "idempotency_key": "demo-refund-0001",
    }
    _step("Alice requests a refund", lambda: crm.request_refund(alice, refund))
    _step(
        "The agent retries the same request (idempotent)", lambda: crm.request_refund(alice, refund)
    )
    ticket = {
        "category": "product_issue",
        "order_number": "ORD-10001",
        "subject": "Headphones crackle",
        "description": "The left earcup crackles.",
    }
    _step("Alice opens a ticket", lambda: crm.create_ticket(alice, ticket))
    _step("Duplicate ticket is not created twice", lambda: crm.create_ticket(alice, ticket))
    _step(
        "Expired refund window (ORD-10002)",
        lambda: crm.check_refund_eligibility(alice, "ORD-10002"),
    )
    _step("Alice tries to read Brian's order ORD-10004", lambda: crm.get_order(alice, "ORD-10004"))
    forged = dataclasses.replace(alice, customer_id=2)
    _step("A forged session claiming to be Brian", lambda: crm.list_orders(forged))
    _step(
        "Agent tries to inject customer_id into a tool call",
        lambda: crm.create_ticket(alice, {**ticket, "customer_id": 2}),
    )

    emma = identity.authenticate("emma.rossi@example.com", "778899")
    _step("Suspended account can read orders", lambda: crm.list_orders(emma))
    _step(
        "Suspended account cannot request refunds",
        lambda: crm.request_refund(emma, {"order_number": "ORD-10012", "reason": "damaged"}),
    )

    print("\n=== Audit trail (most recent first) ===")
    with transaction(app.session_factory) as session:
        for e in list_recent_audit_events(session, limit=15):
            print(
                f"{e.occurred_at:%H:%M:%S}  {e.outcome.value:<9} {e.action:<26} "
                f"{e.actor:<24} {e.entity_id or ''}"
            )
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())

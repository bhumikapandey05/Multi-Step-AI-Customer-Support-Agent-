"""Interactive terminal chat loop.

Approval prompts are answered here, by the human at the keyboard. The answer
goes straight to :meth:`SupportAgent.approve` / :meth:`SupportAgent.reject`
and never passes through the LLM.

Only an exact ``yes``/``y`` approves. ``/later`` leaves the action pending in the
saved workflow and ends the chat, so it can be decided after a restart with
``support-crm chat --thread <id>``. Anything else, including end of input,
declines.
"""

import json
from collections.abc import Callable
from typing import Any

from support_agent.agent.agent import (
    AgentError,
    AgentReply,
    ExecutionStatus,
    PendingApproval,
    SupportAgent,
    ToolExecution,
)

InputFn = Callable[[str], str]
OutputFn = Callable[[str], None]

HELP = """Commands:
  /help     show this help
  /pending  show actions waiting for your confirmation
  /quit     end the chat (the conversation is saved and can be resumed)
At a confirmation prompt, type 'yes' to approve, '/later' to decide after a restart,
or anything else to decline."""

LATER = "/later"

_ICONS = {
    ExecutionStatus.SUCCESS: "ok",
    ExecutionStatus.ERROR: "error",
    ExecutionStatus.AWAITING_APPROVAL: "needs approval",
    ExecutionStatus.DECLINED: "declined",
    ExecutionStatus.CACHED: "cached",
}


def _short(value: Any, limit: int) -> str:
    text = json.dumps(value, default=str, separators=(",", ":"))
    return text if len(text) <= limit else text[: limit - 3] + "..."


def format_execution(execution: ToolExecution, *, verbose: bool) -> str:
    args = _short(execution.arguments, 200)
    approved = " (approved by you)" if execution.approved else ""
    line = f"  [tool] {execution.tool}({args}) -> {_ICONS[execution.status]}{approved}"
    result = execution.result
    if verbose:
        body = json.dumps(result, indent=2, default=str)
        return line + "\n" + "\n".join("         " + row for row in body.splitlines())
    if result.get("ok") is False:
        return f"{line}: {result.get('error')}: {result.get('message')}"
    return line + (
        ""
        if execution.status is ExecutionStatus.AWAITING_APPROVAL
        else f": {_short(result.get('data'), 160)}"
    )


def run_chat(
    agent: SupportAgent,
    *,
    input_fn: InputFn = input,
    output: OutputFn = print,
    show_tools: bool = True,
    verbose: bool = False,
    greeting: str | None = None,
) -> int:
    output(greeting or "Support assistant ready. Type /help for commands, /quit to exit.")

    def show(reply: AgentReply) -> AgentReply:
        if show_tools:
            for execution in reply.tool_executions:
                output(format_execution(execution, verbose=verbose))
        output(f"\nassistant> {reply.text}\n")
        return reply

    def confirm_pending(pending: list[PendingApproval]) -> bool:
        """Ask about each pending action until none remain. False means "decide later"."""
        while pending:
            approval = pending[0]
            output(f"  CONFIRM [{approval.action_id}] {approval.summary}")
            try:
                answer = input_fn(
                    f"  Approve this action? Type 'yes' to confirm ({LATER} to decide later): "
                )
            except (EOFError, KeyboardInterrupt):
                answer = ""
            answer = answer.strip().lower()
            if answer == LATER:
                output("  Saved. Nothing was changed; this action is still waiting for you.")
                return False
            try:
                if answer in ("yes", "y"):
                    reply = show(agent.approve(approval.action_id))
                else:
                    output("  Declined. Nothing was changed.")
                    reply = show(agent.reject(approval.action_id))
            except AgentError as exc:  # e.g. decided at the same time in another session
                output(f"  {exc}")
                pending = agent.pending_approvals
                continue
            pending = reply.pending_approvals
        return True

    # Pick up where a previous session stopped.
    if agent.needs_resume:
        output("  Finishing an action you already confirmed before the chat was interrupted...")
        resumed = agent.resume()
        if resumed is not None:
            show(resumed)
    waiting = agent.pending_approvals
    if waiting:
        output("  This conversation has an action waiting for your confirmation.")
        if not confirm_pending(waiting):
            return 0

    while True:
        try:
            text = input_fn("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            output("")
            return 0
        if not text:
            continue
        if text in ("/quit", "/exit"):
            return 0
        if text == "/help":
            output(HELP)
            continue
        if text == "/pending":
            pending = agent.pending_approvals
            output(
                "\n".join(f"  [{p.action_id}] {p.summary}" for p in pending)
                or "  Nothing is waiting for confirmation."
            )
            continue
        try:
            reply = show(agent.send(text))
        except AgentError as exc:
            output(f"  {exc}")
            continue
        if not confirm_pending(reply.pending_approvals):
            return 0

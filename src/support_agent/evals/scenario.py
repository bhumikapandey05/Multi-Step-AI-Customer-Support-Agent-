"""Scenario files: what the customer does, what the (mocked) model does, what must happen.

Scenarios live in ``scenarios/<set>.json``. Every model is strict
(``extra="forbid"``), so a typo in a scenario file fails loudly instead of
silently weakening a check.

* ``turns`` are the simulated customer (and staff) actions, in order.
* ``mock`` is the scripted model for offline runs, one step per model call.
  Some scripts deliberately misbehave (wrong customer's order, smuggled fields,
  made-up tools) to test that the *system* blocks them. Live runs ignore it.
* ``expect`` describes the outcome. Global safety invariants (no data leaks,
  no unapproved or unauthorized writes, no duplicates) are always checked too.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCENARIO_DIR = Path(__file__).parent / "scenarios"
SETS = ("dev", "heldout")

Category = Literal[
    "order_tracking",
    "refunds",
    "tickets",
    "ambiguous",
    "unauthorized_access",
    "adversarial",
    "recovery",
    "staff_approval",
]
WriteKind = Literal["refund_requests", "tickets", "refund_decisions", "approval_decisions"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ToolCallSpec(_Strict):
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)


class MockStep(_Strict):
    """One scripted model response. ``repeat`` expands it into several identical steps."""

    text: str | None = None
    calls: list[ToolCallSpec] = Field(default_factory=list)
    error: Literal["unavailable"] | None = None
    repeat: int = Field(default=1, ge=1, le=50)


class Fault(_Strict):
    kind: Literal["crash_after_write", "transient_errors"]
    target: Literal["request_refund", "create_ticket"]
    count: int = Field(default=1, ge=1, le=10)


class StaffStep(_Strict):
    email: str
    decision: Literal["approve", "reject"]
    order: str  # decides the newest refund request created in this scenario for this order
    note: str | None = None


class Turn(_Strict):
    say: str | None = None
    say_chars: int | None = Field(default=None, ge=1)  # send a message of this many characters
    confirm: Literal["approve", "reject", "leave"] = "leave"  # answer to every confirmation
    restart: bool = False  # new process: new checkpoint connection and a fresh sign-in
    resume: bool = False  # continue an interrupted step
    advance_minutes: int = Field(default=0, ge=0)
    fault: Fault | None = None  # armed before this turn's actions
    staff: StaffStep | None = None
    race_approve: bool = False  # two sessions approve the pending action at the same moment


class Expect(_Strict):
    final_stop_reason: str | None = None
    tools_called: list[str] = Field(default_factory=list)  # model must propose these
    tools_not_called: list[str] = Field(default_factory=list)  # model must not propose these
    tool_errors: dict[str, str] = Field(default_factory=dict)  # tool -> error code seen
    writes: dict[WriteKind, int] = Field(default_factory=dict)  # exact new rows
    refund_status: dict[str, str] = Field(default_factory=dict)  # order -> newest status
    manual_review: dict[str, bool] = Field(default_factory=dict)  # order -> flag
    reply_contains_any: list[str] = Field(default_factory=list)
    reply_excludes: list[str] = Field(default_factory=list)
    pending_at_end: bool | None = None
    input_rejected: bool = False
    staff_refused: bool | None = None


class Scenario(_Strict):
    id: str = Field(pattern=r"^[a-z0-9-]+$")
    category: Category
    title: str
    customer: str | None  # None = guest
    turns: list[Turn] = Field(min_length=1)
    expect: Expect
    mock: list[MockStep] = Field(default_factory=list)
    require_ticket_approval: bool = True
    # Some scenarios script model behaviour that cannot be forced on a live model
    # (an outage, a runaway tool loop); those only run offline.
    modes: list[Literal["offline", "live"]] = Field(default_factory=lambda: ["offline", "live"])

    @model_validator(mode="after")
    def _turns_do_something(self) -> "Scenario":
        for turn in self.turns:
            if not turn.model_dump(exclude_defaults=True):
                raise ValueError(f"{self.id}: a turn does nothing")
        return self

    @property
    def mock_steps(self) -> list[MockStep]:
        return [step for step in self.mock for _ in range(step.repeat)]


class ScenarioSet(BaseModel):
    name: str
    sha256: str
    scenarios: list[Scenario]


def load_set(name: str, directory: Path = SCENARIO_DIR) -> ScenarioSet:
    path = directory / f"{name}.json"
    raw = path.read_bytes()
    scenarios = [Scenario.model_validate(s) for s in json.loads(raw)["scenarios"]]
    ids = [s.id for s in scenarios]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate scenario ids in {path.name}")
    return ScenarioSet(name=name, sha256=hashlib.sha256(raw).hexdigest(), scenarios=scenarios)


def load_sets(names: list[str], directory: Path = SCENARIO_DIR) -> list[ScenarioSet]:
    sets = [load_set(n, directory) for n in names]
    all_ids = [s.id for st in sets for s in st.scenarios]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("A scenario id appears in more than one set")
    return sets

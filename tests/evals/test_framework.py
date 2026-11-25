"""The evaluation framework itself: scenario files, determinism, detectors, metrics, safety."""

import hashlib
import json
import tempfile
from collections import Counter
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from support_agent.agent.agent import SupportAgent
from support_agent.cli import EXIT_OK, main
from support_agent.config import Settings
from support_agent.crm.service import CRMService
from support_agent.db.engine import transaction
from support_agent.db.models import (
    Customer,
    Order,
    RefundDecision,
    RefundReason,
    RefundRequest,
    RefundStatus,
    StaffMember,
    StaffRole,
)
from support_agent.evals.checks import evaluate
from support_agent.evals.llm import FreeTierGuard, InstrumentedLLM, MockLLM
from support_agent.evals.metrics import Prices, percentile, summarize, wilson
from support_agent.evals.report import render_markdown
from support_agent.evals.run import heldout_frozen
from support_agent.evals.runner import run_scenario
from support_agent.evals.sandbox import EvalSandbox, UnsafeEvaluationTarget
from support_agent.evals.scenario import SCENARIO_DIR, Scenario, load_set, load_sets
from tests.agent.conftest import ScriptedLLM, say
from tests.conftest import TEST_SECRET

CATEGORIES = {
    "order_tracking", "refunds", "tickets", "ambiguous",
    "unauthorized_access", "adversarial", "recovery", "staff_approval",
}  # fmt: skip


def scenario(set_name: str, scenario_id: str) -> Scenario:
    return next(s for s in load_set(set_name).scenarios if s.id == scenario_id)


def check(result, name):
    return next(c for c in result.checks if c.name == name)


# ------------------------------------------------------------------ scenario files


def test_scenario_sets_cover_the_requirements():
    sets = load_sets(["dev", "heldout"])
    every = [s for st in sets for s in st.scenarios]
    assert len(every) >= 40
    assert {s.category for s in every} == CATEGORIES
    assert {s.category for s in load_set("heldout").scenarios} == CATEGORIES


def test_heldout_set_matches_its_frozen_fingerprint():
    assert heldout_frozen() is True


def test_scenario_schema_is_strict():
    base = scenario("dev", "track-list-orders").model_dump()
    with pytest.raises(ValidationError):
        Scenario.model_validate({**base, "expect": {**base["expect"], "writez": {}}})
    with pytest.raises(ValidationError):
        Scenario.model_validate({**base, "turns": [{"say": "hi", "confrim": "approve"}]})
    with pytest.raises(ValidationError):
        Scenario.model_validate({**base, "turns": [{}]})


def test_duplicate_ids_across_sets_are_rejected(tmp_path):
    raw = (SCENARIO_DIR / "dev.json").read_text()
    (tmp_path / "dev.json").write_text(raw)
    (tmp_path / "heldout.json").write_text(raw)
    with pytest.raises(ValueError, match="more than one set"):
        load_sets(["dev", "heldout"], tmp_path)


# ------------------------------------------------------------------ determinism and isolation


def _fingerprint(result):
    return (result.passed, [(c.name, c.passed, c.detail) for c in result.checks], result.duplicates)


@pytest.mark.parametrize(
    "scenario_id",
    [
        "refund-eligible-approved",
        "recover-crash-after-write",
        "staff-manual-review-needs-supervisor",
    ],
)
def test_offline_runs_are_deterministic(scenario_id):
    s = scenario("dev", scenario_id)
    first = run_scenario(s, mode="offline", set_name="dev")
    second = run_scenario(s, mode="offline", set_name="dev")
    assert first.passed and _fingerprint(first) == _fingerprint(second)


def test_sandbox_refuses_protected_paths(tmp_path, monkeypatch):
    target = tmp_path / "sandbox"
    target.mkdir()
    monkeypatch.setattr(tempfile, "mkdtemp", lambda **_: str(target))
    with pytest.raises(UnsafeEvaluationTarget), EvalSandbox(protected=[target / "crm.db"]):
        pass


def test_eval_cli_never_touches_the_real_database(tmp_path, capsys):
    real = tmp_path / "real"
    settings = Settings(
        _env_file=None, database_url=f"sqlite:///{real / 'crm.db'}", auth_secret_key=TEST_SECRET,
        pin_hash_iterations=1_000, log_level="ERROR", gemini_api_key=None,
    )  # fmt: skip
    assert main(["init-db"], settings=settings) == EXIT_OK
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in real.iterdir()}

    code = main(
        ["eval", "--scenario", "refund-eligible-approved", "--scenario", "ticket-late-package",
         "--out", str(tmp_path / "report")],
        settings=settings,
    )  # fmt: skip

    assert code == EXIT_OK
    after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in real.iterdir()}
    assert after == before  # same files, same bytes; no checkpoints.db created either
    report = (tmp_path / "report" / "report.md").read_text()
    assert "2/2" in report and "Offline run" in report
    assert json.loads((tmp_path / "report" / "results.json").read_text())["summary"]["scored"] == 2


# ------------------------------------------------------------------ detectors must fire


def test_leak_detector_fires_when_ownership_checks_are_broken(monkeypatch):
    def no_ownership(self, op, order_number):  # a deliberately broken CRM
        from sqlalchemy import select
        from sqlalchemy.orm import selectinload

        return op.session.scalar(
            select(Order)
            .where(Order.order_number == order_number)
            .options(selectinload(Order.items))
        )

    monkeypatch.setattr(CRMService, "_get_owned_order", no_ownership)
    result = run_scenario(
        scenario("dev", "unauth-other-customers-order"), mode="offline", set_name="t"
    )
    leak = check(result, "no_cross_customer_leak")
    assert not leak.passed and "Adjustable Dumbbell Set" in leak.detail
    assert not result.passed and result.violations


def test_confirmation_detector_fires_when_the_approval_gate_is_removed(monkeypatch):
    monkeypatch.setattr(SupportAgent, "_needs_approval", lambda self, tool: False)
    s = scenario("dev", "refund-eligible-approved")
    result = run_scenario(s, mode="offline", set_name="t")
    gate = check(result, "writes_confirmed_by_customer")
    assert not gate.passed and "1 created, 0 confirmed" in gate.detail


def test_duplicate_and_staff_detectors_fire_on_bad_rows():
    s = scenario("dev", "track-list-orders")
    with EvalSandbox() as sb:
        with transaction(sb.app.session_factory) as db:
            alice = db.query(Customer).filter_by(email="alice.nguyen@example.com").one()
            order = db.query(Order).filter_by(order_number="ORD-10001").one()
            sam = db.query(StaffMember).filter_by(role=StaffRole.SUPPORT_AGENT).one()
            now = datetime(2026, 6, 15, tzinfo=UTC)
            rows = [
                RefundRequest(
                    refund_number=f"RFD-0000BAD{i}", customer_id=alice.id, order_id=order.id,
                    amount_cents=12_999, reason=RefundReason.DEFECTIVE, policy_version="2026.1",
                    status=status, created_at=now, updated_at=now,
                )
                for i, status in enumerate((RefundStatus.REJECTED, RefundStatus.PENDING_REVIEW))
            ]  # fmt: skip
            db.add_all(rows)
            db.flush()
            db.add(
                RefundDecision(
                    refund_request_id=rows[0].id, decision="rejected", decided_by_id=sam.id,
                    required_role=StaffRole.REFUND_APPROVER, amount_cents=12_999,
                    idempotency_key="bad-decision-1", request_fingerprint="x", decided_at=now,
                )
            )  # fmt: skip
        checks, duplicates, _ = evaluate(
            s, sb, [], InstrumentedLLM(MockLLM([])), None, pending_at_end=False, mode="offline"
        )
    by_name = {c.name: c for c in checks}
    assert duplicates == 1 and not by_name["no_duplicate_actions"].passed
    assert not by_name["staff_decisions_within_role"].passed
    assert "support_agent decided" in by_name["staff_decisions_within_role"].detail
    assert not by_name["writes_confirmed_by_customer"].passed  # nobody confirmed these
    assert not by_name["writes:refund_requests"].passed  # expected 0 new


def test_unused_mock_steps_are_flagged():
    s = scenario("dev", "track-list-orders")
    padded = s.model_copy(update={"mock": [*s.mock, *s.mock]})
    result = run_scenario(padded, mode="offline", set_name="t")
    assert not check(result, "mock_script_consumed").passed and not result.passed


def test_wrong_expectation_fails_with_a_clear_detail():
    s = scenario("dev", "refund-eligible-approved")
    wrong = s.model_copy(
        update={"expect": s.expect.model_copy(update={"writes": {"refund_requests": 2}})}
    )
    result = run_scenario(wrong, mode="offline", set_name="t")
    assert check(result, "writes:refund_requests").detail == "expected 2 new, got 1"


# ------------------------------------------------------------------ live mode plumbing


class UsageLLM(ScriptedLLM):
    def generate(self, **kwargs):
        response = super().generate(**kwargs)
        return type(response)(
            text=response.text,
            usage={
                "prompt_token_count": 1000,
                "candidates_token_count": 50,
                "total_token_count": 1050,
            },
        )


def test_live_mode_reports_usage_and_skips_offline_only_scenarios(tmp_path, capsys):
    settings = Settings(_env_file=None, auth_secret_key=TEST_SECRET, gemini_api_key="fake-key",
                        log_level="ERROR")  # fmt: skip
    llm = UsageLLM(say("You can request a refund within 30 days of delivery."))
    code = main(
        ["eval", "--mode", "live", "--scenario", "ambiguous-guest-policy", "--scenario",
         "adv-runaway-tool-loop", "--rpm", "600", "--out", str(tmp_path),
         "--price-input-per-1m", "0.10", "--price-output-per-1m", "0.40"],
        settings=settings, llm_factory=lambda s: llm,
    )  # fmt: skip
    data = json.loads((tmp_path / "results.json").read_text())
    usage = data["summary"]["llm_usage"]
    assert usage["calls"] == 1 and usage["total_tokens"] == 1050
    assert usage["estimated_cost_usd"] == pytest.approx(1000 / 1e6 * 0.10 + 50 / 1e6 * 0.40)
    skipped = {s["id"]: s["skipped_reason"] for s in data["scenarios"] if not s["scored"]}
    assert "offline" in skipped["adv-runaway-tool-loop"]
    # The single-step fake can't do the required policy lookup, so this is a fail, not a pass.
    assert (
        code != EXIT_OK
        and "| Tokens (prompt / output / total) | 1,000 / 50 / 1,050"
        in (tmp_path / "report.md").read_text()
    )


def test_free_tier_guard_paces_and_caps_calls():
    sleeps, now = [], [0.0]

    def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    guard = FreeTierGuard(
        ScriptedLLM(say("a"), say("b"), say("c")),
        requests_per_minute=6,
        max_calls=2,
        sleep=fake_sleep,
        monotonic=lambda: now[0],
    )
    kwargs = {"system_prompt": "", "messages": (), "tools": ()}
    guard.generate(**kwargs)
    guard.generate(**kwargs)
    assert sleeps == [10.0]  # 6 per minute -> 10 s apart
    from support_agent.evals.llm import BudgetExhausted

    with pytest.raises(BudgetExhausted):
        guard.generate(**kwargs)


def test_budget_exhaustion_marks_live_scenarios_not_scored():
    guard = FreeTierGuard(ScriptedLLM(), requests_per_minute=600, max_calls=1)
    guard.used = 1  # budget already spent
    result = run_scenario(scenario("dev", "track-list-orders"), mode="live", set_name="dev",
                          live_llm=guard)  # fmt: skip
    assert not result.scored and "BudgetExhausted" in result.skipped_reason
    summary = summarize([result])
    assert summary["task_success"]["total"] == 0 and summary["not_scored"][0]["id"] == result.id


# ------------------------------------------------------------------ metrics and report


def test_wilson_and_percentiles():
    assert wilson(0, 0) is None
    lo, hi = wilson(39, 39)
    assert hi == 1.0 and 0.90 < lo < 0.92
    assert wilson(5, 10) == pytest.approx((0.2366, 0.7634), abs=1e-4)
    assert percentile([1, 2, 3, 4], 50) == 2.5 and percentile([], 95) is None


def test_no_cost_without_prices_and_report_sections():
    result = run_scenario(scenario("dev", "refund-window-expired"), mode="offline", set_name="dev")
    summary = summarize([result])
    assert summary["llm_usage"]["estimated_cost_usd"] is None
    assert summary["llm_usage"]["total_tokens"] is None  # mocks report no usage
    md = render_markdown([result], meta={"mode": "offline", "model": "m", "sets": []},
                         prices=Prices(1, 1))  # fmt: skip
    for section in ("## Headline metrics", "## Failure analysis", "## Limitations"):
        assert section in md
    assert "not available (no provider usage data" in md


def test_failure_analysis_lists_failed_checks():
    s = scenario("dev", "refund-eligible-approved")
    wrong = s.model_copy(update={"expect": s.expect.model_copy(update={"writes": {"tickets": 1}})})
    result = run_scenario(wrong, mode="offline", set_name="dev")
    md = render_markdown([result], meta={"mode": "offline", "model": "m", "sets": []})
    assert "### `refund-eligible-approved`" in md and "expected 1 new, got 0" in md


def test_every_dev_scenario_has_a_category_and_runs_offline():
    results = [run_scenario(s, mode="offline", set_name="dev") for s in load_set("dev").scenarios]
    assert Counter(r.scored for r in results)[True] == len(results)
    assert all(r.framework_error is None for r in results)


def test_model_latency_excludes_free_tier_pacing():
    now = [0.0]

    def slow_sleep(seconds: float) -> None:
        now[0] += seconds

    class TimedLLM(ScriptedLLM):
        def generate(self, **kwargs):
            now[0] += 0.5  # the model itself takes 0.5 s
            return super().generate(**kwargs)

    guard = FreeTierGuard(TimedLLM(say("a"), say("b")), requests_per_minute=6, max_calls=5,
                          sleep=slow_sleep, monotonic=lambda: now[0])  # fmt: skip
    llm = InstrumentedLLM(guard, clock=lambda: now[0])
    kwargs = {"system_prompt": "", "messages": (), "tools": ()}
    llm.generate(**kwargs)
    llm.generate(**kwargs)  # waits 9.5 s for the 10 s gap, then 0.5 s of model time
    assert [c.latency_ms for c in llm.calls] == [500.0, 500.0]
    assert llm.calls[1].paced_ms == 9500.0

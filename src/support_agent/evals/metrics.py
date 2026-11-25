"""Aggregate metrics computed from scenario results. Nothing here is estimated or invented.

* Rates are over *scored* scenarios only. Live scenarios hit by a model outage or
  the call budget are reported as not scored, not as failures.
* Each rate comes with a Wilson 95% interval, because the sample is small.
* Token counts come only from provider-reported usage. A cost estimate is only
  computed when the caller supplies per-token prices; otherwise it is ``None``.
"""

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from support_agent.evals.runner import ScenarioResult


@dataclass(frozen=True)
class Prices:
    """USD per 1M tokens, as supplied by the user for their model and tier."""

    input_per_million: float
    output_per_million: float


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float] | None:
    if total == 0:
        return None
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return (round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4))


def percentile(values: list[float], q: float) -> float | None:
    """Linear-interpolated percentile (q in 0..100)."""
    if not values:
        return None
    ordered = sorted(values)
    k = (len(ordered) - 1) * q / 100
    lo, hi = math.floor(k), math.ceil(k)
    return round(ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo), 3)


def _rate(successes: int, total: int) -> dict[str, Any]:
    return {
        "successes": successes,
        "total": total,
        "rate": round(successes / total, 4) if total else None,
        "ci95": wilson(successes, total),
    }


def _latency(values: list[float]) -> dict[str, Any]:
    return {
        "count": len(values),
        "p50_ms": percentile(values, 50),
        "p95_ms": percentile(values, 95),
        "max_ms": round(max(values), 3) if values else None,
    }


def summarize(results: list["ScenarioResult"], prices: Prices | None = None) -> dict[str, Any]:
    ran = [r for r in results if r.turns]  # everything that actually executed
    scored = [r for r in results if r.scored]
    by_category: dict[str, list[ScenarioResult]] = defaultdict(list)
    for r in scored:
        by_category[r.category].append(r)

    tool_sel = [r.tool_selection_correct for r in scored if r.tool_selection_correct is not None]
    recovery = [r for r in scored if r.category == "recovery"]
    calls = [c for r in ran for c in r.llm_calls]
    with_usage = [c for c in calls if c.usage]
    prompt = sum(c.usage.get("prompt_token_count", 0) for c in with_usage)
    output = sum(c.usage.get("candidates_token_count", 0) for c in with_usage)
    total = sum(c.usage.get("total_token_count", 0) for c in with_usage)

    cost = None
    if prices is not None and with_usage:
        cost = round(
            prompt / 1e6 * prices.input_per_million + output / 1e6 * prices.output_per_million, 6
        )

    return {
        "scenarios": len(results),
        "scored": len(scored),
        "not_scored": [{"id": r.id, "reason": r.skipped_reason} for r in results if not r.scored],
        "task_success": _rate(sum(r.passed for r in scored), len(scored)),
        "by_category": {
            cat: _rate(sum(r.passed for r in rs), len(rs))
            for cat, rs in sorted(by_category.items())
        },
        "tool_selection": _rate(sum(tool_sel), len(tool_sel)),
        "authorization_violations": {
            "total": sum(len(r.violations) for r in scored),
            "scenarios_affected": sum(1 for r in scored if r.violations),
            "by_check": dict(sorted(_count(c.name for r in scored for c in r.violations).items())),
        },
        "duplicate_actions": {
            "total": sum(r.duplicates for r in scored),
            "duplicates_prevented_by_idempotency": sum(r.prevented_duplicates for r in scored),
        },
        "workflow_recovery": _rate(sum(r.passed for r in recovery), len(recovery)),
        "framework_errors": [
            {"id": r.id, "error": r.framework_error} for r in results if r.framework_error
        ],
        "latency": {
            "turn": _latency([t.latency_ms for r in ran for t in r.turns]),
            "llm_call": _latency([c.latency_ms for c in calls]),
            "free_tier_pacing_total_ms": round(sum(c.paced_ms for c in calls), 3),
        },
        "llm_usage": {
            "calls": len(calls),
            "errors": sum(1 for c in calls if c.error),
            "calls_with_usage": len(with_usage),
            "prompt_tokens": prompt if with_usage else None,
            "output_tokens": output if with_usage else None,
            "total_tokens": total if with_usage else None,
            "estimated_cost_usd": cost,
            "prices_per_million": None
            if prices is None
            else {"input": prices.input_per_million, "output": prices.output_per_million},
        },
    }


def _count(names: Any) -> dict[str, int]:
    out: dict[str, int] = defaultdict(int)
    for n in names:
        out[n] += 1
    return out

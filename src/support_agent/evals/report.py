"""Readable Markdown report plus machine-readable JSON results."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from support_agent.evals.checks import to_dict
from support_agent.evals.metrics import Prices, summarize

if TYPE_CHECKING:
    from support_agent.evals.runner import ScenarioResult

LIMITATIONS_COMMON = [
    "**Small sample.** Every rate has a Wilson 95% interval, and with a few dozen scenarios "
    "these intervals are wide. A difference of one or two scenarios is not meaningful.",
    "**Leak detection is canary-based.** It flags other customers' order/ticket/refund numbers, "
    "names, emails, and product names from the seed data. A leak of information that is not one "
    "of those identifiers would not be detected.",
    "**Scenarios are hand-written** by the same team that built the agent, so they may share its "
    "blind spots. They are not a sample of real customer traffic.",
    "**The system cannot stop the model from saying something false.** Text claims (for example "
    '"your refund is approved") are only caught where a scenario checks the reply; writes and '
    "data access are enforced by code and checked on every scenario.",
]
LIMITATIONS_OFFLINE = [
    "**Offline mode does not measure Gemini.** The model is a scripted mock written per scenario. "
    "Tool-selection correctness and reply wording therefore reflect the scripts, and are 100% by "
    "construction when the scripts are followed. What offline mode *does* measure is everything "
    "around the model: authorization, confirmation gating, staff limits, idempotency, "
    "checkpoint recovery, input limits, and failure handling, including when the scripted model "
    "deliberately misbehaves.",
    "**Offline latency is local compute only** (SQLite, LangGraph, and the mock). It is not an "
    "estimate of production latency.",
    "**No token usage or cost in offline mode**: no real model is called.",
]
LIMITATIONS_LIVE = [
    "**Live runs are not deterministic.** Gemini's choices vary between runs; rerun before "
    "drawing conclusions.",
    "**Free-tier pacing** (a minimum gap between requests) is included in turn latency for "
    "live runs. Per-call model latency is reported separately.",
    "**Cost** is only estimated when per-token prices are supplied (`--price-input-per-1m`, "
    "`--price-output-per-1m`). Free-tier usage is not billed.",
]


def _pct(rate: dict[str, Any]) -> str:
    if not rate["total"]:
        return "n/a"
    ci = rate["ci95"]
    return (
        f"{rate['successes']}/{rate['total']} = **{rate['rate'] * 100:.1f}%** "
        f"(95% CI {ci[0] * 100:.0f} to {ci[1] * 100:.0f}%)"
    )


def _ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value:,.1f} ms"


def _scenario_path(r: "ScenarioResult") -> str:
    steps = []
    for t in r.turns:
        for reply in t.replies:
            tools = [
                f"{e['tool']}{'✓' if e['ok'] else ''}{'✗' + e['error'] if e['error'] else ''}"
                for e in reply.executions
            ]
            steps.append(f"{'+'.join(tools) or 'text'}→{reply.stop_reason}")
        if t.crashed:
            steps.append("CRASH")
        if t.input_rejected:
            steps.append("input rejected")
        if t.race:
            steps.append("race[" + ", ".join(t.race) + "]")
        if t.staff:
            steps.append(
                f"staff:{'refused ' + str(t.staff['error']) if t.staff['refused'] else 'ok'}"
            )
    return " · ".join(steps)


def render_markdown(
    results: list["ScenarioResult"],
    *,
    meta: dict[str, Any],
    prices: Prices | None = None,
) -> str:
    summary = summarize(results, prices)
    lines: list[str] = []
    w = lines.append
    mode = meta["mode"]

    w("# Agent evaluation report\n")
    w("| | |\n|---|---|")
    for key in ("mode", "model", "started_at", "duration_s", "command"):
        w(f"| {key.replace('_', ' ').title()} | {meta.get(key, '')} |")
    for s in meta["sets"]:
        w(f"| Set `{s['name']}` | {s['count']} scenarios, sha256 `{s['sha256'][:16]}…` |")
    w("")

    if mode == "offline":
        w(
            "> **Offline run.** The model is a deterministic scripted mock. These numbers describe "
            "the agent's *system* (authorization, confirmations, idempotency, recovery), not "
            "Gemini's judgment. See Limitations.\n"
        )

    w("## Headline metrics\n")
    lat, usage = summary["latency"], summary["llm_usage"]
    av, dup = summary["authorization_violations"], summary["duplicate_actions"]
    w("| Metric | Result |\n|---|---|")
    w(f"| Task success | {_pct(summary['task_success'])} |")
    w(f"| Tool-selection correctness | {_pct(summary['tool_selection'])} |")
    w(f"| Authorization violations | **{av['total']}** in {av['scenarios_affected']} scenarios |")
    replayed = dup["duplicates_prevented_by_idempotency"]
    w(f"| Duplicate CRM actions | **{dup['total']}** ({replayed} repeats safely replayed) |")
    w(f"| Workflow recovery | {_pct(summary['workflow_recovery'])} |")
    turn, call = lat["turn"], lat["llm_call"]
    w(f"| Turn latency p50 / p95 / max | {_ms(turn['p50_ms'])} / {_ms(turn['p95_ms'])} / "
      f"{_ms(turn['max_ms'])} ({turn['count']} turns) |")  # fmt: skip
    w(f"| Model call latency p50 / p95 | {_ms(call['p50_ms'])} / {_ms(call['p95_ms'])} "
      f"({usage['calls']} calls, {usage['errors']} errors) |")  # fmt: skip
    if lat["free_tier_pacing_total_ms"]:
        paced = _ms(lat["free_tier_pacing_total_ms"])
        w(f"| Free-tier pacing waits (in turn latency) | {paced} total |")
    if usage["calls_with_usage"]:
        w(f"| Tokens (prompt / output / total) | {usage['prompt_tokens']:,} / "
          f"{usage['output_tokens']:,} / {usage['total_tokens']:,} "
          f"(usage reported on {usage['calls_with_usage']}/{usage['calls']} calls) |")  # fmt: skip
        cost = usage["estimated_cost_usd"]
        shown = (
            "not computed (no prices supplied)"
            if cost is None
            else f"${cost:.4f} at supplied prices"
        )
        w(f"| Estimated cost | {shown} |")
    else:
        w("| Tokens / cost | not available (no provider usage data in this run) |")
    w(f"| Scenarios | {summary['scenarios']} total, {summary['scored']} scored |")
    w("")

    w("## By scenario set\n")
    w("| Set | Task success | Auth violations | Duplicates | Recovery |\n|---|---|---|---|---|")
    for s in meta["sets"]:
        sub = summarize([r for r in results if r.set == s["name"]], prices)
        w(f"| `{s['name']}` | {_pct(sub['task_success'])} | "
          f"{sub['authorization_violations']['total']} | {sub['duplicate_actions']['total']} | "
          f"{_pct(sub['workflow_recovery'])} |")  # fmt: skip
    w("")
    w("## By category\n")
    w("| Category | Task success |\n|---|---|")
    for cat, rate in summary["by_category"].items():
        w(f"| {cat.replace('_', ' ')} | {_pct(rate)} |")
    w("")

    w("## Failure analysis\n")
    failed = [r for r in results if r.scored and not r.passed]
    if not failed:
        w("No scored scenario failed.\n")
    for r in failed:
        w(f"### `{r.id}` ({r.set}, {r.category})\n")
        w(f"{r.title}\n")
        if r.framework_error:
            w(f"- **Scenario crashed:** `{r.framework_error}`")
        for c in r.failed_checks:
            w(f"- **{c.name}** ({c.kind}): {c.detail}")
        w(f"- Path: {_scenario_path(r)}\n")
    if summary["not_scored"]:
        w("### Not scored\n")
        for item in summary["not_scored"]:
            w(f"- `{item['id']}`: {item['reason']}")
        w("")

    w("## All scenarios\n")
    w("| Scenario | Set | Category | Result | Path |\n|---|---|---|---|---|")
    for r in results:
        result = "PASS" if r.passed else ("not scored" if not r.scored else "**FAIL**")
        w(f"| `{r.id}` | {r.set} | {r.category} | {result} | {_scenario_path(r)} |")
    w("")

    w("## Limitations\n")
    for item in LIMITATIONS_COMMON + (
        LIMITATIONS_OFFLINE if mode == "offline" else LIMITATIONS_LIVE
    ):
        w(f"- {item}")
    w("- **Held-out set.** `heldout.json` is fingerprinted above (sha256). It is for final "
      "measurement only: failures there are reported, not fixed by editing the scenarios or "
      "tuning the agent against them.")  # fmt: skip
    w("")
    return "\n".join(lines)


def results_json(
    results: list["ScenarioResult"], *, meta: dict[str, Any], prices: Prices | None
) -> dict[str, Any]:
    return {
        "meta": meta,
        "summary": summarize(results, prices),
        "scenarios": [
            {
                "id": r.id,
                "title": r.title,
                "set": r.set,
                "category": r.category,
                "passed": r.passed,
                "scored": r.scored,
                "skipped_reason": r.skipped_reason,
                "framework_error": r.framework_error,
                "duplicates": r.duplicates,
                "prevented_duplicates": r.prevented_duplicates,
                "checks": [to_dict(c) for c in r.checks],
                "turns": [
                    {
                        **asdict(t),
                        "replies": [
                            {
                                **asdict(rep),
                                "executions": [
                                    {k: v for k, v in e.items() if k != "result_json"}
                                    for e in rep.executions
                                ],
                            }
                            for rep in t.replies
                        ],
                    }
                    for t in r.turns
                ],
                "llm_calls": [asdict(c) for c in r.llm_calls],
            }
            for r in results
        ],
    }


def write_report(
    out_dir: Path,
    results: list["ScenarioResult"],
    *,
    meta: dict[str, Any],
    prices: Prices | None = None,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    md = out_dir / "report.md"
    js = out_dir / "results.json"
    md.write_text(render_markdown(results, meta=meta, prices=prices), encoding="utf-8")
    js.write_text(
        json.dumps(results_json(results, meta=meta, prices=prices), indent=2, default=str),
        encoding="utf-8",
    )
    return md, js

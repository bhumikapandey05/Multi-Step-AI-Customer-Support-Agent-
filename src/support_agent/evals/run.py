"""Run scenario sets and write the report."""

import contextlib
import hashlib
import time
from collections.abc import Callable
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from support_agent.agent.llm import LLMClient
from support_agent.evals.metrics import Prices, summarize
from support_agent.evals.report import write_report
from support_agent.evals.runner import Mode, ScenarioResult, run_scenario
from support_agent.evals.scenario import SCENARIO_DIR, load_sets


def heldout_frozen(directory: Path = SCENARIO_DIR) -> bool | None:
    """True if heldout.json still matches the fingerprint recorded before its first run."""
    manifest = directory / "heldout.sha256"
    if not manifest.exists():
        return None
    expected = manifest.read_text().split()[0]
    return hashlib.sha256((directory / "heldout.json").read_bytes()).hexdigest() == expected


def _versions() -> dict[str, str]:
    out = {}
    for pkg in ("langgraph", "google-genai", "sqlalchemy", "pydantic"):
        with contextlib.suppress(metadata.PackageNotFoundError):
            out[pkg] = metadata.version(pkg)
    return out


def run_evaluation(
    *,
    mode: Mode,
    set_names: list[str],
    out_dir: Path,
    scenario_ids: list[str] | None = None,
    live_llm: LLMClient | None = None,
    model_name: str = "scripted mock",
    protected: list[Path] | None = None,
    prices: Prices | None = None,
    command: str = "",
    keep: bool = False,
    progress: Callable[[str], None] = lambda _: None,
    directory: Path = SCENARIO_DIR,
) -> tuple[list[ScenarioResult], dict[str, Any], Path, Path]:
    sets = load_sets(set_names, directory)
    started = datetime.now(UTC)
    t0 = time.perf_counter()
    results: list[ScenarioResult] = []
    selected = set(scenario_ids or [])
    unknown = selected - {s.id for st in sets for s in st.scenarios}
    if unknown:
        raise ValueError(f"Unknown scenario ids: {sorted(unknown)}")
    for st in sets:
        for scenario in st.scenarios:
            if selected and scenario.id not in selected:
                continue
            result = run_scenario(
                scenario,
                mode=mode,
                set_name=st.name,
                live_llm=live_llm,
                protected=protected or [],
                keep=keep,
            )
            results.append(result)
            label = "PASS" if result.passed else ("SKIP" if not result.scored else "FAIL")
            progress(f"  {label:<4} {st.name:<7} {scenario.id}")
    meta = {
        "mode": mode,
        "model": model_name,
        "started_at": started.isoformat(timespec="seconds"),
        "duration_s": round(time.perf_counter() - t0, 2),
        "command": command,
        "sets": [
            {
                "name": st.name,
                "sha256": st.sha256,
                "count": sum(1 for r in results if r.set == st.name),
            }
            for st in sets
        ],
        "heldout_matches_frozen_fingerprint": heldout_frozen(directory),
        "versions": _versions(),
    }
    md, js = write_report(out_dir, results, meta=meta, prices=prices)
    return results, summarize(results, prices), md, js

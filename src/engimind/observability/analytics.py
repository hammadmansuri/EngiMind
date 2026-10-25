"""Pure-Python analytics over the JSON-lines logs (no Streamlit, no pandas).

The log is the single source of truth: runs, metrics and traces are all derived
from it, so the dashboard can never disagree with the files on disk.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from engimind.observability.logging_setup import LOG_FILE_NAME

STATUS_INCOMPLETE = "incomplete"
FINISHED_STATUSES = ("success", "refused", "failed", "cancelled")
COUNTERS = (
    "llm_calls",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "embedding_calls",
    "embedding_tokens",
    "tool_calls",
    "tool_errors",
    "retries",
    "handoffs",
)
_ROTATED = re.compile(re.escape(LOG_FILE_NAME) + r"\.(\d+)$")

Event = dict[str, Any]
Run = dict[str, Any]


def log_files(directory: str | Path) -> list[Path]:
    """Log files oldest first (rotated `.N` files, highest N is oldest)."""
    directory = Path(directory)
    rotated = []
    for path in directory.glob(LOG_FILE_NAME + ".*"):
        match = _ROTATED.search(path.name)
        if match:
            rotated.append((int(match.group(1)), path))
    files = [path for _, path in sorted(rotated, reverse=True)]
    current = directory / LOG_FILE_NAME
    if current.exists():
        files.append(current)
    return files


def parse_lines(lines: Iterable[str]) -> list[Event]:
    events: list[Event] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict) and "event" in event:
            events.append(event)
    return events


def load_events(directory: str | Path, max_events: int | None = None) -> list[Event]:
    """Read every log file, skipping unreadable lines, ordered by timestamp."""
    events: list[Event] = []
    for path in log_files(directory):
        try:
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                events.extend(parse_lines(handle))
        except OSError:
            continue
    events.sort(key=lambda event: str(event.get("ts", "")))
    if max_events is not None and len(events) > max_events:
        events = events[-max_events:]
    return events


def _number(value: Any) -> float:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0


def build_runs(events: Iterable[Event]) -> list[Run]:
    """Merge start/end events into one record per agent run.

    Each run has its own counters (`llm_calls`, ...) and inclusive totals
    (`incl_*`) that add up the whole subtree, so a router run reports the cost
    of everything it delegated to. Summing own counters across runs never
    double counts; summing `incl_*` across a tree does.
    """
    runs: dict[str, Run] = {}
    for event in events:
        name = event.get("event")
        run_id = event.get("run_id")
        if name not in (
            "agent.run.started",
            "agent.run.completed",
            "agent.run.failed",
        ) or not run_id:
            continue
        run = runs.setdefault(
            run_id,
            {"run_id": run_id, "status": STATUS_INCOMPLETE, "children": []},
        )
        for key, value in event.items():
            if key not in ("event", "level", "logger"):
                run[key] = value
        if name == "agent.run.started":
            run.setdefault("ended", False)
            run.setdefault("started_at", event.get("ts"))
        else:
            run["ended"] = True
            run["end_ts"] = event.get("ts")
        run.setdefault("status", STATUS_INCOMPLETE)

    for run in runs.values():
        if not run.get("ended"):
            run["status"] = STATUS_INCOMPLETE
        for key in COUNTERS:
            run[key] = int(_number(run.get(key)))
        parent = runs.get(run.get("parent_run_id") or "")
        if parent is not None:
            parent["children"].append(run["run_id"])

    def totals(run: Run, seen: set[str]) -> dict[str, int]:
        if "_totals" in run:
            return run["_totals"]
        seen.add(run["run_id"])
        result = {key: run[key] for key in COUNTERS}
        for child_id in run["children"]:
            if child_id in seen:
                continue
            for key, value in totals(runs[child_id], seen).items():
                result[key] += value
        run["_totals"] = result
        return result

    for run in runs.values():
        for key, value in totals(run, set()).items():
            run[f"incl_{key}"] = value
        run.pop("_totals", None)
    return sorted(runs.values(), key=lambda run: str(run.get("started_at", "")))


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return round(ordered[index], 2)


def summarize_runs(runs: list[Run]) -> dict[str, Any]:
    """Headline KPIs. Rates are over finished runs; refusals are not failures."""
    finished = [run for run in runs if run["status"] in FINISHED_STATUSES]
    counts = {status: 0 for status in (*FINISHED_STATUSES, STATUS_INCOMPLETE)}
    for run in runs:
        counts[run["status"]] = counts.get(run["status"], 0) + 1
    durations = [_number(run.get("duration_ms")) for run in finished]
    count = len(finished)
    return {
        "runs": len(runs),
        "finished": count,
        **{f"{status}": counts[status] for status in counts},
        "success_rate": round(counts["success"] / count, 4) if count else 0.0,
        "failure_rate": round(counts["failed"] / count, 4) if count else 0.0,
        "p50_ms": percentile(durations, 0.5),
        "p95_ms": percentile(durations, 0.95),
        "avg_ms": round(sum(durations) / count, 2) if count else 0.0,
        "tokens": sum(run["total_tokens"] for run in runs),
        "llm_calls": sum(run["llm_calls"] for run in runs),
        "tool_calls": sum(run["tool_calls"] for run in runs),
        "retries": sum(run["retries"] for run in runs),
        "handoffs": sum(run["handoffs"] for run in runs),
    }


def agent_metrics(runs: list[Run]) -> list[dict[str, Any]]:
    """Per-agent table using each run's OWN counters (no double counting)."""
    grouped: dict[str, list[Run]] = defaultdict(list)
    for run in runs:
        grouped[str(run.get("agent", "?"))].append(run)
    rows = []
    for agent, items in grouped.items():
        summary = summarize_runs(items)
        rows.append(
            {
                "agent": agent,
                "runs": summary["runs"],
                "success": summary["success"],
                "refused": summary["refused"],
                "failed": summary["failed"],
                "success_rate": summary["success_rate"],
                "failure_rate": summary["failure_rate"],
                "p50_ms": summary["p50_ms"],
                "p95_ms": summary["p95_ms"],
                "tokens": summary["tokens"],
                "llm_calls": summary["llm_calls"],
                "tool_calls": summary["tool_calls"],
                "retries": summary["retries"],
                "handoffs": summary["handoffs"],
            }
        )
    return sorted(rows, key=lambda row: row["agent"])


def tool_metrics(events: Iterable[Event]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Event]] = defaultdict(list)
    for event in events:
        if event.get("event") in ("tool.call.completed", "tool.call.failed"):
            grouped[str(event.get("tool", "?"))].append(event)
    rows = []
    for tool, items in grouped.items():
        durations = [_number(item.get("duration_ms")) for item in items]
        failed = sum(1 for item in items if item["event"].endswith("failed"))
        rows.append(
            {
                "tool": tool,
                "calls": len(items),
                "failed": failed,
                "failure_rate": round(failed / len(items), 4),
                "p50_ms": percentile(durations, 0.5),
                "p95_ms": percentile(durations, 0.95),
            }
        )
    return sorted(rows, key=lambda row: row["tool"])


def llm_metrics(events: Iterable[Event]) -> list[dict[str, Any]]:
    """Per-model usage from individual `llm.call` events (chat calls)."""
    grouped: dict[str, list[Event]] = defaultdict(list)
    for event in events:
        if event.get("event") in ("llm.call.completed", "llm.call.failed"):
            grouped[str(event.get("model", "?"))].append(event)
    rows = []
    for model, items in grouped.items():
        durations = [_number(item.get("duration_ms")) for item in items]
        rows.append(
            {
                "model": model,
                "calls": len(items),
                "failed": sum(1 for item in items if item["event"].endswith("failed")),
                "prompt_tokens": int(sum(_number(i.get("prompt_tokens")) for i in items)),
                "completion_tokens": int(
                    sum(_number(i.get("completion_tokens")) for i in items)
                ),
                "total_tokens": int(sum(_number(i.get("total_tokens")) for i in items)),
                "p50_ms": percentile(durations, 0.5),
                "p95_ms": percentile(durations, 0.95),
            }
        )
    return sorted(rows, key=lambda row: row["model"])


def error_events(events: Iterable[Event]) -> list[Event]:
    """Every failure signal: failed runs/tools/llm calls and ERROR/WARNING lines."""
    return [
        event
        for event in events
        if str(event.get("event", "")).endswith(".failed")
        or event.get("level") in ("ERROR", "CRITICAL")
        or event.get("event") == "retry.attempted"
    ]


def error_summary(events: Iterable[Event]) -> list[dict[str, Any]]:
    """Group errors by (agent, error type, message) with counts and last seen."""
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    for event in error_events(events):
        if event.get("event") == "retry.attempted":
            continue
        key = (
            str(event.get("agent", "-")),
            str(event.get("error_type", event.get("event"))),
            str(event.get("error", ""))[:200],
        )
        row = grouped.setdefault(
            key,
            {
                "agent": key[0],
                "error_type": key[1],
                "message": key[2],
                "count": 0,
                "last_seen": "",
                "last_run_id": "",
            },
        )
        row["count"] += 1
        if str(event.get("ts", "")) >= row["last_seen"]:
            row["last_seen"] = str(event.get("ts", ""))
            row["last_run_id"] = str(event.get("run_id", ""))
    return sorted(grouped.values(), key=lambda row: -row["count"])


def parse_ts(value: Any) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def filter_events(
    events: Iterable[Event],
    *,
    agents: Iterable[str] | None = None,
    run_id: str | None = None,
    trace_id: str | None = None,
    levels: Iterable[str] | None = None,
    since: timedelta | None = None,
    text: str | None = None,
    now: datetime | None = None,
) -> list[Event]:
    agent_set = set(agents) if agents else None
    level_set = set(levels) if levels else None
    cutoff = (now or datetime.now(timezone.utc)) - since if since else None
    needle = text.lower() if text else None
    result = []
    for event in events:
        if agent_set is not None and event.get("agent") not in agent_set:
            continue
        if run_id and event.get("run_id") != run_id:
            continue
        if trace_id and event.get("trace_id") != trace_id:
            continue
        if level_set is not None and event.get("level") not in level_set:
            continue
        if cutoff is not None:
            ts = parse_ts(event.get("ts"))
            if ts is None or ts < cutoff:
                continue
        if needle and needle not in json.dumps(event, default=str).lower():
            continue
        result.append(event)
    return result


def filter_runs(
    runs: Iterable[Run],
    *,
    agents: Iterable[str] | None = None,
    statuses: Iterable[str] | None = None,
    run_id: str | None = None,
    trace_id: str | None = None,
    since: timedelta | None = None,
    now: datetime | None = None,
) -> list[Run]:
    agent_set = set(agents) if agents else None
    status_set = set(statuses) if statuses else None
    cutoff = (now or datetime.now(timezone.utc)) - since if since else None
    result = []
    for run in runs:
        if agent_set is not None and run.get("agent") not in agent_set:
            continue
        if status_set is not None and run.get("status") not in status_set:
            continue
        if run_id and not str(run.get("run_id", "")).startswith(run_id.strip()):
            continue
        if trace_id and run.get("trace_id") != trace_id.strip():
            continue
        if cutoff is not None:
            ts = parse_ts(run.get("started_at"))
            if ts is None or ts < cutoff:
                continue
        result.append(run)
    return result


def trace_timeline(events: Iterable[Event], trace_id: str) -> list[dict[str, Any]]:
    """Ordered rows for one workflow: runs, handoffs, tools, LLM calls, retries."""
    trace_events = [e for e in events if e.get("trace_id") == trace_id]
    runs = {run["run_id"]: run for run in build_runs(trace_events)}

    def depth(run_id: str | None) -> int:
        level, seen = 0, set()
        while run_id in runs and run_id not in seen:
            seen.add(run_id)
            run_id = runs[run_id].get("parent_run_id")
            level += 1
        return level

    origin = parse_ts(trace_events[0].get("ts")) if trace_events else None
    rows = []
    for event in trace_events:
        name = str(event.get("event"))
        if name == "agent.run.completed" or name == "agent.run.failed":
            kind, label = "run", f"{event.get('agent')} → {event.get('status')}"
        elif name == "agent.run.started":
            kind, label = "run", f"{event.get('agent')} started"
        elif name == "agent.handoff":
            kind = "handoff"
            label = f"{event.get('from_agent')} → {event.get('to_agent')}"
        elif name.startswith("tool.call"):
            kind, label = "tool", str(event.get("tool"))
        elif name.startswith("llm.call"):
            kind, label = "llm", str(event.get("model"))
        elif name == "retry.attempted":
            kind, label = "retry", str(event.get("kind"))
        else:
            kind, label = "log", name
        ts = parse_ts(event.get("ts"))
        offset = (ts - origin).total_seconds() * 1000 if ts and origin else 0.0
        run_depth = depth(event.get("run_id"))
        rows.append(
            {
                "offset_ms": round(offset, 1),
                "depth": run_depth,
                "kind": kind,
                "name": label,
                "event": name,
                "agent": event.get("agent"),
                "run_id": event.get("run_id"),
                "duration_ms": event.get("duration_ms"),
                "tokens": event.get("total_tokens"),
                "status": event.get("status"),
                "error": event.get("error"),
            }
        )
    return rows


def trace_summaries(runs: list[Run]) -> list[dict[str, Any]]:
    """One row per trace: root agent, span count, worst status, totals."""
    grouped: dict[str, list[Run]] = defaultdict(list)
    for run in runs:
        if run.get("trace_id"):
            grouped[run["trace_id"]].append(run)
    rows = []
    for trace_id, items in grouped.items():
        items.sort(key=lambda run: str(run.get("started_at", "")))
        ids = {run["run_id"] for run in items}
        roots = [run for run in items if run.get("parent_run_id") not in ids]
        statuses = {run["status"] for run in items}
        worst = next(
            (s for s in ("failed", "cancelled", STATUS_INCOMPLETE, "refused") if s in statuses),
            "success",
        )
        rows.append(
            {
                "trace_id": trace_id,
                "started_at": items[0].get("started_at"),
                "root_agent": roots[0].get("agent") if roots else items[0].get("agent"),
                "runs": len(items),
                "status": worst,
                "duration_ms": max(_number(run.get("duration_ms")) for run in roots or items),
                "tokens": sum(run["total_tokens"] for run in items),
                "tool_calls": sum(run["tool_calls"] for run in items),
                "handoffs": sum(run["handoffs"] for run in items),
            }
        )
    return sorted(rows, key=lambda row: str(row["started_at"]), reverse=True)


def parse_pricing(text: str) -> dict[str, tuple[float, float]]:
    """Parse `model, input_per_1M, output_per_1M` lines into a price table."""
    prices: dict[str, tuple[float, float]] = {}
    for line in text.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3 or not parts[0] or parts[0].startswith("#"):
            continue
        try:
            prices[parts[0]] = (float(parts[1]), float(parts[2]))
        except ValueError:
            continue
    return prices


def llm_cost(event: Event, pricing: dict[str, tuple[float, float]]) -> float | None:
    """USD cost of one LLM call: a logged `cost_usd`, else tokens x price table."""
    logged = event.get("cost_usd")
    if isinstance(logged, (int, float)) and not isinstance(logged, bool):
        return float(logged)
    price = pricing.get(str(event.get("model")))
    if price is None:
        return None
    return (
        _number(event.get("prompt_tokens")) * price[0]
        + _number(event.get("completion_tokens")) * price[1]
    ) / 1_000_000


def cost_summary(
    events: Iterable[Event], pricing: dict[str, tuple[float, float]]
) -> dict[str, Any]:
    """Total cost, cost per run and per model, plus models that have no price."""
    total = 0.0
    by_run: dict[str, float] = defaultdict(float)
    by_model: dict[str, float] = defaultdict(float)
    unpriced: set[str] = set()
    for event in events:
        if event.get("event") != "llm.call.completed":
            continue
        cost = llm_cost(event, pricing)
        model = str(event.get("model", "?"))
        if cost is None:
            if _number(event.get("total_tokens")):
                unpriced.add(model)
            continue
        total += cost
        by_model[model] += cost
        if event.get("run_id"):
            by_run[str(event["run_id"])] += cost
    return {
        "total": total,
        "by_run": dict(by_run),
        "by_model": dict(by_model),
        "unpriced": sorted(unpriced),
    }


def time_series(runs: Iterable[Run], bucket: timedelta) -> list[dict[str, Any]]:
    """Runs bucketed by start time: volume, outcomes, latency and tokens."""
    seconds = max(1, int(bucket.total_seconds()))
    grouped: dict[datetime, list[Run]] = defaultdict(list)
    for run in runs:
        ts = parse_ts(run.get("started_at"))
        if ts is None:
            continue
        start = datetime.fromtimestamp((ts.timestamp() // seconds) * seconds, timezone.utc)
        grouped[start].append(run)
    rows = []
    for start in sorted(grouped):
        items = grouped[start]
        durations = [
            _number(r.get("duration_ms")) for r in items if r["status"] in FINISHED_STATUSES
        ]
        rows.append(
            {
                "time": start,
                "runs": len(items),
                "success": sum(1 for r in items if r["status"] == "success"),
                "failed": sum(1 for r in items if r["status"] == "failed"),
                "p50_ms": percentile(durations, 0.5),
                "p95_ms": percentile(durations, 0.95),
                "tokens": sum(r["total_tokens"] for r in items),
                "retries": sum(r["retries"] for r in items),
            }
        )
    return rows


def handoff_metrics(events: Iterable[Event]) -> list[dict[str, Any]]:
    """Inter-agent handoff counts per (from, to) pair."""
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for event in events:
        if event.get("event") != "agent.handoff":
            continue
        key = (str(event.get("from_agent", "-")), str(event.get("to_agent", "-")))
        row = grouped.setdefault(
            key, {"from_agent": key[0], "to_agent": key[1], "count": 0, "last_seen": ""}
        )
        row["count"] += 1
        row["last_seen"] = max(row["last_seen"], str(event.get("ts", "")))
    return sorted(grouped.values(), key=lambda row: -row["count"])


def agent_status(
    runs: Iterable[Run],
    *,
    active_within: timedelta = timedelta(minutes=15),
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """Live state per agent: running, failing (last run failed), healthy or idle."""
    now = now or datetime.now(timezone.utc)
    grouped: dict[str, list[Run]] = defaultdict(list)
    for run in runs:
        grouped[str(run.get("agent", "?"))].append(run)
    rows = []
    for agent, items in grouped.items():
        items.sort(key=lambda run: str(run.get("started_at", "")))
        last = items[-1]
        started = parse_ts(last.get("started_at"))
        recent = started is not None and now - started <= active_within
        if last["status"] == STATUS_INCOMPLETE and recent:
            state = "running"
        elif not recent:
            state = "idle"
        elif last["status"] == "failed":
            state = "failing"
        else:
            state = "healthy"
        rows.append(
            {
                "agent": agent,
                "state": state,
                "last_run": last.get("started_at"),
                "last_status": last["status"],
                "last_error": last.get("error"),
                "runs": len(items),
            }
        )
    return sorted(rows, key=lambda row: row["agent"])


def run_events(events: Iterable[Event], run_id: str) -> list[Event]:
    """Every event of one run, in order."""
    return [event for event in events if event.get("run_id") == run_id]

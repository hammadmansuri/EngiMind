"""Streamlit dashboard over the observability logs (read-only).

Everything shown is derived from the JSON-lines logs on disk; nothing is mocked.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

from engimind.observability import analytics
from engimind.observability.logging_setup import log_directory

WINDOWS = {
    "Last 15 minutes": timedelta(minutes=15),
    "Last hour": timedelta(hours=1),
    "Last 24 hours": timedelta(hours=24),
    "Last 7 days": timedelta(days=7),
    "All time": None,
}
BUCKETS = [
    (timedelta(hours=1), timedelta(minutes=1)),
    (timedelta(hours=6), timedelta(minutes=5)),
    (timedelta(days=2), timedelta(hours=1)),
    (timedelta(days=60), timedelta(days=1)),
]
REFRESH_CHOICES = {"5 s": 5, "10 s": 10, "30 s": 30, "60 s": 60}
STATE_ICONS = {"running": "🔵", "healthy": "🟢", "failing": "🔴", "idle": "⚪"}
MAX_EVENTS = 200_000
MAX_LOG_ROWS = 500
PRICING_HELP = (
    "One model per line: `model, USD per 1M input tokens, USD per 1M output "
    "tokens`. Logs record tokens, not prices, so cost is an estimate from this "
    "table (a logged `cost_usd` field wins when present)."
)


@st.cache_data(show_spinner="Reading logs…")
def _load(directory: str, signature: tuple) -> list[dict]:
    # `signature` (file names, sizes, mtimes) invalidates the cache on new logs.
    return analytics.load_events(directory, max_events=MAX_EVENTS)


def _signature(directory: Path) -> tuple:
    items = []
    for path in analytics.log_files(directory):
        try:
            stat = path.stat()
        except OSError:
            continue
        items.append((path.name, stat.st_size, stat.st_mtime_ns))
    return tuple(items)


def _frame(rows: list[dict], columns: list[str] | None = None) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if columns and not frame.empty:
        frame = frame[[c for c in columns if c in frame.columns]]
    return frame


def _table(rows: list[dict], columns: list[str] | None = None, **kwargs) -> None:
    st.dataframe(_frame(rows, columns), hide_index=True, width="stretch", **kwargs)


def _bucket_for(since: timedelta | None, runs: list[dict]) -> timedelta:
    span = since
    if span is None:
        times = [t for t in (analytics.parse_ts(r.get("started_at")) for r in runs) if t]
        span = (max(times) - min(times)) if times else timedelta(hours=1)
    for limit, bucket in BUCKETS:
        if span <= limit:
            return bucket
    return timedelta(days=7)


def _usd(value: float) -> str:
    return f"${value:,.4f}" if value < 1 else f"${value:,.2f}"


def render_dashboard() -> None:
    st.title("Observability")
    directory = log_directory().resolve()

    try:
        agents = sorted(
            {
                str(run.get("agent"))
                for run in analytics.build_runs(_load(str(directory), _signature(directory)))
            }
        )
    except Exception:
        agents = []  # the body reports the read error

    with st.sidebar:
        st.header("Controls")
        auto = st.toggle("Auto-refresh", value=True)
        interval = REFRESH_CHOICES[
            st.selectbox("Refresh every", list(REFRESH_CHOICES), index=1, disabled=not auto)
        ]
        if st.button("Refresh now", width="stretch"):
            st.rerun()
        st.header("Filters")
        window = st.selectbox("Time window", list(WINDOWS), index=2)
        picked_agents = st.multiselect("Agent", agents)
        picked_status = st.multiselect(
            "Status", [*analytics.FINISHED_STATUSES, analytics.STATUS_INCOMPLETE]
        )
        run_id = st.text_input("Run ID (prefix)").strip()
        trace_id = st.text_input("Trace ID").strip()
        text = st.text_input("Search logs", help="Matches any field of any event.")
        pricing_text = st.text_area(
            "Model pricing (USD / 1M tokens)",
            placeholder="gpt-4o-mini, 0.15, 0.60",
            help=PRICING_HELP,
        )

    body = st.fragment(run_every=interval if auto else None)(_body)
    body(
        directory,
        window,
        tuple(picked_agents),
        tuple(picked_status),
        run_id,
        trace_id,
        text,
        pricing_text,
    )


def _body(
    directory: Path,
    window: str,
    picked_agents: tuple,
    picked_status: tuple,
    run_id: str,
    trace_id: str,
    text: str,
    pricing_text: str,
) -> None:
    try:
        events = _load(str(directory), _signature(directory))
        all_runs = analytics.build_runs(events)
    except Exception as exc:  # a broken log must not take the page down
        st.error(f"Could not read logs from `{directory}`: {exc}")
        st.button("Retry")
        return

    stamp = datetime.now().strftime("%H:%M:%S")
    st.caption(f"Reading logs from `{directory}` · {len(events):,} events · updated {stamp}")

    if not events:
        st.info(
            "No logs yet. Ask a question or index a document; logs appear in the "
            "folder above (`LOG_DIR` overrides it)."
        )
        return
    if not all_runs:
        st.warning(
            "Logs exist but contain no agent runs yet, so run-level metrics are "
            "empty. Raw events are still searchable in the Logs tab."
        )

    since = WINDOWS[window]
    runs = analytics.filter_runs(
        all_runs,
        agents=picked_agents,
        statuses=picked_status,
        run_id=run_id,
        trace_id=trace_id,
        since=since,
    )
    scoped = analytics.filter_events(
        events, agents=picked_agents or None, trace_id=trace_id or None, since=since
    )
    if run_id or picked_status:
        keep = {run["run_id"] for run in runs}
        scoped = [e for e in scoped if e.get("run_id") in keep]
    pricing = analytics.parse_pricing(pricing_text)
    costs = analytics.cost_summary(scoped, pricing)

    overview, runs_tab, traces_tab, errors_tab, logs_tab = st.tabs(
        ["Overview", "Runs & drill-down", "Traces & handoffs", "Errors & retries", "Logs"]
    )
    with overview:
        _overview(runs, all_runs, scoped, costs, since, bool(pricing))
    with runs_tab:
        _runs(runs, events, costs, run_id)
    with traces_tab:
        _traces(runs, events, scoped, trace_id)
    with errors_tab:
        _errors(scoped)
    with logs_tab:
        _logs(scoped, text)


def _overview(runs, all_runs, events, costs, since, has_pricing) -> None:
    if not runs:
        st.info("No agent runs match the current filters.")
        return
    summary = analytics.summarize_runs(runs)

    first = st.columns(4)
    first[0].metric(
        "Runs", f"{summary['runs']:,}", help=f"{summary['incomplete']} running or incomplete"
    )
    first[1].metric("Success rate", f"{summary['success_rate']:.1%}")
    first[2].metric("Failure rate", f"{summary['failure_rate']:.1%}")
    first[3].metric("Refused", summary["refused"])
    second = st.columns(4)
    second[0].metric("Latency p50", f"{summary['p50_ms']:,.0f} ms")
    second[1].metric("Latency p95", f"{summary['p95_ms']:,.0f} ms")
    second[2].metric("Tokens", f"{summary['tokens']:,}")
    second[3].metric(
        "Est. LLM cost",
        _usd(costs["total"]) if (has_pricing or costs["total"]) else "—",
        help="Needs model pricing in the sidebar unless logs carry cost_usd.",
    )
    third = st.columns(4)
    third[0].metric("LLM calls", f"{summary['llm_calls']:,}")
    third[1].metric("Tool calls", f"{summary['tool_calls']:,}")
    third[2].metric("Retries", f"{summary['retries']:,}")
    third[3].metric("Handoffs", f"{summary['handoffs']:,}")
    if costs["unpriced"]:
        st.caption("No price set for: " + ", ".join(costs["unpriced"]))
    st.caption(
        "Rates are over finished runs. Refused answers are valid outcomes, not "
        "failures. Counters are per run and never double counted."
    )

    st.subheader("Agent status")
    status_rows = analytics.agent_status(all_runs)
    columns = st.columns(min(4, len(status_rows)) or 1)
    for i, row in enumerate(status_rows):
        with columns[i % len(columns)].container(border=True):
            st.markdown(f"**{STATE_ICONS[row['state']]} {row['agent']}**")
            st.caption(f"{row['state']} · last run {row['last_status']}")
            st.caption(str(row["last_run"] or ""))

    st.subheader("Trends")
    bucket = _bucket_for(since, runs)
    series = analytics.time_series(runs, bucket)
    if series:
        frame = _frame(series).set_index("time")
        left, right = st.columns(2)
        left.caption(f"Runs by outcome (per {bucket})")
        left.bar_chart(frame[["success", "failed"]])
        right.caption("Latency p50 / p95 (ms)")
        right.line_chart(frame[["p50_ms", "p95_ms"]])
        left, right = st.columns(2)
        left.caption("Tokens")
        left.bar_chart(frame[["tokens"]])
        right.caption("Retries")
        right.bar_chart(frame[["retries"]])

    st.subheader("By agent")
    agent_rows = analytics.agent_metrics(runs)
    _table(
        agent_rows,
        column_config={
            "success_rate": st.column_config.ProgressColumn(
                "success_rate", min_value=0.0, max_value=1.0, format="%.2f"
            ),
        },
    )
    if agent_rows:
        chart = _frame(agent_rows).set_index("agent")
        left, right = st.columns(2)
        left.caption("Latency p95 (ms) by agent")
        left.bar_chart(chart[["p95_ms"]])
        right.caption("Tokens by agent")
        right.bar_chart(chart[["tokens"]])

    left, right = st.columns(2)
    with left:
        st.subheader("Tools")
        tool_rows = analytics.tool_metrics(events)
        if tool_rows:
            _table(tool_rows)
        else:
            st.caption("No tool calls in this view.")
    with right:
        st.subheader("Models")
        llm_rows = analytics.llm_metrics(events)
        for row in llm_rows:
            row["est_cost_usd"] = round(costs["by_model"].get(row["model"], 0.0), 6)
        if llm_rows:
            _table(llm_rows)
        else:
            st.caption("No model calls in this view.")


def _runs(runs, events, costs, preselect: str) -> None:
    if not runs:
        st.info("No runs match the filters.")
        return
    rows = [
        {
            "started_at": run.get("started_at"),
            "agent": run.get("agent"),
            "status": run.get("status"),
            "duration_ms": run.get("duration_ms"),
            "tokens": run["total_tokens"],
            "cost_usd": round(costs["by_run"].get(run["run_id"], 0.0), 6),
            "llm_calls": run["llm_calls"],
            "tools": run["tool_calls"],
            "retries": run["retries"],
            "handoffs": run["handoffs"],
            "error": run.get("error"),
            "run_id": run["run_id"],
            "trace_id": run.get("trace_id"),
        }
        for run in reversed(runs)
    ]
    event = st.dataframe(
        _frame(rows),
        hide_index=True,
        width="stretch",
        on_select="rerun",
        selection_mode="single-row",
        key="runs_table",
    )
    selected = event.selection.rows if event and event.selection else []
    ids = [row["run_id"] for row in rows]
    if selected and selected[0] < len(ids):
        run_id = ids[selected[0]]
    else:
        run_id = next((i for i in ids if preselect and i.startswith(preselect)), None)
    if run_id is None:
        st.caption("Select a row to drill into that execution.")
        return
    run = next(r for r in runs if r["run_id"] == run_id)
    _run_detail(run, events, costs)


def _run_detail(run, events, costs) -> None:
    st.subheader(f"Run {run['run_id']}")
    cols = st.columns(5)
    cols[0].metric("Agent", str(run.get("agent")))
    cols[1].metric("Status", str(run["status"]))
    cols[2].metric("Duration", f"{run.get('duration_ms') or 0:,.0f} ms")
    cols[3].metric("Tokens", f"{run['total_tokens']:,}")
    cols[4].metric("Est. cost", _usd(costs["by_run"].get(run["run_id"], 0.0)))
    if run.get("error"):
        st.error(f"{run.get('error_type', 'Error')}: {run['error']}")
    if run.get("parent_run_id"):
        st.caption(f"Parent run: `{run['parent_run_id']}`")
    if run["children"]:
        st.caption("Child runs: " + ", ".join(f"`{c}`" for c in run["children"]))

    trace = run.get("trace_id")
    timeline = analytics.trace_timeline(events, trace) if trace else []
    mine = [row for row in timeline if row["run_id"] == run["run_id"]]
    if mine:
        st.markdown("**Execution steps**")
        _table(mine, ["offset_ms", "kind", "name", "duration_ms", "tokens", "status", "error"])
    raw = analytics.run_events(events, run["run_id"])
    with st.expander(f"Raw events ({len(raw)})"):
        st.code(json.dumps(raw, indent=2, default=str), language="json")


def _traces(runs, events, scoped, selected: str) -> None:
    summaries = analytics.trace_summaries(runs)
    if not summaries:
        st.info("No traces match the filters.")
    else:
        _table(summaries)
        ids = [row["trace_id"] for row in summaries]
        index = ids.index(selected) if selected in ids else 0
        chosen = st.selectbox("Inspect trace", ids, index=index)
        timeline = analytics.trace_timeline(events, chosen)
        for row in timeline:
            row["name"] = "\u2003" * row["depth"] + row["name"]
        _table(
            timeline,
            ["offset_ms", "kind", "name", "agent", "duration_ms", "tokens", "status", "error", "run_id"],
        )
    st.subheader("Inter-agent handoffs")
    handoffs = analytics.handoff_metrics(scoped)
    if handoffs:
        _table(handoffs)
    else:
        st.caption("No handoffs in this view.")


def _errors(events) -> None:
    summary = analytics.error_summary(events)
    if not summary:
        st.success("No errors in this view.")
    else:
        _table(summary)
    retries = [e for e in events if e.get("event") == "retry.attempted"]
    st.subheader("Retries")
    if retries:
        _table(
            list(reversed(retries)),
            ["ts", "agent", "kind", "run_id", "attempt", "wait_seconds", "reason"],
        )
    else:
        st.caption("No retries in this view.")


def _logs(events, text: str) -> None:
    levels = st.multiselect(
        "Level", ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], default=[], key="f_levels"
    )
    shown = analytics.filter_events(events, levels=levels or None, text=text or None)
    if not shown:
        st.info("No log events match the filters.")
        return
    st.caption(f"{len(shown):,} events (newest first, showing up to {MAX_LOG_ROWS})")
    rows = list(reversed(shown))[:MAX_LOG_ROWS]
    _table(
        rows,
        ["ts", "level", "event", "agent", "run_id", "trace_id", "request_id", "duration_ms", "error"],
    )
    with st.expander("Raw JSON (first 50)"):
        st.json(rows[:50])

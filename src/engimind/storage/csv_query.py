from __future__ import annotations

import re
from dataclasses import dataclass

from engimind.ingestion.csv_loader import CsvRow, CsvTable


@dataclass(frozen=True)
class CsvQueryResult:
    answer: str
    source_rows: tuple[tuple[CsvTable, CsvRow], ...]


def query_csv_tables(
    question: str, tables: list[CsvTable]
) -> CsvQueryResult | None:
    """Answer simple exact-match CSV filters and counts across all loaded tables."""
    if not tables:
        return None

    question_text = _normalize(question)
    count_requested = bool(
        re.search(r"\b(how many|count|number of|total)\b", question_text)
    )
    date_requested = bool(
        re.search(r"\b(when|date|dates|what day|which day)\b", question_text)
    )

    matches: list[tuple[CsvTable, CsvRow]] = []
    matched_value = ""
    matched_column = ""
    for table in tables:
        filter_column = _find_filter_column(question_text, table.columns)
        if filter_column is None:
            continue
        candidate = _find_value(question_text, table, filter_column)
        if candidate is None:
            continue
        value, rows = candidate
        matches.extend((table, row) for row in rows)
        matched_value = value
        matched_column = filter_column

    if not matched_column:
        return None

    if count_requested:
        answer = (
            f"{matched_value} appears in the `{matched_column}` column "
            f"{len(matches)} time(s) across the loaded CSV data."
        )
    elif date_requested:
        date_columns = [
            column
            for table in tables
            for column in table.columns
            if re.search(r"\b(date|day|time)\b", _normalize(column))
        ]
        if date_columns:
            entries = []
            for table, row in matches:
                date_column = next(
                    (
                        column
                        for column in date_columns
                        if column in row.values and row.values[column].strip()
                    ),
                    None,
                )
                if date_column:
                    entries.append(
                        f"{row.values[date_column]} "
                        f"[{table.source or 'CSV'} row {row.row_number}]"
                    )
            answer = (
                f"{matched_value} matched {len(matches)} row(s). Dates: "
                + (", ".join(entries) if entries else "no date values are available.")
            )
        else:
            answer = (
                f"{matched_value} matched {len(matches)} row(s), but the CSV "
                "does not have a date column."
            )
    else:
        return None

    return CsvQueryResult(answer=answer, source_rows=tuple(matches))


def _find_filter_column(question: str, columns: tuple[str, ...]) -> str | None:
    normalized_columns = {_normalize(column): column for column in columns}
    mentioned = [
        (normalized, column)
        for normalized, column in normalized_columns.items()
        if normalized and re.search(rf"\b{re.escape(normalized)}\b", question)
    ]
    if mentioned:
        return max(mentioned, key=lambda item: len(item[0]))[1]

    if re.search(r"\b(toss|coin)\b", question):
        return next(
            (
                column
                for column in columns
                if "toss" in _normalize(column) and "winner" in _normalize(column)
            ),
            None,
        )
    if re.search(r"\b(win|wins|won|winner|victor)\b", question):
        return next(
            (
                column
                for column in columns
                if _normalize(column) in {"winner", "match winner", "winning team"}
            ),
            None,
        )
    return None


def _find_value(
    question: str, table: CsvTable, column: str
) -> tuple[str, list[CsvRow]] | None:
    values = {row.values.get(column, "").strip() for row in table.rows}
    candidates = [
        value
        for value in values
        if value and re.search(rf"\b{re.escape(_normalize(value))}\b", question)
    ]
    if not candidates:
        return None
    value = max(candidates, key=len)
    normalized_value = _normalize(value)
    rows = [
        row
        for row in table.rows
        if _normalize(row.values.get(column, "")) == normalized_value
    ]
    return value, rows


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()

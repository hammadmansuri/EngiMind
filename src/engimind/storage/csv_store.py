from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from engimind.ingestion.csv_loader import CsvRow, CsvTable


def _database_path(chroma_path: str | Path) -> Path:
    directory = Path(chroma_path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory / "csv_tables.sqlite3"


@contextmanager
def _connection(database: Path) -> Iterator[sqlite3.Connection]:
    connection = sqlite3.connect(database)
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def save_csv_table(
    collection: str, table: CsvTable, chroma_path: str | Path
) -> None:
    database = _database_path(chroma_path)
    with _connection(database) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS csv_tables (
                collection TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                columns_json TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS csv_rows (
                collection TEXT NOT NULL,
                row_number INTEGER NOT NULL,
                values_json TEXT NOT NULL,
                PRIMARY KEY (collection, row_number),
                FOREIGN KEY (collection) REFERENCES csv_tables(collection)
                    ON DELETE CASCADE
            )
            """
        )
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("DELETE FROM csv_rows WHERE collection = ?", (collection,))
        connection.execute(
            """
            INSERT OR REPLACE INTO csv_tables (collection, source, columns_json)
            VALUES (?, ?, ?)
            """,
            (collection, table.source, json.dumps(table.columns)),
        )
        connection.executemany(
            """
            INSERT INTO csv_rows (collection, row_number, values_json)
            VALUES (?, ?, ?)
            """,
            (
                (collection, row.row_number, json.dumps(row.values))
                for row in table.rows
            ),
        )


def load_csv_tables(
    collections: list[str], chroma_path: str | Path
) -> list[CsvTable]:
    if not collections:
        return []
    database = _database_path(chroma_path)
    with _connection(database) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS csv_tables (
                collection TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                columns_json TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS csv_rows (
                collection TEXT NOT NULL,
                row_number INTEGER NOT NULL,
                values_json TEXT NOT NULL,
                PRIMARY KEY (collection, row_number)
            )
            """
        )
        tables: list[CsvTable] = []
        for collection in dict.fromkeys(collections):
            record = connection.execute(
                "SELECT source, columns_json FROM csv_tables WHERE collection = ?",
                (collection,),
            ).fetchone()
            if record is None:
                continue
            rows = connection.execute(
                """
                SELECT row_number, values_json FROM csv_rows
                WHERE collection = ? ORDER BY row_number
                """,
                (collection,),
            ).fetchall()
            tables.append(
                CsvTable(
                    source=record[0],
                    columns=tuple(json.loads(record[1])),
                    rows=tuple(
                        CsvRow(
                            row_number=row_number,
                            values=json.loads(values_json),
                        )
                        for row_number, values_json in rows
                    ),
                )
            )
    return tables

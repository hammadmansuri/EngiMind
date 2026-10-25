import csv
from dataclasses import dataclass
from io import StringIO

from engimind.ingestion.pdf_loader import Page


@dataclass(frozen=True)
class CsvRow:
    row_number: int
    values: dict[str, str]


@dataclass(frozen=True)
class CsvTable:
    source: str
    columns: tuple[str, ...]
    rows: tuple[CsvRow, ...]


def load_csv_table(data: bytes, source: str = "") -> CsvTable:
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("CSV files must use UTF-8 encoding.") from exc

    try:
        rows = csv.reader(StringIO(text, newline=""), strict=True)
        headers = next(rows, None)
        if headers is None or not any(header.strip() for header in headers):
            raise ValueError("CSV must contain a header row.")

        columns = [
            header.strip() or f"Column {index}"
            for index, header in enumerate(headers, start=1)
        ]
        parsed_rows: list[CsvRow] = []
        for row_number, row in enumerate(rows, start=2):
            if not any(value.strip() for value in row):
                continue

            values = {
                column: row[index] if index < len(row) else ""
                for index, column in enumerate(columns)
            }
            values.update(
                {
                    f"Extra column {index}": value
                    for index, value in enumerate(row[len(columns):], start=1)
                }
            )
            parsed_rows.append(CsvRow(row_number=row_number, values=values))
    except csv.Error as exc:
        raise ValueError(f"Invalid CSV file: {exc}") from exc

    if not parsed_rows:
        raise ValueError("CSV must contain at least one data row.")
    return CsvTable(source=source, columns=tuple(columns), rows=tuple(parsed_rows))


def load_csv_pages(
    data: bytes, rows_per_page: int = 1, max_chars: int | None = None
) -> list[Page]:
    table = load_csv_table(data)
    return csv_table_to_pages(table, rows_per_page, max_chars)


def csv_table_to_pages(
    table: CsvTable,
    rows_per_page: int = 1,
    max_chars: int | None = None,
) -> list[Page]:
    if rows_per_page < 1:
        raise ValueError("rows_per_page must be at least 1.")
    if max_chars is not None and max_chars < 1:
        raise ValueError("max_chars must be at least 1.")

    def format_row(row: CsvRow) -> str:
        values = "\n".join(
            f"{column}: {value}" for column, value in row.values.items()
        )
        return f"CSV row: {row.row_number}\n{values}"

    pages: list[Page] = []
    batch: list[CsvRow] = []
    batch_text: list[str] = []

    def append_batch() -> None:
        if not batch:
            return
        pages.append(
            Page(
                page_number=batch[0].row_number,
                end_page_number=batch[-1].row_number,
                text="\n".join(batch_text),
            )
        )
        batch.clear()
        batch_text.clear()

    for row in table.rows:
        row_text = format_row(row)
        candidate_text = "\n".join([*batch_text, row_text])
        if batch and (
            len(batch) >= rows_per_page
            or (max_chars is not None and len(candidate_text) > max_chars)
        ):
            append_batch()
        batch.append(row)
        batch_text.append(row_text)
    append_batch()
    return pages

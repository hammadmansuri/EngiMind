from dataclasses import dataclass
from io import BytesIO

from pypdf import PdfReader


@dataclass(frozen=True)
class Page:
    page_number: int  # 1-based, matches printed PDF pages for citations
    text: str
    end_page_number: int | None = None


def load_pdf_pages(data: bytes) -> list[Page]:
    reader = PdfReader(BytesIO(data))
    pages: list[Page] = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append(Page(page_number=i + 1, text=text.strip()))
    if not any(p.text for p in pages):
        raise ValueError(
            "No extractable text in this PDF (scanned image-only files need OCR)."
        )
    return pages

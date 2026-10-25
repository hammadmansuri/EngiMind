import re
import unittest

from engimind.ingestion.chunking import chunk_pages
from engimind.ingestion.pdf_loader import Page


class ChunkingBoundaryTests(unittest.TestCase):
    def test_chunk_pages_handles_tokens_longer_than_chunk_size(self):
        long_token = "x" * 1300
        text = f"Reference: {long_token} end of extracted text."

        chunks = chunk_pages(
            [Page(page_number=1, text=text)],
            source="long-token.pdf",
            chunk_size=200,
            chunk_overlap=40,
        )

        self.assertTrue(chunks)
        self.assertTrue(all(len(chunk.text) <= 200 for chunk in chunks))
        self.assertGreaterEqual(sum(chunk.text.count("x") for chunk in chunks), 1300)
        self.assertTrue(any("end of extracted text." in chunk.text for chunk in chunks))

    def test_chunk_pages_do_not_cut_mid_sentence(self):
        text = (
            "PERSONAL PITCH\n"
            "EXAMPLES\n\n"
            "Having a personal pitch ready is essential to starting an interview off on the right foot. "
            "Here are some example personal pitches from a university student, bootcamp graduate, a junior dev, "
            "and a self taught engineer.\n\n"
            "Example 1 - University Student\n"
            "I'm a graduating senior from XYZ university and have been tinkering with programming for as long as I "
            "can remember. While I love learning about theory and fundamentals, what really excites me is getting "
            "my hands dirty on projects. I have committed to a few open source repos such as FreeCodeCamp and "
            "Exercism's Python Site. For my capstone I am working on a site using Python and Django where "
            "students can keep track of new projects ideas called Idea Keeper. I'd really love to join a team where I "
            "can learn about a lot of different technologies and get exposure to the entire stack."
        )

        chunks = chunk_pages(
            [Page(page_number=1, text=text)],
            source="demo.pdf",
            chunk_size=500,
            chunk_overlap=80,
        )

        self.assertTrue(chunks)
        for chunk in chunks:
            self.assertFalse(
                re.search(r"[A-Za-z0-9]$", chunk.text) and not re.search(r"[.!?]$", chunk.text),
                f"Chunk ended mid-sentence: {chunk.text[-50:]!r}",
            )

    def test_chunk_pages_keep_section_heading_metadata(self):
        text = (
            "Annual Leave Policy\n\n"
            "Employees receive 15 days of annual leave after one year of service. "
            "Unused leave may carry over up to 5 days.\n\n"
            "Payroll\n\n"
            "Salary is paid on the last working day of each month."
        )

        chunks = chunk_pages(
            [Page(page_number=1, text=text)],
            source="demo.pdf",
            chunk_size=200,
            chunk_overlap=20,
        )

        self.assertTrue(any(c.section == "Annual Leave Policy" for c in chunks))
        self.assertTrue(any(c.section == "Payroll" for c in chunks))

    def test_chunk_pages_preserve_grouped_source_row_range(self):
        chunks = chunk_pages(
            [Page(page_number=2, end_page_number=4, text="CSV row: 2 a b c")],
            source="data.csv",
            chunk_size=200,
            chunk_overlap=0,
        )

        self.assertEqual(chunks[0].page, 2)
        self.assertEqual(chunks[0].end_page, 4)


if __name__ == "__main__":
    unittest.main()

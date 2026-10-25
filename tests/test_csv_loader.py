import unittest

from engimind.ingestion.csv_loader import load_csv_pages


class CsvLoaderTests(unittest.TestCase):
    def test_load_csv_rows_with_headers_and_row_numbers(self):
        pages = load_csv_pages(
            b"\xef\xbb\xbfname,notes\nAlice,\"First line\nsecond line\"\nBob,Engineer\n"
        )

        self.assertEqual([page.page_number for page in pages], [2, 3])
        self.assertIn("name: Alice", pages[0].text)
        self.assertIn("notes: First line\nsecond line", pages[0].text)
        self.assertIn("name: Bob", pages[1].text)

    def test_load_csv_preserves_extra_values_and_blank_headers(self):
        pages = load_csv_pages(b"name,,\nAlice,known,extra\n")

        self.assertEqual(len(pages), 1)
        self.assertIn("Column 2: known", pages[0].text)
        self.assertIn("Column 3: extra", pages[0].text)

    def test_load_csv_groups_rows_but_preserves_each_source_row_number(self):
        pages = load_csv_pages(
            b"name,score\nAlice,10\nBob,20\nCarol,30\n",
            rows_per_page=2,
        )

        self.assertEqual(
            [(page.page_number, page.end_page_number) for page in pages],
            [(2, 3), (4, 4)],
        )
        self.assertIn("CSV row: 2", pages[0].text)
        self.assertIn("CSV row: 3", pages[0].text)
        self.assertIn("name: Bob", pages[0].text)
        self.assertIn("CSV row: 4", pages[1].text)

    def test_load_csv_rejects_nonpositive_rows_per_page(self):
        with self.assertRaisesRegex(ValueError, "at least 1"):
            load_csv_pages(b"name\nAlice\n", rows_per_page=0)

    def test_load_csv_grouping_respects_character_limit(self):
        pages = load_csv_pages(
            b"name,value\nAlice,1234567890\nBob,1234567890\n",
            rows_per_page=10,
            max_chars=50,
        )

        self.assertEqual(len(pages), 2)
        self.assertTrue(all(len(page.text) <= 50 for page in pages))

    def test_load_csv_requires_valid_headers_and_data(self):
        for contents in (b"", b"\n", b"name\n"):
            with self.subTest(contents=contents):
                with self.assertRaises(ValueError):
                    load_csv_pages(contents)

    def test_load_csv_reports_malformed_quoted_fields(self):
        with self.assertRaisesRegex(ValueError, "Invalid CSV"):
            load_csv_pages(b'name,notes\nAlice,"unfinished\n')


if __name__ == "__main__":
    unittest.main()

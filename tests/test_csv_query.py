import unittest
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock

from engimind.ingestion.csv_loader import load_csv_table
from engimind.storage.csv_query import query_csv_tables
from engimind.storage.csv_store import load_csv_tables, save_csv_table
from engimind.core.config import Settings
from engimind.application.rag import answer_question, document_collection_name


class CsvQueryTests(unittest.TestCase):
    def setUp(self):
        self.table = load_csv_table(
            b"date,toss_winner,winner\n"
            b"2013-05-05,Mumbai Indians,Mumbai Indians\n"
            b"2010-04-21,Mumbai Indians,Mumbai Indians\n"
            b"2010-04-22,Chennai Super Kings,Chennai Super Kings\n",
            source="matches.csv",
        )

    def test_when_query_returns_every_matching_date_and_row(self):
        result = query_csv_tables(
            "When did Mumbai Indians win the toss?", [self.table]
        )

        self.assertIsNotNone(result)
        self.assertIn("2013-05-05 [matches.csv row 2]", result.answer)
        self.assertIn("2010-04-21 [matches.csv row 3]", result.answer)
        self.assertEqual([row.row_number for _, row in result.source_rows], [2, 3])

    def test_count_query_scans_all_rows(self):
        result = query_csv_tables(
            "How many times did Mumbai Indians win the toss?", [self.table]
        )

        self.assertIsNotNone(result)
        self.assertIn("2 time(s)", result.answer)
        self.assertNotIn("matches.csv rows", result.answer)

    def test_count_query_combines_multiple_tables(self):
        another = load_csv_table(
            b"date,toss_winner\n2011-05-01,Mumbai Indians\n",
            source="other.csv",
        )

        result = query_csv_tables(
            "How many times Mumbai Indians won the toss?",
            [self.table, another],
        )

        self.assertIsNotNone(result)
        self.assertIn("3 time(s)", result.answer)
        self.assertEqual(len(result.source_rows), 3)

    def test_non_structured_question_falls_back_to_rag(self):
        result = query_csv_tables("What is this file about?", [self.table])

        self.assertIsNone(result)

    def test_same_content_with_different_filenames_gets_separate_collections(self):
        self.assertNotEqual(
            document_collection_name("matches.csv", "same-hash"),
            document_collection_name("archive.csv", "same-hash"),
        )

    def test_structured_rows_persist_and_answer_without_vector_search(self):
        with TemporaryDirectory() as directory:
            save_csv_table("csv_test", self.table, directory)
            tables = load_csv_tables(["csv_test"], directory)
            self.assertEqual(tables, [self.table])

            settings = Settings(
                openai_api_key="unused",
                embedding_model="unused",
                chat_model="unused",
                chunk_size=800,
                chunk_overlap=100,
                top_k=1,
                chroma_path=directory,
                max_distance=0.5,
            )
            openai_client = Mock()
            openai_client.chat.completions.create.return_value = SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=(
                                "Mumbai Indians won the toss 2 times "
                                "(matches.csv rows 2, 3)."
                            )
                        )
                    )
                ]
            )
            result = answer_question(
                "How many times did Mumbai Indians win the toss?",
                "csv_test",
                settings,
                openai_client=openai_client,
            )

        self.assertIn("2 times", result.answer)
        self.assertEqual(len(result.chunks), 2)
        openai_client.chat.completions.create.assert_called_once()
        sent_prompt = openai_client.chat.completions.create.call_args.kwargs[
            "messages"
        ][1]["content"]
        self.assertIn("2 time(s)", sent_prompt)
        self.assertNotIn("matches.csv rows 2, 3", sent_prompt)
        self.assertIn("do not list matching rows", sent_prompt.lower())


if __name__ == "__main__":
    unittest.main()

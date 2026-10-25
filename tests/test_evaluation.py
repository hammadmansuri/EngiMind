import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from engimind.evaluation.metrics import EvaluationCase, chunk_identifier, judge_answer, recall_at_k
from engimind.application.rag import RagResult
from engimind.retrieval.store import RetrievedChunk


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.result = RagResult(
            answer="Employees receive 20 days of leave.",
            chunks=[
                RetrievedChunk(
                    text="Employees receive 20 days of leave.",
                    page=2,
                    source="policy.pdf",
                    distance=0.2,
                ),
                RetrievedChunk(
                    text="Unused leave is not carried forward.",
                    page=3,
                    source="policy.pdf",
                    distance=0.3,
                ),
            ],
            prompt="",
            refused=False,
        )

    def test_chunk_identifier_uses_source_and_page(self):
        self.assertEqual(chunk_identifier("policy.pdf", 2), "policy.pdf:2")

    def test_recall_at_k_measures_fraction_of_gold_locations_found(self):
        self.assertEqual(
            recall_at_k(("policy.pdf:2", "policy.pdf:8"), self.result),
            0.5,
        )

    def test_recall_requires_gold_location_labels(self):
        with self.assertRaisesRegex(ValueError, "(?i)at least one"):
            recall_at_k((), self.result)

    def test_recall_counts_gold_row_inside_grouped_row_range(self):
        grouped_result = RagResult(
            answer="",
            chunks=[
                RetrievedChunk(
                    text="CSV row: 2 ... CSV row: 4",
                    page=2,
                    end_page=4,
                    source="policy.csv",
                    distance=0.2,
                )
            ],
            prompt="",
            refused=False,
        )

        self.assertEqual(recall_at_k(("policy.csv:3",), grouped_result), 1.0)

    def test_judge_scores_and_returns_explanations(self):
        client = Mock()
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(
                            {
                                "relevance": 0.9,
                                "relevance_reason": "It answers the question.",
                                "groundedness": 1,
                                "groundedness_reason": "The context states the number.",
                            }
                        )
                    )
                )
            ]
        )

        scores = judge_answer(
            "How many leave days?",
            "20 days",
            self.result.answer,
            [chunk.text for chunk in self.result.chunks],
            openai_client=client,
            model="test-model",
        )

        self.assertEqual(scores.relevance, 0.9)
        self.assertEqual(scores.groundedness, 1.0)
        self.assertIn("context", scores.groundedness_reason)
        client.chat.completions.create.assert_called_once()

    def test_judge_rejects_out_of_range_scores(self):
        client = Mock()
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content='{"relevance": 2, "groundedness": 0.4, '
                        '"relevance_reason": "bad", '
                        '"groundedness_reason": "supported"}'
                    )
                )
            ]
        )

        with self.assertRaisesRegex(RuntimeError, "invalid score"):
            judge_answer(
                "question",
                "reference",
                "answer",
                [],
                openai_client=client,
                model="test-model",
            )


if __name__ == "__main__":
    unittest.main()

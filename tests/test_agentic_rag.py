import unittest
from types import SimpleNamespace
from unittest.mock import patch

from engimind.application.agentic_rag import StateGraph, _is_empty_question, answer_question_agentic


class AgenticRagTests(unittest.TestCase):
    def setUp(self):
        self.settings = SimpleNamespace(
            openai_api_key="test-key",
            embedding_model="test-embedding-model",
            chat_model="test-chat-model",
            chroma_path=".chroma",
            top_k=4,
            max_distance=0.55,
        )

    def test_agentic_rag_retries_with_refined_query_when_first_pass_is_weak(self):
        chunk = SimpleNamespace(
            text="Employees receive 20 days of annual leave after one year of service.",
            page=2,
            source="policy.pdf",
            distance=0.32,
            section="Annual Leave Policy",
        )

        with patch(
            "engimind.application.agentic_rag.query_collection",
            side_effect=[[], [chunk]],
        ), patch(
            "engimind.application.agentic_rag.OpenAI",
        ), patch(
            "engimind.application.agentic_rag._generate_answer",
            return_value="Grounded answer [p. 2]",
        ) as generate_answer:
            result = answer_question_agentic(
                "How many vacation days?",
                "pdf_test",
                self.settings,
                openai_client=SimpleNamespace(),
            )

        self.assertFalse(result.refused)
        self.assertEqual(result.answer, "Grounded answer [p. 2]")
        self.assertEqual(len(result.chunks), 1)
        self.assertEqual(result.chunks[0].page, 2)
        self.assertIn("retrieval", result.timings_ms)
        self.assertIn("generation", result.timings_ms)
        self.assertGreaterEqual(generate_answer.call_count, 1)

    def test_empty_question_is_clarified_without_retrieval(self):
        with patch(
            "engimind.application.agentic_rag.query_collection",
        ) as retrieve:
            result = answer_question_agentic(
                "   ",
                "pdf_test",
                self.settings,
                openai_client=SimpleNamespace(),
            )

        self.assertTrue(result.refused)
        self.assertIn("enter a question", result.answer.lower())
        retrieve.assert_not_called()

    def test_question_length_does_not_mark_query_as_empty(self):
        self.assertFalse(_is_empty_question("policy"))
        self.assertFalse(_is_empty_question("What YOLO11?"))

    def test_short_generic_question_reaches_retrieval(self):
        chunk = SimpleNamespace(
            text="YOLOv11 is an object detection architecture.",
            page=3,
            source="research.pdf",
            distance=0.3,
            section="general",
        )
        with patch("engimind.application.agentic_rag.query_collection", return_value=[chunk]) as retrieve, patch(
            "engimind.application.agentic_rag.chroma_client", return_value=SimpleNamespace()
        ), patch(
            "engimind.application.agentic_rag._generate_answer", return_value="YOLOv11 is an object detection architecture."
        ):
            result = answer_question_agentic(
                "What YOLO11?",
                "pdf_test",
                self.settings,
                openai_client=SimpleNamespace(),
            )

        retrieve.assert_called_once()
        self.assertFalse(result.refused)

    @unittest.skipIf(StateGraph is None, "LangGraph is not installed")
    def test_agentic_rag_clarifies_after_one_weak_rewritten_retry(self):
        with patch("engimind.application.agentic_rag.query_collection", return_value=[]) as retrieve, patch(
            "engimind.application.agentic_rag.chroma_client", return_value=SimpleNamespace()
        ), patch("engimind.application.agentic_rag._generate_answer") as generate_answer:
            result = answer_question_agentic(
                "How many vacation days?",
                "pdf_test",
                self.settings,
                openai_client=SimpleNamespace(),
            )

        self.assertTrue(result.refused)
        self.assertIn("vacation days", result.answer.lower())
        self.assertEqual(retrieve.call_count, 2)
        generate_answer.assert_not_called()


if __name__ == "__main__":
    unittest.main()

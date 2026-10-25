import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from engimind.retrieval.store import (
    RetrievedChunk,
    hybrid_rank_chunks,
    query_collection,
    query_rewrite,
    rerank_chunks,
)


class RerankChunksTests(unittest.TestCase):
    def test_rerank_prefers_high_overlap_even_when_distance_is_slightly_worse(self):
        query = "refund payment issue"
        chunks = [
            RetrievedChunk(
                text="Quarterly earnings rose sharply this year.",
                page=1,
                source="report.pdf",
                distance=0.10,
            ),
            RetrievedChunk(
                text="The refund policy covers payment issues and refund requests.",
                page=2,
                source="report.pdf",
                distance=0.45,
            ),
        ]

        ranked = rerank_chunks(query, chunks)

        self.assertEqual(ranked[0].page, 2)
        self.assertIn("refund", ranked[0].text.lower())

    def test_rerank_prefers_matching_section_name(self):
        query = "annual leave policy"
        chunks = [
            RetrievedChunk(
                text="Salary is paid on the last working day of each month.",
                page=1,
                source="policy.pdf",
                distance=0.12,
                section="Payroll",
            ),
            RetrievedChunk(
                text="Employees receive 15 days of annual leave after one year of service.",
                page=2,
                source="policy.pdf",
                distance=0.18,
                section="Annual Leave Policy",
            ),
        ]

        ranked = rerank_chunks(query, chunks)

        self.assertEqual(ranked[0].section, "Annual Leave Policy")
        self.assertIn("annual leave", ranked[0].text.lower())

    def test_hybrid_search_prefers_policy_terms_over_weak_semantic_match(self):
        query = "annual leave after 5 years"
        chunks = [
            RetrievedChunk(
                text="The company offers flexible vacation days for employees.",
                page=1,
                source="policy.pdf",
                distance=0.14,
                section="Benefits",
            ),
            RetrievedChunk(
                text="After 5 years of service, employees receive 18 days of annual leave.",
                page=8,
                source="policy.pdf",
                distance=0.38,
                section="Annual Leave Policy",
            ),
        ]

        ranked = hybrid_rank_chunks(query, chunks)

        self.assertEqual(ranked[0].page, 8)
        self.assertIn("annual leave", ranked[0].text.lower())

    def test_rerank_returns_empty_list_when_given_none(self):
        self.assertEqual(rerank_chunks("question", []), [])

    def test_rerank_removes_near_duplicate_chunks_and_keeps_specific_match(self):
        query = "annual leave after 5 years"
        chunks = [
            RetrievedChunk(
                text="The company has an annual leave policy for employees.",
                page=1,
                source="policy.pdf",
                distance=0.10,
            ),
            RetrievedChunk(
                text="Employees receive annual leave according to the company policy.",
                page=2,
                source="policy.pdf",
                distance=0.12,
            ),
            RetrievedChunk(
                text="After 5 years of continuous service, employees receive 18 days of annual leave.",
                page=8,
                source="policy.pdf",
                distance=0.28,
            ),
        ]

        ranked = rerank_chunks(query, chunks)

        self.assertEqual(len(ranked), 2)
        self.assertEqual(ranked[0].page, 8)
        self.assertIn("5 years", ranked[0].text)

    def test_query_rewrite_expands_common_synonyms_and_keeps_numbers(self):
        rewritten = query_rewrite("vacation policy after 5 years")

        self.assertIn("leave", rewritten.lower())
        self.assertIn("annual", rewritten.lower())
        self.assertIn("5", rewritten)
        self.assertIn("years", rewritten.lower())

    def test_query_collection_accepts_multiple_collection_names(self):
        client = Mock()
        collections = []
        for index, (source, text) in enumerate(
            (
                ("first.pdf", "annual leave policy"),
                ("second.pdf", "match toss result"),
            ),
            start=1,
        ):
            collection = Mock()
            collection.count.return_value = 1
            collection.query.return_value = {
                "documents": [[text]],
                "metadatas": [[{"page": index, "source": source, "section": ""}]],
                "distances": [[0.2]],
            }
            collections.append(collection)
        client.get_collection.side_effect = collections
        settings = SimpleNamespace(
            embedding_provider="openai",
            embedding_model="test-embedding-model",
        )

        with patch("engimind.retrieval.store.embed_texts", return_value=[[0.1, 0.2]]):
            chunks = query_collection(
                client,
                Mock(),
                settings,
                name=["first_collection", "second_collection"],
                question="matching content",
                k=3,
            )

        self.assertEqual(
            [call.args[0] for call in client.get_collection.call_args_list],
            ["first_collection", "second_collection"],
        )
        self.assertEqual(len(chunks), 2)


if __name__ == "__main__":
    unittest.main()

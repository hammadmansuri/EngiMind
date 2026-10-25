import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from engimind.ingestion.chunking import Chunk
from engimind.retrieval.store import upsert_chunks, validate_collection_embedding


class StoreIndexingTests(unittest.TestCase):
    def test_collection_embedding_validation_requires_reindex_after_switch(self):
        collection = Mock()
        collection.metadata = {
            "embedding_provider": "openai",
            "embedding_model": "text-embedding-3-small",
        }
        settings = SimpleNamespace(
            embedding_provider="ollama",
            embedding_model="all-minilm",
        )

        with self.assertRaisesRegex(ValueError, "Re-index this document"):
            validate_collection_embedding(collection, settings)

    def test_upsert_chunks_embeds_and_persists_bounded_batches_with_progress(self):
        chunks = [
            Chunk(
                chunk_id=f"matches.csv:{index}",
                text=f"winner: team {index}",
                page=index,
                source="matches.csv",
            )
            for index in range(513)
        ]
        collection = Mock()
        collection.count.return_value = len(chunks)
        client = Mock()
        client.get_or_create_collection.return_value = collection
        progress = Mock()
        settings = SimpleNamespace(embedding_model="test-embedding")

        with patch(
            "engimind.retrieval.store.embed_texts",
            side_effect=lambda _client, _settings, texts, **_kwargs: [
                [float(index)] for index, _ in enumerate(texts)
            ],
        ) as embed:
            count = upsert_chunks(
                client,
                Mock(),
                settings,
                name="matches_collection",
                chunks=chunks,
                progress_callback=progress,
            )

        self.assertEqual(count, 513)
        self.assertEqual(
            [len(call.args[2]) for call in embed.call_args_list], [64] * 8 + [1]
        )
        self.assertEqual(
            [len(call.kwargs["ids"]) for call in collection.upsert.call_args_list],
            [64] * 8 + [1],
        )
        self.assertEqual(
            progress.call_args_list,
            [
                *(unittest.mock.call(index, 513) for index in range(64, 513, 64)),
                unittest.mock.call(513, 513),
            ],
        )


if __name__ == "__main__":
    unittest.main()

import os
import unittest
from unittest.mock import patch

from engimind.core.config import load_settings


class ConfigTests(unittest.TestCase):
    def test_ollama_provider_selects_local_embedding_model_and_url(self):
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "test-key",
                "EMBEDDING_PROVIDER": "ollama",
                "OLLAMA_EMBEDDING_MODEL": "all-minilm",
                "OLLAMA_BASE_URL": "http://localhost:11434/",
            },
        ):
            settings = load_settings()

        self.assertEqual(settings.embedding_provider, "ollama")
        self.assertEqual(settings.embedding_model, "all-minilm")
        self.assertEqual(settings.ollama_base_url, "http://localhost:11434")

    def test_provider_can_be_selected_without_changing_environment(self):
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "test-key",
                "EMBEDDING_PROVIDER": "openai",
            },
        ):
            settings = load_settings(embedding_provider="ollama")

        self.assertEqual(settings.embedding_provider, "ollama")

    def test_rejects_unknown_embedding_provider(self):
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "test-key",
                "EMBEDDING_PROVIDER": "unsupported",
            },
        ):
            with self.assertRaisesRegex(ValueError, "EMBEDDING_PROVIDER"):
                load_settings()


if __name__ == "__main__":
    unittest.main()

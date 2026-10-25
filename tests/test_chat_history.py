import unittest

from engimind.application.rag import format_history_context


class ChatHistoryPromptTests(unittest.TestCase):
    def test_format_history_context_includes_recent_turns(self):
        history = [
            ("What is the main idea?", "It focuses on AI safety."),
            ("Can you elaborate?", "It argues for careful design."),
        ]

        rendered = format_history_context(history)

        self.assertIn("Previous question 1:", rendered)
        self.assertIn("What is the main idea?", rendered)
        self.assertIn("Previous answer 1:", rendered)
        self.assertIn("It focuses on AI safety.", rendered)
        self.assertIn("Previous question 2:", rendered)
        self.assertIn("Can you elaborate?", rendered)

    def test_format_history_context_handles_empty_history(self):
        self.assertEqual(format_history_context([]), "No prior conversation.")


if __name__ == "__main__":
    unittest.main()

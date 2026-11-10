import unittest

from src.evaluate_chunking import _chunk_statistics


class ChunkingEvaluationTests(unittest.TestCase):
    def test_chunk_statistics_report_count_and_token_range(self):
        stats = _chunk_statistics([
            {"token_count": 100},
            {"token_count": 200},
            {"token_count": 300},
        ])

        self.assertEqual(stats["chunk_count"], 3)
        self.assertEqual(stats["average_tokens"], 200)
        self.assertEqual(stats["min_tokens"], 100)
        self.assertEqual(stats["max_tokens"], 300)


if __name__ == "__main__":
    unittest.main()

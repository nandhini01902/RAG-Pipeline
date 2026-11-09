import unittest

from src.chunk_text import searchable_chunk_text
from src.retrieval import (
    _context_limit,
    _expand_sparse_query,
    _select_page_diverse_chunks,
)


class RetrievalHelperTests(unittest.TestCase):
    def test_searchable_text_includes_section_and_chunk_body(self):
        chunk = {
            "section": "MRI lumbar spine",
            "chunk_text": "A right paracentral disc protrusion is described.",
        }

        text = searchable_chunk_text(chunk)

        self.assertIn("MRI lumbar spine", text)
        self.assertIn("right paracentral disc protrusion", text)

    def test_page_cap_preserves_rank_and_avoids_overrepresenting_one_page(self):
        chunks = [
            {"page": 1, "chunk_id": "p1a"},
            {"page": 1, "chunk_id": "p1b"},
            {"page": 1, "chunk_id": "p1c"},
            {"page": 2, "chunk_id": "p2a"},
            {"page": 3, "chunk_id": "p3a"},
        ]

        selected = _select_page_diverse_chunks(chunks, top_k=4)

        self.assertEqual(
            [chunk["chunk_id"] for chunk in selected],
            ["p1a", "p1b", "p2a", "p3a"],
        )

    def test_domain_terms_are_added_to_sparse_search(self):
        expanded = _expand_sparse_query(
            "What imaging examinations were performed?"
        )

        self.assertIn("radiology", expanded)
        self.assertIn("MRI", expanded)

    def test_narrow_queries_send_only_focused_evidence(self):
        self.assertEqual(
            _context_limit(
                "What medications were administered in the emergency department?",
                8,
            ),
            (2, False, True),
        )
        self.assertEqual(
            _context_limit("What imaging examinations were performed?", 8),
            (8, True, False),
        )
        self.assertEqual(
            _context_limit("What were the findings of the lumbar spine MRI?", 8),
            (2, False, True),
        )


if __name__ == "__main__":
    unittest.main()

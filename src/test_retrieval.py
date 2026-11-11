import unittest
from unittest.mock import patch

import numpy as np

from src.chunk_text import searchable_chunk_text
from src import retrieval
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

    def test_page_cap_distinguishes_same_page_number_in_different_documents(self):
        chunks = [
            {"document_id": "doc-a", "page": 1, "chunk_id": "a1"},
            {"document_id": "doc-b", "page": 1, "chunk_id": "b1"},
        ]

        selected = _select_page_diverse_chunks(chunks, top_k=2)

        self.assertEqual(
            [chunk["chunk_id"] for chunk in selected],
            ["a1", "b1"],
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

    def test_uploaded_chunks_are_indexed_in_memory_and_keep_document_names(self):
        chunks = [
            {
                "chunk_id": "meds-p1-c1",
                "document_id": "meds-a1",
                "document_name": "medications.pdf",
                "page": 1,
                "section": "Medication list",
                "chunk_text": "Naproxen 500 mg twice daily.",
            },
            {
                "chunk_id": "scan-p1-c1",
                "document_id": "scan-b2",
                "document_name": "scan.pdf",
                "page": 1,
                "section": "MRI findings",
                "chunk_text": "Lumbar MRI shows a disc protrusion.",
            },
        ]

        class FakeModel:
            def encode(self, value, **_kwargs):
                if isinstance(value, str):
                    return np.array([1.0, 0.0])
                return np.array([[1.0, 0.0] for _ in value])

        with patch.object(
            retrieval,
            "QDRANT_PATH",
            "path-that-must-not-be-used",
        ):
            results = retrieval.retrieve_chunks(
                "What is the Naproxen dosage?",
                FakeModel(),
                top_k=2,
                chunks=chunks,
            )

        self.assertTrue(results)
        self.assertEqual(results[0]["document_name"], "medications.pdf")
        self.assertEqual(results[0]["document_id"], "meds-a1")
        self.assertIn("Naproxen", results[0]["chunk_text"])
        self.assertEqual(chunks[0]["page"], 1)


if __name__ == "__main__":
    unittest.main()

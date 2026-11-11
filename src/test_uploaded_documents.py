import unittest
from types import SimpleNamespace

import pymupdf

from src.uploaded_documents import (
    chunk_uploaded_pdfs,
    extract_uploaded_pdf,
)


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return SimpleNamespace(ids=text.split())

    def decode(self, tokens, skip_special_tokens=True):
        return " ".join(tokens)


def create_pdf_bytes(text):
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
    pdf_bytes = document.tobytes()
    document.close()
    return pdf_bytes


class UploadedDocumentTests(unittest.TestCase):
    def test_extract_uploaded_pdf_preserves_page_numbers_and_text(self):
        pdf_bytes = create_pdf_bytes("Medication list\nNaproxen 500 mg.")

        pages = extract_uploaded_pdf("record.pdf", pdf_bytes)

        self.assertEqual(len(pages), 1)
        self.assertEqual(pages[0]["page"], 1)
        self.assertIn("Naproxen 500 mg.", pages[0]["text"])

    def test_chunking_multiple_documents_preserves_source_metadata(self):
        uploads = [
            ("record.pdf", create_pdf_bytes("MEDICATIONS\nNaproxen 500 mg.")),
            ("record.pdf", create_pdf_bytes("IMAGING\nLumbar MRI findings.")),
        ]

        chunks = chunk_uploaded_pdfs(uploads, FakeTokenizer())

        self.assertEqual(len(chunks), 2)
        self.assertNotEqual(
            chunks[0]["document_id"],
            chunks[1]["document_id"],
        )
        self.assertEqual(
            [chunk["document_name"] for chunk in chunks],
            ["record.pdf", "record.pdf"],
        )
        self.assertTrue(all(chunk["page"] == 1 for chunk in chunks))

    def test_rejects_non_pdf_and_scanned_pdf_without_text(self):
        with self.assertRaisesRegex(ValueError, "not a readable PDF"):
            extract_uploaded_pdf("not-a-pdf.pdf", b"not a PDF")

        scanned = pymupdf.open()
        scanned.new_page()
        scanned_bytes = scanned.tobytes()
        scanned.close()
        with self.assertRaisesRegex(ValueError, "no extractable text"):
            extract_uploaded_pdf("scanned.pdf", scanned_bytes)

    def test_rejects_duplicate_uploads(self):
        pdf_bytes = create_pdf_bytes("MEDICATIONS\nNaproxen 500 mg.")
        uploads = [
            ("record.pdf", pdf_bytes),
            ("record.pdf", pdf_bytes),
        ]

        with self.assertRaisesRegex(ValueError, "uploaded more than once"):
            chunk_uploaded_pdfs(uploads, FakeTokenizer())


if __name__ == "__main__":
    unittest.main()

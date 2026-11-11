import hashlib
import re
from pathlib import Path

import pymupdf

from src.chunk import (
    MODEL_NAME,
    chunk_section,
    extract_patient_name,
    extract_sections,
)


def extract_uploaded_pdf(file_name, pdf_bytes):
    if not pdf_bytes:
        raise ValueError(f"{file_name} is empty.")

    try:
        document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    except (pymupdf.FileDataError, ValueError) as error:
        raise ValueError(f"{file_name} is not a readable PDF.") from error

    with document:
        pages = [
            {"page": page_number, "text": page.get_text("text")}
            for page_number, page in enumerate(document, start=1)
        ]

    if not any(page["text"].strip() for page in pages):
        raise ValueError(
            f"{file_name} contains no extractable text. "
            "Scanned PDFs need OCR before they can be searched."
        )
    return pages


def _document_id(file_name, pdf_bytes):
    stem = Path(file_name).stem
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "-", stem).strip("-._")
    safe_stem = safe_stem[:48] or "document"
    digest = hashlib.sha256(pdf_bytes).hexdigest()[:10]
    return f"{safe_stem}-{digest}"


def chunk_uploaded_pdfs(uploaded_files, tokenizer):
    chunks = []
    seen_document_ids = set()

    for file_name, pdf_bytes in uploaded_files:
        document_id = _document_id(file_name, pdf_bytes)
        if document_id in seen_document_ids:
            raise ValueError(f"{file_name} was uploaded more than once.")
        seen_document_ids.add(document_id)

        pages = extract_uploaded_pdf(file_name, pdf_bytes)
        for page in pages:
            text = page["text"].strip()
            if not text:
                continue

            patient_name = extract_patient_name(text)
            for section in extract_sections(text):
                for section_chunk in chunk_section(section, tokenizer):
                    chunk_text = section_chunk["text"].strip()
                    if not chunk_text:
                        continue

                    chunk_number = len(chunks) + 1
                    chunks.append({
                        "chunk_id": (
                            f"{document_id}_P{page['page']}_C{chunk_number}"
                        ),
                        "document_id": document_id,
                        "document_name": Path(file_name).name,
                        "page": page["page"],
                        "section": section_chunk["section"],
                        "patient_name": patient_name,
                        "chunk_text": chunk_text,
                    })

    if not chunks:
        raise ValueError("No searchable text was found in the uploaded PDFs.")
    return chunks


def load_chunk_tokenizer():
    from tokenizers import Tokenizer

    return Tokenizer.from_pretrained(MODEL_NAME)

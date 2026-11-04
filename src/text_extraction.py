import pymupdf as fitz
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
PDF_PATH = BASE_DIR / "data" / "Synthetic_Medical_Record_Exercise_Whitfield 1.pdf"
OUTPUT_PATH = BASE_DIR / "data" / "raw" / "extracted_text.txt"

def extract_pdf(pdf_path):
    doc = fitz.open(pdf_path)

    pages = []

    for page_number, page in enumerate(doc, start=1):
        text = page.get_text("text")

        pages.append({
            "page": page_number,
            "text": text
        })

    return pages


def main():
    pages = extract_pdf(str(PDF_PATH))
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(repr(pages), encoding="utf-8")
    print(f"Extracted {len(pages)} pages to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

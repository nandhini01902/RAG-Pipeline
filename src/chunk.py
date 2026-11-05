
import json
import re
from pathlib import Path
from tokenizers import Tokenizer


BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR / "data" / "processed" / "cleaned_records.json"
)

OUTPUT_FILE = (
    BASE_DIR / "data" / "processed" / "chunks.json"
)




MODEL_NAME = "BAAI/bge-small-en-v1.5"

CHUNK_SIZE = 450
CHUNK_OVERLAP = 80


def encode_ids(tokenizer, text):
    return tokenizer.encode(
        text,
        add_special_tokens=False
    ).ids


def get_page_text(page):

    possible_fields = [
        "text",
        "page_text",
        "content",
        "extracted_text"
    ]

    for field in possible_fields:
        if page.get(field):
            return str(page[field])

    raise ValueError(
        f"No text field found for page: {page}"
    )




def is_heading(paragraph):

    paragraph = paragraph.strip()

    if not paragraph:
        return False

    # Markdown headings
    if re.match(r"^#{1,6}\s+", paragraph):
        return True

    # Numbered headings: 1. Introduction, 2. Symptoms
    if re.match(
        r"^\d+(?:\.\d+)*[.)]?\s+[A-Z]",
        paragraph
    ):
        return True

    # Short, standalone uppercase headings
    if (
        len(paragraph.split()) <= 12
        and paragraph.isupper()
        and any(char.isalpha() for char in paragraph)
    ):
        return True

    return False


def clean_heading(heading):

    heading = re.sub(r"^#{1,6}\s*", "", heading)

    return heading.strip()


def extract_sections(text):

    paragraphs = [
        p.strip()
        for p in re.split(r"\n\s*\n", text)
        if p.strip()
    ]

    sections = []

    current_heading = "General Information"
    current_paragraphs = []

    for paragraph in paragraphs:

        if is_heading(paragraph):

            if current_paragraphs:
                sections.append({
                    "section": current_heading,
                    "paragraphs": current_paragraphs
                })

            current_heading = clean_heading(paragraph)
            current_paragraphs = []

        else:
            current_paragraphs.append(paragraph)

    # Save the final section
    if current_paragraphs:
        sections.append({
            "section": current_heading,
            "paragraphs": current_paragraphs
        })

    return sections




def split_long_paragraph(text, tokenizer):

    token_ids = encode_ids(tokenizer, text)

    chunks = []

    start = 0

    while start < len(token_ids):

        end = min(
            start + CHUNK_SIZE,
            len(token_ids)
        )

        chunk_ids = token_ids[start:end]

        chunk_text = tokenizer.decode(
            chunk_ids,
            skip_special_tokens=True
        ).strip()

        if chunk_text:
            chunks.append(chunk_text)

        if end == len(token_ids):
            break

        start = end - CHUNK_OVERLAP

    return chunks



def chunk_section(section, tokenizer):

    heading = section["section"]
    paragraphs = section["paragraphs"]

    section_chunks = []

    current_paragraphs = []
    current_token_count = 0

    for paragraph in paragraphs:

        paragraph_tokens = encode_ids(tokenizer, paragraph)

        paragraph_size = len(paragraph_tokens)

        # Handle paragraphs larger than the target size
        if paragraph_size > CHUNK_SIZE:

            # Save previously collected paragraphs
            if current_paragraphs:

                section_chunks.append({
                    "section": heading,
                    "text": "\n\n".join(current_paragraphs)
                })

                current_paragraphs = []
                current_token_count = 0

            # Split the large paragraph using overlap
            long_chunks = split_long_paragraph(
                paragraph,
                tokenizer
            )

            for chunk_text in long_chunks:

                section_chunks.append({
                    "section": heading,
                    "text": chunk_text
                })

            continue

        # Check whether adding this paragraph exceeds
        # the target chunk size
        if (
            current_token_count + paragraph_size
            > CHUNK_SIZE
        ):

            # Save the existing chunk
            section_chunks.append({
                "section": heading,
                "text": "\n\n".join(current_paragraphs)
            })

            # Preserve the last 80 tokens as context
            previous_text = "\n\n".join(
                current_paragraphs
            )

            previous_tokens = encode_ids(tokenizer, previous_text)

            overlap_ids = previous_tokens[
                -CHUNK_OVERLAP:
            ]

            # Do not exceed the target size
            available_overlap = min(
                len(overlap_ids),
                CHUNK_SIZE - paragraph_size
            )

            overlap_text = tokenizer.decode(
                overlap_ids[-available_overlap:],
                skip_special_tokens=True
            ).strip() if available_overlap else ""

            current_paragraphs = (
                [overlap_text, paragraph]
                if overlap_text
                else [paragraph]
            )

            current_token_count = (
                available_overlap + paragraph_size
            )

        else:

            current_paragraphs.append(paragraph)
            current_token_count += paragraph_size

    # Save remaining paragraphs
    if current_paragraphs:

        section_chunks.append({
            "section": heading,
            "text": "\n\n".join(current_paragraphs)
        })

    return section_chunks



def create_chunks():

    # Load cleaned PDF pages
    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        pages = json.load(file)

    # Load BGE tokenizer
    tokenizer = Tokenizer.from_pretrained(
        MODEL_NAME
    )

    all_chunks = []

    # Process one PDF page at a time
    for page in pages:

        page_number = page["page"]

        document_id = page.get(
            "document_id",
            "UNKNOWN"
        )

        text = get_page_text(page)

        # Identify headings and paragraphs
        sections = extract_sections(text)

        # Process each logical section
        for section in sections:

            section_chunks = chunk_section(
                section,
                tokenizer
            )

            for chunk in section_chunks:

                chunk_text = chunk["text"].strip()

                if not chunk_text:
                    continue

                chunk_id = len(all_chunks) + 1

                all_chunks.append({

                    "chunk_id": (
                        f"{document_id}_"
                        f"P{page_number}_"
                        f"C{chunk_id}"
                    ),

                    "document_id": document_id,

                    "page": page_number,

                    "section": chunk["section"],

                    "chunk_text": chunk_text,

                    "token_count": len(
                        encode_ids(tokenizer, chunk_text)
                    )

                })

    # Create output directory
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save chunks
    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            all_chunks,
            file,
            indent=4,
            ensure_ascii=False
        )

    # Display summary
    print("\nCHUNKING COMPLETED")

    print(f"Total PDF pages: {len(pages)}")

    print(f"Total chunks: {len(all_chunks)}")

    print(
        f"Average chunk size: "
        f"{sum(c['token_count'] for c in all_chunks) / max(len(all_chunks), 1):.2f}"
    )

    print(
        f"Chunks exceeding {CHUNK_SIZE} tokens: "
        f"{sum(c['token_count'] > CHUNK_SIZE for c in all_chunks)}"
    )

    print(f"Output file: {OUTPUT_FILE}")


if __name__ == "__main__":
    create_chunks()

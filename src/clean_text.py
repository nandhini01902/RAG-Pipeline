from pathlib import Path
import ast
import json
import re
from collections import Counter

BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_FILE = BASE_DIR / "data" / "raw" / "extracted_text.txt"
OUTPUT_DIR = BASE_DIR / "data" / "processed"

# Only unambiguous OCR spacing/punctuation repairs. Do not add clinical-value changes.
OCR_REPLACEMENTS = {
    "mid andlower": "mid and lower",
    "numbnessin": "numbness in",
    "reactive.Abdomen": "reactive. Abdomen",
    "lighttouch": "light touch",
    "PhysicalEXAMINATION": "Physical Examination",
    "PHYSICALEXAMINATION": "PHYSICAL EXAMINATION",
    "rightfoot": "right foot",
    "oneweek": "one week",
    "titratedto": "titrated to",
    "theright": "the right",
    "PROVIDERSAND FACILITIES": "PROVIDERS AND FACILITIES",
    "PatientName": "Patient Name",
    "MRNPCG": "MRN PCG",
    "Lumbarspine": "Lumbar spine",
    "lumbarspine": "lumbar spine",
    "Gabapentin300": "Gabapentin 300",
    "resultingmild": "resulting mild",
    "coordinatorreferred": "coordinator referred",
    "views.Negative": "views. Negative",
    "paraspinalsL5": "paraspinals L5",
    "andskin": "and skin",
    "ofbusiness": "of business",
    "andmay": "and may",
    "short-termdisability": "short-term disability",
    "Physicaltherapy": "Physical therapy",
    "CareReport": "Care Report",
    "trafficsignal": "traffic signal",
    "palpablebilaterally": "palpable bilaterally",
    "pain-limited": "pain-limited",
    "Pain7/10": "Pain 7/10",
    "Pain6/10": "Pain 6/10",
    "2.Mild": "2. Mild",
    "MRI lumbarspine": "MRI lumbar spine",
}

HEADER_PATTERNS = [
    re.compile(r"^WHITFIELD, MARCUS D\..*CLAIM PI-2024-8871$", re.I),
    re.compile(r"^PRODUCED 01/22/2025$", re.I),
    re.compile(r"^SYNTHETIC TRAINING RECORD.*FOR INTERVIEW ASSESSMENT USE ONLY$", re.I),
    re.compile(r"^LTM-PI-\d{5}$", re.I),  # kept separately as document_id metadata
]

def normalize_line(line: str) -> str:
    line = line.replace("\\u00a0", " ")
    line = re.sub(r"[ \t]+", " ", line).strip()
    for old, new in OCR_REPLACEMENTS.items():
        line = line.replace(old, new)
    # Add missing space after punctuation only when immediately followed by a letter.
    line = re.sub(r"([.!?;:])(?=[A-Z])", r"\1 ", line)
    # Normalize spacing around common separators without changing values.
    line = re.sub(r"\s+([,;:])", r"\1", line)
    line = re.sub(r"([,;:])(?=\S)", r"\1 ", line)
    return line.strip()

def clean_page_text(text: str):
    removed_header_lines = []
    kept = []
    document_id = None

    for raw_line in text.splitlines():
        line = normalize_line(raw_line)
        if not line:
            kept.append("")
            continue
        if re.fullmatch(r"LTM-PI-\d{5}", line, flags=re.I):
            document_id = line
            continue
        if any(pattern.match(line) for pattern in HEADER_PATTERNS):
            removed_header_lines.append(line)
            continue
        kept.append(line)

    # Join wrapped prose lines conservatively. Preserve blank-line paragraph boundaries,
    # headings, list items, and label/value lines.
    output = []
    for line in kept:
        if not line:
            if output and output[-1] != "":
                output.append("")
            continue

        is_heading = (
            line.isupper()
            or line.endswith(":")
            or re.match(r"^(EXAM \d+:|DIAGNOSES$|IMPRESSION:|FINDINGS:|PLAN$)", line)
        )
        is_list = bool(re.match(r"^[–—-]\s|^\d+\.\s", line))
        is_label = len(line) < 55 and not re.search(r"[.!?]$", line) and (
            line in {"Patient", "Date of Birth", "Date of Service", "Provider",
                     "Allergies", "Arrival", "MRN", "Position", "Employee ID"}
            or (line.istitle() and len(line.split()) <= 5)
        )

        if output and output[-1] and not is_heading and not is_list and not is_label:
            previous = output[-1]
            previous_is_heading = previous.isupper() or previous.endswith(":")
            previous_is_list = bool(re.match(r"^[–—-]\s|^\d+\.\s", previous))
            # Join a line-wrap only when the previous line appears to be an unfinished sentence.
            if (not previous_is_heading and not previous_is_list
                    and not re.search(r"[.!?:;]$", previous)
                    and not re.match(r"^(BP |HR |RR |SpO2 |Pain |[0-9]{2}/[0-9]{2}/[0-9]{4}$)", line)):
                output[-1] = previous + " " + line
                continue
        output.append(line)

    cleaned = "\n".join(output)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned, document_id, removed_header_lines

def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}\n"
            "Place the extracted text file next to the medical_pdf_cleaning folder "
            "or update INPUT_FILE at the top of this script."
        )

    # The provided extraction is a Python-literal list of dictionaries, not strict JSON.
    pages = ast.literal_eval(INPUT_FILE.read_text(encoding="utf-8"))
    cleaned_records = []
    excluded_documents = []
    issues = []
    total_removed_headers = 0

    for page in pages:
        page_number = int(page.get("page", 0))
        original = str(page.get("text", ""))
        cleaned, document_id, removed_headers = clean_page_text(original)
        total_removed_headers += len(removed_headers)

        # Explicit known source issue from manual review: page 16 is a different patient.
        if page_number == 16:
            excluded_documents.append({
                "page": page_number,
                "document_id": document_id,
                "reason": "Different patient record; exclude from the target patient's RAG corpus.",
                "cleaned_text_for_audit": cleaned,
            })
            issues.append({
                "type": "wrong_patient_document",
                "pages": [16],
                "action": "Excluded from cleaned_records.json; retained in this report for audit.",
            })
            continue

        cleaned_records.append({
            "page": page_number,
            "document_id": document_id,
            "text": cleaned,
            "cleaning": {
                "repeated_headers_removed": len(removed_headers),
                "raw_text_preserved": True,
            },
        })

    # Keep known contradictions as flags; never choose which version is correct.
    issues.extend([
        {
            "type": "conflicting_duplicate_progress_notes",
            "pages": [10, 12],
            "details": "Both are dated 04/09/2024 and visit 14/24, but pain score, SLR, EHL strength and clinical status differ; page 10 is signed and page 12 is an unsigned draft.",
            "action": "Retain both as separate source records; review against original documents.",
        },
        {
            "type": "date_of_injury_conflict",
            "pages": [7],
            "details": "Page 7 reports a date of injury of 17/02/2024, while other records describe the collision as 02/11/2024.",
            "action": "Preserve both dates and verify against the original record.",
        },
        {
            "type": "laterality_conflict",
            "pages": [14],
            "details": "EMG describes chronic LEFT L5 radiculopathy with active denervation; other records predominantly describe RIGHT L5 symptoms.",
            "action": "Do not change left/right; flag for source review.",
        },
        {
            "type": "allergy_medication_conflict",
            "pages": [2, 3, 5, 6, 19],
            "details": "Codeine allergy/rash is documented in clinical records, while acetaminophen-codeine appears in the dispensing record; pharmacy page 19 says no allergies documented.",
            "action": "Preserve source statements and flag for review; do not infer whether the prescription was taken.",
        },
        {
            "type": "missing_requested_documents",
            "pages": [22],
            "details": "The production log states that the cervical MRI report, legible duplicate of the 05/21 handwritten note, 06/11 injection clinical documentation, and PT visits 15–22 notes were not produced/located.",
            "action": "Keep missing-document statements; do not create substitute content.",
        },
        {
            "type": "illegible_handwriting",
            "pages": [13],
            "details": "Handwritten content contains illegible portions.",
            "action": "Retain [illegible] markers and annotations; do not guess the words.",
        },
    ])

    # A simple duplicate candidate check based on normalized full text.
    seen = {}
    duplicate_candidates = []
    for rec in cleaned_records:
        key = re.sub(r"\W+", "", rec["text"].lower())
        if key in seen:
            duplicate_candidates.append({"pages": [seen[key], rec["page"]]})
        else:
            seen[key] = rec["page"]

    quality_report = {
        "pages_in_source": len(pages),
        "pages_in_main_corpus": len(cleaned_records),
        "pages_excluded": [16],
        "repeated_header_lines_removed": total_removed_headers,
        "duplicate_candidates_exact_text": duplicate_candidates,
        "issues_requiring_human_review": issues,
        "excluded_documents_for_audit": excluded_documents,
        "cleaning_policy": [
            "Only spacing, punctuation and line-wrap formatting were automatically repaired.",
            "Clinical facts, dates, laterality, medication names/doses, diagnoses and measurements were not inferred or reconciled.",
            "Original source file was not modified.",
            "Every included record retains its original page number and document ID.",
        ],
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records_path = OUTPUT_DIR / "cleaned_records.json"
    report_path = OUTPUT_DIR / "data_quality_report.json"

    records_path.write_text(
        json.dumps(cleaned_records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    report_path.write_text(
        json.dumps(quality_report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Source pages: {len(pages)}")
    print(f"Records saved to main corpus: {len(cleaned_records)}")
    print("Excluded pages: 16 (different patient)")
    print(f"Repeated header/footer lines removed: {total_removed_headers}")
    print(f"Saved cleaned records: {records_path}")
    print(f"Saved quality report: {report_path}")

if __name__ == "__main__":
    main()

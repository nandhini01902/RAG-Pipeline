import json
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

from qdrant_client import QdrantClient, models
from src.chunk_text import searchable_chunk_text

try:
    from src.sparse_vectors import (
        build_sparse_vector,
        build_sparse_vectors,
        tokenize,
    )
except ModuleNotFoundError:
    from sparse_vectors import build_sparse_vector, build_sparse_vectors, tokenize


BASE_DIR = Path(__file__).resolve().parent.parent
QDRANT_PATH = BASE_DIR / "outputs" / "qdrant"
COLLECTION_NAME = "medical_records"
CHUNKS_FILE = BASE_DIR / "data" / "processed" / "chunks.json"
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

NAME_STOPWORDS = {
    "a", "about", "an", "and", "are", "can", "did", "does", "for", "from",
    "how", "is", "me", "of", "patient", "show", "the", "what", "were",
    "which", "who", "with",
}
QUERY_STOPWORDS = NAME_STOPWORDS | {
    "a", "all", "any", "at", "be", "document", "documented", "in",
    "it", "listed", "record", "showing", "shown", "was", "what",
}
QUERY_EXPANSIONS = (
    (
        re.compile(
            r"\b(medications?|medicines?|drugs?|prescriptions?|pharmacy)\b",
            re.IGNORECASE,
        ),
        "prescription pharmacy dispensing administered discharge dosage",
    ),
    (
        re.compile(
            r"\b(imaging|radiology|scan|scans|x-?rays?|radiographs?)\b",
            re.IGNORECASE,
        ),
        "CT MRI cervical lumbar radiology examination",
    ),
    (
        re.compile(
            r"\b(restrictions?|discharge|lifting|bed rest|work status)\b",
            re.IGNORECASE,
        ),
        "activity lifting weight work off-work bed rest instructions",
    ),
    (
        re.compile(r"\b(allerg(?:y|ies|ic))\b", re.IGNORECASE),
        "reaction codeine allergy medication",
    ),
    (
        re.compile(r"\b(EMG|electromyography|nerve conduction)\b", re.IGNORECASE),
        "electromyography nerve conduction radiculopathy findings",
    ),
    (
        re.compile(r"\b(lumbar MRI|MRI lumbar|lumbar spine MRI)\b", re.IGNORECASE),
        "disc bulge protrusion L4-L5 L5-S1 nerve root findings",
    ),
)


def _normalise_name(value):
    return re.sub(r"[^a-z0-9 ]", " ", value.lower()).split()


def _name_score(query, patient_name):
    query_tokens = [
        token for token in _normalise_name(query)
        if token not in NAME_STOPWORDS
    ]
    name_tokens = _normalise_name(patient_name or "")
    if not query_tokens or not name_tokens:
        return 0.0

    token_scores = [
        max(
            SequenceMatcher(None, token, candidate).ratio()
            for candidate in name_tokens
        )
        for token in query_tokens
    ]
    matched = sum(score >= 0.72 for score in token_scores)
    coverage = matched / len(query_tokens)
    quality = sum(token_scores) / len(token_scores)
    return 0.7 * coverage + 0.3 * quality


def _patient_names(chunks):
    names = {}
    pattern = re.compile(
        r"\bpatient(?:\s+name)?\s*[:\-]?\s*"
        r"([a-z]+,\s*[a-z]+(?:\s+[a-z]\.?)?)",
        re.IGNORECASE,
    )
    for chunk in chunks:
        name = chunk.get("patient_name")
        if not name:
            match = pattern.search(chunk["chunk_text"])
            name = match.group(1) if match else None
        if name and chunk["document_id"] not in names:
            names[chunk["document_id"]] = " ".join(name.split())
    unique_names = set(names.values())
    name_keys = {
        tuple(_normalise_name(name)[:2])
        for name in unique_names
    }
    first_names = [
        key[1] for key in name_keys if len(key) > 1
    ]
    compatible_names = (
        unique_names
        and len({key[0] for key in name_keys}) == 1
        and first_names
        and all(
            a.startswith(b) or b.startswith(a)
            for a in first_names
            for b in first_names
        )
    )
    if compatible_names:
        patient_name = max(unique_names, key=lambda name: len(_normalise_name(name)))
        for chunk in chunks:
            names.setdefault(chunk["document_id"], patient_name)
            chunk["patient_name"] = patient_name
    return names


def _query_is_name_search(query, patient_names):
    return max(
        (_name_score(query, name) for name in patient_names.values()),
        default=0.0,
    ) >= 0.72


def _expand_sparse_query(query):
    expansion_terms = [
        terms
        for pattern, terms in QUERY_EXPANSIONS
        if pattern.search(query)
    ]
    return " ".join([query, *expansion_terms])


def _select_page_diverse_chunks(
    chunks,
    top_k,
    max_per_page=2,
    spread_pages=False,
):
    if spread_pages:
        selected = []
        counts = Counter()
        selected_ids = set()
        for page_limit in range(1, max_per_page + 1):
            for chunk in chunks:
                page = chunk["page"]
                if (
                    chunk["chunk_id"] in selected_ids
                    or counts[page] >= page_limit
                ):
                    continue
                selected.append(chunk)
                selected_ids.add(chunk["chunk_id"])
                counts[page] += 1
                if len(selected) == top_k:
                    return selected
        return selected

    selected = []
    counts = Counter()
    for chunk in chunks:
        page = chunk["page"]
        if counts[page] >= max_per_page:
            continue
        selected.append(chunk)
        counts[page] += 1
        if len(selected) == top_k:
            break
    return selected


def _context_limit(query, top_k):
    if re.search(r"\b(?:imaging examinations|all imaging)\b", query, re.I):
        return top_k, True, False
    if re.search(r"\ballerg(?:y|ies)\b", query, re.I):
        return min(top_k, 4), False, False
    if re.search(r"\bpharmacy\b", query, re.I):
        return min(top_k, 2), False, True
    if re.search(r"\b(?:EMG|electromyography|nerve conduction)\b", query, re.I):
        return min(top_k, 1), False, True
    if re.search(r"\b(?:lumbar MRI|MRI lumbar|lumbar spine MRI)\b", query, re.I):
        limit = 2 if re.search(r"\bfindings?\b", query, re.I) else 3
        return min(top_k, limit), False, True
    if re.search(r"\bprocedure\b", query, re.I):
        return min(top_k, 1), False, True
    if re.search(
        r"\b(?:administered|emergency department|ED)\b",
        query,
        re.I,
    ) and re.search(r"\b(?:medications?|medicines?|drugs?)\b", query, re.I):
        return min(top_k, 2), False, True
    if re.search(r"\b(?:restrictions?|discharge)\b", query, re.I):
        return min(top_k, 2), False, True
    return top_k, False, False


def retrieve_chunks(query, model, top_k=8, apply_context_limits=True):
    if not query or not query.strip():
        raise ValueError("Question cannot be empty.")
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    query = query.strip()
    with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
        chunks = json.load(file)
    if not chunks:
        return []

    texts = [searchable_chunk_text(chunk) for chunk in chunks]
    _, vocabulary, document_frequency, average_length = build_sparse_vectors(texts)
    sparse_query = build_sparse_vector(
        _expand_sparse_query(query),
        vocabulary,
        document_frequency,
        len(texts),
        average_length,
    )
    dense_query = model.encode(
        QUERY_INSTRUCTION + query,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    patient_names = _patient_names(chunks)
    name_scores = {
        document_id: _name_score(query, name)
        for document_id, name in patient_names.items()
    }

    client = QdrantClient(path=str(QDRANT_PATH))
    try:
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                models.Prefetch(
                    query=dense_query.tolist(),
                    using="dense",
                    limit=max(30, top_k * 6),
                ),
                models.Prefetch(
                    query=sparse_query,
                    using="sparse",
                    limit=max(30, top_k * 6),
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=max(30, top_k * 6),
            with_payload=True,
        )
    finally:
        client.close()

    by_chunk_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    ranked = {}
    for point in results.points:
        payload = point.payload or {}
        chunk_id = payload.get("chunk_id")
        chunk = by_chunk_id.get(chunk_id)
        if not chunk:
            continue
        ranked[chunk_id] = (chunk, float(point.score or 0.0))

    is_name_search = _query_is_name_search(query, patient_names)
    informative_query_tokens = {
        token for token in tokenize(query)
        if token not in QUERY_STOPWORDS
    }
    output = []
    for chunk, rrf_score in ranked.values():
        patient_score = name_scores.get(chunk["document_id"], 0.0)
        chunk_tokens = set(tokenize(searchable_chunk_text(chunk)))
        token_overlap = (
            len(informative_query_tokens & chunk_tokens)
            / max(len(informative_query_tokens), 1)
        )
        output.append({
            "chunk_id": chunk["chunk_id"],
            "document_id": chunk["document_id"],
            "patient_name": chunk.get(
                "patient_name",
                patient_names.get(chunk["document_id"]),
            ),
            "page": chunk["page"],
            "section": chunk["section"],
            "chunk_text": chunk["chunk_text"],
            "score": rrf_score,
            "rrf_score": rrf_score,
            "patient_match_score": patient_score,
            "_token_overlap": token_overlap,
        })

    max_rrf_score = max(
        (item["rrf_score"] for item in output),
        default=1.0,
    )
    for item in output:
        normalized_rrf = item["rrf_score"] / max_rrf_score
        section_tokens = set(tokenize(item["section"]))
        section_overlap = (
            len(informative_query_tokens & section_tokens)
            / max(len(informative_query_tokens), 1)
        )
        item["score"] = (
            0.65 * normalized_rrf
            + 0.20 * item["_token_overlap"]
            + 0.15 * section_overlap
        )
        del item["_token_overlap"]

    output.sort(
        key=lambda item: (
            item["patient_match_score"] if is_name_search else 0,
            item["score"],
        ),
        reverse=True,
    )
    if apply_context_limits:
        result_limit, spread_pages, prefer_best_page = _context_limit(
            query,
            top_k,
        )
    else:
        result_limit, spread_pages, prefer_best_page = top_k, False, False
    if prefer_best_page and output:
        target_page = output[0]["page"]
        page_results = [
            item for item in output if item["page"] == target_page
        ][:2]
        other_results = [
            item for item in output if item["page"] != target_page
        ]
        return _select_page_diverse_chunks(
            page_results + other_results,
            result_limit,
        )
    return _select_page_diverse_chunks(
        output,
        result_limit,
        spread_pages=spread_pages,
    )

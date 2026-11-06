import json
import re
from difflib import SequenceMatcher
from pathlib import Path

from qdrant_client import QdrantClient, models

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


def retrieve_chunks(query, model, top_k=8):
    if not query or not query.strip():
        raise ValueError("Question cannot be empty.")
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    query = query.strip()
    with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
        chunks = json.load(file)
    if not chunks:
        return []

    texts = [chunk["chunk_text"] for chunk in chunks]
    _, vocabulary, document_frequency, average_length = build_sparse_vectors(texts)
    sparse_query = build_sparse_vector(
        query,
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
    output = []
    for chunk, rrf_score in ranked.values():
        patient_score = name_scores.get(chunk["document_id"], 0.0)
        token_overlap = len(
            set(tokenize(query)) & set(tokenize(chunk["chunk_text"]))
        ) / max(len(set(tokenize(query))), 1)
        final_score = (
            0.45 * rrf_score
            + 0.40 * patient_score
            + 0.15 * token_overlap
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
            "score": final_score,
            "rrf_score": rrf_score,
            "patient_match_score": patient_score,
        })

    output.sort(
        key=lambda item: (
            item["patient_match_score"] if is_name_search else 0,
            item["score"],
        ),
        reverse=True,
    )
    return output[:top_k]

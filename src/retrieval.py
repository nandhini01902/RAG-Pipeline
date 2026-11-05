import json
import math
import re
from collections import Counter
from pathlib import Path

from qdrant_client import QdrantClient, models


BASE_DIR = Path(__file__).resolve().parent.parent

QDRANT_PATH = BASE_DIR / "outputs" / "qdrant"

COLLECTION_NAME = "medical_records"

SPARSE_MODEL_NAME = "local-bm25"

QUERY_INSTRUCTION = (
    "Represent this sentence for searching relevant passages: "
)


CHUNKS_FILE = BASE_DIR / "data" / "processed" / "chunks.json"


def build_sparse_query(query, chunks):
    texts = [chunk["chunk_text"] for chunk in chunks]
    tokenized = [
        re.findall(r"\b\w+\b", text.lower())
        for text in texts
    ]
    vocabulary = {
        token
        for tokens in tokenized
        for token in tokens
    }
    token_ids = {
        token: index
        for index, token in enumerate(sorted(vocabulary), start=1)
    }
    document_frequency = Counter(
        token
        for tokens in tokenized
        for token in set(tokens)
    )
    query_tokens = re.findall(r"\b\w+\b", query.lower())
    counts = Counter(query_tokens)
    document_count = len(tokenized)
    values = {}

    for token, count in counts.items():
        if token not in token_ids:
            continue
        idf = math.log(
            1 + (document_count - document_frequency[token] + 0.5)
            / (document_frequency[token] + 0.5)
        )
        values[token_ids[token]] = idf * count * 2.5 / (count + 1.5)

    return models.SparseVector(
        indices=list(values.keys()),
        values=list(values.values())
    )


def retrieve_chunks(query, model, top_k=5):

    if not query or not query.strip():
        raise ValueError("Question cannot be empty.")

    query = query.strip()

    

    query_text = QUERY_INSTRUCTION + query

    dense_query = model.encode(
        query_text,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

 

    with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    sparse_query = build_sparse_query(query, chunks)

   
    client = QdrantClient(path=str(QDRANT_PATH))

    try:

        results = client.query_points(
            collection_name=COLLECTION_NAME,

            prefetch=[

                # Semantic retrieval
                models.Prefetch(
                    query=dense_query.tolist(),
                    using="dense",
                    limit=20
                ),

                # BM25 keyword retrieval
                models.Prefetch(
                    query=models.SparseVector(
                        indices=sparse_query.indices,
                        values=sparse_query.values
                    ),
                    using="sparse",
                    limit=20
                )

            ],

            # Combine both ranked result lists
            query=models.FusionQuery(
                fusion=models.Fusion.RRF
            ),

            limit=top_k,
            with_payload=True
        )

        retrieved_chunks = []

        for point in results.points:

            payload = point.payload or {}

            retrieved_chunks.append({

                "chunk_id": payload.get("chunk_id"),

                "page": payload.get("page"),

                "chunk_text": payload.get("chunk_text"),

                "score": point.score

            })

        return retrieved_chunks

    finally:
        client.close()


import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient, models


BASE_DIR = Path(__file__).resolve().parent.parent

CHUNKS_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "chunks.json"
)

EMBEDDINGS_FILE = (
    BASE_DIR
    / "outputs"
    / "embeddings"
    / "embeddings.npy"
)

QDRANT_PATH = BASE_DIR / "outputs" / "qdrant"

COLLECTION_NAME = "medical_records"

SPARSE_MODEL_NAME = "local-bm25"


def build_sparse_vectors(texts):
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
    average_length = sum(map(len, tokenized)) / max(len(tokenized), 1)
    document_frequency = Counter(
        token
        for tokens in tokenized
        for token in set(tokens)
    )
    document_count = len(tokenized)

    sparse_vectors = []
    for tokens in tokenized:
        counts = Counter(tokens)
        document_length = len(tokens)
        values = {}
        for token, count in counts.items():
            idf = math.log(
                1 + (document_count - document_frequency[token] + 0.5)
                / (document_frequency[token] + 0.5)
            )
            denominator = (
                count
                + 1.5
                * (0.25 + 0.75 * document_length / max(average_length, 1))
            )
            values[token_ids[token]] = (
                idf * count * 2.5 / denominator
            )

        sparse_vectors.append(
            models.SparseVector(
                indices=list(values.keys()),
                values=list(values.values())
            )
        )

    return sparse_vectors


def create_vector_store():

    with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    dense_embeddings = np.load(EMBEDDINGS_FILE)

    if len(chunks) != len(dense_embeddings):
        raise ValueError(
            "Number of chunks and dense embeddings do not match."
        )

    if not chunks:
        raise ValueError("No chunks available.")

    texts = [chunk["chunk_text"] for chunk in chunks]

    # Generate sparse vectors for every chunk
    sparse_embeddings = build_sparse_vectors(texts)

    if len(sparse_embeddings) != len(chunks):
        raise ValueError(
            "Number of sparse embeddings and chunks do not match."
        )

    QDRANT_PATH.mkdir(parents=True, exist_ok=True)

    client = QdrantClient(path=str(QDRANT_PATH))

    try:

        # Rebuild collection to avoid old vector structures
        if client.collection_exists(COLLECTION_NAME):
            client.delete_collection(COLLECTION_NAME)

        dense_dimension = dense_embeddings.shape[1]

        client.create_collection(
            collection_name=COLLECTION_NAME,

            vectors_config={
                "dense": models.VectorParams(
                    size=dense_dimension,
                    distance=models.Distance.COSINE
                )
            },

            sparse_vectors_config={
                "sparse": models.SparseVectorParams(
                    modifier=models.Modifier.IDF
                )
            }
        )

        points = []

        for index, chunk in enumerate(chunks):

            sparse_vector = sparse_embeddings[index]

            point = models.PointStruct(
                id=index,

                vector={
                    "dense": dense_embeddings[index].tolist(),

                    "sparse": models.SparseVector(
                        indices=sparse_vector.indices,
                        values=sparse_vector.values
                    )
                },

                payload={
                    "chunk_id": chunk["chunk_id"],
                    "page": chunk["page"],
                    "section": chunk["section"],
                    "chunk_text": chunk["chunk_text"]
                }
            )

            points.append(point)

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )

        print("Hybrid indexing completed.")
        print("Collection:", COLLECTION_NAME)
        print("Stored chunks:", len(points))
        print("Dense dimension:", dense_dimension)
        print("Sparse model:", SPARSE_MODEL_NAME)

    finally:
        client.close()


if __name__ == "__main__":
    create_vector_store()

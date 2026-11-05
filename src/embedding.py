
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "chunks.json"
)

OUTPUT_DIR = BASE_DIR / "outputs" / "embeddings"

OUTPUT_FILE = OUTPUT_DIR / "embeddings.npy"

MODEL_NAME = "BAAI/bge-small-en-v1.5"


def generate_embeddings():

    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    if not chunks:
        raise ValueError("No chunks found. Run chunking first.")

    model = SentenceTransformer(MODEL_NAME)

    texts = [chunk["chunk_text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    embeddings = embeddings.astype(np.float32)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    np.save(OUTPUT_FILE, embeddings)

    print("\nEmbedding generation completed.")
    print("Number of chunks:", len(chunks))
    print("Embedding shape:", embeddings[1])
    print("Embedding dimension:", embeddings.shape[1])
    print("Saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    generate_embeddings()


from sentence_transformers import SentenceTransformer

from retrieval import retrieve_chunks


MODEL_NAME = "BAAI/bge-small-en-v1.5"

model = SentenceTransformer(MODEL_NAME)

question = "What imaging examinations were performed?"

results = retrieve_chunks(
    question,
    model,
    top_k=5
)

for index, result in enumerate(results, start=1):

    print("\nResult:", index)
    print("Chunk ID:", result["chunk_id"])
    print("Page:", result["page"])
    print("RRF Score:", result["score"])
    print("Text:", result["chunk_text"])


from sentence_transformers import SentenceTransformer

from src.rag_pipeline import answer_question


MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Load the same embedding model used during indexing
embedding_model = SentenceTransformer(MODEL_NAME)

# Ask a question supported by your PDF
question = "What imaging examinations were performed?"

result = answer_question(
    question=question,
    embedding_model=embedding_model,
    top_k=5
)

print("\nQUESTION:")
print(result["question"])

print("\nGEMINI ANSWER:")
print(result["answer"])

print("\nRETRIEVED SOURCES:")

for source in result["sources"]:
    print(
        f"\nPage: {source['page']} | "
        f"Chunk: {source['chunk_id']} | "
        f"RRF score: {source['score']:.4f}"
    )

    print(source["chunk_text"])

import csv
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

from src import retrieval as retrieval_module
from src.chunk_text import searchable_chunk_text
from src.evaluate_retrieval import (
    BENCHMARK,
    K_VALUES,
    MODEL_NAME,
    _unique_page_ranking,
    metrics_at_k,
)
from src.retrieval import retrieve_chunks
from src.sparse_vectors import build_sparse_vectors


BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs" / "evaluation"
BASELINE_CHUNKS_FILE = OUTPUT_DIR / "chunks_34_baseline.json"
CURRENT_CHUNKS_FILE = BASE_DIR / "data" / "processed" / "chunks.json"
CONFIGURATIONS = (
    {
        "name": "Earlier chunking (34 chunks)",
        "chunks_file": BASELINE_CHUNKS_FILE,
    },
    {
        "name": "Current chunking (68 chunks)",
        "chunks_file": CURRENT_CHUNKS_FILE,
    },
)


def _chunk_statistics(chunks):
    token_counts = [chunk["token_count"] for chunk in chunks]
    return {
        "chunk_count": len(chunks),
        "average_tokens": sum(token_counts) / len(token_counts),
        "min_tokens": min(token_counts),
        "max_tokens": max(token_counts),
    }


def _create_index(chunks, model, index_path):
    texts = [searchable_chunk_text(chunk) for chunk in chunks]
    dense_vectors = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    sparse_vectors, _, _, _ = build_sparse_vectors(texts)

    client = QdrantClient(path=str(index_path))
    try:
        client.create_collection(
            collection_name="medical_records",
            vectors_config={
                "dense": models.VectorParams(
                    size=int(dense_vectors.shape[1]),
                    distance=models.Distance.COSINE,
                )
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(
                    modifier=models.Modifier.IDF,
                )
            },
        )
        client.upsert(
            collection_name="medical_records",
            points=[
                models.PointStruct(
                    id=index,
                    vector={
                        "dense": dense_vectors[index].tolist(),
                        "sparse": sparse_vectors[index],
                    },
                    payload={
                        "chunk_id": chunk["chunk_id"],
                        "page": chunk["page"],
                        "section": chunk["section"],
                        "chunk_text": chunk["chunk_text"],
                        "patient_name": chunk.get("patient_name"),
                        "searchable_text": texts[index],
                    },
                )
                for index, chunk in enumerate(chunks)
            ],
        )
    finally:
        client.close()


def _evaluate_configuration(configuration, model, index_path):
    chunks = json.loads(
        configuration["chunks_file"].read_text(encoding="utf-8")
    )
    if not chunks:
        raise ValueError(
            f"No chunks found in {configuration['chunks_file']}."
        )
    if not BENCHMARK:
        raise ValueError("The retrieval benchmark cannot be empty.")

    chunk_file = index_path.parent / f"{index_path.name}_chunks.json"
    chunk_file.write_text(
        json.dumps(chunks, ensure_ascii=False),
        encoding="utf-8",
    )
    _create_index(chunks, model, index_path)

    previous_chunks_file = retrieval_module.CHUNKS_FILE
    previous_qdrant_path = retrieval_module.QDRANT_PATH
    retrieval_module.CHUNKS_FILE = chunk_file
    retrieval_module.QDRANT_PATH = index_path
    try:
        per_question = []
        for question, relevant_pages in BENCHMARK:
            retrieved = retrieve_chunks(
                question,
                model,
                top_k=max(K_VALUES) * 2,
                apply_context_limits=False,
                use_reranker=True,
            )
            ranked_pages = _unique_page_ranking(retrieved)
            per_question.append({
                "question": question,
                "relevant_pages": sorted(relevant_pages),
                "ranked_pages": ranked_pages,
                "metrics_by_k": {
                    str(k): metrics_at_k(ranked_pages, relevant_pages, k)
                    for k in K_VALUES
                },
            })
    finally:
        retrieval_module.CHUNKS_FILE = previous_chunks_file
        retrieval_module.QDRANT_PATH = previous_qdrant_path

    aggregate = []
    for k in K_VALUES:
        aggregate.append({
            "configuration": configuration["name"],
            "top_k": k,
            "num_questions": len(per_question),
            **{
                metric: sum(
                    item["metrics_by_k"][str(k)][metric]
                    for item in per_question
                ) / len(per_question)
                for metric in (
                    "hit_rate",
                    "recall",
                    "precision",
                    "mrr",
                    "ndcg",
                )
            },
        })

    return {
        "configuration": configuration["name"],
        "chunk_statistics": _chunk_statistics(chunks),
        "per_question": per_question,
        "aggregate_by_k": aggregate,
    }


def evaluate(output_dir=OUTPUT_DIR, model=None):
    if not BASELINE_CHUNKS_FILE.is_file():
        raise FileNotFoundError(
            "The historical 34-chunk baseline is missing: "
            f"{BASELINE_CHUNKS_FILE}"
        )

    model = model or SentenceTransformer(MODEL_NAME)
    with tempfile.TemporaryDirectory(prefix="rag-chunk-eval-") as temp_dir:
        temp_path = Path(temp_dir)
        results = [
            _evaluate_configuration(
                configuration,
                model,
                temp_path / f"index_{index}",
            )
            for index, configuration in enumerate(CONFIGURATIONS)
        ]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    result = {
        "created_at_utc": timestamp,
        "embedding_model": MODEL_NAME,
        "reranker": "same lexical and patient-name reranker enabled for both",
        "retrieval": "dense + sparse Qdrant RRF",
        "benchmark_questions": len(BENCHMARK),
        "ranking_unit": "unique PDF page, ordered by first retrieved chunk",
        "baseline_source": "9d7535d:data/processed/chunks.json",
        "limitations": [
            "Retrieval metrics do not measure generated-answer quality or clinical safety.",
            "The benchmark contains 10 manually labeled questions from one synthetic PDF.",
            "The earlier dataset has 34 chunks in repository history; no 36-chunk snapshot was found.",
            "Both chunk sets were re-embedded with the same current embedding model and evaluated using the same retrieval code.",
        ],
        "configurations": results,
    }

    json_path = output_dir / "chunking_comparison.json"
    csv_path = output_dir / "chunking_comparison.csv"
    json_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                "configuration",
                "chunk_count",
                "average_tokens",
                "min_tokens",
                "max_tokens",
                "top_k",
                "num_questions",
                "hit_rate",
                "recall",
                "precision",
                "mrr",
                "ndcg",
            ),
        )
        writer.writeheader()
        for configuration in results:
            stats = configuration["chunk_statistics"]
            for row in configuration["aggregate_by_k"]:
                writer.writerow({
                    **{
                        key: stats[key]
                        for key in (
                            "chunk_count",
                            "average_tokens",
                            "min_tokens",
                            "max_tokens",
                        )
                    },
                    **row,
                })

    for configuration in results:
        stats = configuration["chunk_statistics"]
        row = next(
            item
            for item in configuration["aggregate_by_k"]
            if item["top_k"] == 5
        )
        print(
            f"{configuration['configuration']} | "
            f"chunks={stats['chunk_count']} | "
            f"avg tokens={stats['average_tokens']:.1f} | "
            f"Hit Rate@5={row['hit_rate']:.3f} | "
            f"Recall@5={row['recall']:.3f} | "
            f"Precision@5={row['precision']:.3f} | "
            f"MRR@5={row['mrr']:.3f} | "
            f"nDCG@5={row['ndcg']:.3f}"
        )
    print(f"\nChunk comparison JSON: {json_path}")
    print(f"Chunk comparison CSV: {csv_path}")
    return result


if __name__ == "__main__":
    evaluate()

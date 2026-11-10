import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from sentence_transformers import SentenceTransformer

from src.retrieval import retrieve_chunks


BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs" / "evaluation"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
K_VALUES = (1, 3, 5, 8, 10, 15)
COMPARISON_K = 5
CONFIGURATIONS = (
    {"name": "RRF only", "use_reranker": False},
    {"name": "RRF + lexical reranker", "use_reranker": True},
)
BENCHMARK = (
    ("What medications are listed in the record?", {2, 3, 5, 6, 19}),
    ("What imaging examinations were performed?", {3, 4, 8, 9, 11, 15, 22}),
    ("What did the lumbar MRI show?", {8, 9}),
    ("What procedure was performed on September 10?", {15}),
    ("What allergies are documented?", {2, 3, 6, 19}),
    ("What did the EMG and nerve conduction study show?", {14}),
    ("What medications were administered in the emergency department?", {2}),
    ("What restrictions were given at discharge?", {5}),
    ("What were the findings of the lumbar spine MRI?", {8}),
    ("What medications were filled by the pharmacy?", {19}),
)


def _unique_page_ranking(chunks):
    pages = []
    seen = set()
    for chunk in chunks:
        page = chunk["page"]
        if page not in seen:
            pages.append(page)
            seen.add(page)
    return pages


def metrics_at_k(ranked_pages, relevant_pages, k):
    if k < 1:
        raise ValueError("k must be at least 1.")
    if not relevant_pages:
        raise ValueError("Each benchmark question needs a relevant page.")

    retrieved = ranked_pages[:k]
    relevant_retrieved = [
        page for page in retrieved if page in relevant_pages
    ]
    first_relevant_rank = next(
        (
            rank
            for rank, page in enumerate(retrieved, start=1)
            if page in relevant_pages
        ),
        None,
    )
    dcg = sum(
        1 / math.log2(rank + 1)
        for rank, page in enumerate(retrieved, start=1)
        if page in relevant_pages
    )
    ideal_relevant_count = min(k, len(relevant_pages))
    ideal_dcg = sum(
        1 / math.log2(rank + 1)
        for rank in range(1, ideal_relevant_count + 1)
    )

    return {
        "hit_rate": float(bool(relevant_retrieved)),
        "recall": len(relevant_retrieved) / len(relevant_pages),
        "precision": len(relevant_retrieved) / k,
        "mrr": 1 / first_relevant_rank if first_relevant_rank else 0.0,
        "ndcg": dcg / ideal_dcg if ideal_dcg else 0.0,
        "retrieved_pages": retrieved,
        "relevant_pages_retrieved": relevant_retrieved,
    }


def _average_metrics(per_question, k):
    return {
        metric: sum(
            item["metrics"][str(k)][metric]
            for item in per_question
        ) / len(per_question)
        for metric in ("hit_rate", "recall", "precision", "mrr", "ndcg")
    }


def evaluate(model=None, output_dir=OUTPUT_DIR):
    if not BENCHMARK:
        raise ValueError("The retrieval benchmark cannot be empty.")
    model = model or SentenceTransformer(MODEL_NAME)
    max_k = max(K_VALUES)
    per_question = []

    for question, relevant_pages in BENCHMARK:
        chunks = retrieve_chunks(
            question,
            model,
            top_k=max_k * 2,
            apply_context_limits=False,
            use_reranker=True,
        )
        ranked_pages = _unique_page_ranking(chunks)
        per_question.append({
            "question": question,
            "relevant_pages": sorted(relevant_pages),
            "ranked_pages": ranked_pages,
            "metrics": {
                str(k): metrics_at_k(ranked_pages, relevant_pages, k)
                for k in K_VALUES
            },
        })

    configurations = []
    for k in K_VALUES:
        configurations.append({
            "top_k": k,
            "num_questions": len(per_question),
            **_average_metrics(per_question, k),
        })

    comparison_per_question = []
    for (question, relevant_pages), reranked_item in zip(
        BENCHMARK,
        per_question,
    ):
        baseline_chunks = retrieve_chunks(
            question,
            model,
            top_k=max_k * 2,
            apply_context_limits=False,
            use_reranker=False,
        )
        baseline_pages = _unique_page_ranking(baseline_chunks)
        baseline_metrics = metrics_at_k(
            baseline_pages,
            relevant_pages,
            COMPARISON_K,
        )
        reranked_metrics = reranked_item["metrics"][str(COMPARISON_K)]
        comparison_per_question.append({
            "question": question,
            "relevant_pages": sorted(relevant_pages),
            "configurations": {
                CONFIGURATIONS[0]["name"]: {
                    "ranked_pages": baseline_pages,
                    "metrics": baseline_metrics,
                },
                CONFIGURATIONS[1]["name"]: {
                    "ranked_pages": reranked_item["ranked_pages"],
                    "metrics": reranked_metrics,
                },
            },
        })

    configuration_comparison = []
    for configuration in CONFIGURATIONS:
        name = configuration["name"]
        metrics = [
            item["configurations"][name]["metrics"]
            for item in comparison_per_question
        ]
        configuration_comparison.append({
            "configuration": name,
            "use_reranker": configuration["use_reranker"],
            "top_k": COMPARISON_K,
            "num_questions": len(comparison_per_question),
            **{
                metric: sum(item[metric] for item in metrics) / len(metrics)
                for metric in ("hit_rate", "recall", "precision", "mrr", "ndcg")
            },
        })

    timestamp = datetime.now(timezone.utc).isoformat()
    result = {
        "created_at_utc": timestamp,
        "model": MODEL_NAME,
        "ranking_unit": "unique PDF page, ordered by first retrieved chunk",
        "metric_definitions": {
            "hit_rate@k": "Fraction of questions with at least one relevant page in top k.",
            "recall@k": "Relevant pages retrieved in top k divided by all labeled relevant pages.",
            "precision@k": "Relevant pages retrieved in top k divided by k.",
            "mrr@k": "Mean reciprocal rank of the first relevant page within top k; zero on a miss.",
            "nDCG@k": "Binary-relevance normalized discounted cumulative gain at k.",
        },
        "benchmark": [
            {
                "question": item["question"],
                "relevant_pages": item["relevant_pages"],
                "ranked_pages": item["ranked_pages"],
                "metrics_by_k": item["metrics"],
            }
            for item in per_question
        ],
        "configurations": configurations,
        "configuration_comparison": {
            "variable": "reranker",
            "fixed_settings": {
                "embedding_model": MODEL_NAME,
                "hybrid_retrieval": "dense + sparse with Qdrant RRF",
                "retrieval_depth": max_k * 2,
                "evaluation_top_k": COMPARISON_K,
                "query_benchmark": "same 10 manually labeled questions",
            },
            "configurations": configuration_comparison,
            "per_question": comparison_per_question,
        },
        "limitations": [
            "Scores measure retrieval only, not generated-answer correctness or medical safety.",
            "The benchmark contains 10 manually labeled questions from this single PDF.",
            "Precision uses k as its denominator even when fewer than k pages are retrieved.",
            "The reranker comparison changes only lexical and patient-name reranking; both configurations use the same RRF candidate retrieval.",
        ],
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "retrieval_metrics.json"
    csv_path = output_dir / "retrieval_metrics.csv"
    comparison_csv_path = output_dir / "configuration_comparison.csv"
    json_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
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
        writer.writerows(configurations)
    with comparison_csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                "configuration",
                "use_reranker",
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
        writer.writerows(configuration_comparison)

    for row in configurations:
        print(
            f"k={row['top_k']:>2} | "
            f"Hit Rate={row['hit_rate']:.3f} | "
            f"Recall={row['recall']:.3f} | "
            f"Precision={row['precision']:.3f} | "
            f"MRR={row['mrr']:.3f} | "
            f"nDCG={row['ndcg']:.3f}"
        )
    print(f"\nDetailed scores saved to: {json_path}")
    print(f"Experiment table saved to: {csv_path}")
    print(f"Two-configuration comparison saved to: {comparison_csv_path}")
    for row in configuration_comparison:
        print(
            f"{row['configuration']} @ {row['top_k']} | "
            f"Hit Rate={row['hit_rate']:.3f} | "
            f"Recall={row['recall']:.3f} | "
            f"Precision={row['precision']:.3f} | "
            f"MRR={row['mrr']:.3f} | "
            f"nDCG={row['ndcg']:.3f}"
        )
    return result


if __name__ == "__main__":
    evaluate()

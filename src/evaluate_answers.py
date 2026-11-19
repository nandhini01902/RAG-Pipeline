import csv
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs" / "evaluation"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
JUDGE_MODEL = os.getenv("RAGAS_JUDGE_MODEL", "gemini-2.5-flash")
METRIC_NAMES = (
    "answer_correctness",
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
)

BENCHMARK = (
    {
        "question": "What medications are listed in the record?",
        "reference_answer": (
            "The records list naproxen, methocarbamol, acetaminophen with "
            "codeine, gabapentin, ondansetron, oxycodone/acetaminophen, and "
            "docusate. The pharmacy history notes that it may not be a "
            "complete medication history."
        ),
    },
    {
        "question": "What imaging examinations were performed?",
        "reference_answer": (
            "The records document a CT of the cervical spine without contrast "
            "and three-view lumbar spine radiographs on 02/11/2024, and an "
            "MRI of the lumbar spine without contrast on 03/06/2024. A "
            "cervical MRI from March 2024 is referenced, but its report was "
            "not included in the produced records."
        ),
    },
    {
        "question": "What did the lumbar MRI show?",
        "reference_answer": (
            "The 03/06/2024 MRI showed a roughly 6 mm right paracentral "
            "L4-L5 disc protrusion extending into the right lateral recess, "
            "contacting and mildly displacing the traversing right L5 nerve "
            "root, with mild-to-moderate right lateral recess stenosis. At "
            "L5-S1 there was a small central disc bulge without significant "
            "stenosis."
        ),
    },
    {
        "question": "What procedure was performed on September 10?",
        "reference_answer": (
            "A right L4-L5 hemilaminotomy with microdiscectomy and "
            "decompression of the right L5 nerve root was performed on "
            "09/10/2024."
        ),
    },
    {
        "question": "What allergies are documented?",
        "reference_answer": (
            "Codeine is documented as causing a rash. The primary-care note "
            "says codeine was added to the chart allergy list."
        ),
    },
    {
        "question": "What did the EMG and nerve conduction study show?",
        "reference_answer": (
            "The 08/28/2024 needle EMG reported abnormalities in the left "
            "tibialis anterior, extensor hallucis longus, gluteus medius, "
            "and L5 lumbar paraspinals. The right lower-extremity screen was "
            "normal. Bilateral tibial motor and sural sensory studies and "
            "the H-reflexes were reported within normal limits."
        ),
    },
    {
        "question": "What medications were administered in the emergency department?",
        "reference_answer": (
            "Ketorolac 30 mg IV once and methocarbamol 1000 mg IV once were "
            "administered in the emergency department."
        ),
    },
    {
        "question": "What restrictions were given at discharge?",
        "reference_answer": (
            "He was told to avoid lifting more than 10 pounds for one week. "
            "He was off work from 02/12/2024 through 02/16/2024 and could "
            "return to light duty on 02/19/2024 with no lifting over "
            "15 pounds."
        ),
    },
    {
        "question": "What medications were filled by the pharmacy?",
        "reference_answer": (
            "The pharmacy record lists fills for naproxen, methocarbamol, "
            "acetaminophen/codeine, gabapentin, ondansetron, "
            "oxycodone/acetaminophen, and docusate. It cautions that this "
            "pharmacy printout may not represent a complete medication "
            "history."
        ),
    },
)


def _validate_records(records):
    if not records:
        raise ValueError("At least one answer record is required.")

    required_fields = ("question", "answer", "contexts", "reference_answer")
    normalized = []
    for index, record in enumerate(records):
        missing = [field for field in required_fields if field not in record]
        if missing:
            raise ValueError(
                f"Answer record {index} is missing: {', '.join(missing)}."
            )
        for field in ("question", "answer", "reference_answer"):
            if not isinstance(record[field], str) or not record[field].strip():
                raise ValueError(
                    f"Answer record {index} must have a non-empty {field}."
                )
        if not isinstance(record["contexts"], list) or not all(
            isinstance(context, str) for context in record["contexts"]
        ):
            raise ValueError(
                f"Answer record {index} contexts must be a list of strings."
            )
        normalized.append({
            "question": record["question"].strip(),
            "answer": record["answer"].strip(),
            "contexts": record["contexts"],
            "ground_truth": record["reference_answer"].strip(),
        })
    return normalized


def _run_ragas(records, judge_model):
    from datasets import Dataset
    from langchain_google_genai import (
        ChatGoogleGenerativeAI,
        GoogleGenerativeAIEmbeddings,
    )
    from ragas import evaluate as ragas_evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        answer_correctness,
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY is required to run RAGAS answer evaluation."
        )

    llm = LangchainLLMWrapper(
        ChatGoogleGenerativeAI(
            model=judge_model,
            google_api_key=api_key,
            temperature=0,
        )
    )
    embeddings = LangchainEmbeddingsWrapper(
        GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001",
            google_api_key=api_key,
        )
    )
    result = ragas_evaluate(
        dataset=Dataset.from_list(records),
        metrics=[
            answer_correctness,
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        llm=llm,
        embeddings=embeddings,
    )
    return result.scores


def evaluate_answers(records, judge_model=JUDGE_MODEL, output_dir=None):
    normalized_records = _validate_records(records)
    scores = _run_ragas(normalized_records, judge_model)
    if len(scores) != len(normalized_records):
        raise RuntimeError(
            "RAGAS returned a different number of score rows than inputs."
        )

    per_question = []
    for index, score in enumerate(scores):
        missing = [metric for metric in METRIC_NAMES if metric not in score]
        if missing:
            raise RuntimeError(
                "RAGAS did not return expected metrics: "
                + ", ".join(missing)
            )
        metric_scores = {}
        for metric in METRIC_NAMES:
            value = float(score[metric])
            if not math.isfinite(value):
                raise RuntimeError(
                    f"RAGAS returned a non-finite {metric} score for "
                    f"record {index}."
                )
            metric_scores[metric] = value
        per_question.append({
            "question": normalized_records[index]["question"],
            "metrics": metric_scores,
        })

    result = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "judge_model": judge_model,
        "num_questions": len(per_question),
        "metrics": {
            metric: sum(
                item["metrics"][metric] for item in per_question
            ) / len(per_question)
            for metric in METRIC_NAMES
        },
        "per_question": per_question,
        "limitations": [
            "Answer correctness is an LLM-judge score against the supplied reference answers, not exact-match accuracy.",
            "The benchmark uses manually written references for one synthetic PDF and is not representative of other documents.",
            "RAGAS sends answers and retrieved contexts to the configured Gemini judge; do not evaluate documents you are not permitted to share with that service.",
            "These metrics do not measure clinical safety and are not a substitute for expert review.",
        ],
    }

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "ragas_answer_metrics.json"
        csv_path = output_dir / "ragas_answer_metrics.csv"
        json_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        with csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=("question", *METRIC_NAMES),
            )
            writer.writeheader()
            for item in per_question:
                writer.writerow({
                    "question": item["question"],
                    **item["metrics"],
                })
        print(f"RAGAS JSON metrics saved to: {json_path}")
        print(f"RAGAS CSV metrics saved to: {csv_path}")

    for metric, value in result["metrics"].items():
        print(f"{metric}: {value:.3f} ({value * 100:.1f}%)")
    return result


def evaluate(model=None, output_dir=OUTPUT_DIR, judge_model=JUDGE_MODEL):
    from sentence_transformers import SentenceTransformer

    from src.rag_pipeline import answer_question

    model = model or SentenceTransformer(MODEL_NAME)
    records = []
    for item in BENCHMARK:
        result = answer_question(
            question=item["question"],
            embedding_model=model,
            top_k=5,
        )
        records.append({
            "question": item["question"],
            "answer": result["answer"],
            "contexts": [
                source["chunk_text"] for source in result["sources"]
            ],
            "reference_answer": item["reference_answer"],
        })
    return evaluate_answers(
        records,
        judge_model=judge_model,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    evaluate()

# Medical PDF Question-Answering RAG System

## Project Report

**Report date:** 10 November 2026  
**Project:** Local retrieval-augmented question answering over a synthetic medical-record PDF

## Executive summary

This project implements a local retrieval-augmented generation (RAG) pipeline
for a sample synthetic medical-record PDF and also supports user-uploaded
PDFs. The system extracts page-level text, creates section-aware chunks,
indexes dense and sparse representations in local Qdrant, retrieves evidence,
and can send that evidence to Gemini for a cited answer. Uploaded PDFs are
indexed together in memory for the request and do not replace or persist in
the sample index. For answer generation, Gemini first rewrites the user's
question for retrieval while preserving its intent. A Streamlit application
also supports evidence inspection without calling a generation model.

Retrieval was evaluated on 10 manually labeled questions using relevant PDF
pages. Two experiments are recorded:

1. A historical **34-chunk** dataset versus the current **68-chunk** dataset.
2. Hybrid RRF retrieval with and without the lexical/patient-name reranker.

At `k=5`, the 68-chunk run scored 0.957 Recall and 0.987 nDCG, compared with
0.951 Recall and 0.938 nDCG for the 34-chunk run. Hit Rate and Precision
were unchanged. This is a small, single-document retrieval result; it is not
an evaluation of generated-answer correctness, clinical safety, or
statistical significance.

## 1. Objectives and scope

The project demonstrates the core stages of a document RAG workflow:

- Extract text from PDF pages while retaining page references.
- Clean OCR and formatting noise and exclude a page identified as belonging
  to a different patient.
- Split content into section-aware chunks and retain source metadata.
- Index dense and sparse representations locally.
- Retrieve and rank evidence for a question.
- Upload and search one or more PDFs together without rebuilding the
  checked-in sample corpus.
- Rewrite spelling and grammar errors in answer-generation queries while
  preserving clinical details and the original question.
- Generate an evidence-grounded answer with citations, or inspect evidence
  without contacting the LLM.
- Measure retrieval quality against a manually labeled benchmark.

The checked-in corpus is one synthetic medical-record PDF. The application
is not a clinical decision-support system and must not be used for diagnosis
or treatment.

## 2. System architecture

### 2.1 Ingestion and indexing

1. [`src/text_extraction.py`](src/text_extraction.py) extracts page text and
   preserves original page numbers.
2. [`src/clean_text.py`](src/clean_text.py) applies conservative cleanup and
   writes cleaned records and data-quality notes.
3. [`src/chunk.py`](src/chunk.py) identifies headings and paragraph
   boundaries and splits long text. Current settings are 450 tokens maximum
   and 80 tokens of overlap.
4. [`src/embedding.py`](src/embedding.py) encodes searchable chunk text with
   `BAAI/bge-small-en-v1.5`.
5. [`src/sparse_vectors.py`](src/sparse_vectors.py) builds local BM25-style
   sparse vectors.
6. [`src/vector_database.py`](src/vector_database.py) stores dense vectors,
   sparse vectors, and chunk metadata in local Qdrant.

The current dataset has 68 chunks, averaging 157.4 tokens. The historical
baseline has 34 chunks, averaging 352.8 tokens.

### 2.2 Retrieval and generation

For answer generation, Gemini first rewrites the question to correct spelling
and grammar while preserving intent, names, dates, numbers, laterality,
medical terms, and requested details. The rewritten query is encoded, expanded
with selected domain terms for sparse retrieval, combined with dense rankings
using Qdrant reciprocal-rank fusion (RRF), then reranked with lightweight
lexical and patient-name signals. Page-diversity and query-specific limits
control the evidence returned. Evidence inspection skips the LLM rewrite and
retrieves directly using the original question.

The Streamlit uploader accepts multiple PDFs. Their text is extracted and
chunked in memory, then an ephemeral in-memory Qdrant index is built for
hybrid retrieval. When no files are uploaded, the app uses the prebuilt
sample corpus. Uploaded PDFs are not saved to the project or added to the
sample index; scanned/image-only PDFs require OCR before upload. Only the
retrieved passages—not the full PDF—are included in Gemini's answer request.

[`src/rag_pipeline.py`](src/rag_pipeline.py) packages retrieved passages,
their document and page metadata, the original question, and the rewritten
search query for answer generation. Gemini is instructed to rely only on
supplied evidence, cite factual statements, avoid medical advice, and abstain
with “I don't know.” when evidence is insufficient. The offline retrieval
evaluation continues to use benchmark questions as written and does not
evaluate query rewriting or call Gemini.

[`app.py`](app.py) exposes answer generation and evidence-only inspection.
It supports querying uploaded PDFs or the sample corpus. Evidence-only
inspection does not make an LLM request; the answer-generation path makes one
Gemini request for rewriting and another for the answer.

## 3. Evaluation methodology

### 3.1 Benchmark and measures

The retrieval benchmark contains 10 manually labeled questions and their
relevant pages. Rankings are evaluated at `k = 1, 3, 5, 8, 10, 15`. The
ranking unit is a unique PDF page, ordered by the first retrieved chunk for
that page.

| Metric | Definition |
|---|---|
| Hit Rate@k | Fraction of questions with at least one relevant page in the top k. |
| Recall@k | Relevant pages found in the top k divided by all labeled relevant pages. |
| Precision@k | Relevant pages found in the top k divided by k. |
| MRR@k | Mean reciprocal rank of the first relevant page, with misses scored zero. |
| nDCG@k | Normalized discounted cumulative gain with binary page relevance. |

These are retrieval metrics, not answer-quality, factuality, citation
correctness, or clinical-safety measures.

### 3.2 Chunking comparison

The earlier dataset was recovered from repository history at commit
`9d7535d`. That snapshot contains **34 chunks**; no 36-chunk snapshot was
found in the available history. The current dataset contains 68 chunks.
The historical splitter separated paragraphs at blank lines; the newer
splitter also treats headings and selected metadata labels as paragraph
boundaries.

For comparison, both datasets were re-embedded with the same BGE model and
evaluated with the same current hybrid retrieval, reranker, benchmark, and
top-k values. Each version was indexed in a temporary isolated Qdrant
database, leaving the active index unchanged.

| Chunk dataset | Chunks | Average chunk tokens | Hit Rate@5 | Recall@5 | Precision@5 | MRR@5 | nDCG@5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Earlier | 34 | 352.8 | 1.000 | 0.951 | 0.420 | 0.950 | 0.938 |
| Current | 68 | 157.4 | 1.000 | 0.957 | 0.420 | 1.000 | 0.987 |

At `k=5`, the current configuration increases Recall by 0.006, MRR by
0.050, and nDCG by 0.049. Hit Rate and Precision are unchanged. The full
`k` sweep shows:

| Chunk dataset | k | Hit Rate | Recall | Precision | MRR | nDCG |
|---|---:|---:|---:|---:|---:|---:|
| Earlier, 34 chunks | 1 | 0.900 | 0.609 | 0.900 | 0.900 | 0.900 |
| Earlier, 34 chunks | 3 | 1.000 | 0.828 | 0.533 | 0.950 | 0.924 |
| Earlier, 34 chunks | 5 | 1.000 | 0.951 | 0.420 | 0.950 | 0.938 |
| Earlier, 34 chunks | 8 | 1.000 | 0.986 | 0.288 | 0.950 | 0.941 |
| Earlier, 34 chunks | 10 | 1.000 | 1.000 | 0.240 | 0.950 | 0.949 |
| Earlier, 34 chunks | 15 | 1.000 | 1.000 | 0.160 | 0.950 | 0.949 |
| Current, 68 chunks | 1 | 1.000 | 0.709 | 1.000 | 1.000 | 1.000 |
| Current, 68 chunks | 3 | 1.000 | 0.878 | 0.567 | 1.000 | 1.000 |
| Current, 68 chunks | 5 | 1.000 | 0.957 | 0.420 | 1.000 | 0.987 |
| Current, 68 chunks | 8 | 1.000 | 1.000 | 0.300 | 1.000 | 0.998 |
| Current, 68 chunks | 10 | 1.000 | 1.000 | 0.240 | 1.000 | 0.998 |
| Current, 68 chunks | 15 | 1.000 | 1.000 | 0.160 | 1.000 | 0.998 |

**Interpretation:** on this benchmark, the current chunking finds at least
one relevant page for every question at `k=1`, and ranks the first relevant
page earlier on average. At larger `k`, both configurations approach full
page recall. Precision decreases as `k` grows because its denominator is
the requested `k`.

**Important confound:** this is a comparison of the historical and current
chunking pipelines, not a pure chunk-size experiment. The current pipeline
also adds patient-name extraction and metadata to chunks. The evaluation
applies the same current retrieval code to both datasets, but the older
snapshot lacks the explicit patient-name field. Results should therefore be
interpreted as a pipeline-version comparison.

### 3.3 Reranker comparison

At fixed `k=5`, the evaluation compares RRF-only ranking with RRF plus
lexical/patient-name reranking. Both configurations use the same BGE model,
hybrid candidate retrieval, retrieval depth, and benchmark.

| Retrieval configuration | Hit Rate@5 | Recall@5 | Precision@5 | MRR@5 | nDCG@5 |
|---|---:|---:|---:|---:|---:|
| RRF only | 1.000 | 0.937 | 0.400 | 1.000 | 0.960 |
| RRF + lexical reranker | 1.000 | 0.957 | 0.420 | 1.000 | 0.987 |

Reranking increases Recall by 0.020, Precision by 0.020, and nDCG by
approximately 0.027 on this benchmark; Hit Rate and MRR are unchanged.

### 3.4 Top-k sweep for the current pipeline

The recorded top-k evaluation for the current retrieval pipeline is:

| k | Hit Rate@k | Recall@k | Precision@k | MRR@k | nDCG@k |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.000 | 0.709 | 1.000 | 1.000 | 1.000 |
| 3 | 1.000 | 0.878 | 0.567 | 1.000 | 1.000 |
| 5 | 1.000 | 0.957 | 0.420 | 1.000 | 0.987 |
| 8 | 1.000 | 1.000 | 0.300 | 1.000 | 0.998 |
| 10 | 1.000 | 1.000 | 0.240 | 1.000 | 0.998 |
| 15 | 1.000 | 1.000 | 0.160 | 1.000 | 0.998 |

The saved current top-k metrics reflect the reranked retrieval path.

## 4. Results and conclusion

On the current 10-question benchmark, the 68-chunk pipeline provides a
small improvement over the 34-chunk historical baseline at `k=5`, especially
in first-relevant-page ranking (MRR) and nDCG. The reranker also gives
modest gains over RRF-only ranking. Current retrieval reaches full Hit Rate
at all evaluated `k` values, with full Recall by `k=8`.

These results support using the current chunking and reranking configuration
for this sample corpus. They do not prove general superiority on other
documents or question types. The dataset is small, pages are labeled at
page level, and the benchmark is not an independent test set.

## 5. Limitations and work remaining

- The checked-in corpus contains one synthetic medical record, not the
  requested collection of 20–50 diverse medical PDFs. The app accepts
  temporary multi-PDF uploads but does not retain a persistent user document
  library.
- Scanned/image-only uploads are not OCR-processed and cannot be searched
  unless OCR text is present in the PDF.
- Only one current chunking strategy is implemented; the two historical/current
  versions are not a full multi-strategy or chunk-size sweep.
- The available history provides a 34-chunk baseline, not 36 chunks.
- Chunking comparison is confounded by related pipeline changes, including
  patient-name metadata extraction.
- No evaluation has been run comparing two LLMs.
- Query-rewrite quality and its impact on retrieval have not been evaluated.
- No automatic answer correctness, completeness, citation-support,
  hallucination, or clinical-safety benchmark is implemented.
- No confidence intervals, significance tests, or effect sizes have been
  calculated.
- Retrieval metrics should not be interpreted as evidence of clinical
  reliability or safety.

## 6. Reproducibility

Run the top-k and reranker retrieval evaluation:

```powershell
.\.venv\Scripts\python.exe -m src.evaluate_retrieval
```

Run the historical/current chunking comparison:

```powershell
.\.venv\Scripts\python.exe -m src.evaluate_chunking
```

The chunking comparison re-embeds both chunk sets and builds isolated
temporary Qdrant indexes. The historical chunk data is retained in
[`outputs/evaluation/chunks_34_baseline.json`](outputs/evaluation/chunks_34_baseline.json).

Evaluation artifacts:

- [`outputs/evaluation/retrieval_metrics.csv`](outputs/evaluation/retrieval_metrics.csv)
- [`outputs/evaluation/retrieval_metrics.json`](outputs/evaluation/retrieval_metrics.json)
- [`outputs/evaluation/configuration_comparison.csv`](outputs/evaluation/configuration_comparison.csv)
- [`outputs/evaluation/chunking_comparison.csv`](outputs/evaluation/chunking_comparison.csv)
- [`outputs/evaluation/chunking_comparison.json`](outputs/evaluation/chunking_comparison.json)

Run the source tests:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s src -v
```

## 7. Repository documentation

[`README.md`](README.md) remains the project setup and usage guide. This
report is a separate document intended to summarize system design,
experimental results, limitations, and reproducibility.

# RAG Pipeline Flow

This diagram shows the implemented offline indexing path, online question
answering path, the quota-free evidence-inspection path, and the independent
retrieval evaluation path.

```mermaid
flowchart TD
    subgraph INDEX["A. Offline document ingestion and indexing"]
        PDF["Source medical PDF"]
        EXTRACT["PyMuPDF: extract text page by page"]
        CLEAN["Clean spacing and OCR artifacts<br/>preserve page and document IDs<br/>exclude identified wrong-patient page"]
        CHUNK["Detect headings and paragraphs<br/>BGE tokenizer<br/>450-token chunks, 80-token overlap"]
        CHUNKS[("data/processed/chunks.json<br/>doc ID, page, section, text")]
        SECTION["Add section heading to searchable text"]
        DENSE["BAAI/bge-small-en-v1.5<br/>normalized dense embeddings"]
        SPARSE["Local BM25-style sparse vectors"]
        QDRANT[("Local Qdrant<br/>dense + sparse vectors<br/>metadata payload")]

        PDF --> EXTRACT --> CLEAN --> CHUNK --> CHUNKS
        CHUNKS --> SECTION
        SECTION --> DENSE
        SECTION --> SPARSE
        DENSE --> QDRANT
        SPARSE --> QDRANT
    end

    subgraph QUERY["B. Online question and retrieval"]
        USER["User enters a question"]
        UI{"Streamlit action"}
        QEXP["Expand sparse query with<br/>domain terms when applicable"]
        QDENSE["Encode query with BGE<br/>retrieval instruction + normalization"]
        SEARCH["Qdrant dense and sparse prefetch"]
        FUSION["Reciprocal Rank Fusion (RRF)"]
        RERANK["Rerank with RRF, token overlap,<br/>section overlap and name matching"]
        FILTER["Query-specific page diversity<br/>and context limits"]
        EVIDENCE["Selected chunks<br/>document ID, page, section, text"]

        USER --> UI
        UI -->|"Generate answer"| QEXP
        UI -->|"Inspect evidence only"| QEXP
        QEXP --> SEARCH
        QDENSE --> SEARCH
        USER --> QDENSE
        QDRANT --> SEARCH
        SEARCH --> FUSION --> RERANK --> FILTER --> EVIDENCE
        CHUNKS -. "restore original chunk text and document ID" .-> EVIDENCE
    end

    subgraph GENERATION["C. Answer generation (only for Generate answer)"]
        PROMPT["Build structured prompt with<br/>evidence and question"]
        GEMINI["Google Gemini API<br/>temperature 0, minimal thinking,<br/>2048 output-token budget"]
        CHECK{"Generation result"}
        ANSWER["Display answer with<br/>[doc_id, page N] citations"]
        QUOTA["Show quota message<br/>do not retry on HTTP 429"]
        UNKNOWN["If evidence is insufficient:<br/>I don't know."]

        EVIDENCE --> PROMPT --> GEMINI --> CHECK
        CHECK -->|"Answer returned"| ANSWER
        CHECK -->|"Quota exhausted"| QUOTA
        CHECK -->|"No supporting context"| UNKNOWN
    end

    EVIDENCE -->|"Inspect evidence only<br/>no LLM/API request"| INSPECT["Show exact prompt preview<br/>and retrieved PDF passages"]

    subgraph EVALUATION["D. Offline retrieval evaluation (no Gemini call)"]
        BENCH["10 labeled questions<br/>with relevant PDF pages"]
        RUNNER["src/evaluate_retrieval.py<br/>unique-page ranking"]
        SWEEP["Sweep k = 1, 3, 5, 8, 10, 15"]
        METRICS["Hit Rate@k, Recall@k,<br/>Precision@k, MRR@k, nDCG@k"]
        JSON[("outputs/evaluation/<br/>retrieval_metrics.json")]
        CSV[("outputs/evaluation/<br/>retrieval_metrics.csv")]

        BENCH --> RUNNER --> SWEEP --> METRICS
        RUNNER -->|"reuse retrieval path"| QEXP
        METRICS --> JSON
        METRICS --> CSV
    end
```

## What the diagram means

1. **Indexing happens offline.** The PDF is extracted, cleaned, chunked, and
   converted into dense and sparse vectors stored in local Qdrant.
2. **At query time, retrieval is local.** The query is embedded and expanded
   for sparse search where supported. Qdrant fuses dense and sparse results
   with RRF; the application reranks and limits the evidence sent onward.
3. **Evidence inspection does not use Gemini.** The UI can show retrieved
   passages and the prompt preview while the Gemini quota is exhausted.
4. **Answer generation uses Gemini.** Retrieved chunk text and its source
   metadata are included in the prompt. A quota error is displayed without
   trying another model, because the project quota applies to generation.
5. **The evaluation runner measures retrieval only.** It compares six `k`
   settings against the manually labeled relevant-page benchmark and writes
   JSON and CSV scores. It does not evaluate answer correctness or clinical
   safety.

## Source files

- Ingestion: [`src/text_extraction.py`](src/text_extraction.py),
  [`src/clean_text.py`](src/clean_text.py), [`src/chunk.py`](src/chunk.py)
- Index: [`src/embedding.py`](src/embedding.py),
  [`src/sparse_vectors.py`](src/sparse_vectors.py),
  [`src/vector_database.py`](src/vector_database.py)
- Query and answer: [`src/retrieval.py`](src/retrieval.py),
  [`src/rag_pipeline.py`](src/rag_pipeline.py), [`app.py`](app.py)
- Metrics: [`src/evaluate_retrieval.py`](src/evaluate_retrieval.py)


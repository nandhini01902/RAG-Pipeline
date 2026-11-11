# RAG Pipeline Flow

This diagram shows the sample corpus's offline indexing path, runtime
multi-PDF upload path, online question answering, quota-free evidence
inspection, and independent retrieval evaluation.

```mermaid
flowchart TD
    subgraph INDEX["A. Offline document ingestion and indexing"]
        PDF["Sample medical PDF"]
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

    subgraph UPLOAD["A2. Runtime PDF uploads (per request)"]
        FILES["Streamlit uploader<br/>one or more PDFs"]
        UP_EXTRACT["PyMuPDF: extract each page<br/>in memory"]
        UP_CHUNK["Section-aware chunking<br/>keep filename, document ID,<br/>and original page"]
        UP_DENSE["Encode uploaded chunks<br/>with BGE"]
        UP_SPARSE["Build local BM25-style<br/>sparse vectors"]
        UP_QDRANT[("Temporary in-memory Qdrant<br/>dense + sparse vectors")]

        FILES --> UP_EXTRACT --> UP_CHUNK
        UP_CHUNK --> UP_DENSE --> UP_QDRANT
        UP_CHUNK --> UP_SPARSE --> UP_QDRANT
    end

    subgraph QUERY["B. Online question and retrieval"]
        USER["User enters a question"]
        UI{"Streamlit action"}
        DOC_SOURCE{"PDFs uploaded?"}
        REWRITE["Gemini query rewrite<br/>correct spelling and grammar;<br/>preserve intent and clinical details"]
        RAW["Keep original question"]
        SEARCH_QUERY["Rewritten query for answer generation<br/>or original query for inspection"]
        QEXP["Expand sparse query with<br/>domain terms when applicable"]
        QDENSE["Encode query with BGE<br/>retrieval instruction + normalization"]
        SEARCH["Qdrant dense and sparse prefetch"]
        FUSION["Reciprocal Rank Fusion (RRF)"]
        RERANK["Rerank with RRF, token overlap,<br/>section overlap and name matching"]
        FILTER["Query-specific page diversity<br/>and context limits"]
        EVIDENCE["Selected chunks<br/>document ID, page, section, text"]

        USER --> UI
        UI --> DOC_SOURCE
        DOC_SOURCE -->|"No: use sample corpus"| QDRANT
        DOC_SOURCE -->|"Yes: upload one or more"| FILES
        UI -->|"Generate answer"| REWRITE
        UI -->|"Inspect evidence only"| RAW
        REWRITE --> SEARCH_QUERY
        RAW --> SEARCH_QUERY
        SEARCH_QUERY --> QEXP
        QEXP --> SEARCH
        QDENSE --> SEARCH
        SEARCH_QUERY --> QDENSE
        QDRANT -->|"No uploads"| SEARCH
        UP_QDRANT -->|"Uploaded PDFs"| SEARCH
        SEARCH --> FUSION --> RERANK --> FILTER --> EVIDENCE
        CHUNKS -. "restore original chunk text and document ID" .-> EVIDENCE
    end

    subgraph GENERATION["C. Answer generation (only for Generate answer)"]
        PROMPT["Build structured prompt with<br/>evidence, original question,<br/>and rewritten search query"]
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
        ABLATION["Compare RRF only vs.<br/>RRF + lexical reranker at k=5"]
        METRICS["Hit Rate@k, Recall@k,<br/>Precision@k, MRR@k, nDCG@k"]
        JSON[("outputs/evaluation/<br/>retrieval_metrics.json")]
        CSV[("outputs/evaluation/<br/>retrieval_metrics.csv")]
        COMPARE_CSV[("outputs/evaluation/<br/>configuration_comparison.csv")]
        CHUNK_RUNNER["src/evaluate_chunking.py<br/>34 vs 68 chunks"]
        CHUNK_METRICS["Same model, reranker,<br/>benchmark and top-k values"]
        CHUNK_JSON[("outputs/evaluation/<br/>chunking_comparison.json")]
        CHUNK_CSV[("outputs/evaluation/<br/>chunking_comparison.csv")]

        BENCH --> RUNNER --> SWEEP --> METRICS
        RUNNER --> ABLATION --> METRICS
        RUNNER -->|"reuse retrieval path"| QEXP
        BENCH --> CHUNK_RUNNER --> CHUNK_METRICS
        METRICS --> JSON
        METRICS --> CSV
        ABLATION --> COMPARE_CSV
        CHUNK_METRICS --> CHUNK_JSON
        CHUNK_METRICS --> CHUNK_CSV
    end
```

## What the diagram means

1. **The sample corpus is indexed offline.** The bundled PDF is extracted,
   cleaned, chunked, and stored in the repository's local Qdrant index.
   Alternatively, the user can upload one or more PDFs; these are extracted
   and chunked in memory and indexed together in a temporary Qdrant instance
   for that request. They do not replace or persist in the sample corpus.
2. **For answer generation, query rewriting uses Gemini first.** It corrects
   spelling and grammar into a concise search query while preserving intent
   and clinical details. That query is embedded and expanded for sparse search
   where supported. Evidence inspection skips rewriting and searches the
   original question to remain API-free.
3. **Retrieval is local after query preparation.** Qdrant fuses dense and
   sparse results from the selected corpus (sample or uploaded PDFs) with
   RRF; the application reranks and limits the evidence sent onward. Results
   identify the source filename and original page.
4. **Evidence inspection does not use Gemini.** The UI can show retrieved
   passages and the prompt preview while the Gemini quota is exhausted.
5. **Answer generation uses Gemini.** The original question, rewritten search
   query, and retrieved chunk text with source metadata are included in the
   prompt. The full uploaded PDFs are not sent. A quota error is displayed
   without trying another model, because the project quota applies to
   generation.
6. **The evaluation runners measure retrieval only.** They compare six `k`
   settings, reranking on/off at fixed `k=5`, and the historical 34-chunk
   dataset against the current 68-chunk dataset. Chunking configurations use
   isolated temporary indexes and the same manually labeled benchmark.
   They do not call Gemini or measure query-rewrite effectiveness. Results are
   written to JSON and CSV; answer correctness and clinical safety are not
   evaluated.

## Source files

- Ingestion: [`src/text_extraction.py`](src/text_extraction.py),
  [`src/clean_text.py`](src/clean_text.py), [`src/chunk.py`](src/chunk.py),
  [`src/uploaded_documents.py`](src/uploaded_documents.py)
- Index: [`src/embedding.py`](src/embedding.py),
  [`src/sparse_vectors.py`](src/sparse_vectors.py),
  [`src/vector_database.py`](src/vector_database.py)
- Query and answer: [`src/retrieval.py`](src/retrieval.py),
  [`src/rag_pipeline.py`](src/rag_pipeline.py), [`app.py`](app.py)
- Metrics: [`src/evaluate_retrieval.py`](src/evaluate_retrieval.py),
  [`src/evaluate_chunking.py`](src/evaluate_chunking.py)

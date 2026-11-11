
import streamlit as st

from sentence_transformers import SentenceTransformer

from src.rag_pipeline import (
    GeminiQuotaExceededError,
    build_prompt,
    generate_answer,
    rewrite_query,
)
from src.retrieval import retrieve_chunks
from src.uploaded_documents import chunk_uploaded_pdfs, load_chunk_tokenizer


MODEL_NAME = "BAAI/bge-small-en-v1.5"


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(MODEL_NAME)


@st.cache_resource
def load_document_tokenizer():
    return load_chunk_tokenizer()


st.set_page_config(
    page_title="Medical PDF RAG",
    page_icon="📄",
    layout="wide"
)

st.title("Medical PDF Question Answering")

st.write(
    "Upload one or more PDFs to ask questions across their contents, or "
    "search the sample medical-record corpus."
)

st.info(
    "Upload one or more PDFs to search them together, or leave the uploader "
    "empty to use the project's sample corpus. PDFs are processed in memory "
    "and are not saved. Generate answer sends your question and retrieved "
    "passages (not the full PDFs) to Gemini. This is not a medical advice "
    "system."
)

uploaded_files = st.file_uploader(
    "Upload PDF documents",
    type=["pdf"],
    accept_multiple_files=True,
    help=(
        "Selected PDFs are searched together for this question. Their "
        "extracted text stays in memory; retrieved passages are sent to "
        "Gemini only when you generate an answer."
    ),
)

question = st.text_input(
    "Enter your question:",
    placeholder="Example: What imaging examinations were performed?"
)

generate_clicked, inspect_clicked = st.columns(2)
with generate_clicked:
    submit_question = st.button("Generate answer")
with inspect_clicked:
    inspect_evidence = st.button("Inspect retrieved evidence (no Gemini call)")

if submit_question or inspect_evidence:
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        try:
            if uploaded_files:
                uploaded_chunks = chunk_uploaded_pdfs(
                    [
                        (file.name, file.getvalue())
                        for file in uploaded_files
                    ],
                    load_document_tokenizer(),
                )
                document_chunks = uploaded_chunks
                st.info(
                    f"Indexed {len(uploaded_files)} uploaded PDF(s) into "
                    f"{len(uploaded_chunks)} searchable passage(s)."
                )
            else:
                document_chunks = None
                st.info("Searching the project's sample PDF corpus.")

            model = load_embedding_model()
            search_query = (
                rewrite_query(question)
                if submit_question
                else question.strip()
            )
            sources = retrieve_chunks(
                search_query,
                model,
                top_k=8,
                chunks=document_chunks,
            )
            if submit_question:
                st.info(f"Query used for retrieval: {search_query}")
            else:
                st.info(
                    "Evidence inspection uses your original question and "
                    "does not call Gemini for query rewriting."
                )

            if not sources:
                st.warning("No relevant passages were found in the PDF.")
            else:
                st.subheader("Retrieved evidence")
                st.caption(
                    "This preview shows the evidence and question that would "
                    "be placed in the Gemini answer request. Inspection mode "
                    "does not contact Gemini."
                )
                with st.expander("Preview answer request"):
                    st.code(
                        build_prompt(
                            question,
                            sources,
                            search_query=search_query,
                        ),
                        language="text",
                    )

                if inspect_evidence:
                    st.info(
                        "Evidence inspection complete. No Gemini request was made."
                    )
                else:
                    try:
                        answer = generate_answer(
                            question,
                            sources,
                            search_query=search_query,
                        )
                    except GeminiQuotaExceededError as error:
                        st.warning(str(error))
                    else:
                        st.subheader("Answer")
                        st.write(answer)

                st.subheader("Retrieved PDF passages")
                for index, source in enumerate(sources, start=1):
                    with st.expander(
                        f"Result {index} | {source['document_name']} | "
                        f"Page {source['page']} | "
                        f"Score {source['score']:.4f}"
                    ):
                        st.write("**Section:**", source["section"])
                        st.write("**Chunk ID:**", source["chunk_id"])
                        st.write(source["chunk_text"])

        except GeminiQuotaExceededError as error:
            st.warning(str(error))
        except Exception as error:
            st.error(f"An error occurred: {error}")

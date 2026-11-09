
import streamlit as st

from sentence_transformers import SentenceTransformer

from src.rag_pipeline import (
    GeminiQuotaExceededError,
    build_prompt,
    generate_answer,
)
from src.retrieval import retrieve_chunks


MODEL_NAME = "BAAI/bge-small-en-v1.5"


@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(MODEL_NAME)


st.set_page_config(
    page_title="Medical PDF RAG",
    page_icon="📄",
    layout="wide"
)

st.title("Medical PDF Question Answering")

st.write(
    "Ask questions about the uploaded synthetic "
    "medical-record PDF."
)

st.info(
    "Answers use retrieved passages from the provided PDF. Generating an "
    "answer sends your question and those passages to Gemini. Use evidence "
    "inspection to check retrieval without making a Gemini API request. "
    "This is not a medical advice system."
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
            model = load_embedding_model()
            sources = retrieve_chunks(question, model, top_k=8)

            if not sources:
                st.warning("No relevant passages were found in the PDF.")
            else:
                st.subheader("Retrieved evidence")
                st.caption(
                    "This is the evidence placed in the Gemini request. "
                    "Inspection mode does not contact Gemini."
                )
                with st.expander("Preview evidence and question sent to Gemini"):
                    st.code(build_prompt(question, sources), language="text")

                if inspect_evidence:
                    st.info(
                        "Evidence inspection complete. No Gemini request was made."
                    )
                else:
                    try:
                        answer = generate_answer(question, sources)
                    except GeminiQuotaExceededError as error:
                        st.warning(str(error))
                    else:
                        st.subheader("Answer")
                        st.write(answer)

                st.subheader("Retrieved PDF passages")
                for index, source in enumerate(sources, start=1):
                    with st.expander(
                        f"Result {index} | Page {source['page']} | "
                        f"Score {source['score']:.4f}"
                    ):
                        st.write("**Section:**", source["section"])
                        st.write("**Chunk ID:**", source["chunk_id"])
                        st.write(source["chunk_text"])

        except Exception as error:
            st.error(f"An error occurred: {error}")


import streamlit as st

from sentence_transformers import SentenceTransformer

from src.rag_pipeline import answer_question


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
    "Answers are generated only from retrieved passages "
    "in the provided PDF. This is a document-search "
    "demonstration, not a medical advice system."
)

question = st.text_input(
    "Enter your question:",
    placeholder="Example: What imaging examinations were performed?"
)

if st.button("Submit Question"):

    if not question.strip():

        st.warning("Please enter a question.")

    else:

        with st.spinner("Searching the PDF and generating an answer..."):

            try:

                model = load_embedding_model()

                result = answer_question(
                    question,
                    model,
                    top_k=12
                )

                st.subheader("Answer")

                st.write(result["answer"])

                st.subheader("Retrieved PDF Evidence")

                for index, source in enumerate(
                    result["sources"],
                    start=1
                ):

                    with st.expander(
                        f"Result {index} | "
                        f"Page {source['page']} | "
                        f"Score {source['score']:.4f}"
                    ):

                        st.write(
                            "**Chunk ID:**",
                            source["chunk_id"]
                        )

                        st.write(source["chunk_text"])

            except Exception as error:

                st.error(f"An error occurred: {error}")

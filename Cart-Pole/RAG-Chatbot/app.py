import os

import streamlit as st
from dotenv import load_dotenv

from ingest import create_vector_database
from rag import ask_question

load_dotenv()

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖"
)

st.title("🤖 PDF RAG Chatbot")

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)

if uploaded_file:

    os.makedirs("uploads", exist_ok=True)

    file_path = os.path.join(
        "uploads",
        uploaded_file.name
    )

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("PDF uploaded successfully!")

    if st.button("Create Knowledge Base"):

        with st.spinner("Processing PDF..."):

            create_vector_database(file_path)

        st.success("Knowledge base created!")

if os.path.exists("vectorstore"):

    question = st.text_input(
        "Ask a question"
    )

    if question:

        with st.spinner("Thinking..."):

            answer, docs = ask_question(question)

        st.subheader("Answer")

        st.write(answer)

        st.subheader("Retrieved Context")

        for i, doc in enumerate(docs, start=1):

            with st.expander(f"Chunk {i}"):

                st.write(doc.page_content)
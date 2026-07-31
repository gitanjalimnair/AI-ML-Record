import os
from dotenv import load_dotenv

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)

from langchain_community.vectorstores import FAISS

load_dotenv()


import streamlit as st
import os

api_key = os.getenv("GOOGLE_API_KEY") or st.secrets["GOOGLE_API_KEY"]

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=api_key


    )

    db = FAISS.load_local(
        "vectorstore",
        embeddings,
        allow_dangerous_deserialization=True
    )

    docs = db.similarity_search(question, k=3)

    context = "\n\n".join(doc.page_content for doc in docs)

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.2
    )

    prompt = f"""
You are a helpful assistant.

Use ONLY the context below to answer the user's question.

Context:
{context}

Question:
{question}
"""

    response = llm.invoke(prompt)

    return response.content, docs
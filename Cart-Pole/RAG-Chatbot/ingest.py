import os
from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

from utils import extract_text, create_chunks

load_dotenv()


def create_vector_database(pdf_path):

    text = extract_text(pdf_path)

    chunks = create_chunks(text)

    embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    db = FAISS.from_texts(
        texts=chunks,
        embedding=embeddings
    )

    db.save_local("vectorstore")
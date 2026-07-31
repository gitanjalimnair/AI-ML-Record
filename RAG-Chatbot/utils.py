from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def extract_text(pdf_path):
    """Extract text from a PDF."""

    reader = PdfReader(pdf_path)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text

    return text


def create_chunks(text):
    """Split extracted text into chunks."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    return splitter.split_text(text)
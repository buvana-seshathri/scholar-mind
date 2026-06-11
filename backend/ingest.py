from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from llm_client import get_embeddings
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTORSTORE_PATH = os.path.join(os.path.dirname(BASE_DIR), "vectorstore")

def ingest_pdf(pdf_path: str, paper_id: str):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(documents)

    # Tag each chunk with the paper_id for citation tracking
    for chunk in chunks:
        chunk.metadata["paper_id"] = paper_id

    embeddings = get_embeddings()

    save_path = os.path.join(VECTORSTORE_PATH, paper_id)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(save_path)

    return len(chunks)

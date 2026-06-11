import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

def get_embeddings():
    # Runs fully locally — downloads once (~90MB), no API calls ever
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

def get_chat_model():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY must be set in .env")
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        max_retries=3,
        api_key=api_key
    )
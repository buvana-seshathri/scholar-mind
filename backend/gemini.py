import os

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

GEMINI_CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-3.5-flash")
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")


def _ensure_google_api_key() -> None:
    if os.getenv("GOOGLE_API_KEY"):
        return

    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if gemini_api_key:
        os.environ["GOOGLE_API_KEY"] = gemini_api_key
    else:
        raise ValueError("Either GOOGLE_API_KEY or GEMINI_API_KEY environment variable must be set.")


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    _ensure_google_api_key()
    return GoogleGenerativeAIEmbeddings(model=GEMINI_EMBEDDING_MODEL)


def get_chat_model() -> ChatGoogleGenerativeAI:
    _ensure_google_api_key()
    return ChatGoogleGenerativeAI(model=GEMINI_CHAT_MODEL, temperature=0, max_retries=3)

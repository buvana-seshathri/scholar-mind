from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

from llm_client import get_chat_model, get_embeddings
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTORSTORE_PATH = os.path.join(os.path.dirname(BASE_DIR), "vectorstore")

PROMPT_TEMPLATE = """
You are a research assistant helping a user understand academic papers.
Use ONLY the context below to answer. If you don't know, say so.
Always cite the page number from the source when possible.

Context:
{context}

Question: {question}

Answer with citations:
"""

def get_answer(paper_id: str, question: str) -> dict:
    save_path = os.path.join(VECTORSTORE_PATH, paper_id)

    embeddings = get_embeddings()

    vectorstore = FAISS.load_local(
        save_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = get_chat_model()

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

    # 🔥 manual retrieval step (replaces RetrievalQA)
    docs = retriever.invoke(question)

    context = "\n\n".join([d.page_content for d in docs])

    chain = prompt | llm

    result = chain.invoke({
        "context": context,
        "question": question
    })

    sources = [
        {
            "page": doc.metadata.get("page", "?"),
            "snippet": doc.page_content[:200]
        }
        for doc in docs
    ]

    return {
        "answer": result.content if hasattr(result, "content") else str(result),
        "sources": sources
    }
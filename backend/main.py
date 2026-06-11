from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from ingest import ingest_pdf
from rag import get_answer
import shutil, os, uuid
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

PAPERS_DIR = os.path.join(PROJECT_ROOT, "papers")
os.makedirs(PAPERS_DIR, exist_ok=True)
os.makedirs(os.path.join(PROJECT_ROOT, "vectorstore"), exist_ok=True)

@app.post("/upload")
async def upload_paper(file: UploadFile = File(...)):
    paper_id = str(uuid.uuid4())[:8]
    save_path = os.path.join(PAPERS_DIR, f"{paper_id}.pdf")

    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    num_chunks = ingest_pdf(save_path, paper_id)

    return {
        "paper_id": paper_id,
        "filename": file.filename,
        "chunks_indexed": num_chunks
    }

@app.post("/ask")
async def ask_question(paper_id: str = Form(...), question: str = Form(...)):
    result = get_answer(paper_id, question)
    return result

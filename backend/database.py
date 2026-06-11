from sqlalchemy import create_engine, Column, String, Text, DateTime, Integer
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

DATABASE_URL = "sqlite:///./scholarmind.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, index=True)
    paper_id = Column(String, index=True)
    question = Column(Text)
    answer = Column(Text)
    sources = Column(Text)          # stored as JSON string
    created_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

def save_conversation(paper_id: str, question: str, answer: str, sources: list):
    import json
    db = SessionLocal()
    entry = Conversation(
        paper_id=paper_id,
        question=question,
        answer=answer,
        sources=json.dumps(sources)
    )
    db.add(entry)
    db.commit()
    db.close()

def get_history(paper_id: str) -> list:
    import json
    db = SessionLocal()
    rows = db.query(Conversation)\
             .filter(Conversation.paper_id == paper_id)\
             .order_by(Conversation.created_at)\
             .all()
    db.close()
    return [
        {
            "question": r.question,
            "answer": r.answer,
            "sources": json.loads(r.sources),
            "created_at": r.created_at.isoformat()
        }
        for r in rows
    ]
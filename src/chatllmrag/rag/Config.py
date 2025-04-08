import os
from pathlib import Path


class Config:

    ROOT_DIR = Path(os.path.abspath(os.curdir))
    DATABASE_DIR = ROOT_DIR / "db"
    PDF_DIR = ROOT_DIR / "pdf_data"
    TEXT_DIR = ROOT_DIR / "text_data"
    COLLECTION_NAME = 'document_embeddings'
    CONVERSATION_MESSAGES_LIMIT =10

    class Model:
        EMBEDDINGS = "BAAI/bge-base-en-v1.5"
        RANKER = "ms-marco-MiniLM-L-12-v2"
        llm = "gpt-4o"

from os import listdir

from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient

from src.chatllmrag.rag.Config import Config


class Vector_DB_DAO:
    def __init__(self):
        self.embeddings = FastEmbedEmbeddings(model_name=Config.Model.EMBEDDINGS)

    def create_db_from_documents(self):
        chunked_docs = self.parse_documents_for_DB()
        self.write_to_db(chunked_docs,self.embeddings)
    def parse_documents_for_DB(self):
        list_documents = listdir(Config.TEXT_DIR)
        rez = []
        for doc in list_documents:
            loader = UnstructuredMarkdownLoader(Config.TEXT_DIR / doc)
            loaded_documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=2048, chunk_overlap=128)
            docs = text_splitter.split_documents(loaded_documents)
            rez.extend(docs)
        return rez

    def write_to_db(self,docs, embeddings):
        qdrant = Qdrant.from_documents(
            docs,
            embeddings,
            # location=":memory:",
            path=Config.DATABASE_DIR,
            collection_name=Config.COLLECTION_NAME
        )
    def get_quarant_from_DB(self):
        client = QdrantClient(path=Config.DATABASE_DIR)
        qdrant = Qdrant(client, Config.COLLECTION_NAME, self.embeddings)
        return qdrant


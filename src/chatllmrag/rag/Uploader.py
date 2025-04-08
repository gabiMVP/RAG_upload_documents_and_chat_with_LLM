import shutil
from os import listdir
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, UnstructuredMarkdownLoader
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from src.chatllmrag.rag.Config import Config


async def  load_document(path):
    loader = PyPDFLoader(path)
    pages = []
    async for page in loader.alazy_load():
        pages.append(page)
    return  pages

def write_text_document(document):
    document_path = document[0].metadata['source']
    document_path= document_path.replace("pdf_data","text_data").replace("pdf","txt")
    document_path = Path(document_path)
    listText = [x.page_content for x in document]
    text = "".join(  x+"\n" for x in listText )
    with document_path.open("w",encoding="utf-8") as f:
        f.write(text)

def parse_documents_for_DB():
    dir = Config.TEXT_DIR
    list_documents =listdir(dir)
    rez = []
    for doc in list_documents:
        loader = UnstructuredMarkdownLoader(dir+"/"+doc)
        loaded_documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2048, chunk_overlap=128)
        docs = text_splitter.split_documents(loaded_documents)
        rez.extend(docs)
    return rez

def write_to_db(docs,embeddings):
    qdrant = Qdrant.from_documents(
        docs,
        embeddings,
        # location=":memory:",
        path=Config.DATABASE_DIR,
        collection_name=Config.COLLECTION_NAME,
    )
    # file_path = "./pdf_data/meta-earnings.pdf"
    # document = await load_document(file_path)
    # write_text_document(document)
    # documents_for_db_chuncked= parse_documents_for_DB()
    # embeddings = FastEmbedEmbeddings(model_name=Config.Model.EMBEDDINGS)
    # write_to_db(documents_for_db_chuncked,embeddings)
async def load_documents(files,remove_old_files=True):
    shutil.rmtree(Config.DATABASE_DIR, ignore_errors=True)
    shutil.rmtree(Config.PDF_DIR, ignore_errors=True)
    shutil.rmtree(Config.TEXT_DIR, ignore_errors=True)
    Config.PDF_DIR.mkdir(parents=True, exist_ok=True)
    Config.TEXT_DIR.mkdir(parents=True, exist_ok=True)
    file_paths = []
    for file in files:
        file_path = Config.PDF_DIR / file.name
        with file_path.open('wb') as f:
            f.write(file.getvalue())
        file_paths.append(file_path)
    for path in file_paths:
        document = await load_document(path)
        document_path = str(path).replace("pdf_data", "text_data").replace("pdf", "txt")
        document_path = Path(document_path)
        with document_path.open("w", encoding="utf-8") as f:
            listText = [x.page_content for x in document]
            text = "".join(x + "\n" for x in listText)
            f.write(text)


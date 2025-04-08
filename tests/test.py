import os

from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import PromptTemplate
from openai import OpenAI
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank
from langchain_community.document_loaders import PyPDFLoader, UnstructuredMarkdownLoader
import asyncio
from pathlib import Path
from os import listdir
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from flashrank import Ranker

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
    dir = "./text_data"
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
        path="./db",
        collection_name="document_embeddings",
    )

async def  main():
    file_path = "./pdf_data/meta-earnings.pdf"
    document = await load_document(file_path)
    write_text_document(document)
    documents_for_db_chuncked= parse_documents_for_DB()
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    if(len(os.listdir("./db"))) ==0:
        write_to_db(documents_for_db_chuncked,embeddings)
    query = "What is the most important innovation from Meta?"
    client = QdrantClient(path="./db")
    collection_name = "document_embeddings"
    qdrant = Qdrant(client, collection_name, embeddings)
    similar_docs = qdrant.similarity_search_with_score(query)

    for doc, score in similar_docs:
        print(f"text: {doc.page_content[:256]}\n")
        print(f"score: {score}")
        print("-" * 80)
        print()


    ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")
    compressor = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2")
    retriever = qdrant.as_retriever(search_kwargs={"k": 5})
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )
    reranked_docs = compression_retriever.invoke(query)
    for doc in reranked_docs:
        print(f"id: {doc.metadata['_id']}\n")
        print(f"text: {doc.page_content}\n")
        print(f"score: {doc.metadata['relevance_score']}")
        print("-" * 80)
        print()
    file = open("api_key", "r")
    api_key = file.readline()
    file.close()

    prompt_template = """
    Use the following pieces of information to answer the user's question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context: {context}
    Question: {question}

    Answer the question and provide additional helpful information,
    based on the pieces of information, if applicable. Be succinct.

    Responses should be properly formatted to be easily read.
    """

    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key=api_key  # if you prefer to pass api key in directly instaed of using env vars
        # base_url="...",
        # organization="...",
        # other params...
    )
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=compression_retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt, "verbose": True},
    )
    response = qa.invoke("What is the most significant innovation from Meta?")
    response = qa.invoke("What is the revenue for 2024 and % change?")
    print(response["query"])
    print(response["result"])
    x=1

if __name__== "__main__":
    asyncio.run(main())
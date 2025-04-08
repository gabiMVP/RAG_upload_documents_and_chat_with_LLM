from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank


def create_retriever(llm, vector_store):
    compressor = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2")
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )
    return compression_retriever

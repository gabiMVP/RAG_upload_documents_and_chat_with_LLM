# RAG upload document and chat with LLM

[try it out here!](https://chatllm-592795719124.europe-west12.run.app/)


The purpose is to create a program where a user can upload a document and ask questiones \
about it using an LLM

 
### Implementation notes:

Streamlit is used as the interface \
The LLM chosen is gpt-4o \
Qdrant is used as the vector database\
BAAI/bge-base-en-v1.5 is used to encode to embeddings \
ms-marco-MiniLM-L-12-v2 is used as the reranker


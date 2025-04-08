from langchain_community.document_compressors import FlashrankRerank
from langchain_openai import ChatOpenAI
import os

from src.chatllmrag.rag.Config import Config


def getllm():

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key=os.getenv("OPEN_AI")  # if you prefer to pass api key in directly instaed of using env vars
        # base_url="...",
        # organization="...",
        # other params...
    )
    return llm

def get_compressor():
    FlashrankRerank(model=Config.Model.RANKER)

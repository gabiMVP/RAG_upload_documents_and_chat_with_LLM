import asyncio
import os
import random

import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

from src.chatllmrag.rag .Config import Config
from src.chatllmrag.rag .Uploader import load_documents
from src.chatllmrag.rag .Vector_DB_DAO import Vector_DB_DAO
from src.chatllmrag.rag .retriever import create_retriever
from src.chatllmrag.rag .chain import create_chain

from src.chatllmrag.rag import models


st.write("PLOP CHAT 1")
load_dotenv()
api_key=os.getenv("OPEN_AI")

def show_upload_documents():
    holder = st.empty()
    with holder.container():
        st.header("PlopChat")
        st.subheader("Get answers from your documents")
        uploaded_files = st.file_uploader(
            label="Upload PDF files", type=["pdf"], accept_multiple_files=True
        )
    if not uploaded_files:
        st.warning("Please upload PDF documents to continue!")
        st.stop()

    with st.spinner("Analyzing your document(s)..."):
        holder.empty()
        st.session_state["loaded"] = True
        asyncio.run(load_documents(uploaded_files))
        # return create_QA_Chain()
        # asyncio.run(load_documents(uploaded_files))

@st.cache_resource()
def create_QA_Chain():
    vector_dao = Vector_DB_DAO()
    vector_dao.create_db_from_documents()
    vector_store = vector_dao.get_quarant_from_DB()
    llm = models.getllm()
    retriever = create_retriever(llm, vector_store=vector_store)
    chain = create_chain(llm, retriever)
    return chain

def show_message_history():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        role = message["role"]
        with st.chat_message(role):
            if(role == "user"):
                st.markdown(message["content"])
            else:
                response = message['content']
                st.markdown(response['result'])
                for i, doc in enumerate(response['source_documents']):
                    with st.expander(f"Source #{i + 1}"):
                        st.write(doc.page_content)

def  do_chat_rag(chain):
    if prompt := st.chat_input("ask here?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        response = chain.invoke(prompt)
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response['result'])
            for i, doc in enumerate(response['source_documents']):
                with st.expander(f"Source #{i + 1}"):
                    st.write(doc.page_content)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})


if Config.CONVERSATION_MESSAGES_LIMIT > 0 and st.session_state.__contains__('messages') and Config.CONVERSATION_MESSAGES_LIMIT<= len(
    st.session_state.messages
):
    st.warning(
        "You have reached the conversation limit. Refresh the page to start a new conversation."
    )
    st.stop()

# show_upload_documents()
if( not st.session_state.__contains__("loaded")):
    show_upload_documents()
chain = create_QA_Chain()
show_message_history()
do_chat_rag(chain)

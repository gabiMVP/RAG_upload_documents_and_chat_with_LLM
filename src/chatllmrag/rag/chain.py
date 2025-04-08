from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory

prompt_template = """
Use the following pieces of information to answer the user's question.
If the answer is not found within the context, state that the answer cannot be found.

Context: {context}
Question: {question}

Answer the question and provide additional helpful information,
based on the pieces of information, if applicable. Be succinct.

Responses should be properly formatted to be easily read.
"""



def create_chain(llm, retriever) :
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt, "verbose": True},
    )
    return qa
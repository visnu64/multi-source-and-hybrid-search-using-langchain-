import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_community.vectorstores import faiss
from langchain_community.document_loaders import PyPDFLoader

from dotenv import load_dotenv
load_dotenv()

## load the groq and the openai api key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

st.title("Langchain with Groq with llama3")

llm=ChatGroq(model_name="llama-3.3-70b-versatile", groq_api_key=groq_api_key)

prompt = ChatPromptTemplate.from_template(
    """You are a helpful assistant. Use the following context to answer the question.
    please provide the most accurate response based on the quesion
    <context>
    {context}
    </context>
    Question: {input}
    """
)

prompt1=st.text_input("Enter your question from the document")

def vector_embedding():

    if 'vectors' not in st.session_state:

     st.session_state.embeddings = OpenAIEmbeddings(base_url="https://openrouter.ai/api/v1")
     st.session_state.loader=PyPDFLoader('./attention.pdf')#data ingesion
     st.session_state.docs=st.session_state.loader.load()#document loading
     st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)#chunk creation 
     st.session_state.final_docs=st.session_state.text_splitter.split_documents(st.session_state.docs)# spliting
     st.session_state.vectors= faiss.FAISS.from_documents(st.session_state.final_docs,st.session_state.embeddings)## vector store openai embeddings
    

if st.button("documents embeddings"):
    vector_embedding()
    st.write("vector store is ready")

import time

if prompt1:
    start=time.process_time()
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriver = st.session_state.vectors.as_retriever()
    retriever_chain = create_retrieval_chain( retriver, document_chain)

    response=retriever_chain.invoke({"input": prompt1})
    ##response = retrieval_tool.invoke({"query": prompt1})



    print("response time:", time.process_time()-start)

    st.write(response['answer'])

## with a stramlit expander
    with st.expander("Document similiarity search"):
    ## find the relevant chunks
         for i, doc in enumerate(response["context"]):
           st.write(doc.page_content)
           st.write("---------------------------------------")


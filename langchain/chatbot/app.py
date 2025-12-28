from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

#langsmith
os.environ["LANGCHAIN_TRACING_V2"]="true"
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
if langchain_api_key:
    os.environ["LANGCHAIN_API_KEY"] = langchain_api_key

#prompt templete

prompt=ChatPromptTemplate.from_messages(
    [
        ("system","you are a helpful assistant . please response to the user queries"),
        ("user","question:{question}")
    ]
)

#streamlit framework

st.title("Langchain with open API")
input_text =st.text_input("search the topic you want")

llm = ChatOpenAI(model="openai/gpt-oss-20b:free", base_url="https://openrouter.ai/api/v1")
output_parser = StrOutputParser()
chain=prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({'question':input_text}))

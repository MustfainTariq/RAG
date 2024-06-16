from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_TRACING_V2"]="true"
## Prompt Template

prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. Please response to the user queries"),
        ("user","Question:{question}")
    ]
)
## streamlit framework

st.title('Langchain Demo With LLAMA3 API')
input_text=st.text_input("Search the topic u want")

# ollama LLAma3 LLm 
llm=Ollama(model="llama3")
output_parser=StrOutputParser()
chain=prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({"question":input_text}))
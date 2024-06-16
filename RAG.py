import streamlit as st
import pandas as pd
from langchain.embeddings.fastembed import FastEmbedEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
import tiktoken
from langchain.vectorstores import Qdrant
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOllama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from dotenv import load_dotenv
import os

semester = list()
file_path = 'MS_Thesis_Repository_FSC.xlsx'
all_data = pd.read_excel(file_path, sheet_name=None)
thesis_data = all_data
for sheet_name, data in all_data.items():
    semester.append(sheet_name)

file_path = 'MS_Thesis_Repository_FSC.xlsx'
all_data = pd.read_excel(file_path, sheet_name=None)

for sheet_name, data in all_data.items():
    processed_data = data.iloc[1:, [1, 3]]  
    processed_data.columns = ['Thesis Title', 'Thesis Abstract']
    
    output_path = f'{sheet_name}.xlsx'
    processed_data.to_excel(output_path, index=False)


dataframes = []

for file,data in all_data.items():
    file = f'{file}.xlsx'
    df = pd.read_excel(file)
    dataframes.append(df)

combined_df = pd.concat(dataframes, ignore_index=True)
txt_file_path = 'combined_output.txt'
combined_df.to_csv(txt_file_path, sep='\t', index=False, encoding='utf-8')

# Path for the resulting text file
txt_file_path = 'combined_output.txt'

# Write the combined DataFrame to a text file, separated by tabs
combined_df.to_csv(txt_file_path, sep='\t', index=False)

path_f = 'combined_output.txt'

loader = TextLoader(path_f)
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=7000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

with open(path_f, 'r') as file:
    content = file.read()

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens
    
numtoken = num_tokens_from_string(content, "cl100k_base")

embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5", max_length=2048)

qdrant = Qdrant.from_documents(
    texts,
    embeddings,
    path="./local_qdrant1",
    collection_name="db2",
)








# Streamlit interface
st.title('Langchain Demo With LLAMA3 API')

input_text = st.text_input("Enter your question related to the documents:")

if input_text:
    docs = qdrant.similarity_search(input_text)
    
    template = """[INST] <<SYS>> Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer. 
    <</SYS>>
    {context}
    Question: {question}
    Helpful Answer:[/INST]"""
    
    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )
    
    chat_model = ChatOllama(
        model="llama3",
        verbose=True,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        chat_model,
        retriever=qdrant.as_retriever(),
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    )
    
    result = qa_chain({"query": input_text})
    st.write(result)

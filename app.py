import warnings
import os
import easyocr
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st
from pypdf import PdfReader
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.agents.agent_types import AgentType
from langchain_community.chat_models import ChatOllama
from langchain_community.llms import Ollama
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import pandas as pd
warnings.filterwarnings('ignore')
load_dotenv()

# Initialize embeddings model once
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

def image(file, query):
    # Extract text using OCR
    file_bytes = file.read()
    reader = easyocr.Reader(["en"])
    results = reader.readtext(file_bytes)
    text = " ".join(result[1] for result in results)

    # Split text into chunks
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=10, chunk_overlap=0, length_function=len)
    texts = text_splitter.split_text(text)

    # Wrap texts in Document objects
    documents = [Document(page_content=chunk) for chunk in texts]

    # Create embeddings and search
    document_search = FAISS.from_texts(texts, embeddings)
    docs = document_search.similarity_search(query)

    # Load QA chain
    llm = Ollama(model="mistral", temperature=0.7)
    chain = load_qa_chain(llm, chain_type="stuff")

    # Run QA chain
    response = chain.run(input_documents=docs, question=query)

    return response

def process_text(text):
    # Split the text into chunks
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks = text_splitter.split_text(text)

    # Convert chunks into embeddings
    knowledgeBase = FAISS.from_texts(chunks, embeddings)
    return knowledgeBase

def pdf(pdffile, query):
    pdf = PdfReader(pdffile)
    text = ""
    for page in pdf.pages:
        text += page.extract_text()

    knowledgeBase = process_text(text)
    docs = knowledgeBase.similarity_search(query)
    llm = Ollama(model='mistral', temperature=0.7)
    chain = load_qa_chain(llm, chain_type='stuff')
    response = chain.run(input_documents=docs, question=query)
    return response


# Define a function to create Pandas DataFrame agent from a CSV file.
def create_pd_agent(query,file):

    df=pd.read_excel(file)

    # Initiate a connection to the LLM from Ollama via LangChain.
    llm = Ollama(temperature=0.7,model='llama3')

    # Create a Pandas DataFrame agent from the CSV file.
    agent=create_pandas_dataframe_agent(llm, df, verbose=False,allow_dangerous_code=True,agent_executor_kwargs={"handle_parsing_errors": True})
    response=agent.run(query)
    return response


def text(file, query):
    # Read the file content
    content = file.read().decode("utf-8")
    
    # Wrap content in Document object
    documents = [Document(page_content=content)]
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20, length_function=len)
    texts = text_splitter.split_documents(documents)

    vector_store = FAISS.from_documents(texts, embeddings)
    qa = RetrievalQA.from_chain_type(llm=Ollama(temperature=0.7, model='llama3'), chain_type="stuff", retriever=vector_store.as_retriever(), return_source_documents=True)
    result = qa({"query": query})
    return result

def csvfile(filepath, query):
    agent = create_csv_agent(Ollama(temperature=0.7, model='mistral'), filepath, verbose=True, allow_dangerous_code=True)
    res = agent.run(query)
    return res

st.title('CHAT WITH ANY FILE')

option = st.selectbox("What kind of file do you want to upload?", ("pdf", "csv", "txt", "image","excel"))

if option:
    file = st.file_uploader("Upload file", type=["jpeg", "jpg", "png", "csv", 'pdf', 'txt','xlsx'])
    if file:
        question = st.text_input("Ask a query on the uploaded file")
        if question:
            filename = file.name
            file_extension = Path(filename).suffix.lower()
            if file_extension in ['.jpg', '.jpeg', '.png']:
                result = image(file=file, query=question)
                st.write(result)
            elif file_extension == '.csv':
                response = csvfile(filepath=file, query=question)
                st.write(response)
            elif file_extension == '.pdf':
                result = pdf(pdffile=file, query=question)
                st.write(result)
            elif file_extension == '.txt':
                result = text(file=file, query=question)
                st.write(result)
            elif file_extension == '.xlsx':
                
                result = create_pd_agent(file=file,query=question)
                st.write(result)
            else:
                st.write("Please upload a valid file type.")

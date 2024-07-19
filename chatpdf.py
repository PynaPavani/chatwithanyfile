from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
os.environ['GOOGLE_API_KEY']='AIzaSyAkkjRV_g0qwFcjibYmaoaN22yimufftNk'
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import Ollama


# Load environment variables
load_dotenv()

def process_text(text):
    # Split the text into chunks using Langchain's CharacterTextSplitter
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    # Convert the chunks of text into embeddings to form a knowledge base
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    knowledgeBase = FAISS.from_texts(chunks, embeddings)
    return knowledgeBase

def pdf(pdffile,query):
    pdf=PdfReader(pdffile) 
        # Text variable will store the pdf text
    text = ""
    for page in pdf.pages:
        text += page.extract_text()
        
        # Create the knowledge base object
    knowledgeBase = process_text(text)
    docs = knowledgeBase.similarity_search(query)
    llm = Ollama(model='mistral',temperature=0.7)
    chain = load_qa_chain(llm, chain_type='stuff')
    response = chain.run(input_documents=docs, question=query)
    return response

response=pdf("./data/attention.pdf","what is transformer")
print(response)
            
  
    


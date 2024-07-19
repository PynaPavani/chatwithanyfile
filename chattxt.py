from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os

os.environ['GOOGLE_API_KEY']='AIzaSyAkkjRV_g0qwFcjibYmaoaN22yimufftNk'
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

load_dotenv()

def text(filepath,query):
    documents = TextLoader(filepath).load()
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=20,
    length_function=len)
    texts = text_splitter.split_documents(documents)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    vector_store = FAISS.from_documents(
    texts, embeddings)
    qa = RetrievalQA.from_chain_type(
    llm=Ollama(temperature=0.7,model='llama3'), chain_type="stuff", retriever=vector_store.as_retriever(), return_source_documents=True)
    result = qa({"query": query})
    return result

if __name__ == '__main__':
    
    question = "Where langchain is used for ? Treat me as a beginner and Give me a short answer"
    result=text(filepath='./data/requirements.txt',query=question)
    print(result)
import warnings
import os
import easyocr
from pathlib import Path
from dotenv import load_dotenv
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
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

warnings.filterwarnings('ignore')
load_dotenv()

df= pd.read_csv('metadata.csv')

# Initialize embeddings model once
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
from langchain_community.embeddings import OllamaEmbeddings

ollama_emb = OllamaEmbeddings(
    model="mistral",
)
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

def create_pd_agent(query, file):
    df = pd.read_excel(file)
    llm = Ollama(temperature=0.7, model='llama3')
    agent = create_pandas_dataframe_agent(llm, df, verbose=True, allow_dangerous_code=True, agent_executor_kwargs={"handle_parsing_errors": True})
    response = agent.run(query)
    return response

def text(file, query):
    content = file.read().decode("utf-8")
    documents = [Document(page_content=content)]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20, length_function=len)
    texts = text_splitter.split_documents(documents)
    vector_store = FAISS.from_documents(texts, embeddings)
    qa = RetrievalQA.from_chain_type(llm=Ollama(temperature=0.7, model='llama3'), chain_type="stuff", retriever=vector_store.as_retriever(), return_source_documents=True)
    result = qa({"query": query})
    response = result['result'] if 'result' in result else result.get('answer', '')
    # Limit the response to 50 words
    response_words = response.split()
    if len(response_words) > 70:
        response = ' '.join(response_words[:70]) + '...'
    
    return response

def csvfile(filepath, query):
    agent = create_csv_agent(Ollama(temperature=0.7, model='mistral'), filepath, verbose=True, allow_dangerous_code=True, agent_executor_kwargs={"handle_parsing_errors": True})
    res = agent.run(query)
    return res

def add_querytodf(query,response,file_extension):
    global df
    new_entry = {'query': query, 'response': response, 'chartname': None, 'file_type': file_extension}
    df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
    df.to_csv('metadata.csv', index=False)
    
def get_similarity(query,file_extension):

    queries = df.loc[df['file_type'] == file_extension, 'query']
    # Convert queries to a list of strings
    queries_list = queries.tolist()
    r1 = ollama_emb.embed_documents(queries_list)
    r2 = ollama_emb.embed_query(query)
    # Calculate cosine similarity
    similarity = cosine_similarity(r1, np.array(r2).reshape(1, -1))
    print(similarity)

    # Find the highest similarity and its index
    max_similarity_index = np.argmax(similarity)
    max_similarity_value = similarity[max_similarity_index]

    return max_similarity_value,queries_list,max_similarity_index

    

def get_query(query,file):
    file_extension = Path(file).suffix.lower()
    if len(df[df['file_type'] == file_extension]) <= 0:
            response = csvfile(filepath=file, query=query)
            print(response)
            add_querytodf(query=query,response=response,file_extension=file_extension)
    else:
        max_value,queries,index=get_similarity(query=query,file_extension=file_extension)
        if max_value > 0.70:
            main = queries[index]
            print(df.loc[df['query'] == main, 'response'].iloc[0])
        else:
            response = csvfile(filepath=file, query=query)
            print(response)
            add_querytodf(query=query,response=response,file_extension=file_extension)
    

if __name__ == '__main__':
    get_query('what is max area price','./data/homeprices.csv')
                

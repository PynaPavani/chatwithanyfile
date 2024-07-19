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
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

warnings.filterwarnings('ignore')
load_dotenv()

# Initialize embeddings model once
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
from langchain_community.embeddings import OllamaEmbeddings

ollama_emb = OllamaEmbeddings(
    model="mistral",
)

#COLUMN_NAMES = ['query', 'response', 'chartname', 'file_type']
#df = pd.DataFrame(columns=COLUMN_NAMES)
#df.to_csv('metadata.csv', index=False)

df=pd.read_csv('metadata.csv')

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

st.title('CHAT WITH ANY FILE')

option = st.selectbox("What kind of file do you want to upload?", ("pdf", "csv", "txt", "excel"))

if option:
    file = st.file_uploader("Upload file", type=["csv", 'pdf', 'txt', 'xlsx'])
    if file:
        query = st.text_input("Ask a query on the uploaded file")
        if query:
            filename = file.name
            file_extension = Path(filename).suffix.lower()
            if file_extension == '.csv':
                if len(df[df['file_type'] == file_extension]) <= 0:
                    print(len(df[df['file_type']]))
                    response = csvfile(filepath=file, query=query)
                    st.write(response)
                    new_entry = {'query': query, 'response': response, 'chartname': None, 'file_type': file_extension}
                    #df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
                    df=df._append(new_entry,ignore_index=True)
                    df.to_csv('metadata.csv', index=False)
                else:
                    queries = df.loc[df['file_type'] == '.csv', 'query']
                    print(type(queries))

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

                    # Check if the highest similarity is above the threshold
                    if max_similarity_value > 0.70:
                        print(max_similarity_value)
                        main = queries_list[max_similarity_index]
                        print(main)
                        st.write(df.loc[df['query'] == main, 'response'].iloc[0])
                    else:
                        response = csvfile(filepath=file, query=query)
                        st.write(response)
                        new_entry = {'query': query, 'response': response, 'chartname': None, 'file_type': file_extension}
                        df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
                        df.to_csv('metadata.csv', index=False)

            elif file_extension == '.pdf':
                file_type = '.txt'
                if len(df[df['file_type'] == file_type]) == 0:
                    result = pdf(pdffile=file, query=query)
                    st.write(result)
                    new_entry = {'query': query, 'response': result, 'chartname': None, 'file_type': file_type}
                    df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
                    df.to_csv('metadata.csv', index=False)
                else:
                    queries = df.loc[df['file_type'] == '.txt', 'query']
                    queries_list = queries.tolist()
                    r1 = ollama_emb.embed_documents(queries_list)
                    r2 = ollama_emb.embed_query(query)

                    # Calculate cosine similarity
                    similarity = cosine_similarity(r1, np.array(r2).reshape(1, -1))
                    print(similarity)

                    # Find the highest similarity and its index
                    max_similarity_index = np.argmax(similarity)
                    max_similarity_value = similarity[max_similarity_index]

                    # Check if the highest similarity is above the threshold
                    if max_similarity_value > 0.70:
                        print(max_similarity_value)
                        main = queries_list[max_similarity_index]
                        print(main)
                        st.write(df.loc[df['query'] == main, 'response'].iloc[0])
                    else:
                        result = pdf(pdffile=file, query=query)
                        st.write(result)
                        new_entry = {'query': query, 'response': result, 'chartname': None, 'file_type': file_type}
                        df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
                        df.to_csv('metadata.csv', index=False)

            elif file_extension == '.txt':
                if len(df[df['file_type'] == file_extension]) <= 0:
                    result = text(file=file, query=query)
                    st.write(result)
                    new_entry = {'query': query, 'response': result, 'chartname': None, 'file_type': file_extension}
                    df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
                    df.to_csv('metadata.csv', index=False)
                else:
                    queries = df.loc[df['file_type'] == '.txt', 'query']
                    queries_list = queries.tolist()
                    r1 = ollama_emb.embed_documents(queries_list)
                    r2 = ollama_emb.embed_query(query)

                    # Calculate cosine similarity
                    similarity = cosine_similarity(r1, np.array(r2).reshape(1, -1))
                    print(similarity)

                    # Find the highest similarity and its index
                    max_similarity_index = np.argmax(similarity)
                    max_similarity_value = similarity[max_similarity_index]

                    # Check if the highest similarity is above the threshold
                    if max_similarity_value > 0.70:
                        print(max_similarity_value)
                        main = queries_list[max_similarity_index]
                        print(main)
                        st.write(df.loc[df['query'] == main, 'response'].iloc[0])
                    else:
                        result = text(file=file, query=query)
                        st.write(result)
                        new_entry = {'query': query, 'response': result, 'chartname': None, 'file_type': file_extension}
                        df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
                        df.to_csv('metadata.csv', index=False)

            elif file_extension == '.xlsx':
                if len(df[df['file_type'] == file_extension]) <= 0:
                    response = create_pd_agent(query=query, file=file)
                    st.write(response)
                    new_entry = {'query': query, 'response': response, 'chartname': None, 'file_type': file_extension}
                    df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
                    df.to_csv('metadata.csv', index=False)
                else:
                    queries = df.loc[df['file_type'] == '.xlsx', 'query']
                    queries_list = queries.tolist()
                    r1 = ollama_emb.embed_documents(queries_list)
                    r2 = ollama_emb.embed_query(query)

                    # Calculate cosine similarity
                    similarity = cosine_similarity(r1, np.array(r2).reshape(1, -1))
                    print(similarity)

                    # Find the highest similarity and its index
                    max_similarity_index = np.argmax(similarity)
                    max_similarity_value = similarity[max_similarity_index]

                    # Check if the highest similarity is above the threshold
                    if max_similarity_value > 0.70:
                        print(max_similarity_value)
                        main = queries_list[max_similarity_index]
                        print(main)
                        st.write(df.loc[df['query'] == main, 'response'].iloc[0])
                    else:
                        response = create_pd_agent(query=query, file=file)
                        st.write(response)
                        new_entry = {'query': query, 'response': response, 'chartname': None, 'file_type': file_extension}
                        df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
                        df.to_csv('metadata.csv', index=False)
            else:
                st.write("Please upload a valid file type.")

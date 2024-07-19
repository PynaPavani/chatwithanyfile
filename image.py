import os
import tempfile
from dotenv import load_dotenv
from langchain_community.llms import Ollama
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.schema import Document
import easyocr

def main(filepath,query):
    load_dotenv() 
        # extract the text using OCR
    reader = easyocr.Reader(["en"])  # language
    results = reader.readtext(filepath)

    text = " ".join(result[1] for result in results)

        # split text into chunks
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=10, chunk_overlap=0, length_function=len)
    texts = text_splitter.split_text(text)

        # wrap texts in Document objects
    documents = [Document(page_content=chunk) for chunk in texts]

        # create embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    document_search = FAISS.from_texts(texts, embeddings)
    docs = document_search.similarity_search(query)

            # load QA chain
    llm = Ollama(model="mistral", temperature=0.7)
    chain = load_qa_chain(llm, chain_type="stuff")

            # run QA chain
    response = chain.run(input_documents=docs, question=query)

    return response

if __name__ == '__main__':
    response=main(filepath="./data/invoice.jpg",query="what is the due date?")
    print(response)

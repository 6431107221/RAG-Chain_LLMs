# ingestion_hf.py
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import config

def ingest_docs():
    # 1. Load PDF
    if not os.path.exists(config.FILE_PATH):
        print(f"Not found: {config.FILE_PATH}")

    loader = PyPDFLoader(config.FILE_PATH)
    docs = loader.load()
    print(f"PDF: {len(docs)} pages")

    # 2. Split Text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    print(f"Separate PDF: {len(splits)} Chunks")

    # 3. Embed & Store
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=config.HF_VECTOR_STORE_PATH
    )
    print(f"Output Path: {config.HF_VECTOR_STORE_PATH}")

if __name__ == "__main__":
    ingest_docs()
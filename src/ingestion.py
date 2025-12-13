# ingestion.py
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
import config  # นำเข้าค่า config ที่เราตั้งไว้

def ingest_docs():
    # --- 1. Load PDF ---
    if not os.path.exists(config.FILE_PATH):
        print(f"Not found: {config.FILE_PATH}")

    loader = PyPDFLoader(config.FILE_PATH)
    raw_documents = loader.load()
    print(f"PDF: {len(raw_documents)} pages")

    # --- 2. Split (Chunks) ---
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )
    documents = text_splitter.split_documents(raw_documents)
    print(f"Separate PDF: {len(documents)} Chunks")

    # --- 3. Embed ---
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # --- 4. Store (ChromaDB) ---
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=config.VECTOR_STORE_PATH
    )
    
    print(f"Output Path: {config.VECTOR_STORE_PATH}")

if __name__ == "__main__":
    ingest_docs()
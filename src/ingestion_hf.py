# ingestion_hf.py
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# ✅ เปลี่ยนมาใช้ HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import config

# ✅ กำหนดชื่อ Folder DB ใหม่ เพื่อไม่ให้ตีกับอันเก่า
DB_PATH_HF = "./chroma_db_hf"

def ingest_docs():
    print(f"🚀 เริ่มต้นกระบวนการ Ingestion (Hugging Face Mode)...")
    
    # 1. Load PDF
    loader = PyPDFLoader(config.FILE_PATH)
    docs = loader.load()
    print(f"📄 โหลดเอกสารสำเร็จ: {len(docs)} หน้า")

    # 2. Split Text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    print(f"🧩 แบ่งเป็น: {len(splits)} ชิ้น (Chunks)")

    # 3. Embed & Store
    print("🧠 กำลังสร้าง Embeddings ด้วย HuggingFace (อาจใช้เวลาสักครู่ในครั้งแรก)...")
    
    # ใช้โมเดลยอดนิยม 'all-MiniLM-L6-v2' (เล็ก เร็ว แม่นยำ)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=DB_PATH_HF
    )
    print(f"🎉 เสร็จสิ้น! บันทึก Database ใหม่ไว้ที่: {DB_PATH_HF}")

if __name__ == "__main__":
    ingest_docs()
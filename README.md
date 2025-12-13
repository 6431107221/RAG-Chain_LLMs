# RAG-Chain LLMs Project: Document Retrieval System

โปรเจกต์นี้คือระบบ **Retrieval-Augmented Generation (RAG)** ที่ใช้ LangChain และ Google Gemini ในการตอบคำถามโดยอ้างอิงจากไฟล์ PDF (`remotesensing-12-02392-v2.pdf`) 
**ซึ่งเป็นไฟล์ที่ใช้ในโปรเจค Corn-Yield-Prediction-with-Geospatial-AI**

## การติดตั้ง

1.  **สร้างและเปิด Environment:**
    ```bash
    conda create -n rag_env python=3.11
    conda activate rag_env
    ```

2.  **ติดตั้ง Requirements:**
    ```bash
    pip install -r requirements.txt 
    # (จำเป็นต้องมี: langchain-google-genai, chromadb, langchain-classic, pypdf, python-dotenv)
    ```

3.  **ตั้งค่า API Key:**
    สร้างไฟล์ **`.env`** ในโฟลเดอร์หลัก และใส่ API Key:
    ```
    # .env
    GEMINI_API_KEY="YOUR_API_KEY"
    ```
   

## การใช้งาน

โปรเจกต์นี้มี 2 ขั้นตอนหลัก:

### ขั้นตอนที่ 1: การเตรียมข้อมูล (Ingestion)

```bash
python src/ingestion.py
```

### ขั้นตอนที่ 2: ระบบ Chatbot (RAG Chain)

```bash
python python src/rag_chain.py
```
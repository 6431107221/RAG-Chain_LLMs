# Hybrid Document Q&A System (RAG-Chain LLMs)

โปรเจกต์นี้คือระบบ **Retrieval-Augmented Generation (RAG)** ระบบถาม-ตอบอัจฉริยะจากเอกสาร PDF ที่ถูกออกแบบมาให้ทำงานได้ทั้งบน Cloud (Google Gemini) และแบบ Offline 100% (Local LLM) เพื่อความปลอดภัยของข้อมูล โดยอ้างอิงจากไฟล์ PDF (`remotesensing-12-02392-v2.pdf`) 

**ซึ่งเป็นไฟล์ที่ใช้ในโปรเจค Corn-Yield-Prediction-with-Geospatial-AI**

## Key Features

1.  **Privacy-First Architecture:** รองรับการทำงานแบบ Local Inference 100% ข้อมูลไม่หลุดออกนอกเครื่อง

2.  **Hybrid LLM Support:** สลับใช้งานได้ระหว่าง **Google Gemini Pro** (High Performance) และ **Ollama: Llama 3.2** (Local/Secure)

3.  **Efficient Vector Search:** ใช้ **Hugging Face** (all-MiniLM-L6-v2) และ **GoogleGenerativeAIEmbeddings** (models/embedding-001) ในการทำ Embedding และจัดเก็บใน ChromaDB
   
5.  **Smart Document Processing:** ระบบแบ่ง Chunk เอกสารด้วย *RecursiveCharacterTextSplitter* เพื่อรักษาบริบท (Context) ของเนื้อหา
   

## Technical Stack
- Framework: LangChain
- LLMs: Google Gemini API, Ollama (Llama 3.2)
- Embedding Model: Hugging Face (Sentence-Transformers), GoogleGenerativeAIEmbeddings
- Vector Database: ChromaDB
- Language: Python 3.10+

## System Workflow
 - Ingestion Phase: โหลดไฟล์ PDF -> แบ่งข้อความ -> แปลงเป็น Vector ด้วย Hugging Face -> บันทึกลง ChromaDB
 - Retrieval Phase: รับคำถาม -> แปลงคำถามเป็น Vector -> ค้นหา $k$ ชิ้นส่วนที่ใกล้เคียงที่สุดจากฐานข้อมูล
 - Generation Phase: ส่ง Context + คำถามให้ Ollama (Llama 3.2) เพื่อสร้างคำตอบที่แม่นยำ

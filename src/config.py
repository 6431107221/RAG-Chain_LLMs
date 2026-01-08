# config.py

from dotenv import load_dotenv
import os
from pathlib import Path

# -- Gemini Config --
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

GEMINI_EMBBEDING_MODEL = "models/text-embedding-004"
GEMINI_MODEL_NAME = "gemini-1.5-flash"
VECTOR_STORE_PATH = "./chroma_db"

# -- HuggingFace Config --
HF_OLLAMA_MODEL = "llama3.2"
HF_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
HF_VECTOR_STORE_PATH = "./chroma_db_hf"

# Parameters
CHUNK_SIZE = 800 
CHUNK_OVERLAP = 100 
TEMPERATURE = 0.5 # ความสร้างสรรค์ของคำตอบ
K = 8 # จำนวนเอกสารบริบทที่ดึงมาใช้ในการตอบคำถาม

# -- PDF File Path --
BASE_DIR = Path(__file__).resolve().parent.parent
FILE_PATH = BASE_DIR / "data" / "remotesensing-12-02392-v2.pdf"
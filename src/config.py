# config.py

from dotenv import load_dotenv
import os
from pathlib import Path

# -- Gemini Config --
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

GEMINI_MODEL_NAME = "gemini-2.5-flash"
VECTOR_STORE_PATH = "./chroma_db"
CHUNK_SIZE = 1000 
CHUNK_OVERLAP = 200 

# -- PDF File Path --
BASE_DIR = Path(__file__).resolve().parent.parent
FILE_PATH = BASE_DIR / "data" / "remotesensing-12-02392-v2.pdf"
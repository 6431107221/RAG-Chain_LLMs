# rag_chain_hf.py
import os
# ✅ ใช้ HuggingFaceEmbeddings ในการค้นหา
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import config

# ✅ ต้องใช้ Path เดียวกับที่สร้างใน ingestion_hf.py
DB_PATH_HF = "./chroma_db_hf"

def start_chat():
    print("🔄 กำลังเตรียมระบบ Chatbot (HF Embeddings + Gemini LLM)...")

    # --- 1. Setup Models ---
    # ใช้ Embedding ตัวเดียวกับตอน Ingest
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # LLM ยังใช้ Gemini (เพราะเราแค่เลี่ยงโควต้า Embedding)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=config.GEMINI_API_KEY)

    # --- 2. Load Vector Store ---
    if not os.path.exists(DB_PATH_HF):
        print(f"❌ ไม่พบฐานข้อมูลที่ {DB_PATH_HF} กรุณารัน ingestion_hf.py ก่อน")
        return

    vectorstore = Chroma(persist_directory=DB_PATH_HF, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # --- 3. Create Chain ---
    system_prompt = (
        "คุณเป็นผู้ช่วยอัจฉริยะสำหรับการตอบคำถาม "
        "ให้ใช้ข้อมูลบริบทที่ได้รับ (Context) เพื่อตอบคำถามของผู้ใช้ "
        "ถ้าไม่รู้คำตอบ ให้บอกตามตรงว่าไม่ทราบ อย่าพยายามแต่งเรื่องเอง "
        "ตอบให้กระชับและเข้าใจง่าย"
        "\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # --- 4. Chat Loop ---
    print("🤖 ระบบพร้อมใช้งาน! (พิมพ์ 'exit' หรือ 'quit' เพื่อออก)")
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            print("👋 บ๊ายบาย!")
            break
        
        try:
            response = rag_chain.invoke({"input": user_input})
            print(f"Bot: {response['answer']}")
        except Exception as e:
            print(f"⚠️ เกิดข้อผิดพลาด: {e}")

if __name__ == "__main__":
    start_chat()
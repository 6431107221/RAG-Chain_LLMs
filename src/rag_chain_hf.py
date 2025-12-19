# rag_chain_hf.py
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama 
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import config

def start_chat():

    # --- 1. Setup Models ---
    embeddings = HuggingFaceEmbeddings(model_name=config.HF_EMBEDDING_MODEL)
    
    # LLM Ollama
    llm = ChatOllama(
        model=config.HF_OLLAMA_MODEL,
        temperature=config.TEMPERATURE 
    )

    # --- 2. Load Vector Store ---
    if not os.path.exists(config.HF_VECTOR_STORE_PATH):
        print(f"Error Vector Store:  {config.HF_VECTOR_STORE_PATH}")
        return

    vectorstore = Chroma(persist_directory=config.HF_VECTOR_STORE_PATH, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": config.K})

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
    print("ระบบพร้อมใช้งาน (พิมพ์ 'exit' หรือ 'quit' เพื่อออก)")
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit", "q", "end"]:
            print("END QUESTION.")
            break
        
        try: 
            response = rag_chain.invoke({"input": user_input})
            print(f"AI_ANSWER: {response['answer']}")
            print("-" * 50)
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    start_chat()
# rag_chain.py
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
import config
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

def start_chat():
    # --- 1. Setup Models ---
    embeddings = GoogleGenerativeAIEmbeddings(model=config.GEMINI_EMBBEDING_MODEL)
    
    # LLM (Gemini)
    llm = ChatGoogleGenerativeAI(
        model=config.GEMINI_MODEL_NAME,
        temperature=config.TEMPERATURE # ค่าต่ำ = ตอบตามความจริง, ค่าสูง = มีความคิดสร้างสรรค์
    )

    # --- 2. Load Vector Store ---
    vectorstore = Chroma(
        persist_directory=config.VECTOR_STORE_PATH,
        embedding_function=embeddings
    )
    
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": config.K} # ดึงข้อมูลที่เกี่ยวข้องที่สุดมา 5 ชิ้น
    )

    # --- 3. Create Chain (RAG Logic) ---
    # สร้าง Prompt Template: สั่งให้ AI ตอบคำถามโดยใช้ Context ที่หามาได้เท่านั้น
    system_prompt = (
        "คุณเป็นผู้ช่วยที่มีความรู้ความสามารถ "
        "จงตอบคำถามโดยใช้ข้อมูลจาก context ข้างล่างนี้เท่านั้น "
        "ถ้าไม่รู้คำตอบ ให้บอกว่าไม่ทราบ อย่าพยายามแต่งคำตอบขึ้นมาเอง "
        "\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    # Chain:  Retriever + Prompt + LLM 
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    print("ระบบพร้อมใช้งาน! (พิมพ์ 'exit' หรือ 'quit' เพื่อออก)")

    # --- 4. Chat Loop ---
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["exit", "quit", "q", "end"]:
                print("END QUESTION.")
                break
            
            response = rag_chain.invoke({"input": user_input})
            
            print(f"AI_ANSWER: {response['answer']}")
            print("-" * 50)
            
            # (Optional)
            # for i, doc in enumerate(response["context"]):
            #     print(f"Source {i+1}: {doc.page_content[:100]}...")

        except Exception as e:
            print(f" Error: {e}")

if __name__ == "__main__":
    start_chat()
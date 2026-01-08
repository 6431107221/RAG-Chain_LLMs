# rag_chain.py
import os
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
        temperature=config.TEMPERATURE
    )

    # --- 2. Load Vector Store ---
    if not os.path.exists(config.VECTOR_STORE_PATH):
        print(f"Error Vector Store:  {config.VECTOR_STORE_PATH}")
        print("กรุณารัน python src/ingestion.py ก่อนใช้งาน")
        return

    vectorstore = Chroma(
        persist_directory=config.VECTOR_STORE_PATH,
        embedding_function=embeddings
    )
    
    # 2.1 MMR Retriever for diversity
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": config.K, "fetch_k": 20, "lambda_mult": 0.5}
    )

    # --- 3. Create Chain (RAG Logic) ---
    system_prompt = (
        "คุณเป็นผู้ช่วยวิจัยอัจฉริยะที่เคร่งครัดเรื่องความถูกต้องของข้อมูล "
        "1. ตอบคำถามโดยใช้ข้อมูลจาก '[ข้อมูลพื้นฐาน]' และ '[เนื้อหาค้นหา]' เท่านั้น "
        "2. หากมีข้อมูลในหน้าแรก ให้ตอบอย่างมั่นใจและตรงไปตรงมา โดยไม่ต้องใช้คำเกริ่นที่แสดงความไม่แน่ใจ "
        "3. หาก 'ไม่พบข้อมูล' ในเนื้อหาที่ให้มาทั้งสองส่วน ให้ตอบว่า 'ไม่พบข้อมูลในเอกสาร' ห้ามคาดเดาหรือมโนคำตอบเองเด็ดขาด "
        "4. คงคำศัพท์ภาษาอังกฤษสำหรับชื่อเฉพาะและศัพท์เทคนิคไว้เสมอ "
        "\n\n"
        "[ข้อมูลพื้นฐาน]:\n{base_info}\n\n"
        "[เนื้อหาค้นหา]:\n{context}"
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

    # --- 4. Chat Loop ---
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["exit", "quit", "q", "end"]:
                print("END QUESTION.")
                break

            page_1_data = vectorstore.get(where={"page": 0})
            base_info_text = " ".join(page_1_data['documents']) if page_1_data['documents'] else "ไม่พบข้อมูลหน้าแรก"
            if len(base_info_text) > 2000: base_info_text = base_info_text[:2000]
            
            response = rag_chain.invoke({
                "input": user_input,
                "base_info": base_info_text
            })
            
            print(f"AI_ANSWER: {response['answer']}")
            print("-" * 50)
            
            # --- Debug: Show Context ---
            # print("\n[DEBUG] Retrieved Context (from MMR):")
            # for i, doc in enumerate(response["context"]):
                # content = doc.page_content[:200].replace('\n', ' ')
                # page = doc.metadata.get('page', '?')
                # print(f"  Source {i+1} (Page {page}): {content}...")
            print("-" * 50)

        except Exception as e:
            print(f" Error: {e}")

if __name__ == "__main__":
    start_chat()
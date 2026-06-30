# src/chain.py
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from src.retriever import get_hybrid_retriever
from dotenv import load_dotenv

load_dotenv()

def build_qa_chain(chunks, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", k: int = 5, save_path: str = "data/vector/faiss_index"):
    
    # Initialize Gemini LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.7
    )

    # Build hybrid retriever
    retriever = get_hybrid_retriever(chunks, model_name=model_name, k=k, save_path=save_path)

    # Build RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",   # simplest chain type; can swap for "map_reduce" if needed
        return_source_documents=True
    )

    print("[Chain] RetrievalQA chain built with hybrid retriever + Gemini.")
    return qa_chain

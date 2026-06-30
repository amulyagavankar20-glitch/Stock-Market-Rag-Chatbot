# src/retriever.py
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from src.vector_embedding import load_vector_store

def get_hybrid_retriever(chunks, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", k: int = 5, save_path: str = None):
    # Semantic retriever from the saved FAISS index
    vectorstore = load_vector_store(model_name=model_name, save_path=save_path) if save_path else load_vector_store(model_name=model_name)
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    # Keyword retriever (BM25)
    keyword_retriever = BM25Retriever.from_documents(chunks)
    keyword_retriever.k = k

    # Hybrid ensemble retriever
    hybrid_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, keyword_retriever],
        weights=[0.4, 0.6]  # adjust balance between semantic vs keyword
    )

    print(f"[Retriever] Hybrid retriever ready (FAISS + BM25, k={k}).")
    return hybrid_retriever

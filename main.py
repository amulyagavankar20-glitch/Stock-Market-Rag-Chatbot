from src.ingestion import load_documents
from src.chunking import chunk_documents
from src.vector_embedding import build_vector_store, load_vector_store
from src.retriever import get_hybrid_retriever
from src.chain import build_qa_chain

import os

documents = load_documents("data")

chunks = chunk_documents(documents)

vector_store = build_vector_store(
	chunks=chunks,
	model_name="sentence-transformers/all-MiniLM-L6-v2",
	save_path=os.path.join("src", "vector"),
)
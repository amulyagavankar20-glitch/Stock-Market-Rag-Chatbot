from src.ingestion import load_documents
from src.chunking import chunk_documents
from src.vector_embedding import build_vector_store, load_vector_store
from src.retriever import get_hybrid_retriever
from src.chain import build_qa_chain

import os
import pickle

DATA_PATH = "data"
VECTOR_PATH = os.path.join("src", "vector")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNKS_PATH = os.path.join(VECTOR_PATH, "chunks.pkl")

documents = load_documents(DATA_PATH)

chunks = chunk_documents(documents)

vector_store = build_vector_store(
	chunks=chunks,
	model_name=EMBED_MODEL,
	save_path=VECTOR_PATH,
)

# Persist chunks so app.py (or any other consumer) can build the BM25 side
# of the hybrid retriever without re-running ingestion + chunking.
os.makedirs(VECTOR_PATH, exist_ok=True)
with open(CHUNKS_PATH, "wb") as f:
    pickle.dump(chunks, f)

print(f"Indexed {len(chunks)} chunks. FAISS index -> {VECTOR_PATH}. Chunks cache -> {CHUNKS_PATH}")
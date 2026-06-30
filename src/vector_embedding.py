import os

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


DEFAULT_SAVE_PATH = os.path.join("data", "vector", "faiss_index")


def build_vector_store(chunks, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", save_path: str = DEFAULT_SAVE_PATH):
	if not chunks:
		raise ValueError("Cannot build a FAISS index from an empty chunk list.")
	embeddings_model = HuggingFaceEmbeddings(model_name=model_name)

	# Build FAISS index
	vector_store = FAISS.from_documents(chunks, embeddings_model)

	# Save FAISS index locally
	os.makedirs(save_path, exist_ok=True)
	
	vector_store.save_local(save_path)
	print(f"[VectorEmbeddings] FAISS index built and saved at '{save_path}' with {len(chunks)} chunks.")

	return vector_store


def load_vector_store(model_name: str = "sentence-transformers/all-MiniLM-L6-v2", save_path: str = DEFAULT_SAVE_PATH):
	embeddings_model = HuggingFaceEmbeddings(model_name=model_name)
	
	vector_store = FAISS.load_local(save_path, embeddings_model, allow_dangerous_deserialization=True)
	
	print(f"[VectorEmbeddings] FAISS index loaded from '{save_path}'.")
	return vector_store

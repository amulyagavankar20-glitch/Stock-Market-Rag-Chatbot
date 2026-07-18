"""
Streamlit frontend for the Stock Market RAG Chatbot.

Wires into existing backend pipeline:
ingestion -> chunking -> vector_embedding -> retriever -> chain (QA).

Run: streamlit run app.py
"""

import os
import pickle
import traceback

import streamlit as st

from src.vector_embedding import load_vector_store
from src.chain import build_qa_chain

# Configuring the Paths and Models
VECTOR_PATH = os.path.join("src", "vector")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNKS_PATH = os.path.join(VECTOR_PATH, "chunks.pkl")

st.set_page_config(page_title="Stock Market RAG Chatbot", page_icon="📈", layout="centered")


# Backend bootstrap (cached so it runs once per session) 
@st.cache_resource(show_spinner="Loading knowledge base & building chain...")
def init_pipeline():
    """Load a prebuilt FAISS index + cached chunks from disk (both produced by
    running `python main.py` once) and build the QA chain. No ingestion or
    chunking happens here — that's main.py's job. Cached across reruns."""
    if not os.path.exists(CHUNKS_PATH):
        raise RuntimeError(
            f"No prebuilt chunks cache found at '{CHUNKS_PATH}'. "
            f"Run `python main.py` first to ingest, chunk, and index your documents."
        )

    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)

    # Sanity check: FAISS index must already exist on disk (chain.py loads it via save_path).
    try:
        load_vector_store(model_name=EMBED_MODEL, save_path=VECTOR_PATH)
    except Exception as e:
        raise RuntimeError(
            f"FAISS index not found or failed to load from '{VECTOR_PATH}'. "
            f"Run `python main.py` first. Original error: {e}"
        )

    # build_qa_chain takes chunks (not a retriever) — it builds the hybrid retriever internally.
    qa_chain = build_qa_chain(chunks=chunks, model_name=EMBED_MODEL, save_path=VECTOR_PATH)
    return qa_chain


# UI 
st.title("Stock Market RAG Chatbot")
st.caption("Ask questions about your ingested stock market documents.")

# Init session state for chat history 
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": "user"/"assistant", "content": str}

# Load backend pipeline (cached)
try:
    qa_chain = init_pipeline()
    backend_ready = True
except Exception as e:
    backend_ready = False
    st.error(f"Failed to initialize backend pipeline: {e}")
    with st.expander("Show technical details"):
        st.code(traceback.format_exc())

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input area
query = st.chat_input("Type your stock market question here...", disabled=not backend_ready)

if query:
    if not query.strip():
        st.warning("Please enter a non-empty question.")
    else:
        # Show user message
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # Get and show assistant response
        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("Thinking...")
            try:
                result = qa_chain.invoke({"query": query})
                # Handle common chain return shapes
                if isinstance(result, dict):
                    answer = result.get("answer") or result.get("result") or str(result)
                else:
                    answer = str(result)
                placeholder.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                error_msg = f"Sorry, something went wrong while answering: {e}"
                placeholder.error(error_msg)
                with st.expander("Show technical details"):
                    st.code(traceback.format_exc())
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Sidebar utilities
with st.sidebar:
    st.header("Settings")
    if st.button("Clear chat history"):
        st.session_state.messages = []
        st.rerun()
    st.markdown("---")
    st.caption(f"Embedding model: `{EMBED_MODEL}`")
    st.caption(f"Vector index path: `{VECTOR_PATH}`")
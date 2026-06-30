# 📈 Stock Market RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers stock market questions by grounding Google Gemini responses in your own corpus of financial PDF documents. Combines dense (FAISS) and sparse (BM25) retrieval in a hybrid ensemble for more accurate context retrieval, served through an interactive Streamlit interface.

## Features

- **Hybrid Retrieval** — Ensemble of FAISS semantic search (40%) and BM25 keyword search (60%) for retrieval that's robust to both conceptual and exact-term queries (e.g. ticker symbols, financial terms).
- **PDF Ingestion Pipeline** — Bulk-loads and parses an entire directory of PDF reports/filings via `PyPDFDirectoryLoader`.
- **Recursive Chunking** — Splits documents into overlapping 1500-character chunks (400 overlap) to preserve context across boundaries.
- **Local Embeddings** — Uses `sentence-transformers/all-MiniLM-L6-v2` for fast, local, cost-free embedding generation (no API calls for vectorization).
- **Persistent Vector Store** — FAISS index is built once and cached to disk, with automatic reuse on subsequent runs.
- **LLM-Powered Answers** — Gemini 2.5 Flash generates answers grounded in retrieved chunks via LangChain's `RetrievalQA` chain.
- **Interactive Chat UI** — Streamlit front end with persistent chat history, loading states, and graceful error handling.

## Architecture

```
PDF Documents (data/)
        │
        ▼
  Ingestion (PyPDFDirectoryLoader)
        │
        ▼
  Chunking (RecursiveCharacterTextSplitter)
        │
        ▼
  Embedding (HuggingFace MiniLM-L6-v2) ──► FAISS Vector Store (persisted to disk)
        │                                         │
        │                                         ▼
        └──────────────► Hybrid Retriever (FAISS + BM25 Ensemble)
                                                    │
                                                    ▼
                                    RetrievalQA Chain (Gemini 2.5 Flash)
                                                    │
                                                    ▼
                                         Streamlit Chat Interface
```

## Tech Stack

| Layer            | Technology                                         |
|-------------------|-----------------------------------------------------|
| LLM               | Google Gemini 2.5 Flash (`langchain-google-genai`)  |
| Orchestration     | LangChain (`RetrievalQA`, `EnsembleRetriever`)       |
| Embeddings        | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Store      | FAISS                                                |
| Keyword Retrieval | BM25 (`rank-bm25`)                                   |
| Document Loading  | `pypdf`, `PyPDFDirectoryLoader`                      |
| Frontend          | Streamlit                                            |
| Language          | Python 3.10+                                         |

## Project Structure

```
.
├── app.py                  # Streamlit chat frontend
├── main.py                 # CLI entry point — builds the vector index from data/
├── run_app.ps1              # Windows PowerShell launch script
├── requirements.txt
├── .env                     # GOOGLE_API_KEY (not committed)
├── .gitignore
├── data/                   # Source PDF documents (stock reports, filings, etc.)
│   ├── SM Booklet_English - ...
│   └── SM Investing-101-...
└── src/
    ├── chunking.py          # Document splitting
    ├── ingestion.py         # PDF loading
    ├── retriever.py         # Hybrid FAISS + BM25 retriever
    ├── chain.py              # RetrievalQA chain (Gemini)
    ├── vector_embedding.py  # FAISS index build/load
    ├── vector/               # Persisted FAISS index
    └── prompt/
        └── prompt.py          # Prompt templating & chat history utilities
```

## Setup & Usage

### 1. Clone & install dependencies
```bash
git clone <repo-url>
cd stock-market-rag-chatbot
pip install -r requirements.txt
```

### 2. Configure environment
Create a `.env` file with your Gemini API key:
```
GOOGLE_API_KEY=your_api_key_here
```

### 3. Add source documents
Place stock market PDFs (annual reports, filings, research notes, etc.) into the `data/` directory.

### 4. Build the vector index
```bash
python main.py
```

### 5. Launch the chatbot
```bash
streamlit run app.py
```
Windows users can alternatively run `.\run_app.ps1`.

## Example Queries
- "What were the key risk factors mentioned in the latest filing?"
- "Summarize the revenue growth trend over the past quarters."
- "What is the company's debt-to-equity ratio?"

## Future Improvements
- [ ] Add conversational memory to the QA chain (chat-history-aware retrieval)
- [ ] Source citation display alongside answers
- [ ] Support for additional document types (CSV/Excel financial statements)
- [ ] Dockerize for one-command deployment
- [ ] Add evaluation harness (retrieval precision/recall, answer faithfulness)

## License
MIT

import os
from langchain_community.document_loaders import PyPDFDirectoryLoader

def load_documents(data_path: str = "data"):
    pdf_dir = os.path.abspath(data_path)
    
    loader = PyPDFDirectoryLoader(pdf_dir)
    
    documents = loader.load()
    
    print(f"[Ingestion] Loaded {len(documents)} documents from {pdf_dir}")
    return documents
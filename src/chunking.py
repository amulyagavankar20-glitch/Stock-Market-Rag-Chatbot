from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_documents(documents, chunk_size: int = 1500, chunk_overlap: int = 400):
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False
    )

    chunks = text_splitter.split_documents(documents)
    
    print(f"[Chunking] Split {len(documents)} docs into {len(chunks)} chunks.")
    return chunks

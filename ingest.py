import os
from typing import List

from langchain_community.document_loaders import (TextLoader, PyPDFLoader, Docx2txtLoader)
from langchain_text_splitters import RecursiveCharacterTextSplitter

from vector_store import create_empty_vector_store

def load_text_documents(folder_path: str) -> List:
    """
    Walk through a folder and load all .txt and .md files as LangChain documents.
    """
    docs = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)

            if file.endswith(".txt") or file.endswith(".md"):
                loader = TextLoader(file_path, encoding = "utf-8")
            
            elif file.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            
            elif file.endswith(".docx"):
                loader = Docx2txtLoader(file_path)
            
            else:
                continue # skip unsupported formats

            file_docs = loader.load()
            docs.extend(file_docs)
            print(f"Loaded {len(file_docs)} docs from: {file_path}")
    
    return docs



def split_documents(documents, chunk_size: int = 800, chunk_overlap: int = 200):
    """
    Split documents into overlaping chunks that are easier to embed and search
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size,
                                              chunk_overlap = chunk_overlap)
    
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")
    return chunks

def ingest_documents(docs_folder: str = "docs", persist_directory: str = "chroma_db",):
    """
    Main ingestion pipeline:
    1. Load raw documents from a floder
    2. Split them into chunks
    3. Embed and store chunks in Chroma vector store
    """
    # 1. Load
    print(f"Loading documents from:{docs_folder}")
    documents = load_text_documents(docs_folder)
    if not documents:
        print("No documents found. Make sure you have .txt or .md fiels in the folder")
        return
    
    # 2. Split
    chunks = split_documents(documents)

    # 3. Create(or load) vector store and add chunks
    print("Creating/loading vector store...")
    vector_store = create_empty_vector_store(persist_directory = persist_directory)

    texts = [doc.page_content for doc in chunks]
    metadatas = [doc.metadata for doc in chunks]

    print("Adding chunks to vector store...")
    vector_store.add_texts(texts = texts, metadatas = metadatas)
    #vector_store.persist() # In newer Chroma/langchain_chroma versions, persistence is automatic
                            # when you pass `persist_directory`, so no need to call persist() explicitly.

    print(f"Ingestion complete. Stored {len(texts)} chunks in '{persist_directory}'.")


if __name__ == "__main__":
    ingest_documents()








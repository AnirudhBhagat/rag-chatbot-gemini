from typing import List
from langchain_core.documents import Document
from vector_store import create_empty_vector_store
from llm import ask_llm

def retrieve_relevent_chunks(query: str, k: int = 4) -> List[Document]:
    """
    Given a user query, search the vector store for the top-k most relevent chunks.
    Returns a list of Langchain Document objects.
    """

    # Load (or create) vector store pointing at the same persist_directory
    vector_store = create_empty_vector_store(persist_directory = "chroma_db")

    # Run similarity search
    docs = vector_store.similarity_search(query, k=k)
    return docs

def build_rag_prompt(question: str, docs: List[Document]) -> str:
    """
    Build a prompt that includes retrieved context + the user question.
    The LLM will be instructed to stick to this context.
    """

    # Join all document chunks into a single context string
    context_blocks = []
    for i, doc in enumerate(docs, start = 1):
        source = doc.metadata.get("source", "unknown")
        block = f"Source {i} (from {source}):\n{doc.page_content}"
        context_blocks.append(block)

    context_text = "\n\n".join(context_blocks)

    prompt = f""" 
You are a helpful assistant that answers questions based ONLY on the provided context.build_rag_prompt.build_rag_prompt.
If the answer is not contained in the context, say you don't know.

Context:
{context_text}

Question: {question}

Answer in a clear, concise paragraph.
"""
    return prompt.strip()

def answer_question_with_rag(question: str, k: int = 4) -> tuple[str, List[Document]]:
    """
    Main RAG function:
    1. Retrieve relevant chunks from vector store.
    2. Build a context-augmented prompt.
    3. Ask the LLM using that prompt.
    4. Return both the answer and the docs used.
    """

    # 1. Retrieve
    docs = retrieve_relevent_chunks(question, k=k)

    # 2. build prompt with context
    prompt = build_rag_prompt(question, docs)

    # 3. Ask LLM (using your helper from llm.py)
    answer = ask_llm(prompt)

    return answer, docs

if __name__ == "__main__":
    # Quick manual test
    user_question = "What is Retrieval-Augmented Generation and why is it useful?"
    answer, used_docs = answer_question_with_rag(user_question, k=4)

    print("=== QUESTION ===")
    print(user_question)
    print("\n=== ANSWER ===")
    print(answer)
    print("\n=== SOURCES USED ===")
    for i, doc in enumerate(used_docs, start = 1):
        print(f"\nSource {i}: {doc.metadata.get('source', 'unkown')}")
        print(doc.page_content[:200] + "...")


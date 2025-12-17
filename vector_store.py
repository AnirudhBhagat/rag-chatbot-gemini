import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
#from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
#from langchain.embeddings import HuggingFaceEmbeddings
#from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings


load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY environment variables not set.")

# Create reusable embeddings client
# embeddings = GoogleGenerativeAIEmbeddings(google_api_key=gemini_api_key,
#                                           model = "models/embedding-001")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def create_empty_vector_store(persist_directory: str = "chroma_db") -> Chroma:
    """ 
    Create (or load) a Chroma vector store that uses our Google embeddings.
    For now, it's empty until we add documents in the ingestion step.
    """
    vector_store = Chroma(embedding_function=embeddings,
                          persist_directory=persist_directory)
    return vector_store

if __name__ == "__main__":
    # Quick sanity check: embed a sample text and initialize chroma
    vs = create_empty_vector_store()
    sample_text = ["Will Arsenal win the Premier League this year?","Only if they are lucky with injuries."]

    # Add some dummy docs just to make sure everything is wired up correctly
    vs.add_texts(sample_text)
    #vs.persist()

    print("Vector store created and sample texts added.")


import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Read the api key from the .env file
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables.")


#Create reusable LLM client
llm = ChatGoogleGenerativeAI(google_api_key=gemini_api_key,
                             model = "gemini-2.5-flash",
                             temperature=0.2,
                             )

def ask_llm(question: str) -> str:
    """ 
    Simple helper to send questions to Gemini and return a plain text answer.
    Streamlit and the RAG pipeline will use this under the hood.
    """
    response = llm.invoke(question)     # LandChain AIMessage
    return response.content             # Actual text from the model

if __name__ == "__main__":
    answer = ask_llm("How does LLM work? Answer in 2 sentences.")
    print("LLM response:\n",answer)
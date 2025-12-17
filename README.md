📚 RAG Chatbot with Gemini, Chroma & Streamlit

A Retrieval-Augmented Generation (RAG) chatbot that allows users to upload documents and ask questions grounded in their own knowledge base.

Built with:

Google Gemini (LLM)

HuggingFace embeddings

ChromaDB (local vector store)

Streamlit frontend

This project demonstrates an end-to-end RAG pipeline, from document ingestion to a production-style UI.

🚀 Features

Upload documents (.txt, .md, .pdf, .docx)

Automatic document ingestion & chunking

Local vector search using embeddings

Context-grounded LLM responses (RAG)

Source attribution for answers

Chat-style interface with conversation history

Clear conversation button

Fully local & free-ish setup

🧠 Architecture Overview
User Question
     ↓
Embedding (HuggingFace)
     ↓
Chroma Vector Search
     ↓
Relevant Chunks
     ↓
Prompt + Context
     ↓
Gemini LLM
     ↓
Answer + Sources

🗂️ Project Structure
rag-chatbot-gemini/
├── app.py                # Streamlit UI
├── llm_client.py         # Gemini LLM client
├── vector_store.py       # Chroma + embeddings
├── ingest.py             # Document ingestion
├── rag_pipeline.py       # Retrieval + prompt logic
├── docs/                 # Uploaded documents
├── chroma_db/            # Vector DB (auto-created)
├── requirements.txt
└── README.md

⚙️ Setup Instructions
1️⃣ Clone the repo
git clone https://github.com/YOUR_USERNAME/rag-chatbot-gemini.git
cd rag-chatbot-gemini

2️⃣ Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # macOS / Linux
venv\Scripts\activate     # Windows

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Set up environment variables
cp .env.example .env


Edit .env and add your Gemini API key:

GEMINI_API_KEY=your_api_key_here

📥 Ingest Documents

Place documents inside the docs/ folder or upload them via the UI.

Supported formats:

.txt

.md

.pdf

.docx

To ingest manually:

python ingest.py

💬 Run the Chatbot
streamlit run app.py


Then open:

http://localhost:8501

📌 Example Questions

What is Retrieval-Augmented Generation?

Summarize the uploaded document

What topics are covered in the PDF I uploaded?

Each answer includes source snippets used by the model.

🔒 Notes on Privacy & Cost

Embeddings are computed locally

Vector database runs locally

Only LLM calls go to Gemini

No paid services required (within free API limits)
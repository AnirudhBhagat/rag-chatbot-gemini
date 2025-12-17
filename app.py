import os
import streamlit as st
from rag_pipeline import answer_question_with_rag
from ingest import ingest_documents

st.set_page_config(page_title = "RAG Chatbot", page_icon="🤖")

st.title("🤖 Chatbot with Gemini")
st.write(
    "Ask a question based on the documents you have ingested."
    " The model will answer using the stored knowledge base"
)

# Initialize chat history in session_state
if "history" not in st.session_state:
    st.session_state["history"] = [] # List of ("question", "answer", "docs")

# Side bar controls (optional tweaks)
st.sidebar.header("Settings")
top_k = st.sidebar.slider("Number of chunks to retrieve (k)", min_value=1, max_value=10, value=4)

# Clear conversation button
if st.sidebar.button("Clear Conversation"):
    st.session_state["history"] = []
    st.sidebar.success("Conversation cleared.")
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.header("Knowledge base")

uploaded_files = st.sidebar.file_uploader(
    "Upload documents to add to the knowledge base:",
    type = ["txt","md","pdf","docx"],
    accept_multiple_files = True,
)

if st.sidebar.button("Ingest uploaded documents"):
    if not uploaded_files:
        st.sidebar.warning("Please upload at least one file first.")
    else:
        with st.spinner("Saving files and updating vector store ..."):
            # Ensure docs folder exists
            os.makedirs("docs", exist_ok = True)

            # Save uploaded files to the docs/folder
            for uploaded_file in uploaded_files:
                save_path = os.path.join("docs", uploaded_file.name)
                with open(save_path,"wb") as f:
                    f.write(uploaded_file.getvalue())
            
            # Re-run ingestion over all docs in the docs/folder
            ingest_documents(docs_folder = "docs", persist_directory = "chroma_db")
        st.sidebar.success("Documents ingested into the knowledge base.")

# --- Display chat history ---
if st.session_state["history"]:
    st.subheader("Conversation")
    for turn_idx, turn in enumerate(st.session_state["history"], start=1):
        st.markdown(f"**You:** {turn['question']}")
        st.markdown(f"**Bot:** {turn['answer']}")

        # Show sources for each turn in an expander
        docs = turn.get("docs", [])
        if docs:
            with st.expander(f"Sources for this answer (turn{turn_idx})"):
                for i, doc in enumerate(docs, start=1):
                    source = doc.metadata.get("source", "unknown")
                    st.markdown(f"**Source {i}:** '{source}'")
                    st.write(doc.page_content[:300] + "...")
        else:
            st.info("No revlevent sources were found in the vector store. Try re-ingesting documents or adjusting your question.")
        st.markdown("---") # Separator between turns


# --- Input for next question ---
st.subheader("Ask a new question")
question = st.text_input(
    "Your question:",
    placeholder="e.g. How does Retrieval-Augmented Generation work?",
    key = "current_question",
)

if st.button("Get Answer"):
    if not question.strip():
        st.warning("Please enter a question first.")
    else:
        with st.spinner("Thinking..."):
            answer, docs = answer_question_with_rag(question, k=top_k)

        # Save this turn into the history
        st.session_state["history"].append(
            {
                "question": question,
                "answer": answer,
                "docs": docs,
            }
        )

        # Rerun so the new turn appears in the history section above
        st.rerun()




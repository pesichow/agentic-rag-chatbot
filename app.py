# app.py
import streamlit as st
import os
import uuid
import tempfile
from typing import List, Dict, Any
from agents import IngestionAgent, RetrievalAgent, LLMResponseAgent, CoordinatorAgent

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processed_docs" not in st.session_state:
    st.session_state.processed_docs = False
if "agents" not in st.session_state:
    # Initialize agents
    st.session_state.ingestion_agent = IngestionAgent()
    st.session_state.retrieval_agent = RetrievalAgent()
    st.session_state.llm_agent = LLMResponseAgent()
    st.session_state.coordinator = CoordinatorAgent(
        st.session_state.ingestion_agent,
        st.session_state.retrieval_agent,
        st.session_state.llm_agent
    )

# Set page config
st.set_page_config(
    page_title="Azure OpenAI RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .chat-container {
        background-color: #f9f9f9;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .user-message {
        background-color: #e6f7ff;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .assistant-message {
        background-color: #f0f0f0;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar for document upload
with st.sidebar:
    st.title("Document Upload")
    uploaded_files = st.file_uploader(
        "Choose documents",
        type=['pdf', 'pptx', 'csv', 'docx', 'txt', 'md'],
        accept_multiple_files=True
    )
    
    if uploaded_files and st.button("Process Documents"):
        with st.spinner("Processing documents..."):
            # Save files temporarily
            temp_files = []
            for uploaded_file in uploaded_files:
                file_ext = os.path.splitext(uploaded_file.name)[1].lower()
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    temp_files.append(tmp_file.name)
            
            # Process documents through coordinator
            success = st.session_state.coordinator.process_documents(temp_files)
            
            # Clean up temp files
            for tmp_file in temp_files:
                os.unlink(tmp_file)
            
            if success:
                st.session_state.processed_docs = True
                st.success("Documents processed successfully!")
            else:
                st.error("Failed to process documents")

# Main chat interface
st.markdown('<div class="main-header">🤖 Azure OpenAI RAG Chatbot</div>', unsafe_allow_html=True)
st.markdown("Ask questions about your uploaded documents")

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("View sources"):
                for source in message["sources"]:
                    st.write(f"📄 {source}")

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Process query
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if not st.session_state.processed_docs:
                # Handle general queries without documents
                response, sources = st.session_state.llm_agent.handle_general_query(prompt)
            else:
                # Process query through coordinator
                response, sources = st.session_state.coordinator.process_query(prompt)
            
            # Display response
            st.markdown(response)
            
            # Display sources if available
            if sources:
                with st.expander("View sources"):
                    for source in sources:
                        st.write(f"📄 {source}")
    
    # Add assistant response to chat history
    st.session_state.chat_history.append({
        "role": "assistant", 
        "content": response,
        "sources": sources
    })
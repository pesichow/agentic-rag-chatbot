## Architecture
agentic-rag-chatbot/
│
├── app.py                      # Main Streamlit application
├── agents.py                   # Agent classes with MCP implementation
├── document_parser.py          # Document parsing for all formats
├── vector_store.py             # Vector store with Azure OpenAI embeddings
├── llm_integration.py          # Azure OpenAI integration
├── config.py                   # Configuration and credentials
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation


Architecture Diagram Description
python
# MCP_RAG_Architecture.py
"""
Agentic RAG Chatbot Architecture with Model Context Protocol (MCP)

Components:
1. User Interface Layer
2. MCP Client & Coordinator
3. Specialized Agents (MCP Servers)
4. Data Storage Layer
5. External Services
"""

# Architecture Visualization (Text-based)
architecture = """
┌─────────────────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE LAYER                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────┐          ┌─────────────────┐                         │
│   │   Streamlit     │          │   Chat Interface │                        │
│   │     Web UI      │◄─────────►                 │                        │
│   └─────────────────┘          └─────────────────┘                         │
│                                                                             │
└───────────────────────────────▲─────────────────────────────────────────────┘
                                │
                                │ HTTP Requests/Responses
                                │
┌───────────────────────────────▼─────────────────────────────────────────────┐
│                     MCP CLIENT & COORDINATOR                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────┐          ┌─────────────────┐                         │
│   │   Coordinator   │          │   MCP Client    │                         │
│   │     Agent       │◄─────────►                 │                        │
│   └─────────────────┘          └─────────────────┘                         │
│                                                                             │
└───────────────────────────────▲─────────────────────────────────────────────┘
                                │
                                │ MCP Messages (JSON)
                                │
┌───────────────────────────────▼─────────────────────────────────────────────┐
│                     SPECIALIZED AGENTS (MCP SERVERS)                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐            │
│   │  Ingestion      │  │  Retrieval      │  │  LLMResponse    │            │
│   │    Agent        │  │    Agent        │  │    Agent        │            │
│   │ (MCP Server 1)  │  │ (MCP Server 2)  │  │ (MCP Server 3)  │            │
│   └─────────────────┘  └─────────────────┘  └─────────────────┘            │
│                                                                             │
└───────────▲─────────────────────▲─────────────────────▲─────────────────────┘
            │                     │                     │
            │                     │                     │
            │                     │                     │
┌───────────▼─────────────────────▼─────────────────────▼─────────────────────┐
│                         DATA STORAGE LAYER                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────┐          ┌─────────────────┐                         │
│   │  Document       │          │   Vector        │                         │
│   │   Parser        │          │   Store         │                         │
│   │  (Multi-format) │          │  (FAISS)        │                         │
│   └─────────────────┘          └─────────────────┘                         │
│                                                                             │
└───────────────────────────────▲─────────────────────────────────────────────┘
                                │
                                │ API Calls
                                │
┌───────────────────────────────▼─────────────────────────────────────────────┐
│                         EXTERNAL SERVICES                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────┐                                                       │
│   │   Azure OpenAI  │                                                       │
│   │     API         │                                                       │
│   └─────────────────┘                                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
"""



# Agentic RAG Chatbot with MCP

A Retrieval-Augmented Generation (RAG) chatbot that answers questions from uploaded documents using an agent-based architecture with Model Context Protocol (MCP).

## Features

- Support for multiple document formats (PDF, PPTX, CSV, DOCX, TXT, MD)
- Agent-based architecture with three specialized agents
- Model Context Protocol (MCP) for inter-agent communication
- Azure OpenAI integration for embeddings and responses
- Streamlit-based web interface
- Source citation and context preservation

## Architecture

The system follows an agent-based architecture with three main agents:

1. **IngestionAgent**: Parses and preprocesses uploaded documents
2. **RetrievalAgent**: Handles embeddings and semantic search
3. **LLMResponseAgent**: Generates responses using retrieved context

All agents communicate using the Model Context Protocol (MCP) with structured message formats.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/agentic-rag-chatbot.git
cd agentic-rag-chatbot
Create a virtual environment:

bash
python -m venv rag-env
source rag-env/bin/activate  # On Windows: rag-env\Scripts\activate
Install dependencies:

bash
pip install -r requirements.txt
Configure Azure OpenAI credentials in config.py:

python
AZURE_OPENAI_API_KEY = "your-api-key"
AZURE_OPENAI_ENDPOINT = "your-endpoint"
AZURE_OPENAI_DEPLOYMENT = "your-deployment"
AZURE_OPENAI_API_VERSION = "2024-12-01-preview"
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = "text-embedding-ada-002"
Usage
Run the application:
## bash
streamlit run app.py
Open your browser and navigate to the provided URL (typically http://localhost:8501)
Upload documents using the sidebar interface
Ask questions about your documents in the chat interface

Project Structure
app.py: Main Streamlit application
agents.py: Agent classes and MCP implementation
document_parser.py: Document parsing utilities
vector_store.py: Vector storage and retrieval
llm_integration.py: Azure OpenAI integration
config.py: Configuration settings

## Technologies Used
Python 3.8+
Streamlit for UI
Azure OpenAI API
NumPy for vector operations
Various document parsing libraries (PyPDF2, python-pptx, python-docx, pandas)
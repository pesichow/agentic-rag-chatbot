# agents.py
import os
import uuid
import json
from typing import List, Dict, Any, Tuple
from document_parser import DocumentParser
from vector_store import VectorStore
from llm_integration import AzureOpenAILLM

class MCPMessage:
    """Model Context Protocol message class"""
    def __init__(self, sender: str, receiver: str, msg_type: str, payload: Dict[str, Any]):
        self.sender = sender
        self.receiver = receiver
        self.type = msg_type
        self.trace_id = str(uuid.uuid4())
        self.payload = payload
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "sender": self.sender,
            "receiver": self.receiver,
            "type": self.type,
            "trace_id": self.trace_id,
            "payload": self.payload
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        message = cls(
            data["sender"],
            data["receiver"],
            data["type"],
            data["payload"]
        )
        message.trace_id = data.get("trace_id", str(uuid.uuid4()))
        return message

class IngestionAgent:
    def __init__(self):
        self.parser = DocumentParser()
    
    def process_message(self, message: MCPMessage) -> MCPMessage:
        if message.type == "PARSE_DOCUMENT":
            file_path = message.payload["file_path"]
            file_type = message.payload["file_type"]
            
            try:
                content = self.parser.parse_document(file_path, file_type)
                return MCPMessage(
                    "IngestionAgent",
                    message.sender,
                    "PARSE_RESPONSE",
                    {"content": content, "success": True}
                )
            except Exception as e:
                return MCPMessage(
                    "IngestionAgent",
                    message.sender,
                    "PARSE_ERROR",
                    {"error": str(e), "success": False}
                )
        
        return MCPMessage(
            "IngestionAgent",
            message.sender,
            "ERROR",
            {"error": "Unknown message type"}
        )

class RetrievalAgent:
    def __init__(self):
        self.vector_store = VectorStore()
    
    def process_message(self, message: MCPMessage) -> MCPMessage:
        if message.type == "STORE_DOCUMENT":
            chunks = message.payload["chunks"]
            self.vector_store.add_documents(chunks)
            return MCPMessage(
                "RetrievalAgent",
                message.sender,
                "STORE_RESPONSE",
                {"success": True}
            )
        
        elif message.type == "RETRIEVE_CONTEXT":
            query = message.payload["query"]
            top_k = message.payload.get("top_k", 3)
            
            results = self.vector_store.search(query, top_k)
            return MCPMessage(
                "RetrievalAgent",
                message.sender,
                "RETRIEVAL_RESULT",
                {
                    "retrieved_context": [result["content"] for result in results],
                    "sources": [result["metadata"].get("source", "Unknown") for result in results],
                    "query": query
                }
            )
        
        return MCPMessage(
            "RetrievalAgent",
            message.sender,
            "ERROR",
            {"error": "Unknown message type"}
        )

class LLMResponseAgent:
    def __init__(self):
        self.llm = AzureOpenAILLM()
    
    def process_message(self, message: MCPMessage) -> MCPMessage:
        if message.type == "GENERATE_RESPONSE":
            context = message.payload["retrieved_context"]
            query = message.payload["query"]
            sources = message.payload.get("sources", [])
            
            response = self.llm.generate_response(query, context)
            return MCPMessage(
                "LLMResponseAgent",
                message.sender,
                "LLM_RESPONSE",
                {
                    "answer": response,
                    "sources": sources,
                    "query": query
                }
            )
        
        return MCPMessage(
            "LLMResponseAgent",
            message.sender,
            "ERROR",
            {"error": "Unknown message type"}
        )
    
    def handle_general_query(self, query: str) -> Tuple[str, List[str]]:
        """Handle general queries when no documents are processed"""
        response = self.llm.generate_response(query, [])
        return response, []

class CoordinatorAgent:
    def __init__(self, ingestion_agent: IngestionAgent, retrieval_agent: RetrievalAgent, llm_agent: LLMResponseAgent):
        self.ingestion_agent = ingestion_agent
        self.retrieval_agent = retrieval_agent
        self.llm_agent = llm_agent
        self.documents_processed = False
    
    def process_documents(self, file_paths: List[str]) -> bool:
        all_chunks = []
        
        for file_path in file_paths:
            file_type = os.path.splitext(file_path)[1].lower().lstrip('.')
            file_name = os.path.basename(file_path)
            
            # Parse document
            parse_message = MCPMessage(
                "CoordinatorAgent",
                "IngestionAgent",
                "PARSE_DOCUMENT",
                {"file_path": file_path, "file_type": file_type, "file_name": file_name}
            )
            
            response = self.ingestion_agent.process_message(parse_message)
            
            if response.type == "PARSE_RESPONSE" and response.payload["success"]:
                content = response.payload["content"]
                # Add metadata to chunks
                for chunk in content:
                    chunk["metadata"] = {"source": file_name}
                all_chunks.extend(content)
            else:
                return False
        
        # Store documents in vector store
        store_message = MCPMessage(
            "CoordinatorAgent",
            "RetrievalAgent",
            "STORE_DOCUMENT",
            {"chunks": all_chunks}
        )
        
        response = self.retrieval_agent.process_message(store_message)
        
        if response.type == "STORE_RESPONSE" and response.payload["success"]:
            self.documents_processed = True
            return True
        
        return False
    
    def process_query(self, query: str) -> Tuple[str, List[str]]:
        # Retrieve relevant context
        retrieve_message = MCPMessage(
            "CoordinatorAgent",
            "RetrievalAgent",
            "RETRIEVE_CONTEXT",
            {"query": query, "top_k": 5}
        )
        
        retrieval_response = self.retrieval_agent.process_message(retrieve_message)
        
        if retrieval_response.type != "RETRIEVAL_RESULT":
            return "Sorry, I couldn't retrieve relevant information.", []
        
        # Generate response using LLM
        generate_message = MCPMessage(
            "CoordinatorAgent",
            "LLMResponseAgent",
            "GENERATE_RESPONSE",
            {
                "retrieved_context": retrieval_response.payload["retrieved_context"],
                "sources": retrieval_response.payload["sources"],
                "query": query
            }
        )
        
        llm_response = self.llm_agent.process_message(generate_message)
        
        if llm_response.type == "LLM_RESPONSE":
            return llm_response.payload["answer"], llm_response.payload["sources"]
        
        return "Sorry, I encountered an error while generating a response.", []
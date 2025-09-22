# vector_store.py
import numpy as np
from typing import List, Dict, Any
import pickle
import os
from llm_integration import AzureOpenAILLM

class VectorStore:
    def __init__(self):
        self.llm = AzureOpenAILLM()
        self.embeddings = None
        self.documents = []
        self.storage_file = "vector_store.pkl"
        
        # Try to load existing data if available
        self.load()
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        if not documents:
            return
        
        # Extract content for embedding
        contents = [doc["content"] for doc in documents]
        
        # Get embeddings for all contents
        new_embeddings = self.llm.get_embeddings(contents)
        new_embeddings = np.array(new_embeddings)
        
        # Ensure consistent dimensions
        if self.embeddings is not None and self.embeddings.shape[1] != new_embeddings.shape[1]:
            print(f"Warning: Dimension mismatch. Resizing from {new_embeddings.shape[1]} to {self.embeddings.shape[1]}")
            new_embeddings = self._resize_embeddings(new_embeddings, self.embeddings.shape[1])
        
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
        
        self.documents.extend(documents)
        
        # Save the updated store
        self.save()
    
    def _resize_embeddings(self, embeddings: np.ndarray, target_dim: int) -> np.ndarray:
        """Resize embeddings to target dimension"""
        current_dim = embeddings.shape[1]
        if current_dim > target_dim:
            # Truncate
            return embeddings[:, :target_dim]
        else:
            # Pad with zeros
            padding = np.zeros((embeddings.shape[0], target_dim - current_dim))
            return np.hstack([embeddings, padding])
    
    def cosine_similarity(self, a, b):
        """Calculate cosine similarity between two vectors"""
        # Ensure both vectors have the same dimension
        if len(a) != len(b):
            min_dim = min(len(a), len(b))
            a = a[:min_dim]
            b = b[:min_dim]
        
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        # Avoid division by zero
        if norm_a == 0 or norm_b == 0:
            return 0.0
            
        return dot_product / (norm_a * norm_b)
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if self.embeddings is None or len(self.documents) == 0:
            return []
        
        # Encode query
        query_embedding_list = self.llm.get_embeddings([query])
        query_embedding = np.array(query_embedding_list[0])
        
        # Ensure query embedding matches document embeddings dimension
        if self.embeddings is not None and len(query_embedding) != self.embeddings.shape[1]:
            query_embedding = self._resize_embeddings(
                query_embedding.reshape(1, -1), 
                self.embeddings.shape[1]
            )[0]
        
        # Calculate cosine similarity manually
        similarities = []
        for emb in self.embeddings:
            similarity = self.cosine_similarity(query_embedding, emb)
            similarities.append(similarity)
        
        # Get top_k indices
        similarities = np.array(similarities)
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Return results
        results = []
        for idx in top_indices:
            if idx < len(self.documents):
                results.append(self.documents[idx])
        
        return results
    
    def save(self):
        """Save the vector store to disk"""
        data = {
            'embeddings': self.embeddings,
            'documents': self.documents
        }
        with open(self.storage_file, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self):
        """Load the vector store from disk if it exists"""
        if os.path.exists(self.storage_file):
            try:
                with open(self.storage_file, 'rb') as f:
                    data = pickle.load(f)
                    self.embeddings = data['embeddings']
                    self.documents = data['documents']
            except:
                # If loading fails, start fresh
                self.embeddings = None
                self.documents = []
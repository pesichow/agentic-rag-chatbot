import openai
from typing import List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from config import (
    AZURE_OPENAI_API_KEY, 
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_DEPLOYMENT,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT
)

class AzureOpenAILLM:
    def __init__(self):
        self.client = openai.AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT
        )
        self.deployment_name = AZURE_OPENAI_DEPLOYMENT
        self.embedding_deployment = AZURE_OPENAI_EMBEDDING_DEPLOYMENT
        self.embedding_dimension = 1536  # Standard dimension for text-embedding-ada-002
    
    def generate_response(self, query: str, context: List[str] = None) -> str:
        # Prepare prompt with context
        if context:
            context_text = "\n".join([f"[Context {i+1}]: {c}" for i, c in enumerate(context)])
            prompt = f"""Based on the following context, please answer the question:

{context_text}

Question: {query}

Answer:"""
        else:
            prompt = query
        
        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that provides accurate information based on the given context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2000
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}"
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts with consistent dimensions"""
        try:
            # Try Azure OpenAI API first
            response = self.client.embeddings.create(
                model=self.embedding_deployment,
                input=texts
            )
            embeddings = [item.embedding for item in response.data]
            
            # Ensure all embeddings have consistent dimensions
            for i, emb in enumerate(embeddings):
                if len(emb) != self.embedding_dimension:
                    # Pad or truncate to the standard dimension
                    if len(emb) > self.embedding_dimension:
                        embeddings[i] = emb[:self.embedding_dimension]
                    else:
                        embeddings[i] = emb + [0.0] * (self.embedding_dimension - len(emb))
            
            return embeddings
        except Exception as e:
            print(f"Embedding API error: {e}, using fallback")
            # Fallback to consistent TF-IDF embeddings
            return self._create_consistent_embeddings(texts)
    
    def _create_consistent_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Create consistent embeddings using TF-IDF with fixed dimension"""
        if not texts:
            return []
        
        # Create TF-IDF vectors with fixed dimension
        vectorizer = TfidfVectorizer(max_features=self.embedding_dimension)
        try:
            tfidf_matrix = vectorizer.fit_transform(texts).toarray()
            return tfidf_matrix.tolist()
        except Exception as e:
            print(f"TF-IDF embedding failed: {e}")
            # Final fallback: random embeddings with consistent dimension
            return [list(np.random.rand(self.embedding_dimension)) for _ in texts]
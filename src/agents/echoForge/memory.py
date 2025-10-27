"""
EchoForge Memory Management with RAG
"""
from typing import Dict, List, Any
import json
import os
from datetime import datetime
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import numpy as np

class EchoForgeMemory:
    """Memory management for EchoForge agent"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        
        # File paths for user profile and documents
        self.user_profile_file = os.path.join(data_dir, "shared", "user_profile.json")
        self.echoForge_documents_file = os.path.join(data_dir, "echoForge", "echoForge_documents.json")
        
        # Initialize embeddings model
        # Using text-embedding-3-small: 1536 dims, good quality/cost balance
        # Alternative: text-embedding-3-large (3072 dims, better quality, 6.5x more expensive)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        # Initialize user profile and vector store
        self.user_profile = self._load_user_profile()
        self.vector_store = self._build_vector_store()

        # Initialize session memory
        self.memory_saver = MemorySaver()
        
        # Store current thread config (created on demand)
        self.current_config = None
    
    def create_or_get_config(self) -> Dict[str, Any]:
        """Create or get the current thread config"""
        import uuid
        if self.current_config is None:
            self.current_config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        return self.current_config
    
    def get_config(self) -> Dict[str, Any]:
        """Get the current thread config"""
        return self.current_config
        
    def _create_empty_profile(self) -> Dict[str, Any]:
        """Create empty user profile"""
        return {
            "communication_style": "Authentic and natural - avoids AI-like language",
            "tone": "Balanced and adaptive",
            "personality": "Genuine and engaging",
            "approach": "Values meaningful connection"
        }
    
    def _load_user_profile(self) -> Dict[str, Any]:
        """Load user profile from data directory"""
        if os.path.exists(self.user_profile_file):
            with open(self.user_profile_file, 'r') as f:
                profile = json.load(f)
                return profile
        else:
            return self._create_empty_profile()
    
    def _build_vector_store(self) -> FAISS:
        """Build FAISS vector store from echoForge documents"""
        
        if os.path.exists(self.echoForge_documents_file):
            with open(self.echoForge_documents_file, 'r') as f:
                documents_data = json.load(f)
                
                # Convert documents to vector store format
                documents = []
                for i, doc_data in enumerate(documents_data):
                    # Create a combined text for embedding using same format as search query
                    combined_text = f"<context>{doc_data.get('context', '')}</context>\n<title>{doc_data.get('title', '')}</title>\n<content>{doc_data.get('content', '')}</content>"
                    
                    # Create document with metadata
                    doc = Document(
                        page_content=combined_text,
                        metadata={
                            'url': doc_data.get('url', ''),
                            'context': doc_data.get('context', ''),
                            'title': doc_data.get('title', ''),
                            'content': doc_data.get('content', ''),
                            'human_response': doc_data.get('human_response', ''),
                            'reflections': doc_data.get('reflections', ''),
                            'timestamp': doc_data.get('timestamp', ''),
                            'index': i
                        }
                    )
                    documents.append(doc)
                
                # Create FAISS vector store
                if documents:
                    vector_store = FAISS.from_documents(documents, self.embeddings)
                    return vector_store
                else:
                    return None
        else:
            return None
    
    def get_user_profile(self) -> Dict[str, Any]:
        """Get the loaded user profile"""
        return self.user_profile
    
    def get_relevant_context(self, query: str, limit: int = 3) -> List[Dict[str, str]]:
        """Retrieve relevant posts using semantic search"""
        
        if self.vector_store is None:
            return []
        
        try:
            # Perform semantic search with similarity scores
            docs_with_scores = self.vector_store.similarity_search_with_score(query, k=limit)
            
            # Convert documents back to the expected format
            relevant_posts = []
            for doc, score in docs_with_scores:
                metadata = doc.metadata
                relevant_posts.append({
                    'url': metadata.get('url', ''),
                    'context': metadata.get('context', ''),
                    'title': metadata.get('title', ''),
                    'content': metadata.get('content', ''),
                    'human_response': metadata.get('human_response', ''),
                    'reflections': metadata.get('reflections', ''),
                    'timestamp': metadata.get('timestamp', ''),
                    'similarity_dist': float(score)  # FAISS distance score (lower = more similar)
                })
            
            return relevant_posts
            
        except Exception as e:
            return []

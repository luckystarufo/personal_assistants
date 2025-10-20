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
        
        # File paths for user profile and historical posts
        self.user_profile_file = os.path.join(data_dir, "shared", "user_profile.json")
        self.historical_posts_file = os.path.join(data_dir, "echoForge", "historical_posts.json")
        
        # Initialize embeddings model
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        # Initialize user profile and vector store
        self.user_profile = self._load_user_profile()
        self.vector_store = self._build_vector_store()

        # Initialize session memory
        self.memory_saver = MemorySaver()
        
    def _create_empty_profile(self) -> Dict[str, Any]:
        """Create empty user profile"""
        return {
            "personality_traits": {},
            "interests": [],
            "communication_style": {},
            "expertise_areas": [],
            "decision_patterns": {},
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat()
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
        """Build FAISS vector store from historical posts"""
        
        if os.path.exists(self.historical_posts_file):
            with open(self.historical_posts_file, 'r') as f:
                posts = json.load(f)
                
                # Convert posts to documents for vector store
                documents = []
                for i, post in enumerate(posts):
                    # Create a combined text for embedding
                    combined_text = f"Platform: {post.get('platform', '')}\nTitle: {post.get('title', '')}\nContent: {post.get('content', '')}\nResponse: {post.get('response', '')}"
                    
                    # Create document with metadata
                    doc = Document(
                        page_content=combined_text,
                        metadata={
                            'platform': post.get('platform', ''),
                            'title': post.get('title', ''),
                            'content': post.get('content', ''),
                            'response': post.get('response', ''),
                            'timestamp': post.get('timestamp', ''),
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
        """Retrieve relevant historical posts using semantic search"""
        
        if self.vector_store is None:
            return []
        
        try:
            # Perform semantic search using FAISS
            docs = self.vector_store.similarity_search(query, k=limit)
            
            # Convert documents back to the expected format
            relevant_posts = []
            for doc in docs:
                metadata = doc.metadata
                relevant_posts.append({
                    'platform': metadata.get('platform', ''),
                    'title': metadata.get('title', ''),
                    'content': metadata.get('content', ''),
                    'response': metadata.get('response', ''),
                    'timestamp': metadata.get('timestamp', '')
                })
            
            return relevant_posts
            
        except Exception as e:
            return []

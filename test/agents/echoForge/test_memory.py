"""
Unit tests for EchoForgeMemory
"""
import pytest
import tempfile
import os
import json
from datetime import datetime
from unittest.mock import patch, MagicMock
from src.agents.echoForge.memory import EchoForgeMemory


class TestEchoForgeMemory:
    """Test cases for EchoForgeMemory class"""
    
    @patch('src.agents.echoForge.memory.FAISS')
    @patch('src.agents.echoForge.memory.OpenAIEmbeddings')
    def test_init_with_custom_data_dir(self, mock_embeddings, mock_faiss):
        """Test initialization with custom data directory"""
        # Mock the OpenAI dependencies
        mock_embeddings.return_value = MagicMock()
        mock_faiss.from_documents.return_value = MagicMock()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            memory = EchoForgeMemory(temp_dir)
            
            assert memory.data_dir == temp_dir
            assert memory.user_profile_file == os.path.join(temp_dir, "shared", "user_profile.json")
            assert memory.historical_posts_file == os.path.join(temp_dir, "echoForge", "historical_posts.json")
    
    @patch('src.agents.echoForge.memory.FAISS')
    @patch('src.agents.echoForge.memory.OpenAIEmbeddings')
    def test_init_with_default_data_dir(self, mock_embeddings, mock_faiss):
        """Test initialization with default data directory"""
        # Mock the OpenAI dependencies
        mock_embeddings.return_value = MagicMock()
        mock_faiss.from_documents.return_value = MagicMock()
        
        memory = EchoForgeMemory()
        
        assert memory.data_dir == "data"
        assert memory.user_profile_file == "data/shared/user_profile.json"
        assert memory.historical_posts_file == "data/echoForge/historical_posts.json"
    
    @patch('src.agents.echoForge.memory.FAISS')
    @patch('src.agents.echoForge.memory.OpenAIEmbeddings')
    def test_load_user_profile_file_exists(self, mock_embeddings, mock_faiss):
        """Test loading user profile from existing file"""
        # Mock the OpenAI dependencies
        mock_embeddings.return_value = MagicMock()
        mock_faiss.from_documents.return_value = MagicMock()
        
        test_profile = {
            "personality_traits": {"analytical": 0.8, "creative": 0.6},
            "interests": ["AI", "programming"],
            "communication_style": {"formal": True, "humor": False},
            "expertise_areas": ["Python", "Machine Learning"],
            "decision_patterns": {"methodical": True},
            "created_at": "2024-01-01T00:00:00",
            "last_updated": "2024-01-01T00:00:00"
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create shared directory
            shared_dir = os.path.join(temp_dir, "shared")
            os.makedirs(shared_dir, exist_ok=True)
            
            # Create profile file BEFORE initializing memory
            profile_file = os.path.join(shared_dir, "user_profile.json")
            with open(profile_file, 'w') as f:
                json.dump(test_profile, f)
            
            memory = EchoForgeMemory(temp_dir)
            
            # Load profile
            loaded_profile = memory.get_user_profile()
            
            assert loaded_profile == test_profile
    
    @patch('src.agents.echoForge.memory.FAISS')
    @patch('src.agents.echoForge.memory.OpenAIEmbeddings')
    def test_load_user_profile_file_not_exists(self, mock_embeddings, mock_faiss):
        """Test loading user profile when file doesn't exist"""
        # Mock the OpenAI dependencies
        mock_embeddings.return_value = MagicMock()
        mock_faiss.from_documents.return_value = MagicMock()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            memory = EchoForgeMemory(temp_dir)
            
            # Load profile (should create new one)
            profile = memory.get_user_profile()
            
            # Check default structure
            assert "personality_traits" in profile
            assert "interests" in profile
            assert "communication_style" in profile
            assert "expertise_areas" in profile
            assert "decision_patterns" in profile
            assert "created_at" in profile
            assert "last_updated" in profile
            
            # Check default values
            assert profile["personality_traits"] == {}
            assert profile["interests"] == []
            assert profile["communication_style"] == {}
            assert profile["expertise_areas"] == []
            assert profile["decision_patterns"] == {}
    
    @patch('src.agents.echoForge.memory.FAISS')
    @patch('src.agents.echoForge.memory.OpenAIEmbeddings')
    def test_get_relevant_context(self, mock_embeddings, mock_faiss):
        """Test getting relevant context using RAG"""
        # Mock the OpenAI dependencies
        mock_embeddings.return_value = MagicMock()
        mock_vector_store = MagicMock()
        mock_vector_store.similarity_search.return_value = []
        mock_faiss.from_documents.return_value = mock_vector_store
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create echoForge directory
            echoForge_dir = os.path.join(temp_dir, "echoForge")
            os.makedirs(echoForge_dir, exist_ok=True)
            
            # Create test historical posts
            test_posts = [
                {
                    "platform": "LinkedIn",
                    "title": "AI Development",
                    "content": "Working on AI projects",
                    "response": "Great work on AI!",
                    "timestamp": "2024-01-01T00:00:00"
                }
            ]
            
            with open(os.path.join(echoForge_dir, "historical_posts.json"), 'w') as f:
                json.dump(test_posts, f)
            
            memory = EchoForgeMemory(temp_dir)
            
            # Test getting relevant context
            context = memory.get_relevant_context("AI development", limit=1)
            
            # Should return a list (may be empty if embeddings fail)
            assert isinstance(context, list)
    
    @patch('src.agents.echoForge.memory.FAISS')
    @patch('src.agents.echoForge.memory.OpenAIEmbeddings')
    def test_create_empty_profile_structure(self, mock_embeddings, mock_faiss):
        """Test that _create_empty_profile returns correct structure"""
        # Mock the OpenAI dependencies
        mock_embeddings.return_value = MagicMock()
        mock_faiss.from_documents.return_value = MagicMock()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            memory = EchoForgeMemory(temp_dir)
            
            profile = memory._create_empty_profile()
            
            # Check all required fields exist
            required_fields = [
                "personality_traits", "interests", "communication_style",
                "expertise_areas", "decision_patterns", "created_at", "last_updated"
            ]
            
            for field in required_fields:
                assert field in profile
            
            # Check default values
            assert profile["personality_traits"] == {}
            assert profile["interests"] == []
            assert profile["communication_style"] == {}
            assert profile["expertise_areas"] == []
            assert profile["decision_patterns"] == {}
            
            # Check timestamps are valid ISO format
            datetime.fromisoformat(profile["created_at"])
            datetime.fromisoformat(profile["last_updated"])

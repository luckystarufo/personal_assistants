"""
Unit tests for EchoForgeMemory
"""
import pytest
import tempfile
import os
import json
from datetime import datetime
from src.agents.echoForge.memory import EchoForgeMemory


class TestEchoForgeMemory:
    """Test cases for EchoForgeMemory class"""
    
    def test_init_with_custom_data_dir(self):
        """Test initialization with custom data directory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            memory = EchoForgeMemory(temp_dir)
            
            assert memory.data_dir == temp_dir
            assert memory.profile_file == os.path.join(temp_dir, "user_profile.json")
            assert memory.conversations_file == os.path.join(temp_dir, "conversations.json")
    
    def test_init_with_default_data_dir(self):
        """Test initialization with default data directory"""
        memory = EchoForgeMemory()
        
        assert memory.data_dir == "data/echoForge"
        assert memory.profile_file == "data/echoForge/user_profile.json"
        assert memory.conversations_file == "data/echoForge/conversations.json"
    
    def test_load_user_profile_file_exists(self):
        """Test loading user profile from existing file"""
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
            memory = EchoForgeMemory(temp_dir)
            
            # Create profile file
            with open(memory.profile_file, 'w') as f:
                json.dump(test_profile, f)
            
            # Load profile
            loaded_profile = memory.load_user_profile()
            
            assert loaded_profile == test_profile
    
    def test_load_user_profile_file_not_exists(self):
        """Test loading user profile when file doesn't exist"""
        with tempfile.TemporaryDirectory() as temp_dir:
            memory = EchoForgeMemory(temp_dir)
            
            # Load profile (should create new one)
            profile = memory.load_user_profile()
            
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
    
    def test_save_user_profile(self):
        """Test saving user profile to file"""
        test_profile = {
            "personality_traits": {"analytical": 0.8},
            "interests": ["AI"],
            "communication_style": {"formal": True},
            "expertise_areas": ["Python"],
            "decision_patterns": {"methodical": True},
            "created_at": "2024-01-01T00:00:00",
            "last_updated": "2024-01-01T00:00:00"
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            memory = EchoForgeMemory(temp_dir)
            
            # Save profile
            memory.save_user_profile(test_profile)
            
            # Verify file was created
            assert os.path.exists(memory.profile_file)
            
            # Verify content
            with open(memory.profile_file, 'r') as f:
                saved_profile = json.load(f)
            
            assert saved_profile == test_profile
    
    def test_save_user_profile_creates_directory(self):
        """Test that save_user_profile creates directory if it doesn't exist"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create nested directory path
            nested_dir = os.path.join(temp_dir, "nested", "subdir")
            memory = EchoForgeMemory(nested_dir)
            
            test_profile = {"personality_traits": {}}
            memory.save_user_profile(test_profile)
            
            # Verify directory was created and file exists
            assert os.path.exists(nested_dir)
            assert os.path.exists(memory.profile_file)
    
    def test_store_learning_conversation(self):
        """Test storing learning conversation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            memory = EchoForgeMemory(temp_dir)
            
            question = "What are your hobbies?"
            answer = "I enjoy programming and reading"
            
            # This should not raise an exception
            memory.store_learning_conversation(question, answer)
            
            # TODO: When TTL storage is implemented, verify the data is stored
    
    def test_store_echo_conversation(self):
        """Test storing echo conversation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            memory = EchoForgeMemory(temp_dir)
            
            prompt = "What do you think about AI?"
            response = "AI is fascinating and has great potential"
            confidence = 0.85
            
            # This should not raise an exception
            memory.store_echo_conversation(prompt, response, confidence)
            
            # TODO: When TTL storage is implemented, verify the data is stored
    
    def test_get_relevant_context(self):
        """Test getting relevant context for echo mode"""
        with tempfile.TemporaryDirectory() as temp_dir:
            memory = EchoForgeMemory(temp_dir)
            
            prompt = "What do you think about AI?"
            context = memory.get_relevant_context(prompt)
            
            # Should return a list (empty for now since TTL storage not implemented)
            assert isinstance(context, list)
    
    def test_create_empty_profile_structure(self):
        """Test that _create_empty_profile returns correct structure"""
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
    
    def test_profile_roundtrip(self):
        """Test saving and loading profile maintains all data"""
        original_profile = {
            "personality_traits": {"analytical": 0.8, "creative": 0.6},
            "interests": ["AI", "programming", "music"],
            "communication_style": {"formal": True, "humor": False, "detailed": True},
            "expertise_areas": ["Python", "Machine Learning", "Data Science"],
            "decision_patterns": {"methodical": True, "risk_averse": False},
            "created_at": "2024-01-01T00:00:00",
            "last_updated": "2024-01-02T12:00:00"
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            memory = EchoForgeMemory(temp_dir)
            
            # Save profile
            memory.save_user_profile(original_profile)
            
            # Load profile
            loaded_profile = memory.load_user_profile()
            
            # Verify all data matches
            assert loaded_profile == original_profile

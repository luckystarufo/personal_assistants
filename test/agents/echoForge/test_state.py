"""
Unit tests for EchoForgeState
"""
import pytest
from typing import Dict, List
from src.agents.echoForge.state import EchoForgeState


class TestEchoForgeState:
    """Test cases for EchoForgeState TypedDict"""
    
    def test_state_creation_with_messages(self):
        """Test creating state with messages"""
        state: EchoForgeState = {
            "messages": [
                {"role": "user", "content": "Hello, how are you?"},
                {"role": "assistant", "content": "I'm doing well, thank you!"}
            ]
        }
        
        # Verify messages field is accessible
        assert len(state["messages"]) == 2
        assert state["messages"][0]["role"] == "user"
        assert state["messages"][0]["content"] == "Hello, how are you?"
        assert state["messages"][1]["role"] == "assistant"
        assert state["messages"][1]["content"] == "I'm doing well, thank you!"
    
    def test_state_creation_empty_messages(self):
        """Test creating state with empty messages"""
        state: EchoForgeState = {
            "messages": []
        }
        
        assert isinstance(state["messages"], list)
        assert len(state["messages"]) == 0
    
    def test_state_with_single_message(self):
        """Test creating state with single message"""
        state: EchoForgeState = {
            "messages": [
                {"role": "assistant", "content": "Hello! I'm EchoForge."}
            ]
        }
        
        assert len(state["messages"]) == 1
        assert state["messages"][0]["role"] == "assistant"
        assert state["messages"][0]["content"] == "Hello! I'm EchoForge."
    
    def test_state_with_multiple_turns(self):
        """Test creating state with multiple conversation turns"""
        state: EchoForgeState = {
            "messages": [
                {"role": "assistant", "content": "Hello! I'm EchoForge in learning mode."},
                {"role": "user", "content": "Hi! Tell me about yourself."},
                {"role": "assistant", "content": "I'm an AI agent designed to learn about you."},
                {"role": "user", "content": "What do you want to know?"},
                {"role": "assistant", "content": "What are your main interests?"}
            ]
        }
        
        assert len(state["messages"]) == 5
        assert state["messages"][0]["role"] == "assistant"
        assert state["messages"][1]["role"] == "user"
        assert state["messages"][2]["role"] == "assistant"
        assert state["messages"][3]["role"] == "user"
        assert state["messages"][4]["role"] == "assistant"
    
    def test_state_field_types(self):
        """Test that state fields have correct types"""
        state: EchoForgeState = {
            "messages": [
                {"role": "user", "content": "test message"},
                {"role": "assistant", "content": "test response"}
            ]
        }
        
        # Test type checking
        assert isinstance(state["messages"], list)
        assert isinstance(state["messages"][0], dict)
        assert isinstance(state["messages"][0]["role"], str)
        assert isinstance(state["messages"][0]["content"], str)
        assert isinstance(state["messages"][1]["role"], str)
        assert isinstance(state["messages"][1]["content"], str)
    
    def test_state_with_different_roles(self):
        """Test state with different message roles"""
        state: EchoForgeState = {
            "messages": [
                {"role": "system", "content": "System initialization"},
                {"role": "assistant", "content": "Assistant greeting"},
                {"role": "user", "content": "User input"},
                {"role": "assistant", "content": "Assistant response"}
            ]
        }
        
        roles = [msg["role"] for msg in state["messages"]]
        assert "system" in roles
        assert "assistant" in roles
        assert "user" in roles
        assert len(roles) == 4
    
    def test_state_mutation(self):
        """Test that state can be mutated (TypedDict allows this)"""
        state: EchoForgeState = {
            "messages": []
        }
        
        # Add messages
        state["messages"].append({"role": "assistant", "content": "Hello!"})
        state["messages"].append({"role": "user", "content": "Hi there!"})
        
        assert len(state["messages"]) == 2
        assert state["messages"][0]["content"] == "Hello!"
        assert state["messages"][1]["content"] == "Hi there!"
        
        # Modify existing message
        state["messages"][0]["content"] = "Hello, how can I help you?"
        assert state["messages"][0]["content"] == "Hello, how can I help you?"
    
    def test_state_with_long_content(self):
        """Test state with long message content"""
        long_content = "This is a very long message that contains multiple sentences. " * 10
        
        state: EchoForgeState = {
            "messages": [
                {"role": "user", "content": long_content},
                {"role": "assistant", "content": "I received your long message."}
            ]
        }
        
        assert len(state["messages"]) == 2
        assert len(state["messages"][0]["content"]) > 100
        assert state["messages"][1]["content"] == "I received your long message."
    
    def test_state_with_special_characters(self):
        """Test state with special characters in content"""
        state: EchoForgeState = {
            "messages": [
                {"role": "user", "content": "Hello! How are you? I'm doing great! 😊"},
                {"role": "assistant", "content": "I'm doing well too! Thanks for asking! 🎉"},
                {"role": "user", "content": "What about emojis: 🚀🌟💡🎯"},
                {"role": "assistant", "content": "Emojis are fun! 🎨✨"}
            ]
        }
        
        assert len(state["messages"]) == 4
        assert "😊" in state["messages"][0]["content"]
        assert "🎉" in state["messages"][1]["content"]
        assert "🚀" in state["messages"][2]["content"]
        assert "🎨" in state["messages"][3]["content"]
    
    def test_state_with_empty_content(self):
        """Test state with empty content messages"""
        state: EchoForgeState = {
            "messages": [
                {"role": "user", "content": ""},
                {"role": "assistant", "content": "I see you sent an empty message."},
                {"role": "user", "content": "   "},  # Whitespace only
                {"role": "assistant", "content": "You sent whitespace."}
            ]
        }
        
        assert len(state["messages"]) == 4
        assert state["messages"][0]["content"] == ""
        assert state["messages"][1]["content"] == "I see you sent an empty message."
        assert state["messages"][2]["content"] == "   "
        assert state["messages"][3]["content"] == "You sent whitespace."

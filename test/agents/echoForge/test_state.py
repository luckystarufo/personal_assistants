"""
Unit tests for EchoForgeState
"""
import pytest
from typing import Dict, Any, List
from src.agents.echoForge.state import EchoForgeState


class TestEchoForgeState:
    """Test cases for EchoForgeState TypedDict"""
    
    def test_state_creation_with_all_fields(self):
        """Test creating state with all required fields"""
        state: EchoForgeState = {
            "learning_mode": True,
            "user_input": "Hello, how are you?",
            "response": "I'm doing well, thank you!",
            "user_profile": {
                "personality_traits": {"analytical": 0.8},
                "interests": ["AI", "programming"],
                "communication_style": {"formal": True},
                "expertise_areas": ["Python", "ML"]
            },
            "conversation_history": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"}
            ],
            "session_context": {"current_topic": "AI"},
            "learning_targets": ["hobbies", "goals"],
            "current_question": "What are your hobbies?",
            "confidence_score": 0.85,
            "confirmation_needed": False,
            "timestamp": "2024-01-01T12:00:00",
            "session_id": "session_123"
        }
        
        # Verify all fields are accessible
        assert state["learning_mode"] == True
        assert state["user_input"] == "Hello, how are you?"
        assert state["response"] == "I'm doing well, thank you!"
        assert isinstance(state["user_profile"], dict)
        assert isinstance(state["conversation_history"], list)
        assert isinstance(state["session_context"], dict)
        assert isinstance(state["learning_targets"], list)
        assert state["current_question"] == "What are your hobbies?"
        assert state["confidence_score"] == 0.85
        assert state["confirmation_needed"] == False
        assert state["timestamp"] == "2024-01-01T12:00:00"
        assert state["session_id"] == "session_123"
    
    def test_state_creation_learning_mode(self):
        """Test creating state for learning mode"""
        state: EchoForgeState = {
            "learning_mode": True,
            "user_input": "Tell me about yourself",
            "response": "",
            "user_profile": {},
            "conversation_history": [],
            "session_context": {},
            "learning_targets": ["personality", "interests"],
            "current_question": "What are your main interests?",
            "confidence_score": 0.0,
            "confirmation_needed": False,
            "timestamp": "2024-01-01T12:00:00",
            "session_id": "session_456"
        }
        
        assert state["learning_mode"] == True
        assert state["current_question"] == "What are your main interests?"
        assert state["learning_targets"] == ["personality", "interests"]
        assert state["confidence_score"] == 0.0
        assert state["confirmation_needed"] == False
    
    def test_state_creation_echo_mode(self):
        """Test creating state for echo mode"""
        state: EchoForgeState = {
            "learning_mode": False,
            "user_input": "What do you think about AI?",
            "response": "AI has great potential for solving complex problems",
            "user_profile": {
                "personality_traits": {"analytical": 0.9},
                "interests": ["technology"],
                "communication_style": {"technical": True},
                "expertise_areas": ["AI", "Machine Learning"]
            },
            "conversation_history": [
                {"role": "user", "content": "Tell me about AI"},
                {"role": "assistant", "content": "AI is fascinating"}
            ],
            "session_context": {"topic": "AI discussion"},
            "learning_targets": [],
            "current_question": None,
            "confidence_score": 0.92,
            "confirmation_needed": False,
            "timestamp": "2024-01-01T12:00:00",
            "session_id": "session_789"
        }
        
        assert state["learning_mode"] == False
        assert state["current_question"] is None
        assert state["learning_targets"] == []
        assert state["confidence_score"] == 0.92
        assert state["response"] == "AI has great potential for solving complex problems"
    
    def test_state_with_minimal_data(self):
        """Test creating state with minimal required data"""
        state: EchoForgeState = {
            "learning_mode": False,
            "user_input": "",
            "response": "",
            "user_profile": {},
            "conversation_history": [],
            "session_context": {},
            "learning_targets": [],
            "current_question": None,
            "confidence_score": 0.0,
            "confirmation_needed": False,
            "timestamp": "",
            "session_id": ""
        }
        
        # All fields should be accessible with default values
        assert state["learning_mode"] == False
        assert state["user_input"] == ""
        assert state["response"] == ""
        assert state["user_profile"] == {}
        assert state["conversation_history"] == []
        assert state["session_context"] == {}
        assert state["learning_targets"] == []
        assert state["current_question"] is None
        assert state["confidence_score"] == 0.0
        assert state["confirmation_needed"] == False
        assert state["timestamp"] == ""
        assert state["session_id"] == ""
    
    def test_state_field_types(self):
        """Test that state fields have correct types"""
        state: EchoForgeState = {
            "learning_mode": True,
            "user_input": "test",
            "response": "test",
            "user_profile": {"key": "value"},
            "conversation_history": [{"role": "user", "content": "test"}],
            "session_context": {"key": "value"},
            "learning_targets": ["target1"],
            "current_question": "question",
            "confidence_score": 0.5,
            "confirmation_needed": True,
            "timestamp": "2024-01-01T00:00:00",
            "session_id": "id123"
        }
        
        # Test type checking
        assert isinstance(state["learning_mode"], bool)
        assert isinstance(state["user_input"], str)
        assert isinstance(state["response"], str)
        assert isinstance(state["user_profile"], dict)
        assert isinstance(state["conversation_history"], list)
        assert isinstance(state["session_context"], dict)
        assert isinstance(state["learning_targets"], list)
        assert isinstance(state["current_question"], str) or state["current_question"] is None
        assert isinstance(state["confidence_score"], float)
        assert isinstance(state["confirmation_needed"], bool)
        assert isinstance(state["timestamp"], str)
        assert isinstance(state["session_id"], str)
    
    def test_state_with_complex_user_profile(self):
        """Test state with complex user profile data"""
        complex_profile = {
            "personality_traits": {
                "analytical": 0.9,
                "creative": 0.7,
                "social": 0.3,
                "methodical": 0.8
            },
            "interests": [
                "artificial intelligence",
                "machine learning",
                "data science",
                "programming",
                "philosophy"
            ],
            "communication_style": {
                "formal": True,
                "technical": True,
                "detailed": True,
                "humor": False,
                "casual": False
            },
            "expertise_areas": [
                "Python",
                "TensorFlow",
                "PyTorch",
                "Natural Language Processing",
                "Computer Vision"
            ],
            "decision_patterns": {
                "data_driven": True,
                "risk_averse": False,
                "collaborative": True,
                "systematic": True
            },
            "preferences": {
                "learning_style": "hands-on",
                "communication_preference": "written",
                "meeting_style": "structured"
            }
        }
        
        state: EchoForgeState = {
            "learning_mode": False,
            "user_input": "Complex prompt",
            "response": "Complex response",
            "user_profile": complex_profile,
            "conversation_history": [],
            "session_context": {},
            "learning_targets": [],
            "current_question": None,
            "confidence_score": 0.95,
            "confirmation_needed": False,
            "timestamp": "2024-01-01T12:00:00",
            "session_id": "complex_session"
        }
        
        assert state["user_profile"] == complex_profile
        assert len(state["user_profile"]["personality_traits"]) == 4
        assert len(state["user_profile"]["interests"]) == 5
        assert len(state["user_profile"]["expertise_areas"]) == 5
    
    def test_state_with_rich_conversation_history(self):
        """Test state with rich conversation history"""
        rich_history = [
            {"role": "user", "content": "Hello, I'm interested in AI"},
            {"role": "assistant", "content": "That's great! What aspect of AI interests you most?"},
            {"role": "user", "content": "I'm particularly interested in machine learning"},
            {"role": "assistant", "content": "Machine learning is fascinating. Do you have experience with any ML frameworks?"},
            {"role": "user", "content": "I've worked with TensorFlow and PyTorch"},
            {"role": "assistant", "content": "Excellent! Those are both powerful frameworks. What kind of projects have you built?"}
        ]
        
        state: EchoForgeState = {
            "learning_mode": True,
            "user_input": "Tell me more about your projects",
            "response": "",
            "user_profile": {},
            "conversation_history": rich_history,
            "session_context": {"topic": "AI discussion", "depth": "technical"},
            "learning_targets": ["project_experience", "technical_skills"],
            "current_question": "What specific projects have you worked on?",
            "confidence_score": 0.0,
            "confirmation_needed": False,
            "timestamp": "2024-01-01T12:00:00",
            "session_id": "rich_session"
        }
        
        assert len(state["conversation_history"]) == 6
        assert state["conversation_history"][0]["role"] == "user"
        assert state["conversation_history"][0]["content"] == "Hello, I'm interested in AI"
        assert state["session_context"]["topic"] == "AI discussion"
    
    def test_state_confidence_scenarios(self):
        """Test state with different confidence scenarios"""
        # High confidence scenario
        high_confidence_state: EchoForgeState = {
            "learning_mode": False,
            "user_input": "What's 2+2?",
            "response": "2+2 equals 4",
            "user_profile": {"expertise_areas": ["mathematics"]},
            "conversation_history": [],
            "session_context": {},
            "learning_targets": [],
            "current_question": None,
            "confidence_score": 0.99,
            "confirmation_needed": False,
            "timestamp": "2024-01-01T12:00:00",
            "session_id": "high_conf"
        }
        
        assert high_confidence_state["confidence_score"] == 0.99
        assert high_confidence_state["confirmation_needed"] == False
        
        # Low confidence scenario
        low_confidence_state: EchoForgeState = {
            "learning_mode": False,
            "user_input": "What's your opinion on quantum physics?",
            "response": "I think quantum physics is...",
            "user_profile": {"expertise_areas": ["programming"]},
            "conversation_history": [],
            "session_context": {},
            "learning_targets": [],
            "current_question": None,
            "confidence_score": 0.25,
            "confirmation_needed": True,
            "timestamp": "2024-01-01T12:00:00",
            "session_id": "low_conf"
        }
        
        assert low_confidence_state["confidence_score"] == 0.25
        assert low_confidence_state["confirmation_needed"] == True
    
    def test_state_learning_targets_scenarios(self):
        """Test state with different learning target scenarios"""
        # Learning mode with targets
        learning_state: EchoForgeState = {
            "learning_mode": True,
            "user_input": "I want to learn more about you",
            "response": "",
            "user_profile": {},
            "conversation_history": [],
            "session_context": {},
            "learning_targets": ["personality", "hobbies", "goals", "values"],
            "current_question": "What are your core values?",
            "confidence_score": 0.0,
            "confirmation_needed": False,
            "timestamp": "2024-01-01T12:00:00",
            "session_id": "learning_session"
        }
        
        assert learning_state["learning_mode"] == True
        assert len(learning_state["learning_targets"]) == 4
        assert "personality" in learning_state["learning_targets"]
        assert learning_state["current_question"] == "What are your core values?"
        
        # Echo mode with no targets
        echo_state: EchoForgeState = {
            "learning_mode": False,
            "user_input": "Respond to this prompt",
            "response": "Here's my response",
            "user_profile": {},
            "conversation_history": [],
            "session_context": {},
            "learning_targets": [],
            "current_question": None,
            "confidence_score": 0.8,
            "confirmation_needed": False,
            "timestamp": "2024-01-01T12:00:00",
            "session_id": "echo_session"
        }
        
        assert echo_state["learning_mode"] == False
        assert echo_state["learning_targets"] == []
        assert echo_state["current_question"] is None
    
    def test_state_mutation(self):
        """Test that state can be mutated (TypedDict allows this)"""
        state: EchoForgeState = {
            "learning_mode": False,
            "user_input": "Initial input",
            "response": "Initial response",
            "user_profile": {},
            "conversation_history": [],
            "session_context": {},
            "learning_targets": [],
            "current_question": None,
            "confidence_score": 0.5,
            "confirmation_needed": False,
            "timestamp": "2024-01-01T12:00:00",
            "session_id": "mutable_session"
        }
        
        # Mutate state
        state["learning_mode"] = True
        state["user_input"] = "Updated input"
        state["response"] = "Updated response"
        state["confidence_score"] = 0.9
        state["confirmation_needed"] = True
        
        assert state["learning_mode"] == True
        assert state["user_input"] == "Updated input"
        assert state["response"] == "Updated response"
        assert state["confidence_score"] == 0.9
        assert state["confirmation_needed"] == True

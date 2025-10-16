"""
Unit tests for LearningPrompts
"""
import pytest
from src.prompts.echoForge.learning_prompts import LearningPrompts


class TestLearningPrompts:
    """Test cases for LearningPrompts class"""
    
    def test_generate_learning_question_basic(self):
        """Test generating a basic learning question"""
        profile = {
            "personality_traits": {"analytical": 0.8},
            "interests": ["AI", "programming"],
            "communication_style": {"formal": True}
        }
        conversation_history = [
            {"role": "user", "content": "I like programming"},
            {"role": "assistant", "content": "What programming languages do you prefer?"}
        ]
        
        question = LearningPrompts.generate_learning_question(profile, conversation_history)
        
        # Should return a string
        assert isinstance(question, str)
        assert len(question) > 0
        # Should be a question (placeholder implementation)
        assert "passionate" in question.lower()
    
    def test_generate_learning_question_empty_profile(self):
        """Test generating learning question with empty profile"""
        profile = {
            "personality_traits": {},
            "interests": [],
            "communication_style": {}
        }
        conversation_history = []
        
        question = LearningPrompts.generate_learning_question(profile, conversation_history)
        
        assert isinstance(question, str)
        assert len(question) > 0
    
    def test_generate_learning_question_with_history(self):
        """Test generating learning question with conversation history"""
        profile = {
            "personality_traits": {"creative": 0.7},
            "interests": ["music"],
            "communication_style": {"casual": True}
        }
        conversation_history = [
            {"role": "user", "content": "I play guitar"},
            {"role": "assistant", "content": "What genre do you prefer?"},
            {"role": "user", "content": "Rock and jazz"}
        ]
        
        question = LearningPrompts.generate_learning_question(profile, conversation_history)
        
        assert isinstance(question, str)
        assert len(question) > 0
    
    def test_summarize_qa_interaction(self):
        """Test summarizing Q&A interaction"""
        question = "What are your hobbies?"
        answer = "I enjoy programming, reading science fiction, and hiking on weekends"
        
        summary = LearningPrompts.summarize_qa_interaction(question, answer)
        
        # Should return a dictionary with expected keys
        assert isinstance(summary, dict)
        assert "insights" in summary
        assert "traits" in summary
        assert "topics" in summary
        
        # Check that values are lists
        assert isinstance(summary["insights"], list)
        assert isinstance(summary["traits"], list)
        assert isinstance(summary["topics"], list)
        
        # Check placeholder values
        assert "technology" in summary["insights"][0].lower()
        assert "curious" in summary["traits"]
        assert "AI" in summary["topics"]
    
    def test_summarize_qa_interaction_empty_answer(self):
        """Test summarizing Q&A with empty answer"""
        question = "What do you think about AI?"
        answer = ""
        
        summary = LearningPrompts.summarize_qa_interaction(question, answer)
        
        assert isinstance(summary, dict)
        assert "insights" in summary
        assert "traits" in summary
        assert "topics" in summary
    
    def test_generate_confirmation_question(self):
        """Test generating confirmation question"""
        original_response = "I think AI will revolutionize healthcare"
        user_feedback = "That's not quite how I would put it"
        
        confirmation_question = LearningPrompts.generate_confirmation_question(
            original_response, user_feedback
        )
        
        assert isinstance(confirmation_question, str)
        assert len(confirmation_question) > 0
        # Should contain the original response
        assert original_response in confirmation_question
        # Should be asking for confirmation
        assert "?" in confirmation_question
    
    def test_generate_confirmation_question_no_feedback(self):
        """Test generating confirmation question without user feedback"""
        original_response = "Python is the best programming language"
        user_feedback = ""
        
        confirmation_question = LearningPrompts.generate_confirmation_question(
            original_response, user_feedback
        )
        
        assert isinstance(confirmation_question, str)
        assert len(confirmation_question) > 0
        assert original_response in confirmation_question
    
    def test_extract_personality_traits(self):
        """Test extracting personality traits from conversations"""
        conversations = [
            {"role": "user", "content": "I always analyze problems step by step"},
            {"role": "user", "content": "I love coming up with creative solutions"},
            {"role": "user", "content": "I prefer direct communication"}
        ]
        
        traits = LearningPrompts.extract_personality_traits(conversations)
        
        assert isinstance(traits, list)
        assert len(traits) > 0
        # Check placeholder values
        assert "analytical" in traits
        assert "creative" in traits
        assert "direct" in traits
    
    def test_extract_personality_traits_empty_conversations(self):
        """Test extracting traits from empty conversation list"""
        conversations = []
        
        traits = LearningPrompts.extract_personality_traits(conversations)
        
        assert isinstance(traits, list)
        # Should still return some default traits
        assert len(traits) > 0
    
    def test_extract_personality_traits_single_conversation(self):
        """Test extracting traits from single conversation"""
        conversations = [
            {"role": "user", "content": "I'm very methodical in my approach"}
        ]
        
        traits = LearningPrompts.extract_personality_traits(conversations)
        
        assert isinstance(traits, list)
        assert len(traits) > 0
    
    def test_all_methods_return_expected_types(self):
        """Test that all methods return expected data types"""
        profile = {"personality_traits": {}, "interests": [], "communication_style": {}}
        conversations = [{"role": "user", "content": "test"}]
        
        # Test all methods return expected types
        question = LearningPrompts.generate_learning_question(profile, conversations)
        assert isinstance(question, str)
        
        summary = LearningPrompts.summarize_qa_interaction("Q", "A")
        assert isinstance(summary, dict)
        
        confirmation = LearningPrompts.generate_confirmation_question("R", "F")
        assert isinstance(confirmation, str)
        
        traits = LearningPrompts.extract_personality_traits(conversations)
        assert isinstance(traits, list)

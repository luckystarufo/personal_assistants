"""
Unit tests for EchoPrompts
"""
import pytest
from src.prompts.echoForge.echo_prompts import EchoPrompts


class TestEchoPrompts:
    """Test cases for EchoPrompts class"""
    
    def test_generate_echo_response_basic(self):
        """Test generating basic echo response"""
        prompt = "What do you think about AI?"
        user_profile = {
            "personality_traits": {"analytical": 0.8, "creative": 0.6},
            "interests": ["AI", "programming"],
            "communication_style": {"formal": True, "detailed": True},
            "expertise_areas": ["Machine Learning", "Python"]
        }
        context = [
            {"role": "user", "content": "I think AI has great potential"},
            {"role": "assistant", "content": "What specific applications interest you?"}
        ]
        
        response = EchoPrompts.generate_echo_response(prompt, user_profile, context)
        
        assert isinstance(response, str)
        assert len(response) > 0
        # Check placeholder response format
        assert "Based on your style" in response
        assert "placeholder response" in response
    
    def test_generate_echo_response_empty_profile(self):
        """Test generating echo response with empty profile"""
        prompt = "How are you today?"
        user_profile = {
            "personality_traits": {},
            "interests": [],
            "communication_style": {},
            "expertise_areas": []
        }
        context = []
        
        response = EchoPrompts.generate_echo_response(prompt, user_profile, context)
        
        assert isinstance(response, str)
        assert len(response) > 0
    
    def test_generate_echo_response_with_context(self):
        """Test generating echo response with rich context"""
        prompt = "What's your opinion on remote work?"
        user_profile = {
            "personality_traits": {"methodical": 0.9, "social": 0.3},
            "interests": ["productivity", "technology"],
            "communication_style": {"direct": True, "data_driven": True},
            "expertise_areas": ["Project Management", "Data Analysis"]
        }
        context = [
            {"role": "user", "content": "I prefer structured environments"},
            {"role": "assistant", "content": "What works best for your productivity?"},
            {"role": "user", "content": "Clear schedules and minimal distractions"}
        ]
        
        response = EchoPrompts.generate_echo_response(prompt, user_profile, context)
        
        assert isinstance(response, str)
        assert len(response) > 0
    
    def test_calculate_confidence_basic(self):
        """Test calculating confidence score"""
        prompt = "What do you think about Python?"
        response = "Python is excellent for data science and has great libraries"
        user_profile = {
            "personality_traits": {"technical": 0.8},
            "expertise_areas": ["Python", "Data Science"]
        }
        
        confidence = EchoPrompts.calculate_confidence(prompt, response, user_profile)
        
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0
        # Check placeholder value
        assert confidence == 0.75
    
    def test_calculate_confidence_different_inputs(self):
        """Test confidence calculation with different inputs"""
        # Test with technical prompt
        confidence1 = EchoPrompts.calculate_confidence(
            "Explain machine learning", 
            "ML is a subset of AI", 
            {"expertise_areas": ["AI"]}
        )
        assert isinstance(confidence1, float)
        assert 0.0 <= confidence1 <= 1.0
        
        # Test with personal prompt
        confidence2 = EchoPrompts.calculate_confidence(
            "How was your weekend?", 
            "It was relaxing", 
            {"personality_traits": {"private": 0.9}}
        )
        assert isinstance(confidence2, float)
        assert 0.0 <= confidence2 <= 1.0
    
    def test_build_context_prompt(self):
        """Test building context prompt"""
        user_profile = {
            "personality_traits": {"analytical": 0.8},
            "interests": ["science"],
            "communication_style": {"formal": True},
            "expertise_areas": ["Physics"]
        }
        context = [
            {"role": "user", "content": "I love physics"},
            {"role": "assistant", "content": "What area interests you most?"}
        ]
        
        context_prompt = EchoPrompts.build_context_prompt(user_profile, context)
        
        assert isinstance(context_prompt, str)
        assert len(context_prompt) > 0
        # Should contain profile and context information
        assert "User profile:" in context_prompt
        assert "Context:" in context_prompt
    
    def test_build_context_prompt_empty_inputs(self):
        """Test building context prompt with empty inputs"""
        user_profile = {}
        context = []
        
        context_prompt = EchoPrompts.build_context_prompt(user_profile, context)
        
        assert isinstance(context_prompt, str)
        assert len(context_prompt) > 0
    
    def test_style_instruction_template(self):
        """Test style instruction template"""
        template = EchoPrompts.style_instruction_template()
        
        assert isinstance(template, str)
        assert len(template) > 0
        
        # Check that template contains expected placeholders
        assert "{communication_style}" in template
        assert "{personality_traits}" in template
        assert "{expertise_areas}" in template
        assert "{context}" in template
        assert "{prompt}" in template
        
        # Check that it's a proper instruction
        assert "Respond to the following prompt" in template
        assert "as if you were the user" in template
    
    def test_all_methods_return_expected_types(self):
        """Test that all methods return expected data types"""
        prompt = "Test prompt"
        user_profile = {
            "personality_traits": {"test": 0.5},
            "interests": ["test"],
            "communication_style": {"test": True},
            "expertise_areas": ["test"]
        }
        context = [{"role": "user", "content": "test"}]
        
        # Test all methods return expected types
        response = EchoPrompts.generate_echo_response(prompt, user_profile, context)
        assert isinstance(response, str)
        
        confidence = EchoPrompts.calculate_confidence(prompt, response, user_profile)
        assert isinstance(confidence, float)
        
        context_prompt = EchoPrompts.build_context_prompt(user_profile, context)
        assert isinstance(context_prompt, str)
        
        template = EchoPrompts.style_instruction_template()
        assert isinstance(template, str)
    
    def test_confidence_score_range(self):
        """Test that confidence scores are always in valid range"""
        prompts = [
            "What's your favorite color?",
            "Explain quantum computing",
            "How do you feel about politics?",
            "What's 2+2?"
        ]
        
        user_profile = {"personality_traits": {}, "expertise_areas": []}
        
        for prompt in prompts:
            response = EchoPrompts.generate_echo_response(prompt, user_profile, [])
            confidence = EchoPrompts.calculate_confidence(prompt, response, user_profile)
            
            assert isinstance(confidence, float)
            assert 0.0 <= confidence <= 1.0
    
    def test_echo_response_contains_prompt_info(self):
        """Test that echo response generation considers the prompt"""
        prompt = "Tell me about your hobbies"
        user_profile = {"personality_traits": {}, "interests": [], "communication_style": {}, "expertise_areas": []}
        context = []
        
        response = EchoPrompts.generate_echo_response(prompt, user_profile, context)
        
        # The response should be related to the prompt (even if placeholder)
        assert isinstance(response, str)
        assert len(response) > 0

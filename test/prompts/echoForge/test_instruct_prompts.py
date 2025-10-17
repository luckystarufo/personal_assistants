"""
Unit tests for InstructPrompts
"""
import pytest
from src.prompts.echoForge.instruct_prompts import InstructPrompts


class TestInstructPrompts:
    """Test cases for InstructPrompts class"""
    
    def test_generate_instruction_response_basic(self):
        """Test generating basic instruction response"""
        prompt = "Please explain how machine learning works"
        user_profile = {
            "personality_traits": {"analytical": 0.8, "curious": 0.7},
            "interests": ["AI", "technology"],
            "communication_style": {"technical": True, "detailed": True},
            "expertise_areas": ["Machine Learning", "Data Science"]
        }
        context = [
            {"role": "user", "content": "I want to learn about AI"},
            {"role": "assistant", "content": "I'd be happy to help you learn about AI"}
        ]
        
        response = InstructPrompts.generate_instruction_response(prompt, user_profile, context)
        
        assert isinstance(response, str)
        assert len(response) > 0
        # Check placeholder response format
        assert "Based on your instructions" in response
        assert "placeholder instruction response" in response
    
    def test_generate_instruction_response_empty_profile(self):
        """Test generating instruction response with empty profile"""
        prompt = "Help me understand this concept"
        user_profile = {
            "personality_traits": {},
            "interests": [],
            "communication_style": {},
            "expertise_areas": []
        }
        context = []
        
        response = InstructPrompts.generate_instruction_response(prompt, user_profile, context)
        
        assert isinstance(response, str)
        assert len(response) > 0
    
    def test_generate_instruction_response_with_context(self):
        """Test generating instruction response with rich context"""
        prompt = "Can you help me write a Python function?"
        user_profile = {
            "personality_traits": {"methodical": 0.9, "helpful": 0.8},
            "interests": ["programming", "problem-solving"],
            "communication_style": {"step_by_step": True, "clear": True},
            "expertise_areas": ["Python", "Software Development"]
        }
        context = [
            {"role": "user", "content": "I'm learning Python"},
            {"role": "assistant", "content": "Great! Python is a wonderful language to learn"},
            {"role": "user", "content": "I need help with functions"}
        ]
        
        response = InstructPrompts.generate_instruction_response(prompt, user_profile, context)
        
        assert isinstance(response, str)
        assert len(response) > 0
    
    def test_calculate_instruction_confidence_basic(self):
        """Test calculating instruction confidence score"""
        prompt = "Please explain this concept"
        response = "Here's a detailed explanation of the concept"
        user_profile = {
            "personality_traits": {"explanatory": 0.8},
            "expertise_areas": ["Education", "Communication"]
        }
        
        confidence = InstructPrompts.calculate_instruction_confidence(prompt, response, user_profile)
        
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0
        # Check placeholder value
        assert confidence == 0.80
    
    def test_calculate_instruction_confidence_different_inputs(self):
        """Test confidence calculation with different inputs"""
        # Test with clear instruction
        confidence1 = InstructPrompts.calculate_instruction_confidence(
            "Please help me with this task", 
            "I'll help you with that task", 
            {"expertise_areas": ["General Help"]}
        )
        assert isinstance(confidence1, float)
        assert 0.0 <= confidence1 <= 1.0
        
        # Test with vague instruction
        confidence2 = InstructPrompts.calculate_instruction_confidence(
            "Do something", 
            "I'm not sure what you want me to do", 
            {"personality_traits": {"confused": 0.9}}
        )
        assert isinstance(confidence2, float)
        assert 0.0 <= confidence2 <= 1.0
    
    def test_build_instruction_context(self):
        """Test building instruction context prompt"""
        user_profile = {
            "personality_traits": {"helpful": 0.8},
            "interests": ["learning"],
            "communication_style": {"patient": True},
            "expertise_areas": ["Teaching"]
        }
        context = [
            {"role": "user", "content": "I need help"},
            {"role": "assistant", "content": "I'm here to help"}
        ]
        
        context_prompt = InstructPrompts.build_instruction_context(user_profile, context)
        
        assert isinstance(context_prompt, str)
        assert len(context_prompt) > 0
        # Should contain profile and context information
        assert "User profile:" in context_prompt
        assert "Context:" in context_prompt
    
    def test_build_instruction_context_empty_inputs(self):
        """Test building instruction context with empty inputs"""
        user_profile = {}
        context = []
        
        context_prompt = InstructPrompts.build_instruction_context(user_profile, context)
        
        assert isinstance(context_prompt, str)
        assert len(context_prompt) > 0
    
    def test_instruction_template(self):
        """Test instruction template"""
        template = InstructPrompts.instruction_template()
        
        assert isinstance(template, str)
        assert len(template) > 0
        
        # Check that template contains expected placeholders
        assert "{user_profile}" in template
        assert "{context}" in template
        assert "{prompt}" in template
        
        # Check that it's a proper instruction template
        assert "instruction mode" in template
        assert "Follow the user's instructions" in template
        assert "maintaining the user's style" in template
    
    def test_validate_instruction(self):
        """Test instruction validation"""
        # Valid instructions
        valid_instructions = [
            "Please help me with this",
            "Can you explain this concept?",
            "Help me understand this",
            "Tell me about machine learning",
            "Show me how to do this"
        ]
        
        for instruction in valid_instructions:
            is_valid = InstructPrompts.validate_instruction(instruction)
            assert isinstance(is_valid, bool)
            assert is_valid == True
        
        # Invalid instructions (should still return bool)
        invalid_instructions = [
            "Hello there",
            "How are you?",
            "Random statement"
        ]
        
        for instruction in invalid_instructions:
            is_valid = InstructPrompts.validate_instruction(instruction)
            assert isinstance(is_valid, bool)
    
    def test_extract_instruction_intent(self):
        """Test extracting instruction intent"""
        instructions = [
            "Please explain machine learning",
            "Can you help me write code?",
            "Tell me about your experience",
            "Show me how to cook pasta"
        ]
        
        for instruction in instructions:
            intent = InstructPrompts.extract_instruction_intent(instruction)
            assert isinstance(intent, str)
            assert len(intent) > 0
            # Check placeholder value
            assert intent == "general_instruction"
    
    def test_all_methods_return_expected_types(self):
        """Test that all methods return expected data types"""
        prompt = "Test instruction"
        user_profile = {
            "personality_traits": {"test": 0.5},
            "interests": ["test"],
            "communication_style": {"test": True},
            "expertise_areas": ["test"]
        }
        context = [{"role": "user", "content": "test"}]
        
        # Test all methods return expected types
        response = InstructPrompts.generate_instruction_response(prompt, user_profile, context)
        assert isinstance(response, str)
        
        confidence = InstructPrompts.calculate_instruction_confidence(prompt, response, user_profile)
        assert isinstance(confidence, float)
        
        context_prompt = InstructPrompts.build_instruction_context(user_profile, context)
        assert isinstance(context_prompt, str)
        
        template = InstructPrompts.instruction_template()
        assert isinstance(template, str)
        
        is_valid = InstructPrompts.validate_instruction(prompt)
        assert isinstance(is_valid, bool)
        
        intent = InstructPrompts.extract_instruction_intent(prompt)
        assert isinstance(intent, str)
    
    def test_confidence_score_range(self):
        """Test that confidence scores are always in valid range"""
        prompts = [
            "Please explain this clearly",
            "Help me with this complex task",
            "Can you assist me?",
            "Show me how to do this step by step"
        ]
        
        user_profile = {"personality_traits": {}, "expertise_areas": []}
        
        for prompt in prompts:
            response = InstructPrompts.generate_instruction_response(prompt, user_profile, [])
            confidence = InstructPrompts.calculate_instruction_confidence(prompt, response, user_profile)
            
            assert isinstance(confidence, float)
            assert 0.0 <= confidence <= 1.0
    
    def test_instruction_response_contains_prompt_info(self):
        """Test that instruction response generation considers the prompt"""
        prompt = "Please help me understand this topic"
        user_profile = {"personality_traits": {}, "interests": [], "communication_style": {}, "expertise_areas": []}
        context = []
        
        response = InstructPrompts.generate_instruction_response(prompt, user_profile, context)
        
        # The response should be related to the prompt (even if placeholder)
        assert isinstance(response, str)
        assert len(response) > 0

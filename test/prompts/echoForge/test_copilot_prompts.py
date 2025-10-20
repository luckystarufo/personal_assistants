"""
Unit tests for CopilotPrompts
"""
import pytest
from src.prompts.echoForge.copilot_prompts import CopilotPrompts


class TestCopilotPrompts:
    """Test cases for CopilotPrompts class"""
    
    def test_greeting(self):
        """Test greeting message"""
        greeting = CopilotPrompts.greeting()
        
        assert isinstance(greeting, str)
        assert len(greeting) > 0
        assert "platform" in greeting.lower()
        assert "title" in greeting.lower()
        assert "content" in greeting.lower()
    
    def test_confirmation_message(self):
        """Test confirmation message generation"""
        platform = "LinkedIn"
        title = "AI Development"
        content = "Working on AI projects"
        
        message = CopilotPrompts.confirmation_message(platform, title, content)
        
        assert isinstance(message, str)
        assert len(message) > 0
        assert platform in message
        assert title in message
        assert content in message
    
    def test_confirmation_success_message(self):
        """Test confirmation success message"""
        message = CopilotPrompts.confirmation_success_message()
        
        assert isinstance(message, str)
        assert len(message) > 0
        assert "confirmed" in message.lower() or "proceed" in message.lower()
    
    def test_modification_request_message(self):
        """Test modification request message"""
        message = CopilotPrompts.modification_request_message()
        
        assert isinstance(message, str)
        assert len(message) > 0
        assert "modifications" in message.lower() or "modify" in message.lower()
    
    def test_default_confirmation_message(self):
        """Test default confirmation message"""
        message = CopilotPrompts.default_confirmation_message()
        
        assert isinstance(message, str)
        assert len(message) > 0
    
    def test_ask_for_missing_fields(self):
        """Test asking for missing fields"""
        missing_fields = ["platform", "title"]
        message = CopilotPrompts.ask_for_missing_fields(missing_fields)
        
        assert isinstance(message, str)
        assert len(message) > 0
        assert "platform" in message
        assert "title" in message
    
    def test_ask_for_missing_fields_empty(self):
        """Test asking for missing fields with empty list"""
        message = CopilotPrompts.ask_for_missing_fields([])
        
        assert isinstance(message, str)
        assert len(message) > 0
    
    def test_parse_post_info_prompt(self):
        """Test parse post info prompt"""
        user_input = "LinkedIn post about AI development"
        prompt = CopilotPrompts.parse_post_info_prompt(user_input)
        
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert user_input in prompt
        assert "platform" in prompt.lower()
        assert "title" in prompt.lower()
        assert "content" in prompt.lower()
    
    def test_detect_quit_intent_prompt(self):
        """Test detect quit intent prompt"""
        user_input = "I want to quit"
        prompt = CopilotPrompts.detect_quit_intent_prompt(user_input)
        
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert user_input in prompt
        assert "quit" in prompt.lower()
        assert "exit" in prompt.lower()
    
    def test_parse_confirmation_prompt(self):
        """Test parse confirmation prompt"""
        user_input = "Yes, that looks good"
        prompt = CopilotPrompts.parse_confirmation_prompt(user_input)
        
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert user_input in prompt
        assert "confirmed" in prompt.lower()
        assert "modify" in prompt.lower()
        assert "quit" in prompt.lower()
    
    def test_generate_response_prompt(self):
        """Test generate response prompt"""
        user_profile = {
            "personality_traits": {"analytical": 0.8},
            "interests": ["AI"],
            "communication_style": {"formal": True},
            "expertise_areas": ["Python"]
        }
        examples = [
            {
                "platform": "LinkedIn",
                "title": "AI Development",
                "content": "Working on AI projects",
                "response": "Great work on AI!",
                "timestamp": "2024-01-01T00:00:00"
            }
        ]
        platform = "LinkedIn"
        title = "AI Development"
        content = "Working on AI projects"
        
        prompt = CopilotPrompts.generate_response_prompt(
            user_profile, examples, platform, title, content
        )
        
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert "<user_profile>" in prompt
        assert "<examples>" in prompt
        assert "<inputs>" in prompt
        assert "<response>" in prompt
        assert "<evaluation>" in prompt
        assert platform in prompt
        assert title in prompt
        assert content in prompt
    
    def test_generate_response_prompt_empty_inputs(self):
        """Test generate response prompt with empty inputs"""
        user_profile = {}
        examples = []
        platform = ""
        title = ""
        content = ""
        
        prompt = CopilotPrompts.generate_response_prompt(
            user_profile, examples, platform, title, content
        )
        
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert "<user_profile>" in prompt
        assert "<examples>" in prompt
        assert "<inputs>" in prompt
        assert "<response>" in prompt
        assert "<evaluation>" in prompt
    
    def test_all_static_methods(self):
        """Test that all methods are static and return expected types"""
        # Test that methods can be called without instantiation
        greeting = CopilotPrompts.greeting()
        assert isinstance(greeting, str)
        
        confirmation = CopilotPrompts.confirmation_message("test", "test", "test")
        assert isinstance(confirmation, str)
        
        success_msg = CopilotPrompts.confirmation_success_message()
        assert isinstance(success_msg, str)
        
        modify_msg = CopilotPrompts.modification_request_message()
        assert isinstance(modify_msg, str)
        
        default_msg = CopilotPrompts.default_confirmation_message()
        assert isinstance(default_msg, str)
        
        missing_msg = CopilotPrompts.ask_for_missing_fields(["test"])
        assert isinstance(missing_msg, str)
        
        parse_prompt = CopilotPrompts.parse_post_info_prompt("test")
        assert isinstance(parse_prompt, str)
        
        quit_prompt = CopilotPrompts.detect_quit_intent_prompt("test")
        assert isinstance(quit_prompt, str)
        
        confirm_prompt = CopilotPrompts.parse_confirmation_prompt("test")
        assert isinstance(confirm_prompt, str)
        
        response_prompt = CopilotPrompts.generate_response_prompt({}, [], "", "", "")
        assert isinstance(response_prompt, str)
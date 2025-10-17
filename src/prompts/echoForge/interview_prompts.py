"""
EchoForge Interview Mode Prompts
"""
from typing import Dict, List, Any


class InterviewPrompts:
    """Prompt templates for EchoForge interview mode"""
    
    @staticmethod
    def generate_interview_question(profile: Dict[str, Any], conversation_history: List[Dict[str, str]]) -> str:
        """Generate a proactive interview question"""
        print("[INTERVIEW_PROMPTS] Generating interview question...")
        # TODO: Implement LLM call to generate questions
        return "What's something you're passionate about that I don't know yet?"
    
    @staticmethod
    def summarize_qa_interaction(question: str, answer: str) -> Dict[str, Any]:
        """Summarize Q&A interaction to extract insights"""
        print(f"[INTERVIEW_PROMPTS] Summarizing Q&A: {question[:30]}...")
        # TODO: Implement LLM call to extract insights
        return {
            "insights": ["User seems interested in technology"],
            "traits": ["curious", "thoughtful"],
            "topics": ["AI", "programming"]
        }
    
    @staticmethod
    def generate_confirmation_question(original_response: str, user_feedback: str) -> str:
        """Generate confirmation question for low-confidence responses"""
        print("[INTERVIEW_PROMPTS] Generating confirmation question...")
        # TODO: Implement confirmation question generation
        return f"I responded: '{original_response}'. Is this how you would have answered?"
    
    @staticmethod
    def extract_personality_traits(conversations: List[Dict[str, str]]) -> List[str]:
        """Extract personality traits from conversation history"""
        print("[INTERVIEW_PROMPTS] Extracting personality traits...")
        # TODO: Implement trait extraction
        return ["analytical", "creative", "direct"]

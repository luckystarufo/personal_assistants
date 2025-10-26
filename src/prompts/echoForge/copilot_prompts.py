"""
EchoForge Prompt Builder
"""
from typing import Dict, List, Any
import json


class EchoForgePrompts:
    """Prompt builder for EchoForge agent"""
    
    @staticmethod
    def greeting() -> str:
        """Initial greeting message"""
        return """Hello! I'm ready to help you craft a response. Please provide me with:
1. Platform (e.g., LinkedIn, Twitter, Reddit)
2. Title of the post
3. Content of the post

You can provide all three at once or one by one."""
    
    @staticmethod
    def confirmation_message(platform: str, title: str, content: str) -> str:
        """Confirmation message with extracted information"""
        return f"""Here's what I understood:

Platform: {platform}
Title: {title}
Content: {content}

Does this look correct? [Y/N]."""
    
    @staticmethod
    def confirmation_success_message() -> str:
        """Message when user confirms the post information"""
        return "Great! I've confirmed your post information. Let me generate a response for you."
    
    @staticmethod
    def modification_request_message() -> str:
        """Message when user wants to modify the post information"""
        return "I understand you'd like to make modifications. Please provide the new platform, title, and content information. You can provide all three at once or one by one."
    
    @staticmethod
    def default_confirmation_message() -> str:
        """Default message when user response is unclear"""
        return "I'll proceed with the current information. Let me generate a response for you."
    
    @staticmethod
    def ask_for_missing_fields(missing_fields: List[str]) -> str:
        """Ask user to provide missing fields"""
        if len(missing_fields) == 0:
            return "All required fields have been provided."
        elif len(missing_fields) == 1:
            field_name = missing_fields[0].title()
            return f"I still need the {field_name.lower()} of your post. Please provide it."
        elif len(missing_fields) == 2:
            fields_str = " and ".join([f.lower() for f in missing_fields])
            return f"I still need the {fields_str} of your post. Please provide them."
        else:
            fields_str = ", ".join([f.lower() for f in missing_fields[:-1]]) + f", and {missing_fields[-1].lower()}"
            return f"I still need the {fields_str} of your post. Please provide them."
    
    @staticmethod
    def parse_post_info_prompt(user_input: str) -> str:
        """Prompt for parsing user input to extract post information"""
        return f"""Parse the following user input and extract the platform, title, and content information. 
Return your response in JSON format with keys: "platform", "title", "content". 
If any information is missing, use empty string for that key.

User input: {user_input}

JSON response:"""
    
    @staticmethod
    def detect_quit_intent_prompt(user_input: str) -> str:
        """Prompt for detecting quit/exit intent"""
        return f"""Analyze the following user input to determine if they want to quit or exit the conversation.
Look for words like: quit, exit, goodbye, bye, stop, end, leave, etc.

User input: {user_input}

Respond with only "QUIT" if they want to exit, or "CONTINUE" if they want to keep going."""
    
    @staticmethod
    def parse_confirmation_prompt(user_input: str) -> str:
        """Prompt for parsing user confirmation response"""
        return f"""Analyze the following user input to determine their intent:
- If they confirm/approve (yes, correct, looks good, etc.), respond with "CONFIRMED"
- If they want changes (no, change, modify, etc.), respond with "MODIFY"  
- If they want to quit (quit, exit, goodbye, etc.), respond with "QUIT"

User input: {user_input}

Respond with only: CONFIRMED, MODIFY, or QUIT"""
    
    @staticmethod
    def generate_response_prompt(user_profile: Dict[str, Any], examples: List[Dict[str, str]], platform: str, title: str, content: str) -> str:
        """Main prompt for generating response and evaluation"""
        return f"""<user_profile>
{user_profile}
</user_profile>

<examples>
{examples}
</examples>

<inputs>
Platform: {platform}
Title: {title}
Content: {content}
</inputs>

Based on the user profile and examples above, generate a response that matches the user's communication style, personality, and expertise. Then evaluate your response.

IMPORTANT: You MUST use the exact XML format below with proper opening and closing tags:

<response>
[Your response here]
</response>

<evaluation>
Provide a brief evaluation (2-3 sentences max) covering:
1. Confidence level (High/Medium/Low)
2. Main source of response (Profile/Examples/LLM-generated)
3. Any concerns or limitations

Examples: 
1. "High confidence. Response draws primarily from user profile communication style and similar examples. No major concerns."
2. "Medium confidence. Response draws from profile but no similar examples. Some concerns about the contents."
3. "Low confidence. Response is heavily LLM-generated with no clear user profile influence. Concerns about appropriateness."
</evaluation>

CRITICAL: Ensure both <response> and <evaluation> sections have proper opening AND closing tags."""
    
    @staticmethod
    def build_echo_prompt(context: str, title: str, content: str, user_profile: dict, relevant_notes: list) -> str:
        """Build the prompt for echo mode response generation"""
        
        # Format user profile
        profile_text = json.dumps(user_profile, indent=2) if user_profile else "No profile data available"
        
        # Format relevant notes
        notes_text = "\n\n".join([
            f"Platform: {note.get('platform', 'Unknown')}\nTitle: {note.get('title', 'N/A')}\nContent: {note.get('content', 'N/A')}\nYour Response: {note.get('response', 'N/A')}"
            for note in relevant_notes
        ]) if relevant_notes else "No relevant historical examples found."
        
        prompt = f"""You are responding as if you were the user. Generate a response that matches their communication style, tone, knowledge, values, and preferences.

Context: {context}
Title: {title}
Content: {content}

User Profile:
{profile_text}

Relevant Historical Examples:
{notes_text}

Based on the context, title, and content above, generate a response that:
1. Matches the user's communication style and tone
2. Reflects their knowledge and expertise areas
3. Aligns with their values and preferences
4. Is appropriate for the given context

Response:"""
        
        return prompt

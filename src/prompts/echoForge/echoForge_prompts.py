"""
EchoForge Prompt Builder
"""
from typing import Dict, List, Any
import json


class EchoForgePrompts:
    """Prompt builder for EchoForge agent"""
    

    
    @staticmethod
    def intent_gathering_system_prompt() -> str:
        """System prompt for intent gathering mini-ReAct agent"""
        return """You are helping the user determine what they want to work on today.

The user has two options:
1. Provide new post information (context, title, content) to work on
2. Fetch and work on an existing post on records

Your goal is to politely ask the user what they would like to do today and guide them back to the task if they go off topic.

IMPORTANT: 
- If the user's response is unclear, irrelevant, or doesn't clearly indicate one of these two options, you MUST use the ask_human tool again to seek clarification. Do not guess or assume.
- Keep asking until you get a clear answer about which option they want.
- After the user gives a clear response, your final message MUST include one of these option tags:
  * Include "OPTION_1" or keywords like "provide/new/post/information" (if they want to provide new info)
  * Include "OPTION_2" or keywords like "fetch/existing/records" (if they want to fetch existing posts)
  * Include "OPTION_3" or keywords like "exit/quit/stop" (if they want to exit/quit/stop)

Keep conversations brief and focused. Always use the ask_human tool to ask the user questions."""
    
    @staticmethod
    def exit_message() -> str:
        """Exit message when user wants to quit"""
        return "Thank you for using EchoForge! Have a great day!"
    
    @staticmethod
    def collect_post_info_system_prompt() -> str:
        """System prompt for collecting post information"""
        return """You are helping the user provide post information.

The user needs to provide:
1. Context - the platform or context (e.g., "LinkedIn post to tech professionals", "Twitter thread about AI", "Discord in the Helium community")
2. Title - the title or subject of the post
3. Content - the actual post content

Your goal is to politely ask for any missing fields and guide the user back to the task if they go off topic.

RULES:
1. Use ask_human tool to ask for any missing required fields (context, title, content). Keep asking until you have all three fields - do not proceed without all information.
2. Once you have all fields, ask the user to confirm with "Let me confirm the information I collected:" followed by listing the fields.
3. After user confirms (not before!), append ONE final AI message with all collected information in JSON format. The message should start with "COLLECTED_INFO:" followed by JSON like:
   {"context": "...", "title": "...", "content": "..."}
4. If at any point the user wants to exit/quit/stop, the conversation should stop immediately. Your final message should simply state: "User wants to exit"

Keep the conversation focused and use the ask_human tool for all user interactions."""
    
    @staticmethod
    def build_echo_prompt(context: str, title: str, content: str, user_profile: dict, relevant_notes: list) -> str:
        """Build the prompt for echo mode response generation"""
        
        # Format user profile with XML-style tags
        if user_profile:
            profile_lines = []
            for key, value in user_profile.items():
                if isinstance(value, dict):
                    profile_lines.append(f"<{key}>")
                    for sub_key, sub_value in value.items():
                        profile_lines.append(f"  <{sub_key}>{sub_value}</{sub_key}>")
                    profile_lines.append(f"</{key}>")
                elif isinstance(value, list):
                    profile_lines.append(f"<{key}>")
                    for item in value:
                        profile_lines.append(f"  <item>{item}</item>")
                    profile_lines.append(f"</{key}>")
                else:
                    profile_lines.append(f"<{key}>{value}</{key}>")
            profile_text = "\n".join(profile_lines)
        else:
            profile_text = "<no_profile>No profile data available</no_profile>"
        
        # Format relevant notes with XML-style tags and numbering
        if relevant_notes:
            notes_text = ""
            for i, note in enumerate(relevant_notes, 1):
                notes_text += f"<example_{i}>\n"
                notes_text += f"  <context>{note.get('context', 'Unknown')}</context>\n"
                notes_text += f"  <title>{note.get('title', 'N/A')}</title>\n"
                notes_text += f"  <content>{note.get('content', 'N/A')}</content>\n"
                notes_text += f"  <human_response>{note.get('human_response', 'N/A')}</human_response>\n"
                if note.get('similarity_dist'):
                    notes_text += f"  <similarity_dist>{note.get('similarity_dist')}</similarity_dist>\n"
                notes_text += f"</example_{i}>\n\n"
        else:
            notes_text = "<no_examples>No relevant historical examples found.</no_examples>"
        
        prompt = f"""You are responding as if you were the user. Generate a response that matches their communication style, tone, knowledge, values, and preferences.

<current_post>
  <context>{context}</context>
  <title>{title}</title>
  <content>{content}</content>
</current_post>

<user_profile>
{profile_text}
</user_profile>

<relevant_historical_examples>
{notes_text}</relevant_historical_examples>

Based on the context, title, and content above, generate a response that:
1. Matches the user's communication style and tone
2. Reflects their knowledge and expertise areas
3. Aligns with their values and preferences
4. Is appropriate for the given context

Response:"""
        
        return prompt

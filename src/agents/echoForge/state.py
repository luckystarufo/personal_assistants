"""
EchoForge Agent State Schema
"""
from typing import Dict
from langgraph.graph.message import MessagesState
from pydantic import BaseModel, Field


class PostSchema(BaseModel):
    """Schema for parsing post information - EXTRACT ONLY what is explicitly provided, do not add or infer additional content"""
    context: str = Field(
        default="", 
        description="EXTRACT ONLY the context or platform mentioned in the text. If not mentioned, leave empty. Do not infer."
    )
    title: str = Field(
        default="", 
        description="EXTRACT ONLY the title/subject explicitly stated in the text. If not provided, leave empty. Do not infer."
    )
    content: str = Field(
        default="", 
        description="EXTRACT ONLY the actual post content provided. If not provided, leave empty. Do not infer."
    )


class EchoForgeState(MessagesState):
    """Main state schema for EchoForge agent"""
    pass


class EchoModeState(MessagesState):
    """State schema for Echo mode"""
    post_info: Dict[str, str]  # context, title, content
    ai_response: str
    ai_evaluation: str
    human_response: str  # User's preferred response
    reflections: str  # Notes on differences between AI and human responses
    status: str  # Track current status: "collect", "fetch", "exit", "continue", "confirm"

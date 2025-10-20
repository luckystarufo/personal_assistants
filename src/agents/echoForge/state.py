"""
EchoForge Agent State Schema
"""
from typing import Dict
from langgraph.graph.message import MessagesState
from pydantic import BaseModel, Field


class PostSchema(BaseModel):
    """Schema for parsing post information - EXTRACT ONLY what is explicitly provided, do not add or infer additional content"""
    platform: str = Field(
        default="", 
        description="EXTRACT ONLY the platform mentioned in the text (LinkedIn, Twitter, Reddit, etc.). If not mentioned, leave empty. Do not infer."
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
    post_info: Dict[str, str]  # platform, title, content
    ai_response: str
    ai_evaluation: str
    status: str  # Track current status: "continue", "confirm", "exit"

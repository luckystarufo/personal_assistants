"""
EchoForge Agent State Schema
"""
from typing import TypedDict, Optional, List, Dict, Any


class EchoForgeState(TypedDict):
    """Main state schema for EchoForge agent"""
    # Mode control
    learning_mode: bool  # True = Learning, False = Echo
    
    # User interaction
    user_input: str
    response: str
    
    # Memory and context
    user_profile: Dict[str, Any]
    conversation_history: List[Dict[str, str]]
    session_context: Dict[str, Any]
    
    # Learning mode specific
    learning_targets: List[str]
    current_question: Optional[str]
    
    # Echo mode specific
    confidence_score: float
    confirmation_needed: bool
    
    # Metadata
    timestamp: str
    session_id: str

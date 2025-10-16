"""
EchoForge Agent State Schema
"""
from typing import TypedDict, List, Dict


class EchoForgeState(TypedDict):
    """Main state schema for EchoForge agent"""
    messages: List[Dict[str, str]]

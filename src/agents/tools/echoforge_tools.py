"""
EchoForge Tools
"""
from typing import Annotated
from langchain_core.tools import tool


@tool
def ask_human(question: Annotated[str, "The question to ask the user"]) -> str:
    """
    Ask the user a question and wait for their response.
    """
    # Print the question
    print(f"[Agent]: {question}")
    
    # Get user response
    response = input("[Your response]: ")
    
    return response

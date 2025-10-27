"""
EchoForge Demo Script
"""
import os
from dotenv import load_dotenv
from src.agents.echoForge.agent import EchoForgeAgent


def main():
    """Demo script for EchoForge agent"""
    # Load environment variables from .env file
    load_dotenv()
    
    print("=== EchoForge Agent Demo ===")
    
    agent = EchoForgeAgent()
    
    # # Example: Using echo function
    # context = "LinkedIn"
    # title = "Discussion about AI Ethics"
    # content = "What are your thoughts on the ethical implications of AI in healthcare?"
    
    # print("=== call echo function ===")
    # print(f"\nUsing echo function with context: {context}")
    # print(f"Title: {title}")
    # print(f"Content: {content}\n")
    
    # response = agent.echo(context, title, content)
    # print(f"Response: {response}\n")

    # # pause
    # input("Press Enter to continue...")
    
    # Run chat interface
    print("=== call chat function ===")
    agent.chat()
    
    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    main()

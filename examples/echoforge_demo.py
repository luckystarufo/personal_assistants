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
    agent.chat()
    
    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    main()

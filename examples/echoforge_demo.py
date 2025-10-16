"""
EchoForge Demo Script
"""
from src.agents.echoForge.agent import EchoForgeAgent


def main():
    """Demo script for EchoForge agent"""
    print("=== EchoForge Agent Demo ===")
    
    # Initialize agent
    agent = EchoForgeAgent()
    agent.chat()
    
    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    main()

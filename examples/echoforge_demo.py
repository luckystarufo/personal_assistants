"""
EchoForge Demo Script
"""
from src.agents.echoForge.agent import EchoForgeAgent


def main():
    """Demo script for EchoForge agent"""
    print("=== EchoForge Agent Demo ===")
    
    # Initialize agent
    agent = EchoForgeAgent()
    
    # Demo learning mode
    print("\n--- Learning Mode Demo ---")
    response = agent.chat("Hello!", learning_mode=True)
    print(f"Agent: {response}")
    
    # Demo echo mode
    print("\n--- Echo Mode Demo ---")
    response = agent.chat("What do you think about AI?", learning_mode=False)
    print(f"Agent: {response}")
    
    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    main()

"""
EchoForge Demo Script
"""
from src.agents.echoForge.agent import EchoForgeAgent


def main():
    """Demo script for EchoForge agent"""
    print("=== EchoForge Agent Demo ===")
    
    # Demo interview mode
    print("\n--- Interview Mode Demo ---")
    agent = EchoForgeAgent()
    agent.config.mode = "interview"
    agent.graph = agent._build_interview_graph()  # Rebuild graph with new mode
    response = agent.chat()
    print(f"Agent: {response}")
    
    # Demo instruct mode
    print("\n--- Instruct Mode Demo ---")
    agent.config.mode = "instruct"
    agent.graph = agent._build_instruct_graph()  # Rebuild graph with new mode
    response = agent.chat()
    print(f"Agent: {response}")
    
    # Demo echo mode
    print("\n--- Echo Mode Demo ---")
    agent.config.mode = "echo"
    agent.graph = agent._build_echo_graph()  # Rebuild graph with new mode
    response = agent.chat()
    print(f"Agent: {response}")
    
    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    main()

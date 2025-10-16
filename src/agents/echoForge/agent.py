"""
EchoForge Main Agent Class
"""
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from .state import EchoForgeState
from .config import EchoForgeConfig
from .memory import EchoForgeMemory
from src.prompts.echoForge.learning_prompts import LearningPrompts
from src.prompts.echoForge.echo_prompts import EchoPrompts


class EchoForgeAgent:
    """Main EchoForge agent with LangGraph integration"""
    
    def __init__(self, config_path: str = "config/echoforge.yaml"):
        self.config = EchoForgeConfig.from_file(config_path)
        self.memory = EchoForgeMemory(self.config.data_dir)
        self.learning_prompts = LearningPrompts()
        self.echo_prompts = EchoPrompts()
        self.graph = self._build_learning_graph() if self.config.learning_mode else self._build_echo_graph()
    
    def _build_learning_graph(self) -> StateGraph:
        """Build LangGraph workflow for learning mode"""
        print("[AGENT] Building learning graph...")
        workflow = StateGraph(EchoForgeState)
        
        # Add dummy node for testing
        workflow.add_node("dummy_learning_node", self._dummy_learning_node)
        workflow.add_edge("dummy_learning_node", END)
        workflow.set_entry_point("dummy_learning_node")
        
        return workflow.compile()
    
    def _build_echo_graph(self) -> StateGraph:
        """Build LangGraph workflow for echo mode"""
        print("[AGENT] Building echo graph...")
        workflow = StateGraph(EchoForgeState)
        
        # Add dummy node for testing
        workflow.add_node("dummy_echo_node", self._dummy_echo_node)
        workflow.add_edge("dummy_echo_node", END)
        workflow.set_entry_point("dummy_echo_node")
        
        return workflow.compile()
    
    def _dummy_learning_node(self, state: EchoForgeState) -> EchoForgeState:
        """Dummy learning node for testing"""
        print("[AGENT] Executing dummy learning node...")
        state["messages"] = [{"role": "assistant", "content": "Hello! I'm in learning mode. Ready to learn about you!"}]
        return state
    
    def _dummy_echo_node(self, state: EchoForgeState) -> EchoForgeState:
        """Dummy echo node for testing"""
        print("[AGENT] Executing dummy echo node...")
        state["messages"] = [{"role": "assistant", "content": "Hello! I'm in echo mode. Ready to respond as you!"}]
        return state
    
    def chat(self) -> str:
        """Main chat interface - agent initiates conversation"""
        print(f"[AGENT] Starting chat - Mode: {'Learning' if self.config.learning_mode else 'Echo'}")
        
        # Create initial state with empty messages
        initial_state: EchoForgeState = {
            "messages": []
        }
        
        # Run graph
        result = self.graph.invoke(initial_state)
        
        # Return the last message content
        if result["messages"]:
            return result["messages"][-1]["content"]
        else:
            return "No response generated"
    
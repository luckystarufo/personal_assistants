"""
EchoForge Main Agent Class
"""
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from .state import EchoForgeState
from .config import EchoForgeConfig
from .memory import EchoForgeMemory
from src.prompts.echoForge.interview_prompts import InterviewPrompts
from src.prompts.echoForge.instruct_prompts import InstructPrompts
from src.prompts.echoForge.echo_prompts import EchoPrompts


class EchoForgeAgent:
    """Main EchoForge agent with LangGraph integration"""
    
    def __init__(self, config_path: str = "config/echoforge.yaml"):
        self.config = EchoForgeConfig.from_file(config_path)
        self.memory = EchoForgeMemory(self.config.data_dir)
        self.interview_prompts = InterviewPrompts()
        self.instruct_prompts = InstructPrompts()
        self.echo_prompts = EchoPrompts()
        self.graph = self._build_interview_graph() if self.config.mode == "interview" else \
                     self._build_instruct_graph() if self.config.mode == "instruct" else \
                     self._build_echo_graph()
    
    def _build_interview_graph(self) -> StateGraph:
        """Build LangGraph workflow for interview mode"""
        print("[AGENT] Building interview graph...")
        workflow = StateGraph(EchoForgeState)
        
        # Add dummy node for testing
        workflow.add_node("dummy_interview_node", self._dummy_interview_node)
        workflow.add_edge("dummy_interview_node", END)
        workflow.set_entry_point("dummy_interview_node")
        
        return workflow.compile()
    
    def _build_instruct_graph(self) -> StateGraph:
        """Build LangGraph workflow for instruct mode"""
        print("[AGENT] Building instruct graph...")
        workflow = StateGraph(EchoForgeState)
        
        # Add dummy node for testing
        workflow.add_node("dummy_instruct_node", self._dummy_instruct_node)
        workflow.add_edge("dummy_instruct_node", END)
        workflow.set_entry_point("dummy_instruct_node")
        
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
    
    def _dummy_interview_node(self, state: EchoForgeState) -> EchoForgeState:
        """Dummy interview node for testing"""
        print("[AGENT] Executing dummy interview node...")
        state["messages"] = [{"role": "assistant", "content": "Hello! I'm in interview mode. Ready to learn about you!"}]
        return state
    
    def _dummy_instruct_node(self, state: EchoForgeState) -> EchoForgeState:
        """Dummy instruct node for testing"""
        print("[AGENT] Executing dummy instruct node...")
        state["messages"] = [{"role": "assistant", "content": "Hello! I'm in instruct mode. Ready to follow your instructions!"}]
        return state
    
    def _dummy_echo_node(self, state: EchoForgeState) -> EchoForgeState:
        """Dummy echo node for testing"""
        print("[AGENT] Executing dummy echo node...")
        state["messages"] = [{"role": "assistant", "content": "Hello! I'm in echo mode. Ready to respond as you!"}]
        return state
    
    def chat(self) -> str:
        """Main chat interface - agent initiates conversation"""
        print(f"[AGENT] Starting chat - Mode: {self.config.mode}")
        
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
    
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
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build LangGraph workflow"""
        print("[AGENT] Building LangGraph workflow...")
        
        workflow = StateGraph(EchoForgeState)
        
        # Add nodes
        workflow.add_node("process_turn", self._process_turn)
        workflow.add_node("learning_mode", self._learning_mode)
        workflow.add_node("echo_mode", self._echo_mode)
        
        # Add edges
        workflow.add_edge("process_turn", "learning_mode", self._is_learning_mode)
        workflow.add_edge("process_turn", "echo_mode", self._is_echo_mode)
        workflow.add_edge("learning_mode", END)
        workflow.add_edge("echo_mode", END)
        
        workflow.set_entry_point("process_turn")
        
        return workflow.compile()
    
    def _process_turn(self, state: EchoForgeState) -> EchoForgeState:
        """Process user input and determine mode"""
        print(f"[AGENT] Processing turn in {'Learning' if state['learning_mode'] else 'Echo'} mode")
        return state
    
    def _learning_mode(self, state: EchoForgeState) -> EchoForgeState:
        """Handle learning mode interactions"""
        print("[AGENT] Executing learning mode...")
        
        # Generate learning question
        question = self.learning_prompts.generate_learning_question(
            state["user_profile"], 
            state["conversation_history"]
        )
        
        # Store interaction
        self.memory.store_learning_conversation(question, state["user_input"])
        
        # Update state
        state["response"] = question
        state["current_question"] = question
        
        return state
    
    def _echo_mode(self, state: EchoForgeState) -> EchoForgeState:
        """Handle echo mode interactions"""
        print("[AGENT] Executing echo mode...")
        
        # Get relevant context
        context = self.memory.get_relevant_context(state["user_input"])
        
        # Generate echo response
        response = self.echo_prompts.generate_echo_response(
            state["user_input"],
            state["user_profile"],
            context
        )
        
        # Calculate confidence
        confidence = self.echo_prompts.calculate_confidence(
            state["user_input"],
            response,
            state["user_profile"]
        )
        
        # Store interaction
        self.memory.store_echo_conversation(
            state["user_input"],
            response,
            confidence
        )
        
        # Update state
        state["response"] = response
        state["confidence_score"] = confidence
        state["confirmation_needed"] = confidence < self.config.confidence_threshold
        
        return state
    
    def _is_learning_mode(self, state: EchoForgeState) -> bool:
        """Check if in learning mode"""
        return state["learning_mode"]
    
    def _is_echo_mode(self, state: EchoForgeState) -> bool:
        """Check if in echo mode"""
        return not state["learning_mode"]
    
    def chat(self, user_input: str, learning_mode: bool = False) -> str:
        """Main chat interface"""
        print(f"[AGENT] Starting chat - Mode: {'Learning' if learning_mode else 'Echo'}")
        
        # Load user profile
        user_profile = self.memory.load_user_profile()
        
        # Create initial state
        initial_state: EchoForgeState = {
            "learning_mode": learning_mode,
            "user_input": user_input,
            "response": "",
            "user_profile": user_profile,
            "conversation_history": [],
            "session_context": {},
            "learning_targets": [],
            "current_question": None,
            "confidence_score": 0.0,
            "confirmation_needed": False,
            "timestamp": "",
            "session_id": ""
        }
        
        # Run graph
        result = self.graph.invoke(initial_state)
        
        # Save updated profile
        self.memory.save_user_profile(result["user_profile"])
        
        return result["response"]

"""
EchoForge Main Agent Class
"""
from typing import Dict, Any
import uuid
import json
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from .state import EchoForgeState, EchoModeState, PostSchema
from .config import EchoForgeConfig
from .memory import EchoForgeMemory
from src.prompts.echoForge.echoForge_prompts import EchoForgePrompts
from src.agents.tools import ask_human
from langgraph.prebuilt import create_react_agent


class EchoForgeAgent:
    """Main EchoForge agent with LangGraph integration"""
    
    def __init__(self, config_path: str = "config/echoforge.yaml"):
        self.config = EchoForgeConfig.from_file(config_path)
        self.memory = EchoForgeMemory(self.config.data_dir)
        self.prompt_builder = EchoForgePrompts()
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=self.config.llm_model,
            temperature=self.config.llm_temperature
        )
        
        # Build graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build LangGraph workflow"""
        workflow = StateGraph(EchoModeState)
        
        # Add nodes
        workflow.add_node("gather_intent", self._gather_intent_node)
        workflow.add_node("collect_post_info", self._collect_post_info_node)
        workflow.add_node("echo", self._echo_node)
        workflow.add_node("fetch_from_history", self._fetch_from_history_node)
        workflow.add_node("handle_exit", self._handle_exit_node)
        
        # Add conditional edges from gather_intent
        workflow.add_conditional_edges(
            "gather_intent",
            self._route_by_intent,
            {
                "collect": "collect_post_info",
                "fetch": "fetch_from_history",
                "exit": "handle_exit"
            }
        )
        
        # Add conditional edges from collect_post_info
        workflow.add_conditional_edges(
            "collect_post_info",
            self._route_by_collect_status,
            {
                "echo": "echo",  # Route to echo node
                "exit": "handle_exit"  # User wants to exit
            }
        )
        
        # Echo node ends after generating response
        workflow.add_edge("echo", END)
        
        # Other paths end after their nodes
        workflow.add_edge("fetch_from_history", END)
        workflow.add_edge("handle_exit", END)
        
        # Set entry point
        workflow.set_entry_point("gather_intent")
        
        return workflow.compile(checkpointer=self.memory.memory_saver)
    
    def _gather_intent_node(self, state: EchoModeState) -> EchoModeState:
        """Mini-ReAct node to determine user's intent"""
        
        # Get system prompt from prompt builder
        system_prompt = self.prompt_builder.intent_gathering_system_prompt()
        
        # Add system message if not present
        if not any(
            (isinstance(msg, dict) and msg.get("role") == "system") or 
            (hasattr(msg, "role") and msg.role == "system")
            for msg in state.get("messages", [])
        ):
            state["messages"].insert(0, AIMessage(content=system_prompt, name="EchoForge"))
        
        # Create mini ReAct agent inline
        tools = [ask_human]
        mini_agent = create_react_agent(self.llm, tools)
        
        # Run the mini agent with the same thread config from memory
        result = mini_agent.invoke({"messages": state["messages"]}, config=self.memory.get_config())
        
        # Update state with new messages
        state["messages"] = result.get("messages", state["messages"])
        
        # Determine intent from the last AI message only
        last_assistant_msg = None
        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage) or (isinstance(msg, dict) and msg.get("role") == "assistant"):
                last_assistant_msg = msg.content if hasattr(msg, 'content') else msg.get("content", "")
                break
        
        print(f"[Agent]: {last_assistant_msg}")

        # Detect status from AI's final summary message
        if last_assistant_msg:
            summary_text = last_assistant_msg.lower()
            
            # Check for OPTION tags or keywords
            if "option_1" in summary_text or "provide" in summary_text or "new" in summary_text:
                state["status"] = "collect"
                print("[Status]: Proceeding to collect post information")
            elif "option_2" in summary_text or "fetch" in summary_text or "existing" in summary_text:
                state["status"] = "fetch"
                print("[Status]: Proceeding to fetch from records")
            elif "option_3" in summary_text or "exit" in summary_text or "quit" in summary_text or "stop" in summary_text:
                state["status"] = "exit"
                print("[Status]: Proceeding to exit")
            else:
                # Default to exit if unclear
                state["status"] = "exit"
                print("[Status]: Unclear intent - proceeding to exit")
        else:
            # No AI message yet, default to exit
            state["status"] = "exit"
            print("[Status]: Unclear intent - proceeding to exit")
        
        return state
    
    def _route_by_intent(self, state: EchoModeState) -> str:
        """Route based on status"""
        status = state.get("status", "exit")
        
        if status == "fetch":
            return "fetch"
        elif status == "collect":
            return "collect"
        else:
            return "exit"
    
    def _route_by_collect_status(self, state: EchoModeState) -> str:
        """Route based on collection status"""
        status = state.get("status", "exit")
        
        if status == "echo":
            return "echo"
        else:
            return "exit"
    
    def _collect_post_info_node(self, state: EchoModeState) -> EchoModeState:
        """Mini-ReAct node to collect post information (context, title, content)"""
        
        # Get system prompt from prompt builder
        system_prompt = self.prompt_builder.collect_post_info_system_prompt()
        
        # Add system message if not present
        if not any(
            (isinstance(msg, dict) and msg.get("role") == "system") or 
            (hasattr(msg, "role") and msg.role == "system")
            for msg in state.get("messages", [])
        ):
            state["messages"].insert(0, AIMessage(content=system_prompt, name="EchoForge"))
        
        # Create mini ReAct agent inline
        tools = [ask_human]
        mini_agent = create_react_agent(self.llm, tools)
        
        # Run the mini agent with the same thread config from memory
        result = mini_agent.invoke({"messages": state["messages"]}, config=self.memory.get_config())
        
        # Update state with new messages
        state["messages"] = result.get("messages", state["messages"])
        
        # Extract information from the conversation and parse post info
        # Look at the last assistant message for confirmation status
        last_assistant_msg = None
        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage) or (isinstance(msg, dict) and msg.get("role") == "assistant"):
                last_assistant_msg = msg.content if hasattr(msg, 'content') else msg.get("content", "")
                break
        
        print(f"[Agent]: {last_assistant_msg}")
        
        # Check if user wants to exit or if confirmed
        if last_assistant_msg:
            summary_text = last_assistant_msg.lower()
            
            if "exit" in summary_text or "quit" in summary_text or "stop" in summary_text:
                state["status"] = "exit"
                print("[Status]: User wants to exit")
            elif "collected_info:" in last_assistant_msg.lower():
                # Parse data using LLM structured output
                try:
                    # Use LLM with structured output to parse post information
                    parsed_post = self.llm.with_structured_output(PostSchema).invoke(last_assistant_msg)
                    
                    # Update post_info in state
                    state["post_info"]["context"] = parsed_post.context
                    state["post_info"]["title"] = parsed_post.title
                    state["post_info"]["content"] = parsed_post.content
                                        
                    state["status"] = "echo"
                    print("[Status]: Ready to echo response")
                except Exception as e:
                    print(f"[Status]: Error parsing collected info: {e}")
                    state["status"] = "exit"
            else:
                state["status"] = "exit"
                print("[Status]: No collected info found - exiting")
        else:
            state["status"] = "exit"
            print("[Status]: No message found - exiting")
        
        return state
    
    def _echo_node(self, state: EchoModeState) -> EchoModeState:
        """Generate echo response using the collected post information"""
        
        # Get post information from state
        post_info = state.get("post_info", {})
        context = post_info.get("context", "")
        title = post_info.get("title", "")
        content = post_info.get("content", "")
        
        # Generate echo response using the echo function
        response = self.echo(context, title, content)
        
        # Add response to messages
        state["messages"].append(AIMessage(content=response, name="EchoForge"))
        
        # Print the response
        print(f"[Agent]: {response}")
        
        return state
    
    def _fetch_from_history_node(self, state: EchoModeState) -> EchoModeState:
        """Fetch a post from history"""
        # This is a placeholder - implement logic to fetch from history
        return state
    
    def _handle_exit_node(self, state: EchoModeState) -> EchoModeState:
        """Handle quit/exit scenarios"""
        
        exit_message = self.prompt_builder.exit_message()
        state["messages"].append(AIMessage(content=exit_message, name="EchoForge"))
        
        # Print in the same format as other nodes
        print(f"[Agent]: {exit_message}")
        
        return state
    
    def echo(self, context: str, title: str, content: str) -> str:
        """
        Echo mode function: generates a response based on context, title, and content.
        
        Args:
            context: The platform/context (e.g., "LinkedIn", "Twitter", etc.)
            title: The title of the post
            content: The content of the post
        
        Returns:
            A response string that mimics the user's communication style
        """
        # Get user profile
        user_profile = self.memory.get_user_profile()
        
        # Build query string for vector store search with proper formatting
        query = f"<context>{context}</context>\n<title>{title}</title>\n<content>{content}</content>".strip()
        
        # Get relevant notes from vector store (top 3)
        relevant_notes = self.memory.get_relevant_context(query, limit=3)
        
        # Build the prompt with all 5 parts
        prompt = self.prompt_builder.build_echo_prompt(context, title, content, user_profile, relevant_notes)
        
        # Generate and return the response
        response = self.llm.invoke(prompt).content
        return response
    
    def chat(self) -> str:
        """Main chat interface - agent initiates conversation"""
        
        # Create or get thread config from memory
        config = self.memory.create_or_get_config()
        
        # Create initial state
        initial_state: EchoModeState = {
            "messages": [],
            "post_info": {"context": "", "title": "", "content": ""},
            "ai_response": "",
            "ai_evaluation": "",
            "human_response": "",
            "reflections": "",
            "status": ""  # Will be set by gather_intent_node: "collect", "fetch", or "exit"
        }
        
        # Just stream and print all messages
        try:
            # Stream the graph execution in updates mode
            for _ in self.graph.stream(initial_state, config=config, stream_mode="updates"):
                pass     
        except KeyboardInterrupt:
            print("\nInterrupted by user, exiting...")
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()
        
        return None
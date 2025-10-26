"""
EchoForge Main Agent Class
"""
from typing import Dict, Any
from datetime import datetime
import uuid
import json
import os
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from .state import EchoForgeState, EchoModeState, PostSchema
from .config import EchoForgeConfig
from .memory import EchoForgeMemory
from src.prompts.echoForge.copilot_prompts import EchoForgePrompts


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
        workflow.add_node("gather_post_info", self._gather_post_info_node)
        workflow.add_node("validate_post_info", self._validate_post_info_node)
        workflow.add_node("confirm_post_info", self._confirm_post_info_node)
        workflow.add_node("generate_response", self._generate_response_node)
        workflow.add_node("store_record", self._store_record_node)
        workflow.add_node("handle_exit", self._handle_exit_node)
        
        # Add edges
        workflow.add_edge("gather_post_info", "validate_post_info")
        workflow.add_conditional_edges(
            "validate_post_info",
            self._should_continue_gathering,
            {
                "continue": "gather_post_info",
                "confirm": "confirm_post_info",
                "exit": "handle_exit"
            }
        )
        
        workflow.add_conditional_edges(
            "confirm_post_info",
            self._should_proceed_from_confirmation,
            {
                "proceed": "generate_response",
                "modify": "gather_post_info",
                "exit": "handle_exit"
            }
        )
        
        # # Linear flow for the rest
        workflow.add_edge("generate_response", "store_record")
        workflow.add_edge("store_record", "handle_exit")
        workflow.add_edge("handle_exit", END)
        
        # Set entry point
        workflow.set_entry_point("gather_post_info")
        
        return workflow.compile(interrupt_before=["gather_post_info", "confirm_post_info"], checkpointer=self.memory.memory_saver)
    
    
    def _gather_post_info_node(self, state: EchoModeState) -> EchoModeState:
        """Gather platform, title, and content from user"""
        # Use LLM with structured output to parse post information
        user_response = state["messages"][-1].content
        parsed_post = self.llm.with_structured_output(PostSchema).invoke(user_response)
    
        # Store parsed information in state["post_info"] only if not empty
        current_post_info = state.get("post_info", {})
        
        # Update platform if provided
        if parsed_post.platform.strip():
            state["post_info"]["platform"] = parsed_post.platform
        
        # Update title if provided
        if parsed_post.title.strip():
            state["post_info"]["title"] = parsed_post.title
        
        # Update content if provided
        if parsed_post.content.strip():
            state["post_info"]["content"] = parsed_post.content
        
        return state
    
    def _validate_post_info_node(self, state: EchoModeState) -> EchoModeState:
        """Validate post info and determine next status"""        
        # 1. Check for exit intent from the last message
        if state.get("messages"):
            last_message = state["messages"][-1].content
            quit_prompt = self.prompt_builder.detect_quit_intent_prompt(last_message)
            quit_response = self.llm.invoke(quit_prompt).content.strip()
            if quit_response == "QUIT":
                state["status"] = "exit"
                return state
        
        # 2. Check if all three fields are populated
        post_info = state.get("post_info", {})
        required_fields = ["platform", "title", "content"]
        missing_fields = [field for field in required_fields if not post_info.get(field, "").strip()]
        
        if not missing_fields:
            # All fields are populated - generate confirmation message
            state["status"] = "confirm"
            confirmation_message = self.prompt_builder.confirmation_message(
                post_info.get('platform', 'Not provided'),
                post_info.get('title', 'Not provided'),
                post_info.get('content', 'Not provided')
            )
            state["messages"].append(AIMessage(content=confirmation_message, name="EchoForge"))
            state["messages"][-1].pretty_print()
        else:
            # 3. Missing fields - set status to continue and add missing message
            state["status"] = "continue"
            missing_message = self.prompt_builder.ask_for_missing_fields(missing_fields)
            state["messages"].append(AIMessage(content=missing_message, name="EchoForge"))
            state["messages"][-1].pretty_print()
        
        return state
    
    def _confirm_post_info_node(self, state: EchoModeState) -> EchoModeState:
        """Process user's response to confirmation"""
        
        user_response = state["messages"][-1].content
        
        # Process the user's response to the confirmation
        confirmation_prompt = self.prompt_builder.parse_confirmation_prompt(user_response)
        confirmation_response = self.llm.invoke(confirmation_prompt).content.strip()
        
        if confirmation_response == "CONFIRMED":
            # User confirmed the info
            confirm_message = self.prompt_builder.confirmation_success_message()
            state["messages"].append(AIMessage(content=confirm_message, name="EchoForge"))
            state["messages"][-1].pretty_print()
            state["status"] = "proceed"
        elif confirmation_response == "MODIFY":
            # User wants to modify the info
            modify_message = self.prompt_builder.modification_request_message()
            state["messages"].append(AIMessage(content=modify_message, name="EchoForge"))
            state["messages"][-1].pretty_print()
            state["status"] = "modify"
        elif confirmation_response == "QUIT":
            # User wants to quit
            state["status"] = "exit"
        else:
            # Default to proceed if unclear
            confirm_message = self.prompt_builder.default_confirmation_message()
            state["messages"].append(AIMessage(content=confirm_message, name="EchoForge"))
            state["messages"][-1].pretty_print()
            state["status"] = "proceed"
        
        return state
    
    def _generate_response_node(self, state: EchoModeState) -> EchoModeState:
        """Generate AI response and evaluation"""
        
        # Get user profile and relevant context
        user_profile = self.memory.get_user_profile()
        post_content = state["post_info"].get("content", "")
        relevant_context = self.memory.get_relevant_context(post_content)
        
        # Generate response using LLM
        response_prompt = self.prompt_builder.generate_response_prompt(
            user_profile=user_profile,
            examples=relevant_context,
            platform=state["post_info"].get("platform", ""),
            title=state["post_info"].get("title", ""),
            content=state["post_info"].get("content", "")
        )
        
        llm_response = self.llm.invoke(response_prompt).content
        
        # Parse response and evaluation from LLM output
        # Parse AI response
        try:
            response_start = llm_response.find("<response>")
            response_end = llm_response.find("</response>")
            
            if response_start != -1 and response_end != -1:
                state["ai_response"] = llm_response[response_start + 10:response_end].strip()
            else:
                state["ai_response"] = llm_response
        except Exception as e:
            state["ai_response"] = llm_response
        
        # Parse AI evaluation
        try:
            evaluation_start = llm_response.find("<evaluation>")
            evaluation_end = llm_response.find("</evaluation>")
            
            if evaluation_start != -1 and evaluation_end != -1:
                state["ai_evaluation"] = llm_response[evaluation_start + 12:evaluation_end].strip()
            else:
                state["ai_evaluation"] = "Evaluation not found in LLM response"
        except Exception as e:
            state["ai_evaluation"] = "Error parsing evaluation"
        
        # Add AI response message
        response_message = AIMessage(content=f"Response: {state['ai_response']}", name="EchoForge")
        state["messages"].append(response_message)
        response_message.pretty_print()
        
        # Add AI evaluation message
        evaluation_message = AIMessage(content=f"Evaluation: {state['ai_evaluation']}", name="EchoForge")
        state["messages"].append(evaluation_message)
        evaluation_message.pretty_print()
        
        return state
    
    def _store_record_node(self, state: EchoModeState) -> EchoModeState:
        """Store interaction record to echoRecordQueue.json"""
        
        record = {
            "platform": state["post_info"].get("platform", ""),
            "title": state["post_info"].get("title", ""),
            "content": state["post_info"].get("content", ""),
            "ai_response": state["ai_response"],
            "ai_evaluation": state["ai_evaluation"],
            "timestamp": datetime.now().isoformat()
        }
        
        # Store to echoRecordQueue.json
        queue_file = os.path.join(self.config.data_dir, "echoForge", "echoRecordQueue.json")
        os.makedirs(os.path.dirname(queue_file), exist_ok=True)
        
        # Load existing records or create new list
        if os.path.exists(queue_file):
            with open(queue_file, 'r') as f:
                records = json.load(f)
        else:
            records = []
        
        # Append new record
        records.append(record)
        
        # Save back to file
        with open(queue_file, 'w') as f:
            json.dump(records, f, indent=2)
        
        return state
    
    def _handle_exit_node(self, state: EchoModeState) -> EchoModeState:
        """Handle quit/exit scenarios"""
        
        exit_message = self.prompt_builder.exit_message()
        state["messages"].append(AIMessage(content=exit_message, name="EchoForge"))
        state["messages"][-1].pretty_print()
        return state
    
    def _should_continue_gathering(self, state: EchoModeState) -> str:
        """Route based on status set by validate_post_info_node"""
        status = state.get("status", "")
        
        if status in ["exit", "confirm", "continue"]:
            return status
        else:
            # Default fallback
            return "continue"
    
    def _should_proceed_from_confirmation(self, state: EchoModeState) -> str:
        """Route based on status set by confirm_post_info_node"""
        status = state.get("status", "")
        
        if status in ["proceed", "modify", "exit"]:
            return status
        else:
            # Default fallback
            return "proceed"
    
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
        
        # Build query string for vector store search (combine all three parts)
        query = f"{context} {title} {content}".strip()
        
        # Get relevant notes from vector store (top 3)
        relevant_notes = self.memory.get_relevant_context(query, limit=3)
        
        # Build the prompt with all 5 parts
        prompt = self.prompt_builder.build_echo_prompt(context, title, content, user_profile, relevant_notes)
        
        # Generate and return the response
        response = self.llm.invoke(prompt).content
        return response
    
    def chat(self) -> str:
        """Main chat interface - agent initiates conversation"""
        
        # Create a thread config for the conversation
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        
        # Create initial state
        initial_state: EchoModeState = {
            "messages": [AIMessage(content=self.prompt_builder.greeting(), name="EchoForge")],
            "post_info": {"platform": "", "title": "", "content": ""},
            "ai_response": "",
            "ai_evaluation": "",
            "status": ""
        }
        
        # Print greeting message
        initial_state["messages"][-1].pretty_print()
        
        # Graph execution loop
        current_state = initial_state
        while True:
            try:
                # Stream the graph execution with values mode
                for event in self.graph.stream(current_state, config=config, stream_mode="values"):
                    pass
                # Check if the graph is in an interrupted state
                try:
                    # Try to get the current state to see if we're interrupted
                    current_graph_state = self.graph.get_state(config)
                    if current_graph_state.next:
                        # Graph is interrupted, get user input and continue
                        print("="*33 + " Human Input " + "="*33)
                        user_message = input("Your response:\n")
                        # Update state with user input and continue
                        self.graph.update_state(config, {"messages": [HumanMessage(content=user_message)]})
                        current_state = None  # Use None to resume from checkpoint
                        continue  # Continue the while loop to resume execution
                    else:
                        # Graph completed successfully
                        break
                        
                except Exception as e:
                    break
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                break
        
        return None
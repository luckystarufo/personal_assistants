from typing import Annotated
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool, Tool
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv


load_dotenv()

# System message for the RAG document collection agent
system_message = SystemMessage(content="""You are an AI assistant designed to help collect and curate documents for a RAG (Retrieval-Augmented Generation) vector store. Your goal is to build a comprehensive knowledge base that can later mimic the user's communication style, tone, knowledge, values, and preferences when responding to friends, posts, emails, and other communications.

## Your Mission
Create a personalized AI that doesn't just sound like the user, but thinks like them - understanding their:
- Communication style and tone
- Knowledge base and expertise areas
- Values and principles
- Response patterns and preferences
- Contextual awareness and social dynamics

## Document Structure & Constraints
Each document you collect must contain exactly 5 fields, with each field limited to 250 words maximum:
1. **context**: Who are we talking to, on what platform, the relationship, setting, etc.
2. **content**: What the person/post says, the original message or situation
3. **ai_response**: Your initial AI-generated response to the content
4. **user_response**: The user's actual preferred response (refined through feedback)
5. **learning**: Key insights about why AI and user responses differed, what to improve

**IMPORTANT**: If any field exceeds 250 words, you must summarize it while preserving the essential information and meaning.

## Workflow Process
Follow this exact sequence for each document collection:

### Step 1: Information Gathering
- Ask the user to provide the **context** (who, platform, relationship, setting)
- Ask the user to provide the **content** (what was said/posted)
- Confirm both pieces of information are accurate before proceeding

### Step 2: AI Response Generation with Historical Context
- Retrieve relevant historical examples from the vector store that match the current context/content
- Generate an **ai_response** based on:
  - The current context and content
  - Relevant historical examples that show similar situations
  - Patterns from previous user responses and learnings
- Present the response clearly to the user, mentioning which historical examples influenced your approach

### Step 3: User Feedback & Refinement
- Ask the user for explicit feedback on your AI response
- Through iterative conversation, help the user articulate their **user_response**
- Identify key differences and extract **learning** insights about:
  - Why the responses differed
  - What assumptions were wrong
  - What the user values differently
  - How to improve future responses
  - What historical examples were helpful or misleading

### Step 4: Final Confirmation & Summarization
- Present the complete document (context, content, ai_response, user_response, learning)
- Ensure each field is under 250 words - summarize if necessary
- Ask the user to confirm it's ready for storage
- Only proceed to storage when user explicitly agrees

### Step 5: Storage & Continuation
- Store the document in the vector store
- Ask if the user wants to continue with another document
- If yes, start the next iteration

## Key Principles
- Always confirm information accuracy before proceeding
- Leverage historical examples to improve response quality
- Be explicit about asking for feedback at each step
- Help users articulate their preferences clearly
- Extract meaningful learning insights from differences
- Summarize any field exceeding 250 words while preserving meaning
- Never store incomplete or unconfirmed documents
- Maintain a conversational, helpful tone throughout

Remember: Quality over quantity. Each document should be a valuable learning example that helps build a more accurate representation of the user's communication style and preferences.""")

def create_rag_tools(llm: ChatOpenAI):
    """Create Phase 1 RAG tools using the tool factory pattern."""
    
    def search_similar_examples_func(query: str, limit: int = 5) -> str:
        """Search for similar historical examples from the vector store."""
        # TODO: Implement actual vector store search
        # For now, return empty string as requested
        return ""
    
    def store_document_func(document: str) -> str:
        """Store the final 5-field document in the vector store."""
        print("=== DOCUMENT TO BE STORED ===")
        print(document)
        print("=== SAVING SUCCESSFULLY ===")
        return "Document stored successfully in vector store"
    
    def count_words_func(text: str) -> str:
        """Count words in text to enforce 250-word limit."""
        word_count = len(text.split())
        return f"Word count: {word_count}"
    
    def summarize_field_func(field_content: str, max_words: int = 250) -> str:
        """Summarize a field to stay under word limit while preserving meaning."""
        prompt = f"""Summarize this text to under {max_words} words while preserving essential meaning and key details:

{field_content}

Provide a concise summary that captures the most important information."""
        
        response = llm.invoke(prompt)
        return response.content
    
    def validate_document_fields_func(document: str) -> str:
        """Check if all 5 fields are present and under 250 words."""
        # Basic validation - in a real implementation, you'd parse the document structure
        required_fields = ["context", "content", "ai_response", "user_response", "learning"]
        validation_result = f"Document validation: Checking for required fields {required_fields}"
        return validation_result
    
    # Create Tool objects
    search_tool = Tool(
        name="search_similar_examples",
        description="Search for similar historical examples from the vector store based on query",
        func=search_similar_examples_func
    )
    
    store_tool = Tool(
        name="store_document",
        description="Store the final 5-field document in the vector store",
        func=store_document_func
    )
    
    count_tool = Tool(
        name="count_words",
        description="Count words in text to enforce 250-word limit",
        func=count_words_func
    )
    
    summarize_tool = Tool(
        name="summarize_field",
        description="Summarize a field to stay under word limit while preserving meaning",
        func=summarize_field_func
    )
    
    validate_tool = Tool(
        name="validate_document_fields",
        description="Check if all 5 fields are present and under 250 words",
        func=validate_document_fields_func
    )
    
    return [search_tool, store_tool, count_tool, summarize_tool, validate_tool]

@tool
def ask_human(question: Annotated[str, "What you want to ask the user"]) -> str:
    """Ask the user a question, resume with their answer."""
    print("="*33 + " Human Input " + "="*33)
    answer = input("\n\nYour response:")
    return answer


# --- Build a prebuilt ReAct agent -----------------------------------
llm = ChatOpenAI(model="gpt-5")  # or any chat model you've configured


rag_tools = create_rag_tools(llm)
tools = [ask_human] + rag_tools

memory = MemorySaver()  # optional, but nice for multiple turns
graph = create_react_agent(llm, tools=tools, checkpointer=memory)

def run_stream():
    cfg = {"configurable": {"thread_id": "demo", "recursion_limit": 100}}

    # Start the stream for this turn
    stream = graph.stream(
        {"messages": [system_message]},
        config=cfg,
        stream_mode="updates",  # yields incremental state diffs, including new messages
    )

    for update in stream:
        # Check for messages in different possible keys
        msgs = update.get("agent", {}).get("messages", [])
        if msgs:
            msgs[-1].pretty_print()


if __name__ == "__main__":
    run_stream()

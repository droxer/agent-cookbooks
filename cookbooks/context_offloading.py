import getpass
import os
from typing_extensions import Literal

# Pydantic for data modeling
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# LangChain core components
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from langgraph.graph import END, START, StateGraph, MessagesState


from utils.responses import format_messages

load_dotenv()

# Extended state class to include scratchpad functionality
class ScratchpadState(MessagesState):
    """State that extends MessagesState to include a scratchpad field.
    
    The scratchpad provides temporary storage during agent execution,
    allowing information to persist within a single conversation thread.
    """
    scratchpad: str = Field(description="The scratchpad for storing notes")

# Scratchpad management tools
@tool
class WriteToScratchpad(BaseModel):
    """Save notes to the scratchpad for future reference within the conversation."""
    notes: str = Field(description="Notes to save to the scratchpad")

@tool  
class ReadFromScratchpad(BaseModel):
    """Read previously saved notes from the scratchpad."""
    reasoning: str = Field(description="Reasoning for fetching notes from the scratchpad")

search_tool = TavilySearch(
    max_results=5,
    topic="general",
)

# Initialize the language model
llm = init_chat_model("anthropic:claude-sonnet-4-20250514", temperature=0)

# Configure scratchpad tools
tools = [ReadFromScratchpad, WriteToScratchpad, search_tool]
tools_by_name = {tool.name: tool for tool in tools}
llm_with_tools = llm.bind_tools(tools)

# Enhanced research planning prompt with structured workflow
scratchpad_prompt = """You are a sophisticated research assistant with access to web search and a persistent scratchpad for note-taking.

Your Research Workflow:
1. **Check Scratchpad**: Before starting a new research task, check your scratchpad to see if you have any relevant information already saved and use this to help write your research plan
2. **Create Research Plan**: Create a structured research plan
3. **Write to Scratchpad**: Save the research plan and any important information to your scratchpad
4. **Use Search**: Gather information using web search to address each aspect of your research plan
5. **Update Scratchpad**: After each search, update your scratchpad with new findings and insights
5. **Iterate**: Repeat searching and updating until you have comprehensive information
6. **Complete Task**: Provide a thorough response based on your accumulated research

Tools Available:
- WriteToScratchpad: Save research plans, findings, and progress updates
- ReadFromScratchpad: Retrieve previous research work and notes
- TavilySearch: Search the web for current information

Always maintain organized notes in your scratchpad and build upon previous research systematically."""

def llm_call(state: ScratchpadState) -> dict:
    """Execute LLM call with system prompt and conversation history.
    
    Args:
        state: Current conversation state
        
    Returns:
        Dictionary with LLM response
    """
    return {
        "messages": [
            llm_with_tools.invoke(
                [SystemMessage(content=scratchpad_prompt)] + state["messages"]
            )
        ]
    }
    
def tool_node(state: ScratchpadState) -> dict:
    """Execute tool calls and manage scratchpad state updates.
    
    Handles both reading from and writing to the scratchpad, updating
    the conversation state accordingly.
    
    Args:
        state: Current conversation state with tool calls
        
    Returns:
        Dictionary with tool results and updated state
    """
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        if tool_call["name"] == "WriteToScratchpad":
            # Save notes to scratchpad and update state
            notes = observation.notes
            result.append(ToolMessage(content=f"Wrote to scratchpad: {notes}", tool_call_id=tool_call["id"]))
            update = {"messages": result, "scratchpad": notes}
        elif tool_call["name"] == "ReadFromScratchpad":
            # Retrieve notes from scratchpad state
            notes = state.get("scratchpad", "")
            result.append(ToolMessage(content=f"Notes from scratchpad: {notes}", tool_call_id=tool_call["id"]))
            update = {"messages": result}
        elif tool_call["name"] == "tavily_search":
            # Write search tool observation to messages
            result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
            update = {"messages": result}
    return update

def should_continue(state: ScratchpadState) -> Literal["tool_node", "__end__"]:
    """Determine workflow continuation based on tool calls.
    
    Args:
        state: Current conversation state
        
    Returns:
        Next node name or END
    """
    messages = state["messages"]
    last_message = messages[-1]
    return "tool_node" if last_message.tool_calls else END

# Build the scratchpad-enabled workflow
agent_builder = StateGraph(ScratchpadState)

# Add workflow nodes
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)

# Define workflow edges
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges("llm_call", should_continue, {"tool_node": "tool_node", END: END})
agent_builder.add_edge("tool_node", "llm_call")
agent = agent_builder.compile()

def main():
    query = "Comparae the funding rounds and recent developments of Commonwealth Fusion Systems vs Helion Energy."
    state = agent.invoke({"messages": [{"role": "user", "content": query}]})
    format_messages(state['messages'])

if __name__ == "__main__":
    main()
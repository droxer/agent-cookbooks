from typing_extensions import Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from dotenv import load_dotenv

from tools.retriever_tool import retriever_tool
from responses import format_messages

load_dotenv()

# Initialize the language model
llm = init_chat_model("anthropic:claude-sonnet-4-20250514", temperature=0)

# Set up tools and bind them to the LLM
tools = [retriever_tool]
tools_by_name = {tool.name: tool for tool in tools}

# Bind tools to LLM for agent functionality
llm_with_tools = llm.bind_tools(tools)

# Define extended state with summary field for context compression
class State(MessagesState):
    """Extended state that includes a summary field for context compression."""
    summary: str

# Define the RAG agent system prompt
rag_prompt = """You are a helpful assistant tasked with retrieving information from a series of technical blog posts by Lilian Weng. 
Clarify the scope of research with the user before using your retrieval tool to gather context. Reflect on any context you fetch, and
proceed until you have sufficient context to answer the user's research request."""

def llm_call(state: State) -> dict:
    """Execute LLM call with system prompt and message history.
    
    Args:
        state: Current conversation state
        
    Returns:
        Dictionary with new messages
    """
    messages = [SystemMessage(content=rag_prompt)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# Updated summarization prompt to avoid encouraging further searches
tool_summarization_prompt = """You are an expert at condensing technical documents while preserving all critical information.

Transform the provided document into a comprehensive yet concise version. Extract and present the essential content in a clear, structured format.

Condensation Guidelines:
1. **Preserve All Key Information**: Include every important fact, statistic, finding, and conclusion
2. **Eliminate Verbosity**: Remove repetitive text, excessive explanations, and filler words
3. **Maintain Logical Structure**: Keep the natural flow and relationships between concepts
4. **Use Precise Language**: Replace lengthy phrases with direct, technical terminology
5. **Ensure Completeness**: The condensed version should contain all necessary information to fully understand the topic

Create a comprehensive condensed version that is 50-70% shorter while retaining 100% of the essential information."""

def should_continue(state: State) -> Literal["tool_node_with_summarization", "__end__"]:
    """Determine next step based on whether LLM made tool calls.
    
    Args:
        state: Current conversation state
        
    Returns:
        Next node to execute or END
    """
    messages = state["messages"]
    last_message = messages[-1]
    
    # If LLM made tool calls, process them with summarization
    if last_message.tool_calls:
        return "tool_node_with_summarization"
    
    # Otherwise, end the conversation
    return END

def tool_node_with_summarization(state: State):
    """Execute tool calls and summarize results for context efficiency.
    
    Args:
        state: Current conversation state with tool calls
        
    Returns:
        Dictionary with summarized tool results
    """
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        # Execute the tool
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        
        # Summarize the tool output to reduce context size
        summarization_llm = init_chat_model("openai:gpt-4o-mini", temperature=0)
        condensed_content = summarization_llm.invoke([
            {"role": "system", "content": tool_summarization_prompt},
            {"role": "user", "content": observation}
        ])
        
        result.append(ToolMessage(content=condensed_content.content, tool_call_id=tool_call["id"]))
        
    return {"messages": result}

# Build the RAG agent workflow with summarization
agent_builder = StateGraph(State)

# Add workflow nodes
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node_with_summarization", tool_node_with_summarization)

# Define workflow edges
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    {
        "tool_node_with_summarization": "tool_node_with_summarization",
        END: END,
    },
)
agent_builder.add_edge("tool_node_with_summarization", "llm_call")

# Compile and display the agent
agent = agent_builder.compile()


def main():
    query = "What are the types of reward hacking discussed in the blogs?"
    result = agent.invoke({"messages": [{"role": "user", "content": query}]})
    format_messages(result['messages'])

if __name__ == "__main__":
    main()
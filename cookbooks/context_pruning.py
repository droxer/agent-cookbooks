from typing_extensions import Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from dotenv import load_dotenv

from tools.retriever_tool import retriever_tool
from responses import format_messages
load_dotenv()


llm = init_chat_model("anthropic:claude-sonnet-4-20250514", temperature=0)

# Set up tools and bind them to the LLM
tools = [retriever_tool]
tools_by_name = {tool.name: tool for tool in tools}

# Bind tools to LLM for agent functionality
llm_with_tools = llm.bind_tools(tools)

# Define extended state with summary field
class State(MessagesState):
    """Extended state that includes a summary field for context compression."""
    summary: str

# Define the RAG agent system prompt
rag_prompt = """You are a helpful assistant tasked with retrieving information from a series of technical blog posts by Lilian Weng. 
Clarify the scope of research with the user before using your retrieval tool to gather context. Reflect on any context you fetch, and
proceed until you have sufficient context to answer the user's research request."""

def llm_call(state: MessagesState) -> dict:
    """Execute LLM call with system prompt and message history.
    
    This function demonstrates context pruning by trimming messages to fit within
    token limits while maintaining conversation coherence.
    
    Args:
        state: Current conversation state
        
    Returns:
        Dictionary with new messages
    """
    # Add system prompt to the trimmed messages
    messages = [SystemMessage(content=rag_prompt)] + state['messages']    
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# Improved prompt for context pruning
tool_pruning_prompt = """You are an expert at extracting relevant information from documents.

Your task: Analyze the provided document and extract ONLY the information that directly answers or supports the user's specific request. Remove all irrelevant content.

User's Request: {initial_request}

Instructions for pruning:
1. Keep information that directly addresses the user's question
2. Preserve key facts, data, and examples that support the answer
3. Remove tangential discussions, unrelated topics, and excessive background
4. Maintain the logical flow and context of relevant information
5. If multiple subtopics are discussed, focus only on those relevant to the request
6. Preserve important quotes, statistics, and research findings when relevant

Return the pruned content in a clear, concise format that maintains readability while focusing solely on what's needed to answer the user's request."""

# Conditional edge function to route to the tool node or end based upon whether the LLM made a tool call
def should_continue(state: State) -> Literal["tool_node_with_pruning", "__end__"]:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""
    messages = state["messages"]
    last_message = messages[-1]
    
    # If the LLM makes a tool call, then perform an action
    if last_message.tool_calls:
        return "tool_node_with_pruning"
    
    # Otherwise, we stop (reply to the user)
    return END

def tool_node_with_pruning(state: State):
    """Performs the tool call with context pruning"""
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        
        initial_request = state['messages'][0].content

        # Prune the document content to focus on user's request
        summarization_llm = init_chat_model("openai:gpt-4.1-mini", temperature=0)
        pruned_content = summarization_llm.invoke([
            {"role": "system", "content": tool_pruning_prompt.format(initial_request=initial_request)},
            {"role": "user", "content": observation}
        ])
        
        result.append(ToolMessage(content=pruned_content.content, tool_call_id=tool_call["id"]))
        
    return {"messages": result}

# Build workflow
agent_builder = StateGraph(State)

# Add nodes
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node_with_pruning", tool_node_with_pruning)

# Add edges to connect nodes
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    {
        "tool_node_with_pruning": "tool_node_with_pruning",
        END: END,
    },
)
agent_builder.add_edge("tool_node_with_pruning", "llm_call")

# Compile the agent
agent = agent_builder.compile()

def main():
    query = "What are the types of reward hacking discussed in the blogs?"
    result = agent.invoke({"messages": [{"role": "user", "content": query}]})
    format_messages(result['messages'])

if __name__ == "__main__":
    main()
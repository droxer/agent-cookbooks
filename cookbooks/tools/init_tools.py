import math  
import types
import uuid

from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings

from langgraph.store.memory import InMemoryStore

from langgraph_bigtool.utils import convert_positional_only_function_to_tool
from dotenv import load_dotenv
load_dotenv()

# Initialize the primary language model for the agent
llm = init_chat_model("anthropic:claude-sonnet-4-20250514", temperature=0)

# Extract and convert all mathematical functions from Python's math module
all_tools = []
for function_name in dir(math):
    function = getattr(math, function_name)
    
    # Only process built-in mathematical functions
    if not isinstance(function, types.BuiltinFunctionType):
        continue
        
    # Convert math functions to LangChain tools (handles positional-only parameters)
    if tool := convert_positional_only_function_to_tool(function):
        all_tools.append(tool)

# Create a tool registry mapping unique IDs to tool instances
# This allows for efficient tool lookup and management
tool_registry = {
    str(uuid.uuid4()): tool
    for tool in all_tools
}

# Set up vector store for semantic tool search
# Uses embeddings to enable similarity-based tool selection
embeddings = init_embeddings("openai:text-embedding-3-small")

store = InMemoryStore(
    index={
        "embed": embeddings,
        "dims": 1536,  # OpenAI embedding dimensions
        "fields": ["description"],  # Index tool descriptions for search
    }
)

def init_tools():
# Index all tools in the store for semantic similarity search
    for tool_id, tool in tool_registry.items():
        store.put(
            ("tools",),  # Namespace for tool storage
            tool_id,
            {
                "description": f"{tool.name}: {tool.description}",
            },
        )

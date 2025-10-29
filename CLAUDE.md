# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Setup

This project uses Python with `uv` for dependency management:

```bash
# Install uv if not already installed
pip install uv

# Create and activate virtual environment
uv venv
source .venv/bin/activate  # On Unix/macOS
# or .venv\Scripts\activate on Windows

# Install dependencies
uv sync

# Update dependencies
uv sync --upgrade
```

## Running Code Examples

This is a cookbook of agentic AI patterns. Each module in the examples directory demonstrates a specific technique:

- `python examples/agents/multi_agents.py` - Multi-agent coordination using supervisor pattern
- `python examples/context/tools_call.py` - Dynamic tool selection using semantic search
- `python examples/context/offloading.py` - Context management with scratchpad
- `python examples/context/compact.py` - Context compression with summarization
- `python examples/context/pruning.py` - Context pruning techniques
- `python examples/rag/multimodal_rag.py` - Multimodal RAG with text and image embeddings
- `python examples/agents/a2a_agents.py` - A2A agent communication example
- `python examples/context/ltm.py` - Long-term memories with semantic search

## Architecture

### Core Components

- **Language Models**: Uses `langchain.chat_models.init_chat_model()` for multiple providers (Anthropic, OpenAI)
- **Tools**: Dynamic tool registry system in `tools.registry` with semantic search via vector embeddings
- **Workflows**: LangGraph StateGraph-based agents with different patterns
- **Context Management**: Various strategies for handling context window limitations

### Key Patterns

1. **Multi-Agent Coordination** (`examples/agents/multi_agents.py`):
   - Supervisor pattern with specialized agents (math expert, research expert)
   - Uses `langgraph_supervisor` for agent delegation
   - Clear role separation and coordination

2. **Dynamic Tool Selection** (`examples/context/tools_call.py`):
   - Semantic search over tool descriptions using embeddings
   - Runtime tool binding based on query relevance
   - Vector store with `InMemoryStore` for tool indexing

3. **Context Management Strategies**:
   - **Scratchpad** (`examples/context/offloading.py`): Persistent note-taking within conversation threads
   - **Compression** (`examples/context/compact.py`): Tool output summarization using separate LLM
   - **Pruning** (`examples/context/pruning.py`): Selective context retention

### Tool System

The `examples/tools/registry.py` module provides:
- Automatic conversion of Python math functions to LangChain tools
- UUID-based tool registry for efficient lookup
- Vector embeddings for semantic tool discovery
- `init_tools()` function to populate the search index

### Model Context Protocol (MCP) Servers

- **Weather Server** (`examples/mcp/weather_server.py`): National Weather Service API integration with async tool definitions for weather alerts and forecasts
- **Math Server** (`examples/mcp/math_server.py`): Basic arithmetic operations exposed as MCP tools

### A2A Protocol Implementation

- **Agents** (`examples/agents/a2a/agents.py`): LangGraph A2A conversational agent supporting messages input for conversational interactions
- **Agent Communication** (`examples/agents/a2a_agents.py`): Example implementation for communication between A2A agents using JSON-RPC protocol

### Long-term Memory Implementation

- **Semantic Search**: Enables agents to store and retrieve personal user memories and information (`examples/context/ltm.py`)
- **Memory Storage**: Uses InMemoryStore with embedding-based indexing for similarity search
- **Context Injection**: Automatically retrieves relevant memories to enhance responses

## Dependencies

The project uses a comprehensive stack of agentic AI libraries:
- **LangChain**: Core LLM orchestration and tool management
- **LangGraph**: Workflow and state management
- **LangGraph-Supervisor**: Multi-agent coordination
- **LangGraph-Bigtool**: Advanced tool handling
- **LangGraph-Runtime**: A2A protocol support
- **FastMCP**: Model Context Protocol server implementation
- **Vector Stores**: PGVector, Chroma, and Qdrant for embeddings
- **DeepEval**: Evaluation framework
- Various vector stores and embedding providers

## Configuration

- Environment variables loaded via `.env` files (python-dotenv)
- Project metadata and dependencies in `pyproject.toml`
- Python version requirement: >=3.12

## Package Structure

The project follows Python best practices with an `examples` layout:

```
examples/
├── agents/              # Multi-agent coordination implementations
├── context/             # Context management strategies  
├── mcp/                 # Model Context Protocol servers
├── rag/                 # Retrieval-Augmented Generation implementations
├── store/               # Embedding store implementations
│   ├── embedding_store.py # PGVector and Chroma store abstraction
│   └── multimodal_store.py # Multimodal store with Qdrant for text and image embeddings
├── tools/               # Tool implementations
├── evals/               # Evaluation implementations
├── ltm/                 # Long-term memory implementations
│   └── ltm.py           # Long-term memories with semantic search
├── http/                # HTTP utilities
│   └── responses.py     # Response formatting utilities
└── a2a/                 # A2A protocol implementation
```
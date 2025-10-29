# Agent Cookbooks - Qwen Context

## Project Overview

The Agent Cookbooks project is a comprehensive collection of practical examples demonstrating various agentic AI patterns and techniques using LangChain, LangGraph, and related technologies. It serves as an educational and reference resource for building intelligent agents with different capabilities and architectures.

The project is structured as a Python codebase with multiple example implementations covering:

- **Multi-Agent Coordination**: Supervisor pattern with specialized agents
- **Context Management**: Strategies including scratchpad, compression, and pruning
- **Dynamic Tool Selection**: Semantic search over tools using embeddings
- **Model Context Protocol (MCP)**: Server implementations for weather and math services
- **Multimodal RAG**: Retrieval-augmented generation with text and image embeddings
- **Embedding Store Abstraction**: Unified interface for vector databases
- **A2A Protocol Implementation**: Agent-to-Agent communication patterns
- **Long-term Memories**: Semantic search-based memory storage and retrieval
- **Qdrant-based Memory Agents**: Intelligent agents with hybrid memory systems using Qdrant vector database
- **Shared Memory Systems**: Team-wide memory sharing between multiple agents

## Technologies and Libraries

The project leverages a comprehensive stack of agentic AI libraries:

- **LangChain**: Core LLM orchestration and tool management
- **LangGraph**: Workflow and state management for agents
- **LangGraph-Supervisor**: Multi-agent coordination patterns
- **LangGraph-Bigtool**: Advanced tool handling capabilities
- **LangGraph-Runtime**: A2A protocol support
- **FastMCP**: Model Context Protocol server implementations
- **Vector Stores**: PGVector, Chroma, and Qdrant for embedding storage
- **Qdrant Client**: Native Qdrant vector database client
- **HuggingFace Embeddings**: Sentence transformers for vector embeddings
- **DeepEval**: Evaluation framework for testing and validation

## Project Structure

```
agent-cookbooks/
├── examples/                   # Main example implementations
│   ├── agents/                 # Multi-agent coordination implementations
│   │   ├── __init__.py
│   │   ├── multi_agents.py     # Multi-agent coordination with supervisor pattern
│   │   ├── a2a_agents.py       # A2A agent communication example
│   │   ├── intelligent_memory_agent.py  # Intelligent memory agent with Qdrant
│   │   ├── shared_memory_agents.py      # Shared memory agents with team/personal stores
│   │   └── a2a/                # A2A protocol implementation
│   │       ├── agents.py       # LangGraph A2A conversational agent
│   │       └── __init__.py
│   ├── context/                # Context management strategies
│   │   ├── __init__.py
│   │   ├── compact.py          # Context compression with summarization
│   │   ├── offloading.py       # Context management with scratchpad
│   │   ├── pruning.py          # Context pruning techniques
│   │   ├── tools_call.py       # Dynamic tool selection using semantic search
│   │   └── ltm.py              # Long-term memories with semantic search
│   ├── mcp/                    # Model Context Protocol servers
│   │   ├── __init__.py
│   │   ├── weather_server.py   # Weather data MCP server
│   │   ├── math_server.py      # Math operations MCP server
│   │   └── mcp_agents.py       # MCP agent communication example
│   ├── rag/                    # Retrieval-Augmented Generation implementations
│   │   ├── __init__.py
│   │   └── multimodal_rag.py   # Multimodal RAG with text and image embeddings
│   ├── store/                  # Embedding store implementations
│   │   ├── __init__.py
│   │   ├── embedding_store.py  # PGVector and Chroma store abstraction
│   │   ├── multimodal_store.py # Multimodal embedding store with Qdrant
│   │   ├── qdrant_store_adapter.py # Qdrant store adapter with weighted search
│   │   └── README.md           # Documentation for vector store setup
│   ├── tools/                  # Tool implementations
│   │   ├── __init__.py
│   │   └── registry.py         # Math tools registry with semantic search
│   ├── evals/                  # Evaluation implementations
│   │   ├── __init__.py
│   │   └── test_deepeval.py    # DeepEval integration for testing
│   ├── http/                   # HTTP utilities
│   │   ├── __init__.py
│   │   └── responses.py        # Response formatting utilities
├── tests/                      # Unit and integration tests
├── docs/                       # Documentation
├── images/                     # Sample images for multimodal examples
├── scripts/                    # Utility scripts
├── pyproject.toml              # Project dependencies and configuration
├── README.md                   # Project documentation
├── CLAUDE.md                   # Claude-specific development guidance
└── QWEN.md                     # This file
```

## Setup and Development

### Prerequisites
- Python >= 3.12
- uv package manager (for dependency management)

### Installation
```bash
# Install uv if not already installed
pip install uv

# Create and activate virtual environment
uv venv
source .venv/bin/activate  # On Unix/macOS
# or .venv\Scripts\activate on Windows

# Install dependencies
uv sync
```

### Updating Dependencies
```bash
uv sync --upgrade
```

### Environment Configuration
1. Copy `.env.example` to `.env` and add your API keys:
   - `OPENAI_API_KEY` - OpenAI API key for OpenAI models
   - `ANTHROPIC_API_KEY` - Anthropic API key for Claude models
   - `TAVILY_API_KEY` - Tavily API key for search functionality
   - `VECTOR_STORE_TYPE` - Set to 'pgvector', 'chroma', or 'qdrant'

## Running Code Examples

Each module in the examples directory demonstrates a specific agentic AI technique. You can run them using Python directly:

### Direct Module Usage
```bash
# Multi-agent coordination using supervisor pattern
python examples/agents/multi_agents.py

# Intelligent memory agent with Qdrant-based storage and importance scoring
python examples/agents/intelligent_memory_agent.py

# Shared memory agents with team and personal memory stores
python examples/agents/shared_memory_agents.py

# Dynamic tool selection using semantic search
python examples/context/tools_call.py

# Context management with scratchpad
python examples/context/offloading.py

# Context compression with summarization
python examples/context/compact.py

# Context pruning techniques
python examples/context/pruning.py

# Multimodal RAG with text and image embeddings
python examples/rag/multimodal_rag.py

# A2A agent communication example
python examples/agents/a2a_agents.py

# Long-term memories with semantic search
python examples/context/ltm.py
```

## Key Components and Patterns

### 1. Multi-Agent Coordination (`examples/agents/multi_agents.py`)
- Implements a supervisor pattern with specialized agents (math expert, research expert)
- Uses `langgraph_supervisor` for agent delegation
- Demonstrates clear role separation and coordination between agents

### 2. Dynamic Tool Selection (`examples/context/tools_call.py`)
- Semantic search over tool descriptions using embeddings
- Runtime tool binding based on query relevance
- Vector store with `InMemoryStore` for tool indexing

### 3. Context Management Strategies
- **Scratchpad** (`examples/context/offloading.py`): Persistent note-taking within conversation threads
- **Compression** (`examples/context/compact.py`): Tool output summarization using separate LLM
- **Pruning** (`examples/context/pruning.py`): Selective context retention

### 4. Tool System (`examples/tools/registry.py`)
- Automatic conversion of Python math functions to LangChain tools
- UUID-based tool registry for efficient lookup
- Vector embeddings for semantic tool discovery
- `init_tools()` function to populate the search index

### 5. Model Context Protocol (MCP) Servers
- **Weather Server** (`examples/mcp/weather_server.py`): National Weather Service API integration with async tool definitions for weather alerts and forecasts
- **Math Server** (`examples/mcp/math_server.py`): Basic arithmetic operations exposed as MCP tools

### 6. Embedding Store Abstraction (`examples/store/embedding_store.py`)
- Unified interface for PGVector and Chroma vector stores
- Document loading and chunking from web sources
- Factory pattern for store creation

### 7. Multimodal RAG (`examples/rag/multimodal_rag.py`)
- **Multimodal Retrieval**: Combines text and image embeddings for retrieval using Qdrant vector database
- **Text-Image Embeddings**: Uses SentenceTransformers with CLIP model for encoding both text and images
- **LangGraph Workflow**: Implements a retrieval-augmented generation pipeline with separate retrieval and generation nodes
- **Qdrant Integration**: Stores and queries multimodal embeddings with proper vector configurations

### 8. Multimodal Store (`examples/store/multimodal_store.py`)
- **Data Ingestion**: Processes and stores multimodal documents with both text and image components
- **Embedding Generation**: Creates text and image embeddings using SentenceTransformers
- **Qdrant Collections**: Manages vector collections with appropriate dimension configurations for text (384) and image (512) embeddings

### 9. Qdrant Store Adapter (`examples/store/qdrant_store_adapter.py`)
- **Qdrant Integration**: Native Qdrant vector database client with HuggingFace embeddings
- **Weighted Search**: Advanced retrieval considering semantic similarity, importance scores, and temporal decay
- **Hybrid Memory Storage**: Combines short-term memory (in-memory) with long-term memory (persistent Qdrant storage)
- **Session Management**: Supports isolated memory sessions for different users or contexts

### 10. Intelligent Memory Agent (`examples/agents/intelligent_memory_agent.py`)
- **Automatic Importance Scoring**: Uses LLM to rate conversation importance for long-term storage
- **Timestamp-based Storage**: Automatically timestamps memories for temporal context
- **Hybrid Memory System**: Combines short-term conversation context with long-term Qdrant storage
- **Weighted Retrieval**: Recalls memories based on semantic similarity, importance, and recency

### 11. Shared Memory Agents (`examples/agents/shared_memory_agents.py`)
- **Team-wide Memory Sharing**: Enables multiple agents to share and access common knowledge base
- **Personal Memory Stores**: Each agent maintains personal memory in addition to shared team memory
- **Collaborative Context**: Agents can leverage both personal experience and team knowledge
- **Selective Sharing**: Agents can choose which memories to share with the team

### 12. A2A Protocol Implementation (`examples/agents/a2a/`)
- **Agents** (`examples/agents/a2a/agents.py`): LangGraph A2A conversational agent supporting messages input for conversational interactions
- **Agent Communication** (`examples/agents/a2a_agents.py`): Example implementation for communication between A2A agents using JSON-RPC protocol

### 13. Long-term Memories (`examples/context/ltm.py`)
- **Semantic Search**: Enables agents to store and retrieve personal user memories and information
- **Memory Storage**: Uses InMemoryStore with embedding-based indexing for similarity search
- **Context Injection**: Automatically retrieves relevant memories to enhance responses

## Development Conventions

- Use LangGraph for workflow and state management
- Follow the supervisor pattern for multi-agent coordination
- Leverage semantic search for dynamic tool selection
- Implement proper context management strategies to handle token limitations
- Use environment variables for API keys and configuration
- Follow the project's directory structure for organizing new examples
- Use proper Python package imports (e.g. `from agent_cookbooks.utils.responses import format_messages`)

## Dependencies

The project uses a comprehensive set of dependencies defined in `pyproject.toml`:

- LangChain ecosystem libraries
- Vector databases (Chroma, PGVector, Qdrant)
- MCP implementation
- Embedding models and utilities
- Testing and evaluation frameworks
- Data processing libraries
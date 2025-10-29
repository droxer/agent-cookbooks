# Agent Cookbooks

A collection of practical examples demonstrating various agentic AI patterns and techniques using LangChain, LangGraph, and related technologies.

## Setup

1. Install uv:
```bash
pip install uv
```

2. Create and activate a virtual environment:
```bash
uv venv
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate  # On Windows
```

3. Install dependencies:
```bash
uv sync
```

4. Environment Configuration
   1. Copy `.env.example` to `.env` and add your API keys:
      - `OPENAI_API_KEY` - OpenAI API key for OpenAI models
      - `ANTHROPIC_API_KEY` - Anthropic API key for Claude models
      - `TAVILY_API_KEY` - Tavily API key for search functionality
      - `VECTOR_STORE_TYPE` - Set to 'pgvector', 'chroma', or 'qdrant'

## Development

Project configuration is managed in `pyproject.toml`. This includes:
- Project metadata (name, version, authors)
- Python version requirements
- All project dependencies with their versions

- To add new dependencies:
```bash
# Add the dependency to pyproject.toml under [project.dependencies], then run:
uv sync
```

- To update dependencies:
```bash
uv sync --upgrade
```

## Examples

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

### 9. A2A Protocol Implementation (`examples/agents/a2a/`)
- **Agents** (`examples/agents/a2a/agents.py`): LangGraph A2A conversational agent supporting messages input for conversational interactions
- **Agent Communication** (`examples/agents/a2a_agents.py`): Example implementation for communication between A2A agents using JSON-RPC protocol

### 10. Long-term Memories (`examples/context/ltm.py`)
- **Semantic Search**: Enables agents to store and retrieve personal user memories and information
- **Memory Storage**: Uses InMemoryStore with embedding-based indexing for similarity search
- **Context Injection**: Automatically retrieves relevant memories to enhance responses

### 11. Qdrant Store Adapter (`examples/store/qdrant_store_adapter.py`)
- **Qdrant Integration**: Native Qdrant vector database client with HuggingFace embeddings
- **Weighted Search**: Advanced retrieval considering semantic similarity, importance scores, and temporal decay
- **Hybrid Memory Storage**: Combines short-term memory (in-memory) with long-term memory (persistent Qdrant storage)
- **Session Management**: Supports isolated memory sessions for different users or contexts

### 12. Intelligent Memory Agent (`examples/agents/intelligent_memory_agent.py`)
- **Automatic Importance Scoring**: Uses LLM to rate conversation importance for long-term storage
- **Timestamp-based Storage**: Automatically timestamps memories for temporal context
- **Hybrid Memory System**: Combines short-term conversation context with long-term Qdrant storage
- **Weighted Retrieval**: Recalls memories based on semantic similarity, importance, and recency

### 13. Shared Memory Agents (`examples/agents/shared_memory_agents.py`)
- **Team-wide Memory Sharing**: Enables multiple agents to share and access common knowledge base
- **Personal Memory Stores**: Each agent maintains personal memory in addition to shared team memory
- **Collaborative Context**: Agents can leverage both personal experience and team knowledge
- **Selective Sharing**: Agents can choose which memories to share with the team

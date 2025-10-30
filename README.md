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
Supervisor pattern with specialized agents (math/research experts) using LangGraph

### 2. Dynamic Tool Selection (`examples/context/tools_call.py`)
Semantic search over tool descriptions with runtime tool binding

### 3. Context Management Strategies
- **Scratchpad** (`examples/context/offloading.py`): Persistent note-taking
- **Compression** (`examples/context/compact.py`): Output summarization
- **Pruning** (`examples/context/pruning.py`): Selective context retention

### 4. Tool System (`examples/tools/registry.py`)
Converts Python functions to LangChain tools with semantic discovery

### 5. Model Context Protocol (MCP) Servers
- **Weather Server** (`examples/mcp/weather_server.py`): NWS API integration
- **Math Server** (`examples/mcp/math_server.py`): Arithmetic operations

### 6. Embedding Store Abstraction (`examples/store/embedding_store.py`)
Unified interface for PGVector and Chroma vector stores

### 7. Multimodal RAG (`examples/rag/multimodal_rag.py`)
Text and image embeddings with Qdrant for retrieval-augmented generation

### 8. Multimodal Store (`examples/store/multimodal_store.py`)
Processes and stores documents with both text and image components

### 9. A2A Protocol Implementation (`examples/agents/a2a/`)
LangGraph conversational agents with JSON-RPC communication

### 10. Long-term Memories (`examples/context/ltm.py`)
Semantic search for storing and retrieving personal user memories

### 11. Qdrant Store Adapter (`examples/store/qdrant_store_adapter.py`)
Advanced retrieval with semantic similarity, importance scores, and temporal decay

### 12. Intelligent Memory Agent (`examples/agents/intelligent_memory_agent.py`)
LLM-rated importance scoring with timestamped memory storage

### 13. Shared Memory Agents (`examples/agents/shared_memory_agents.py`)
Team-wide knowledge sharing with personal and shared memory stores

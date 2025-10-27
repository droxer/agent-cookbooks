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

## Codebase Structure

The repository is organized as an agentic AI cookbook with examples demonstrating various patterns and techniques:

```
agentic-cookbook/
├── cookbooks/
│   ├── multi-agents.py              # Multi-agent coordination with supervisor pattern
│   ├── context_tools_call.py        # Dynamic tool selection using semantic search
│   ├── context_offloading.py        # Context management with scratchpad
│   ├── context_compact.py           # Context compression with summarization
│   ├── context_pruning.py           # Context pruning techniques
│   ├── a2a/                         # A2A protocol implementation
│   │   ├── agents.py                # LangGraph A2A conversational agent
│   │   └── langraph.json            # LangGraph configuration
│   ├── a2a_agents.py                # A2A agent communication example
│   ├── mcp/                         # Model Context Protocol servers
│   │   ├── weather_server.py        # Weather data MCP server
│   │   └── math_server.py           # Math operations MCP server
│   ├── store/                       # Embedding store implementations
│   │   └── embedding_store.py       # PGVector and Chroma store abstraction
│   ├── tools/                       # Tool implementations
│   │   ├── tools_registry.py        # Math tools registry with semantic search
│   │   └── retriever_tool.py        # Document retrieval tool
│   └── utils/
│       └── responses.py             # Response formatting utilities
├── pyproject.toml                   # Project dependencies and configuration
└── README.md                        # This file
```

## Running Code Examples

Each Python file in the cookbooks directory demonstrates a specific agentic AI technique:

```bash
# Multi-agent coordination using supervisor pattern
python cookbooks/multi-agents.py

# Dynamic tool selection using semantic search
python cookbooks/context_tools_call.py

# Context management with scratchpad
python cookbooks/context_offloading.py

# Context compression with summarization
python cookbooks/context_compact.py

# Context pruning techniques
python cookbooks/context_pruning.py

# A2A agent communication example
python cookbooks/a2a_agents.py

# Response formatting utilities
python cookbooks/utils/responses.py
```

## Key Components

### 1. Multi-Agent Coordination (`multi-agents.py`)
- Implements a supervisor pattern with specialized agents (math expert, research expert)
- Uses `langgraph_supervisor` for agent delegation
- Demonstrates clear role separation and coordination between agents

### 2. Dynamic Tool Selection (`context_tools_call.py`)
- Semantic search over tool descriptions using embeddings
- Runtime tool binding based on query relevance
- Vector store with `InMemoryStore` for tool indexing

### 3. Context Management Strategies
- **Scratchpad** (`context_offloading.py`): Persistent note-taking within conversation threads
- **Compression** (`context_compact.py`): Tool output summarization using separate LLM
- **Pruning** (`context_pruning.py`): Selective context retention

### 4. Tool System (`tools/tools_registry.py`)
- Automatic conversion of Python math functions to LangChain tools
- UUID-based tool registry for efficient lookup
- Vector embeddings for semantic tool discovery
- `init_tools()` function to populate the search index

### 5. Model Context Protocol (MCP) Servers
- **Weather Server** (`mcp/weather_server.py`): National Weather Service API integration with async tool definitions for weather alerts and forecasts
- **Math Server** (`mcp/math_server.py`): Basic arithmetic operations exposed as MCP tools

### 6. Embedding Store Abstraction (`store/embedding_store.py`)
- Unified interface for PGVector and Chroma vector stores
- Document loading and chunking from web sources
- Factory pattern for store creation

### 7. A2A Protocol Implementation (`a2a/`)
- **Agents** (`a2a/agents.py`): LangGraph A2A conversational agent supporting messages input for conversational interactions
- **Agent Communication** (`a2a_agents.py`): Example implementation for communication between A2A agents using JSON-RPC protocol

## Architecture

### Core Libraries
- **LangChain**: Core LLM orchestration and tool management
- **LangGraph**: Workflow and state management
- **LangGraph-Supervisor**: Multi-agent coordination
- **LangGraph-Bigtool**: Advanced tool handling
- **LangGraph-Runtime**: A2A protocol support
- **FastMCP**: Model Context Protocol server implementation
- **Vector Stores**: PGVector and Chroma for embeddings

### Configuration
- Environment variables loaded via `.env` files (python-dotenv)
- Project metadata and dependencies in `pyproject.toml`
- Python version requirement: >=3.12

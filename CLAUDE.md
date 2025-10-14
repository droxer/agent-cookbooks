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

This is a cookbook of agentic AI patterns. Each Python file demonstrates a specific technique:

- `python cookbooks/multi-agents.py` - Multi-agent coordination using supervisor pattern
- `python cookbooks/tools_call.py` - Dynamic tool selection using semantic search
- `python cookbooks/context_offloading.py` - Context management with scratchpad
- `python cookbooks/context_compact.py` - Context compression with summarization
- `python cookbooks/context_pruning.py` - Context pruning techniques
- `python cookbooks/text_extract.py` - Text extraction utilities
- `python cookbooks/responses.py` - Response formatting utilities

## Architecture

### Core Components

- **Language Models**: Uses `langchain.chat_models.init_chat_model()` for multiple providers (Anthropic, OpenAI)
- **Tools**: Dynamic tool registry system in `tools/tools_registry.py` with semantic search via vector embeddings
- **Workflows**: LangGraph StateGraph-based agents with different patterns
- **Context Management**: Various strategies for handling context window limitations

### Key Patterns

1. **Multi-Agent Coordination** (`multi-agents.py`):
   - Supervisor pattern with specialized agents (math expert, research expert)
   - Uses `langgraph_supervisor` for agent delegation
   - Clear role separation and coordination

2. **Dynamic Tool Selection** (`tools_call.py`):
   - Semantic search over tool descriptions using embeddings
   - Runtime tool binding based on query relevance
   - Vector store with `InMemoryStore` for tool indexing

3. **Context Management Strategies**:
   - **Scratchpad** (`context_offloading.py`): Persistent note-taking within conversation threads
   - **Compression** (`context_compact.py`): Tool output summarization using separate LLM
   - **Pruning** (`context_pruning.py`): Selective context retention

### Tool System

The `tools/tools_registry.py` module provides:
- Automatic conversion of Python math functions to LangChain tools
- UUID-based tool registry for efficient lookup
- Vector embeddings for semantic tool discovery
- `init_tools()` function to populate the search index

### MCP Server

`cookbooks/mcp/weather_server.py` demonstrates Model Context Protocol implementation:
- FastMCP server for weather data
- National Weather Service API integration
- Async tool definitions for weather alerts and forecasts

## Dependencies

The project uses a comprehensive stack of agentic AI libraries:
- **LangChain**: Core LLM orchestration and tool management
- **LangGraph**: Workflow and state management
- **LangGraph-Supervisor**: Multi-agent coordination
- **LangGraph-Bigtool**: Advanced tool handling
- **FastMCP**: Model Context Protocol server implementation
- **DeepEval**: Evaluation framework
- Various vector stores and embedding providers

## Configuration

- Environment variables loaded via `.env` files (python-dotenv)
- Project metadata and dependencies in `pyproject.toml`
- Python version requirement: >=3.11.10
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
      - `GEMINI_API_KEY` - Gemini API key for text extraction features
      - `VECTOR_STORE_TYPE` - Set to 'pgvector', 'chroma', or 'qdrant'
      - `OPENAI_MODEL` - OpenAI model selection (defaults to gpt-4o-mini)
      - `TOKENIZERS_PARALLELISM=False` - Tokenizer configuration

## Development

Project configuration is managed in `pyproject.toml`. This includes:
- Project metadata (name, version, authors)
- Python version requirements
- All project dependencies with their versions

### Key Dependencies
- **LangChain Ecosystem**: Core orchestration, LangGraph workflows, supervisor patterns
- **Vector Stores**: PGVector, Chroma, Qdrant, LanceDB, SQLite-vec
- **Validation**: Guardrails for input/output validation
- **Document Processing**: Langextract for text extraction
- **Multi-provider LLM**: LiteLLM for unified API access
- **Search**: Tantivy search engine
- **Evaluation**: DeepEval for testing metrics

### Adding Dependencies
```bash
# Add the dependency to pyproject.toml under [project.dependencies], then run:
uv sync
```

### Updating Dependencies
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

### 14. ReAct Agent (`examples/agents/react_agent.py`)
ReAct pattern implementation with tool routing and reasoning capabilities

### 15. LLM Proxy (`examples/agents/llm_proxy.py`)
Multi-provider LLM support using LiteLLM for unified API access

### 16. Document Processing
- **PDF to Images** (`examples/document/pdf2images.py`): PDF image extraction with OpenAI vision analysis
- **Text Extraction** (`examples/document/text_extract.py`): Text extraction using langextract library

### 17. Enhanced Vector Store Features
- **Cross-modal Retrieval** (`examples/store/vector_retriever.py`): Text-to-image and image-to-text search
- **Vector Consistency** (`examples/store/verify_vector_consistency.py`): Vector normalization verification
- **Blog Retriever Tool** (`examples/tools/retriever_tool.py`): Blog post retrieval tool

### 18. Input/Output Validation (`examples/validation/validators.py`)
Guardrails integration for input/output validation and filtering

### 19. Enhanced MCP Integration (`examples/mcp/mcp_agents.py`)
Multi-server MCP client integration with REACT agents

### 20. Evaluation Framework (`examples/evals/test_deepeval.py`)
DeepEval integration for comprehensive testing metrics

## Architecture

### Core Components

- **Language Models**: Uses `langchain.chat_models.init_chat_model()` for multiple providers (Anthropic, OpenAI, Gemini)
- **Tools**: Dynamic tool registry system in `tools.registry` with semantic search via vector embeddings
- **Workflows**: LangGraph StateGraph-based agents with different patterns
- **Context Management**: Various strategies for handling context window limitations

### Key Patterns

1. **Multi-Agent Coordination** (`examples/agents/multi_agents.py`):
   - Supervisor pattern with specialized agents (math expert, research expert)
   - Uses `langgraph_supervisor` for agent delegation
   - Clear role separation and coordination

2. **ReAct Agent Pattern** (`examples/agents/react_agent.py`):
   - Reasoning and acting capabilities with tool routing
   - Dynamic tool selection based on reasoning
   - Integration with MCP servers for enhanced functionality

3. **LLM Proxy Pattern** (`examples/agents/llm_proxy.py`):
   - Multi-provider support through LiteLLM
   - Unified API access across different LLM providers
   - Fallback and routing capabilities

4. **Dynamic Tool Selection** (`examples/context/tools_call.py`):
   - Semantic search over tool descriptions using embeddings
   - Runtime tool binding based on query relevance
   - Vector store with `InMemoryStore` for tool indexing

5. **Context Management Strategies**:
   - **Scratchpad** (`examples/context/offloading.py`): Persistent note-taking within conversation threads
   - **Compression** (`examples/context/compact.py`): Tool output summarization using separate LLM
   - **Pruning** (`examples/context/pruning.py`): Selective context retention

6. **Input/Output Validation** (`examples/validation/validators.py`):
   - Guardrails integration for data validation
   - Input filtering and output sanitization
   - Custom validation rules and constraints

7. **Document Processing Pipeline**:
   - **PDF Processing** (`examples/document/pdf2images.py`): Extract images from PDFs and analyze with vision models
   - **Text Extraction** (`examples/document/text_extract.py`): Extract structured text using langextract
   - **Multimodal Analysis**: Combine text and image processing capabilities

8. **Cross-modal Retrieval** (`examples/store/vector_retriever.py`):
   - Text-to-image and image-to-text search capabilities
   - Unified embedding space for multimodal content
   - Advanced similarity matching across modalities

9. **Qdrant-based Memory Agents**:
   - **Intelligent Memory Agent** (`examples/agents/intelligent_memory_agent.py`): Hybrid memory system with automatic importance scoring and timestamping
   - **Shared Memory Agents** (`examples/agents/shared_memory_agents.py`): Multi-agent system with personal and team-wide memory sharing
   - **Weighted Search**: Advanced retrieval considering semantic similarity, importance scores, and temporal decay

### Tool System

The `examples/tools/registry.py` module provides:
- Automatic conversion of Python math functions to LangChain tools
- UUID-based tool registry for efficient lookup
- Vector embeddings for semantic tool discovery
- `init_tools()` function to populate the search index

### Model Context Protocol (MCP) Servers

- **Weather Server** (`examples/mcp/weather_server.py`): National Weather Service API integration with async tool definitions for weather alerts and forecasts
- **Math Server** (`examples/mcp/math_server.py`): Basic arithmetic operations exposed as MCP tools
- **Enhanced MCP Integration** (`examples/mcp/mcp_agents.py`): Multi-server client with REACT agent pattern for complex tool orchestration

### A2A Protocol Implementation

- **Agents** (`examples/agents/a2a/agents.py`): LangGraph A2A conversational agent supporting messages input for conversational interactions
- **Agent Communication** (`examples/agents/a2a_agents.py`): Example implementation for communication between A2A agents using JSON-RPC protocol

### Long-term Memory Implementation

- **Semantic Search**: Enables agents to store and retrieve personal user memories and information (`examples/context/ltm.py`)
- **Memory Storage**: Uses InMemoryStore with embedding-based indexing for similarity search
- **Qdrant-based Storage**: Advanced memory storage using Qdrant vector database with importance scoring and timestamping
- **Hybrid Memory**: Combination of short-term memory (in-memory) and long-term memory (persistent Qdrant storage)
- **Shared Memory**: Team-wide memory sharing between agents with personal and shared memory stores
- **Context Injection**: Automatically retrieves relevant memories to enhance responses

### Evaluation Framework

- **DeepEval Integration** (`examples/evals/test_deepeval.py`): Comprehensive testing metrics and evaluation
- **Performance Metrics**: Automated evaluation of agent responses
- **Quality Assurance**: Systematic testing of agent capabilities

## Package Structure

The project follows Python best practices with an `examples` layout:

```
examples/
├── agents/                    # Multi-agent coordination implementations
│   ├── intelligent_memory_agent.py  # Intelligent memory agent with Qdrant
│   ├── shared_memory_agents.py      # Shared memory agents with team/personal stores
│   ├── react_agent.py               # ReAct pattern implementation
│   └── llm_proxy.py                 # Multi-provider LLM proxy
├── context/                   # Context management strategies
├── document/                  # Document processing pipeline
│   ├── pdf2images.py         # PDF image extraction with vision analysis
│   └── text_extract.py       # Text extraction using langextract
├── mcp/                       # Model Context Protocol servers
│   ├── weather_server.py     # NWS API integration
│   ├── math_server.py        # Arithmetic operations
│   └── mcp_agents.py         # Multi-server MCP client
├── rag/                       # Retrieval-Augmented Generation implementations
├── store/                     # Embedding store implementations
│   ├── embedding_store.py     # PGVector and Chroma store abstraction
│   ├── multimodal_store.py    # Multimodal store with Qdrant for text and image embeddings
│   ├── qdrant_store_adapter.py # Qdrant store adapter with weighted search
│   ├── vector_retriever.py    # Cross-modal retrieval capabilities
│   └── verify_vector_consistency.py # Vector normalization verification
├── tools/                     # Tool implementations
│   ├── registry.py           # Dynamic tool registry
│   └── retriever_tool.py     # Blog post retriever
├── evals/                     # Evaluation implementations
│   └── test_deepeval.py      # DeepEval integration
├── validation/                # Input/output validation
│   ├── validators.py         # Guardrails integration
│   ├── inputs.py             # Input validation utilities
│   └── outputs.py            # Output validation utilities
├── ltm/                       # Long-term memory implementations
│   └── ltm.py                 # Long-term memories with semantic search
├── http/                      # HTTP utilities
│   └── responses.py           # Response formatting utilities
└── a2a/                       # A2A protocol implementation
    └── agents.py              # A2A conversational agents
```

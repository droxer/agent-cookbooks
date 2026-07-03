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

- `python examples/agents/super_agent.py` - Multi-agent coordination using supervisor pattern
- `python examples/agents/deep_agents.py` - Deep research agent using deepagents with Tavily search
- `python examples/agents/memorized_agent.py` - Intelligent memory agent with Qdrant-based storage and importance scoring
- `python examples/agents/shared_memory_agents.py` - Shared memory agents with team and personal memory stores
- `python examples/agents/react_agent.py` - ReAct agent with tool routing and reasoning
- `python examples/agents/llm_proxy.py` - Multi-provider LLM support using LiteLLM
- `python examples/harness/agent_harness.py` - Agent harness with budgets, permission gating, error recovery, and tracing
- `python examples/harness/durable_agent.py` - Durable (checkpointed) agent with human-in-the-loop interrupt/resume
- `python examples/loop/agentic_loop.py` - Loop engineering with generate → verify → refine and a doneness contract
- `python examples/agents/orchestrator_workers.py` - Parallel context-isolated sub-agents with bounded fan-out
- `python examples/context/context_editing.py` - Clearing stale tool results on a token trigger (offline demo)
- `python examples/evals/trajectory_eval.py` - Trajectory metrics plus LLM-as-judge grading
- `python examples/tools/code_execution.py` - Progressive tool disclosure with code execution as the tool interface
- `python examples/skills/skills_agent.py` - Folder-based agent skills loaded just-in-time
- `python examples/context/tools_call.py` - Dynamic tool selection using semantic search
- `python examples/context/offloading.py` - Context management with scratchpad
- `python examples/context/compact.py` - Context compression with summarization
- `python examples/context/pruning.py` - Context pruning techniques
- `python examples/rag/multimodal_rag.py` - Multimodal RAG with text and image embeddings
- `python examples/agents/a2a_agents.py` - A2A agent communication example
- `python examples/context/ltm.py` - Long-term memories with semantic search
- `python examples/document/pdf2images.py` - PDF image extraction with OpenAI vision analysis
- `python examples/document/text_extract.py` - Text extraction using langextract
- `python examples/validation/validators.py` - Input/output validation with Guardrails
- `python examples/evals/test_deepeval.py` - DeepEval integration for testing metrics

## Architecture

### Core Components

- **Language Models**: Uses `langchain.chat_models.init_chat_model()` for multiple providers (Anthropic, OpenAI, Gemini)
- **Tools**: Dynamic tool registry system in `tools.registry` with semantic search via vector embeddings
- **Workflows**: LangGraph StateGraph-based agents with different patterns
- **Context Management**: Various strategies for handling context window limitations
- **Validation**: Guardrails integration for input/output validation and filtering
- **Document Processing**: PDF extraction and text analysis capabilities

### Key Patterns

1. **Multi-Agent Coordination** (`examples/agents/super_agent.py`):
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
   - **Intelligent Memory Agent** (`examples/agents/memorized_agent.py`): Hybrid memory system with automatic importance scoring and timestamping
   - **Shared Memory Agents** (`examples/agents/shared_memory_agents.py`): Multi-agent system with personal and team-wide memory sharing
   - **Weighted Search**: Advanced retrieval considering semantic similarity, importance scores, and temporal decay

10. **Agent Harness Engineering** (`examples/harness/agent_harness.py`):
    - Bounded agent loop owned by the harness, with enumerated stop reasons
    - Budgets: max turns, max tool calls, wall-clock deadline
    - Per-tool permission gating (allow / confirm / deny) enforced outside the model
    - Tool errors returned as observations; oversized output truncated; model retries with backoff
    - Structured event trace for post-hoc debugging

11. **Loop Engineering** (`examples/loop/agentic_loop.py`):
    - Evaluator-optimizer loop (generate → verify → refine) built as a LangGraph StateGraph
    - Doneness contract: a verifier grades drafts against explicit acceptance criteria with structured output
    - Bounded revision budget with graceful exit to the best-scoring attempt
    - Verifier feedback routed into the next generation attempt; attempt history and stop reason recorded

12. **Orchestrator-Worker Sub-Agents** (`examples/agents/orchestrator_workers.py`):
    - Planner decomposes the task; workers run in parallel via LangGraph `Send`
    - Bounded fan-out enforced in code; workers get fresh, isolated contexts
    - Workers hand back compressed summaries, never raw transcripts

13. **Durable Execution & Human-in-the-Loop** (`examples/harness/durable_agent.py`):
    - Checkpointer persists state each superstep; runs resume by `thread_id` after restarts
    - Sensitive tools pause with `interrupt()`; decisions arrive later via `Command(resume=...)`
    - Denials become observations the model must respect

14. **Context Editing** (`examples/context/context_editing.py`):
    - Replaces stale tool results with a placeholder once a token trigger trips (no LLM call)
    - Keeps the most recent N tool results and preserves tool_call_id pairing

15. **Trajectory Evaluation** (`examples/evals/trajectory_eval.py`):
    - Deterministic tool-sequence checks (order, redundancy, step budget) for CI
    - LLM-as-judge rubric grading groundedness against the trajectory's tool evidence

16. **Code Execution with Tools** (`examples/tools/code_execution.py`):
    - Progressive disclosure: prompt carries tool names/one-liners; `search_tools` returns full signatures on demand
    - `run_code` executes agent-written Python against the tool API; only printed output re-enters context
    - Large intermediate datasets never flow through the context window

17. **Agent Skills** (`examples/skills/skills_agent.py`):
    - Skills are folders (`library/<name>/SKILL.md`): frontmatter metadata + markdown procedure
    - Metadata-only at startup; full instructions loaded via `use_skill` when a task matches
    - Expertise is added by dropping in a folder, with no code changes

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

## Dependencies

The project uses a comprehensive stack of agentic AI libraries:
- **LangChain**: Core LLM orchestration and tool management
- **LangGraph**: Workflow and state management
- **LangGraph-Supervisor**: Multi-agent coordination
- **LangGraph-Bigtool**: Advanced tool handling
- **LangGraph-Runtime**: A2A protocol support
- **FastMCP**: Model Context Protocol server implementation
- **Vector Stores**: PGVector, Chroma, Qdrant, LanceDB, SQLite-vec for embeddings
- **Qdrant Client**: Native Qdrant vector database client
- **HuggingFace Embeddings**: Sentence transformers for vector embeddings
- **Validation**: Guardrails for input/output validation
- **Document Processing**: Langextract for text extraction
- **Multi-provider LLM**: LiteLLM for unified API access
- **Search**: Tantivy search engine
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
├── agents/                    # Agent implementations
│   ├── super_agent.py               # Multi-agent supervisor coordination
│   ├── deep_agents.py               # Deep research agent (deepagents + Tavily)
│   ├── memorized_agent.py           # Intelligent memory agent with Qdrant
│   ├── shared_memory_agents.py      # Shared memory agents with team/personal stores
│   ├── react_agent.py               # ReAct pattern implementation
│   ├── llm_proxy.py                 # Multi-provider LLM proxy
│   ├── orchestrator_workers.py      # Parallel context-isolated sub-agents (Send API)
│   ├── a2a_agents.py                # A2A JSON-RPC client example
│   └── a2a/                         # A2A protocol implementation
│       └── agents.py                # A2A conversational agents
├── harness/                   # Agent harness engineering
│   ├── agent_harness.py       # Budgets, permission gating, error recovery, tracing
│   └── durable_agent.py       # Checkpointed runs with human-in-the-loop interrupts
├── loop/                      # Loop engineering
│   └── agentic_loop.py        # Generate → verify → refine with doneness contract
├── context/                   # Context management strategies
│   ├── tools_call.py          # Dynamic tool selection via semantic search
│   ├── offloading.py          # Scratchpad context offloading
│   ├── compact.py             # Tool output summarization
│   ├── pruning.py             # Selective context retention
│   ├── context_editing.py     # Clearing stale tool results on a token trigger
│   └── ltm.py                 # Long-term memories with semantic search
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
│   ├── retriever_tool.py     # Blog post retriever
│   └── code_execution.py     # Progressive disclosure + run_code data flow
├── skills/                    # Agent skills (progressive disclosure)
│   ├── skills_agent.py        # Skill discovery, loading, and agent loop
│   └── library/               # One folder per skill, each with SKILL.md
├── evals/                     # Evaluation implementations
│   ├── test_deepeval.py      # DeepEval integration
│   └── trajectory_eval.py    # Trajectory metrics + LLM-as-judge
├── validation/                # Input/output validation
│   ├── validators.py         # Guardrails integration
│   ├── inputs.py             # Input validation utilities
│   └── outputs.py            # Output validation utilities
└── http/                      # HTTP utilities
    └── responses.py           # Response formatting utilities
```
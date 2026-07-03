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

### 1. Multi-Agent Coordination (`examples/agents/super_agent.py`)
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

### 12. Intelligent Memory Agent (`examples/agents/memorized_agent.py`)
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

### 21. Deep Agents (`examples/agents/deep_agents.py`)
Deep research agent using the `deepagents` package with web search via Tavily

### 22. Agent Harness Engineering (`examples/harness/agent_harness.py`)
The outer loop that owns the agent: budgets (turns, tool calls, wall-clock deadline),
per-tool permission gating (allow/confirm/deny), tool error recovery, output
truncation, model retries with exponential backoff, and a structured trace

### 23. Loop Engineering (`examples/loop/agentic_loop.py`)
Generate → verify → refine (evaluator-optimizer) loop with an explicit doneness
contract, a bounded revision budget, feedback routing between iterations, and
graceful degradation to the best attempt when the budget is exhausted

### 24. Orchestrator-Worker Sub-Agents (`examples/agents/orchestrator_workers.py`)
Parallel, context-isolated worker agents dispatched via LangGraph `Send`, with a
bounded fan-out budget and compressed findings handoffs back to the orchestrator

### 25. Durable Execution & Human-in-the-Loop (`examples/harness/durable_agent.py`)
Checkpointed agent runs that survive restarts; sensitive tool calls pause with
LangGraph `interrupt()` and resume from the checkpoint with a human decision

### 26. Context Editing (`examples/context/context_editing.py`)
Clearing stale tool results in place once a token trigger trips — the cheap,
LLM-free complement to compaction and pruning (mirrors the Claude API's
`clear_tool_uses` context editing)

### 27. Trajectory Evaluation (`examples/evals/trajectory_eval.py`)
Grading the agent's path, not just its answer: deterministic tool-sequence
metrics (order, redundancy, step budget) plus an LLM-as-judge rubric that
scores groundedness against the trajectory's tool evidence

### 28. Code Execution with Tools (`examples/tools/code_execution.py`)
Progressive tool disclosure (`search_tools` pulls full signatures on demand)
plus a `run_code` execution environment where the agent filters and joins
large datasets in code — only printed results re-enter the context window

### 29. Agent Skills (`examples/skills/skills_agent.py`)
Folder-based expertise (`library/<name>/SKILL.md`) loaded with progressive
disclosure: only frontmatter metadata enters the system prompt; the full
procedure is fetched via `use_skill` when a task matches

## Architecture

### Core Components

- **Language Models**: Uses `langchain.chat_models.init_chat_model()` for multiple providers (Anthropic, OpenAI, Gemini)
- **Tools**: Dynamic tool registry system in `tools.registry` with semantic search via vector embeddings
- **Workflows**: LangGraph StateGraph-based agents with different patterns
- **Context Management**: Various strategies for handling context window limitations

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
    - The harness owns the loop: bounded iteration with enumerated stop reasons
    - Budgets enforced outside the model: max turns, max tool calls, wall-clock deadline
    - Per-tool permission gating (allow / confirm / deny)
    - Tool failures fed back as observations instead of crashing the run
    - Oversized tool output truncated before entering the context window
    - Model retries with exponential backoff and a structured event trace

11. **Loop Engineering** (`examples/loop/agentic_loop.py`):
    - Evaluator-optimizer loop: generate → verify → refine
    - Doneness decided by a verifier against explicit acceptance criteria, not by the generator's self-report
    - Bounded revision budget with graceful exit to the best-scoring attempt
    - Verifier feedback routed into the next generation attempt
    - Full attempt history and stop reason recorded for auditability

12. **Orchestrator-Worker Sub-Agents** (`examples/agents/orchestrator_workers.py`):
    - Planner decomposes the task; workers run in parallel via LangGraph `Send`
    - Bounded fan-out enforced in code, not in the prompt
    - Each worker gets a fresh, isolated context containing only its subtask
    - Workers hand back compressed summaries, never raw transcripts

13. **Durable Execution & Human-in-the-Loop** (`examples/harness/durable_agent.py`):
    - Checkpointer persists state every superstep; runs resume by `thread_id` after a crash or restart
    - Sensitive tools pause the run with `interrupt()`; the pause itself is checkpointed
    - Human decisions arrive asynchronously via `Command(resume=...)` — no one needs to be at the keyboard
    - Denials become observations the model must respect, not exceptions

14. **Context Editing** (`examples/context/context_editing.py`):
    - Replaces stale tool results with a placeholder once a token trigger trips
    - Keeps the most recent N tool results intact and preserves tool_call_id pairing
    - Pure string edit: no LLM call, unlike compaction/summarization

15. **Trajectory Evaluation** (`examples/evals/trajectory_eval.py`):
    - Deterministic layer for CI: required tools in order, no redundant calls, step budget
    - LLM-as-judge layer for samples: groundedness and completeness graded against the tool evidence
    - Catches right-answer-wrong-path failures that final-answer evals miss

16. **Code Execution with Tools** (`examples/tools/code_execution.py`):
    - System prompt lists tool names and one-liners only; `search_tools` discloses full signatures on demand
    - The agent computes in code via `run_code`: large intermediate results stay in the execution environment
    - Only printed output returns to the model, so context cost tracks the answer, not the data
    - Sandboxed execution (stripped builtins, no imports) with the production caveat documented

17. **Agent Skills** (`examples/skills/skills_agent.py`):
    - Skills are folders (`library/<name>/SKILL.md`) with frontmatter metadata + markdown procedure
    - Startup loads metadata only; `use_skill` loads the full instructions just-in-time
    - Adding expertise means dropping in a folder — no code changes, writable by domain experts

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

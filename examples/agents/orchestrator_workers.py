"""Orchestrator-worker pattern: parallel, context-isolated sub-agents.

The single-agent context window is the bottleneck for research-style tasks.
The state-of-the-art answer (used by production deep-research systems) is an
orchestrator that decomposes the task and fans it out to worker agents that
run *in parallel*, each with its own isolated context, and that hand back a
*compressed* summary instead of their full transcript.

Key practices demonstrated:

1. **Bounded fan-out** - the planner is asked for at most ``MAX_WORKERS``
   subtasks and the orchestrator truncates anything beyond that. Effort is an
   explicit budget, not whatever the model feels like.
2. **Context isolation** - each worker starts from a fresh message list
   containing only its subtask. Workers never see each other's transcripts,
   so their contexts stay small and focused.
3. **Compressed handoffs** - a worker returns a distilled summary of its
   findings, not its raw tool transcript. The orchestrator's context grows by
   kilobytes, not megabytes.
4. **Parallelism via LangGraph ``Send``** - subtasks execute concurrently in
   one superstep; results merge through a list reducer.

Run with:
    python examples/agents/orchestrator_workers.py
"""

import operator
from typing import Annotated, List

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from examples.harness.agent_harness import message_text

load_dotenv()

llm = init_chat_model("anthropic:claude-sonnet-4-20250514", temperature=0)

MAX_WORKERS = 3       # fan-out budget: hard cap on parallel sub-agents
MAX_WORKER_TURNS = 4  # per-worker loop budget


@tool
def web_search(query: str) -> str:
    """Search the web for information about a topic (mock)."""
    corpus = {
        "electricity": "Data centers used ~460 TWh in 2024, ~2% of global demand.",
        "cooling": "Liquid cooling adoption grew 3x in 2024; PUE averages fell to 1.4.",
        "chips": "AI accelerator shipments doubled in 2024, led by 5nm-class GPUs.",
    }
    for keyword, result in corpus.items():
        if keyword in query.lower():
            return f"[search: {query!r}] {result}"
    return f"[search: {query!r}] No strong matches; try a more specific query."


# ---------------------------------------------------------------------------
# State
#
# The orchestrator state holds only the plan and each worker's compressed
# findings. Worker transcripts never enter this state.
# ---------------------------------------------------------------------------

class Plan(BaseModel):
    subtasks: List[str] = Field(
        description=f"Independent research subtasks, at most {MAX_WORKERS}. "
                    "Each must be answerable on its own, without the others."
    )


class OrchestratorState(TypedDict):
    task: str
    subtasks: List[str]
    findings: Annotated[List[str], operator.add]  # reducer merges parallel writes
    report: str


class WorkerState(TypedDict):
    """What a worker receives: its subtask and nothing else."""
    subtask: str
    findings: Annotated[List[str], operator.add]


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def plan(state: OrchestratorState) -> dict:
    """Decompose the task into independent subtasks, bounded by MAX_WORKERS."""
    result = llm.with_structured_output(Plan).invoke(
        f"Decompose this research task into at most {MAX_WORKERS} independent "
        f"subtasks that can be researched in parallel:\n\n{state['task']}"
    )
    subtasks = result.subtasks[:MAX_WORKERS]  # enforce the budget in code
    print(f"[orchestrator] plan: {subtasks}")
    return {"subtasks": subtasks}


def fan_out(state: OrchestratorState) -> list[Send]:
    """Dispatch each subtask to a parallel worker with an isolated payload."""
    return [Send("worker", {"subtask": subtask}) for subtask in state["subtasks"]]


def worker(state: WorkerState) -> dict:
    """Research one subtask in an isolated context and return a compressed summary."""
    model = llm.bind_tools([web_search])
    # Fresh context: the worker sees only its own subtask.
    messages = [
        SystemMessage(content=(
            "You are a research worker. Use web_search to investigate the "
            "subtask, then answer with AT MOST 3 bullet points of findings. "
            "Return only the findings - no preamble, no transcript."
        )),
        HumanMessage(content=state["subtask"]),
    ]
    for _ in range(MAX_WORKER_TURNS):
        response = model.invoke(messages)
        messages.append(response)
        if not response.tool_calls:
            break
        for tool_call in response.tool_calls:
            observation = web_search.invoke(tool_call["args"])
            messages.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))

    # Compressed handoff: only the distilled summary leaves the worker.
    summary = message_text(messages[-1])
    print(f"[worker] {state['subtask']!r} -> {len(summary)} chars of findings")
    return {"findings": [f"### {state['subtask']}\n{summary}"]}


def synthesize(state: OrchestratorState) -> dict:
    """Merge the compressed findings into a final report."""
    findings = "\n\n".join(state["findings"])
    response = llm.invoke(
        f"Write a concise report answering this task:\n{state['task']}\n\n"
        f"Base it strictly on these findings from your research team:\n\n{findings}"
    )
    return {"report": message_text(response)}


# ---------------------------------------------------------------------------
# Graph: plan -> (parallel workers) -> synthesize
# ---------------------------------------------------------------------------

builder = StateGraph(OrchestratorState)
builder.add_node("plan", plan)
builder.add_node("worker", worker)
builder.add_node("synthesize", synthesize)

builder.add_edge(START, "plan")
builder.add_conditional_edges("plan", fan_out, ["worker"])
builder.add_edge("worker", "synthesize")
builder.add_edge("synthesize", END)

orchestrator = builder.compile()


def main():
    result = orchestrator.invoke({
        "task": "How is AI growth straining data center infrastructure? "
                "Cover electricity demand, cooling, and chip supply.",
        "subtasks": [],
        "findings": [],
        "report": "",
    })
    print("\n=== REPORT ===")
    print(result["report"])


if __name__ == "__main__":
    main()

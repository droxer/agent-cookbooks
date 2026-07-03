"""Loop engineering: generate -> verify -> refine.

Harness engineering (see ``examples/harness/agent_harness.py``) is about
running one agent turn safely. Loop engineering is about shaping the *outer*
iteration so the agent converges on a good result instead of stopping at its
first draft or spinning forever.

The pattern here is the evaluator-optimizer loop, with the properties every
production agent loop needs:

1. **A doneness contract** - "done" is decided by a verifier checking explicit
   acceptance criteria and returning a structured verdict, not by the
   generator declaring itself finished. Generators grade their own work
   optimistically; a separate verification step catches that.
2. **A revision budget** - the loop is bounded. When the budget runs out the
   loop exits with the best attempt so far, recorded as such, rather than
   looping again or raising.
3. **Feedback routing** - the verifier's critique is injected into the next
   generation attempt, so each iteration improves on the last instead of
   resampling blindly.
4. **Attempt history and stop reasons** - the state records every attempt,
   every verdict, and *why* the loop stopped, so a run can be audited.

Run with:
    python examples/loop/agentic_loop.py
"""

from typing import Literal

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

load_dotenv()

llm = init_chat_model("anthropic:claude-sonnet-4-20250514", temperature=0)

MAX_REVISIONS = 3  # revision budget: total attempts = 1 + MAX_REVISIONS


# ---------------------------------------------------------------------------
# The doneness contract: acceptance criteria + a structured verdict
# ---------------------------------------------------------------------------

ACCEPTANCE_CRITERIA = """
1. The summary is at most 3 sentences long.
2. It states what the technique is AND why it matters.
3. It is understandable by a reader with no machine-learning background:
   no unexplained jargon (e.g. "gradient", "token", "transformer").
4. It contains no marketing language or filler.
"""


class Verdict(BaseModel):
    """Structured output for the verifier: a machine-checkable pass/fail."""
    passed: bool = Field(description="True only if EVERY criterion is met")
    score: float = Field(ge=0.0, le=1.0, description="Overall quality, 0-1")
    failures: list[str] = Field(
        default_factory=list,
        description="Each criterion that failed, with a concrete fix",
    )

verifier = llm.with_structured_output(Verdict)


class Attempt(TypedDict):
    draft: str
    verdict: Verdict


class LoopState(TypedDict):
    task: str
    draft: str
    attempts: list[Attempt]
    revisions_used: int
    stop_reason: str  # "passed" | "budget_exhausted"


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def generate(state: LoopState) -> dict:
    """Produce a draft; on revisions, route the verifier feedback back in."""
    prompt = (
        f"Write a summary for this task:\n{state['task']}\n\n"
        f"It must satisfy ALL of these criteria:\n{ACCEPTANCE_CRITERIA}"
    )
    if state["attempts"]:
        last = state["attempts"][-1]
        failures = "\n".join(f"- {f}" for f in last["verdict"].failures)
        prompt += (
            f"\n\nYour previous draft was rejected:\n{last['draft']}\n\n"
            f"Reviewer feedback (fix every point):\n{failures}"
        )
    response = llm.invoke(prompt)
    # .text is a method in langchain-core 0.3.x and a property in 1.x
    draft = (response.text() if callable(response.text) else response.text).strip()
    print(f"\n--- draft (attempt {len(state['attempts']) + 1}) ---\n{draft}")
    return {"draft": draft}


def verify(state: LoopState) -> dict:
    """Grade the draft against the acceptance criteria - never trust the
    generator's own opinion of its work."""
    verdict = verifier.invoke(
        f"You are a strict reviewer. Grade this summary against the criteria.\n\n"
        f"Criteria:\n{ACCEPTANCE_CRITERIA}\n\nSummary:\n{state['draft']}"
    )
    print(f"--- verdict: passed={verdict.passed} score={verdict.score:.2f} "
          f"failures={verdict.failures}")
    return {
        "attempts": state["attempts"] + [Attempt(draft=state["draft"], verdict=verdict)],
        "revisions_used": len(state["attempts"]),
    }


def finalize(state: LoopState) -> dict:
    """Exit the loop with an explicit stop reason and the best attempt.

    On budget exhaustion we still return the *highest-scoring* draft --
    a bounded loop should degrade gracefully, not discard its work.
    """
    last = state["attempts"][-1]
    if last["verdict"].passed:
        return {"stop_reason": "passed", "draft": last["draft"]}
    best = max(state["attempts"], key=lambda a: a["verdict"].score)
    return {"stop_reason": "budget_exhausted", "draft": best["draft"]}


def route_after_verify(state: LoopState) -> Literal["generate", "finalize"]:
    """The loop's only branch point: doneness contract OR budget, nothing else."""
    if state["attempts"][-1]["verdict"].passed:
        return "finalize"
    if len(state["attempts"]) > MAX_REVISIONS:
        return "finalize"
    return "generate"


# ---------------------------------------------------------------------------
# Graph: generate -> verify -> (loop back | finalize)
# ---------------------------------------------------------------------------

builder = StateGraph(LoopState)
builder.add_node("generate", generate)
builder.add_node("verify", verify)
builder.add_node("finalize", finalize)

builder.add_edge(START, "generate")
builder.add_edge("generate", "verify")
builder.add_conditional_edges("verify", route_after_verify,
                              {"generate": "generate", "finalize": "finalize"})
builder.add_edge("finalize", END)

loop = builder.compile()


def main():
    result = loop.invoke({
        "task": "Summarize what retrieval-augmented generation (RAG) is "
                "for a company-wide newsletter.",
        "draft": "",
        "attempts": [],
        "revisions_used": 0,
        "stop_reason": "",
    })

    print("\n=== LOOP RESULT ===")
    print(f"stop_reason: {result['stop_reason']}")
    print(f"attempts: {len(result['attempts'])}")
    print(f"final draft:\n{result['draft']}")


if __name__ == "__main__":
    main()

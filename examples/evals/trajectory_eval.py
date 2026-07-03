"""Trajectory evaluation: grade the path, not just the answer.

Final-answer evals (see ``test_deepeval.py``) miss the failure modes that
actually dominate agent behavior: the agent that gets the right answer after
nine redundant tool calls, the one that skips retrieval and answers from
parametric memory, the one that calls tools in an order that only worked by
luck. Modern agent evals therefore grade the *trajectory* - the sequence of
tool calls - alongside the answer.

Two complementary layers, both demonstrated here:

1. **Programmatic trajectory metrics** (deterministic, free, run on every CI
   commit): required tools present, called in a valid order, no redundant
   calls, within a step budget.
2. **LLM-as-judge** (probabilistic, costs a call, run on samples): a rubric
   graded with structured output, judging the answer *against the tool
   evidence in the trajectory* - which catches ungrounded answers that string
   metrics cannot.

The programmatic layer runs offline; the judge runs only when an API key is
configured.

Run with:
    python examples/evals/trajectory_eval.py
"""

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()


# ---------------------------------------------------------------------------
# Trajectory capture
#
# In a real system you record this from your agent loop or framework traces
# (LangGraph streams, OTel spans). Here we use two canned trajectories.
# ---------------------------------------------------------------------------

@dataclass
class ToolCall:
    name: str
    args: dict
    result: str


@dataclass
class Trajectory:
    task: str
    tool_calls: list[ToolCall]
    final_answer: str


@dataclass
class TrajectoryReport:
    passed: bool
    checks: dict[str, bool]
    notes: list[str] = field(default_factory=list)


def evaluate_trajectory(
    trajectory: Trajectory,
    required_tools: list[str],
    max_steps: int,
) -> TrajectoryReport:
    """Deterministic trajectory checks - the CI-grade layer."""
    names = [c.name for c in trajectory.tool_calls]
    notes = []

    # Required tools present, in the required (subsequence) order.
    position = 0
    for required in required_tools:
        try:
            position = names.index(required, position) + 1
        except ValueError:
            position = -1
            break
    in_order = position != -1
    if not in_order:
        notes.append(f"required order {required_tools} not found in {names}")

    # No redundant calls: same tool + same args twice is wasted budget.
    seen = set()
    redundant = []
    for call in trajectory.tool_calls:
        key = (call.name, tuple(sorted(call.args.items())))
        if key in seen:
            redundant.append(call.name)
        seen.add(key)
    if redundant:
        notes.append(f"redundant calls: {redundant}")

    within_budget = len(names) <= max_steps
    if not within_budget:
        notes.append(f"{len(names)} calls exceeds budget of {max_steps}")

    checks = {
        "required_tools_in_order": in_order,
        "no_redundant_calls": not redundant,
        "within_step_budget": within_budget,
    }
    return TrajectoryReport(passed=all(checks.values()), checks=checks, notes=notes)


# ---------------------------------------------------------------------------
# LLM-as-judge: grade the answer against the trajectory's evidence
# ---------------------------------------------------------------------------

class JudgeVerdict(BaseModel):
    groundedness: float = Field(ge=0, le=1, description=(
        "Is every claim in the answer supported by the tool evidence? "
        "1.0 = fully grounded, 0.0 = fabricated."))
    completeness: float = Field(ge=0, le=1, description=(
        "Does the answer use all relevant evidence the tools returned?"))
    rationale: str = Field(description="One-paragraph justification citing evidence")


def judge_trajectory(trajectory: Trajectory) -> JudgeVerdict:
    from langchain.chat_models import init_chat_model

    llm = init_chat_model("anthropic:claude-sonnet-4-20250514", temperature=0)
    evidence = "\n".join(
        f"- {c.name}({c.args}) -> {c.result}" for c in trajectory.tool_calls
    )
    return llm.with_structured_output(JudgeVerdict).invoke(
        "You are grading an AI agent's answer STRICTLY against the evidence "
        "its tools returned. Claims not supported by the evidence lower "
        f"groundedness.\n\nTask: {trajectory.task}\n\n"
        f"Tool evidence:\n{evidence}\n\nAgent's answer:\n{trajectory.final_answer}"
    )


# ---------------------------------------------------------------------------
# Demo: one good trajectory, one bad one
# ---------------------------------------------------------------------------

GOOD = Trajectory(
    task="What is the combined 2024 headcount of Meta and Netflix?",
    tool_calls=[
        ToolCall("lookup_headcount", {"company": "Meta"}, "Meta: 67,317 employees"),
        ToolCall("lookup_headcount", {"company": "Netflix"}, "Netflix: 14,000 employees"),
        ToolCall("calculate", {"expression": "67317 + 14000"}, "81317"),
    ],
    final_answer="Meta (67,317) and Netflix (14,000) together employ 81,317 people.",
)

BAD = Trajectory(
    task="What is the combined 2024 headcount of Meta and Netflix?",
    tool_calls=[
        ToolCall("lookup_headcount", {"company": "Meta"}, "Meta: 67,317 employees"),
        ToolCall("lookup_headcount", {"company": "Meta"}, "Meta: 67,317 employees"),
        # Never looked up Netflix, never calculated - answer is unsupported.
    ],
    final_answer="Meta and Netflix together employ roughly 200,000 people.",
)

REQUIRED_TOOLS = ["lookup_headcount", "calculate"]
MAX_STEPS = 5


def main():
    for label, trajectory in [("GOOD", GOOD), ("BAD", BAD)]:
        report = evaluate_trajectory(trajectory, REQUIRED_TOOLS, MAX_STEPS)
        print(f"\n=== {label} trajectory ===")
        print(f"passed: {report.passed}")
        for check, ok in report.checks.items():
            print(f"  {'PASS' if ok else 'FAIL'} {check}")
        for note in report.notes:
            print(f"  note: {note}")

    if os.getenv("ANTHROPIC_API_KEY"):
        print("\n=== LLM-as-judge ===")
        for label, trajectory in [("GOOD", GOOD), ("BAD", BAD)]:
            verdict = judge_trajectory(trajectory)
            print(f"{label}: groundedness={verdict.groundedness:.2f} "
                  f"completeness={verdict.completeness:.2f}\n  {verdict.rationale}")
    else:
        print("\n(no ANTHROPIC_API_KEY set - skipping the LLM-as-judge layer)")


if __name__ == "__main__":
    main()

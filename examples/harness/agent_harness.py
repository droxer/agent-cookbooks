"""Agent harness engineering.

The harness is the code that *owns* the agentic loop. The model proposes tool
calls; the harness decides whether to run them, runs them safely, and decides
when the loop stops. A well-engineered harness gives you agents that fail
predictably instead of crashing, looping forever, or blowing the context window.

This example demonstrates the core harness responsibilities:

1. **The loop belongs to the harness, not the model** - a bounded
   ``while`` loop with explicit, enumerated stop reasons.
2. **Budgets** - max turns, max tool calls, and a wall-clock deadline.
3. **Permission gating** - per-tool allow / deny / confirm policies enforced
   *outside* the model.
4. **Error recovery** - a failing tool never crashes the loop; the error is
   returned to the model as an observation so it can adapt.
5. **Output hygiene** - oversized tool results are truncated before they
   enter the context window.
6. **Retries with backoff** - transient model API errors are retried with
   exponential backoff.
7. **Tracing** - every event is recorded so a run can be debugged after the fact.

Run with:
    python examples/harness/agent_harness.py
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool

load_dotenv()


def message_text(message: BaseMessage) -> str:
    """Extract text from a message across langchain-core versions
    (``.text`` is a method in 0.3.x and a property in 1.x)."""
    return message.text() if callable(message.text) else message.text


# ---------------------------------------------------------------------------
# Tools
#
# One reliable tool, one flaky tool (to show error recovery), and one
# sensitive tool (to show permission gating).
# ---------------------------------------------------------------------------

@tool
def calculate(expression: str) -> str:
    """Evaluate a basic arithmetic expression, e.g. '2 + 3 * 4'."""
    allowed = set("0123456789+-*/(). ")
    if not set(expression) <= allowed:
        raise ValueError(f"Expression contains unsupported characters: {expression!r}")
    return str(eval(expression, {"__builtins__": {}}, {}))


_lookup_attempts = {"count": 0}

@tool
def lookup_headcount(company: str) -> str:
    """Look up the 2024 employee headcount for a company."""
    # Fail on the first call to demonstrate that the harness feeds errors
    # back to the model instead of crashing the loop.
    _lookup_attempts["count"] += 1
    if _lookup_attempts["count"] == 1:
        raise ConnectionError("upstream headcount service timed out")

    headcounts = {
        "meta": "67,317", "apple": "164,000", "amazon": "1,551,000",
        "netflix": "14,000", "alphabet": "181,269",
    }
    key = company.strip().lower()
    if key not in headcounts:
        return f"No headcount data for {company!r}. Known companies: {sorted(headcounts)}"
    return f"{company} 2024 headcount: {headcounts[key]} employees"


@tool
def delete_records(table: str) -> str:
    """Delete all records from a database table."""
    return f"Deleted all records from {table}"


# ---------------------------------------------------------------------------
# Harness configuration and policy
# ---------------------------------------------------------------------------

class Permission(Enum):
    ALLOW = "allow"      # run without asking
    CONFIRM = "confirm"  # ask a human first (simulated here)
    DENY = "deny"        # never run


class StopReason(Enum):
    COMPLETED = "completed"              # model produced a final answer
    MAX_TURNS = "max_turns"              # turn budget exhausted
    MAX_TOOL_CALLS = "max_tool_calls"    # tool-call budget exhausted
    DEADLINE = "deadline"                # wall-clock deadline hit
    MODEL_ERROR = "model_error"          # model API kept failing after retries


@dataclass
class HarnessConfig:
    max_turns: int = 10                  # LLM round-trips per run
    max_tool_calls: int = 20             # total tool executions per run
    deadline_seconds: float = 120.0      # wall-clock budget for the whole run
    max_tool_output_chars: int = 4_000   # truncate tool output beyond this
    max_model_retries: int = 3           # retries on transient model errors
    retry_base_delay: float = 1.0        # backoff: base * 2^attempt seconds
    permissions: dict[str, Permission] = field(default_factory=dict)
    default_permission: Permission = Permission.ALLOW
    confirm: Callable[[str, dict], bool] = lambda name, args: False


@dataclass
class HarnessResult:
    stop_reason: StopReason
    final_answer: str | None
    messages: list[BaseMessage]
    trace: list[dict[str, Any]]
    turns_used: int
    tool_calls_used: int


class AgentHarness:
    """A minimal production-shaped harness around a tool-calling chat model."""

    def __init__(self, model, tools, config: HarnessConfig | None = None):
        self.config = config or HarnessConfig()
        self.tools_by_name = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    # -- observability ------------------------------------------------------

    def _record(self, trace: list, event: str, **data):
        entry = {"event": event, "elapsed": round(time.monotonic() - self._t0, 2), **data}
        trace.append(entry)
        detail = ", ".join(f"{k}={v!r}" for k, v in data.items())
        print(f"[harness +{entry['elapsed']:>5.1f}s] {event}: {detail}")

    # -- model invocation with retry/backoff --------------------------------

    def _invoke_model(self, messages, trace) -> AIMessage | None:
        for attempt in range(self.config.max_model_retries + 1):
            try:
                return self.model.invoke(messages)
            except Exception as exc:  # transient API errors: rate limit, network
                if attempt == self.config.max_model_retries:
                    self._record(trace, "model_failed", error=str(exc))
                    return None
                delay = self.config.retry_base_delay * (2 ** attempt)
                self._record(trace, "model_retry", attempt=attempt + 1,
                             delay=delay, error=str(exc))
                time.sleep(delay)

    # -- tool execution: gate, run, sanitize --------------------------------

    def _execute_tool(self, tool_call: dict, trace) -> ToolMessage:
        name, args = tool_call["name"], tool_call["args"]

        if name not in self.tools_by_name:
            # The model hallucinated a tool. Tell it, don't crash.
            self._record(trace, "tool_unknown", tool=name)
            return ToolMessage(
                content=f"Error: unknown tool {name!r}. "
                        f"Available tools: {sorted(self.tools_by_name)}",
                tool_call_id=tool_call["id"],
            )

        permission = self.config.permissions.get(name, self.config.default_permission)
        if permission is Permission.DENY or (
            permission is Permission.CONFIRM and not self.config.confirm(name, args)
        ):
            self._record(trace, "tool_denied", tool=name, permission=permission.value)
            return ToolMessage(
                content=f"Permission denied: {name!r} is not allowed in this session. "
                        "Do not retry it; continue without it.",
                tool_call_id=tool_call["id"],
            )

        try:
            output = str(self.tools_by_name[name].invoke(args))
        except Exception as exc:
            # Feed the failure back as an observation so the model can adapt
            # (retry, use another tool, or explain the limitation).
            self._record(trace, "tool_error", tool=name, error=str(exc))
            return ToolMessage(
                content=f"Error running {name}: {exc}. "
                        "You may retry once or work around it.",
                tool_call_id=tool_call["id"],
            )

        if len(output) > self.config.max_tool_output_chars:
            kept = self.config.max_tool_output_chars
            self._record(trace, "tool_output_truncated", tool=name,
                         original_chars=len(output), kept_chars=kept)
            output = output[:kept] + f"\n... [truncated {len(output) - kept} chars]"

        self._record(trace, "tool_ok", tool=name, output_chars=len(output))
        return ToolMessage(content=output, tool_call_id=tool_call["id"])

    # -- the loop ------------------------------------------------------------

    def run(self, system_prompt: str, user_input: str) -> HarnessResult:
        self._t0 = time.monotonic()
        trace: list[dict[str, Any]] = []
        messages: list[BaseMessage] = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_input),
        ]
        turns = tool_calls = 0

        def finish(reason: StopReason, answer: str | None) -> HarnessResult:
            self._record(trace, "stop", reason=reason.value)
            return HarnessResult(reason, answer, messages, trace, turns, tool_calls)

        while True:
            # Budget checks happen at the top of the loop, in the harness,
            # where the model can't talk its way past them.
            if turns >= self.config.max_turns:
                return finish(StopReason.MAX_TURNS, None)
            if time.monotonic() - self._t0 > self.config.deadline_seconds:
                return finish(StopReason.DEADLINE, None)

            turns += 1
            response = self._invoke_model(messages, trace)
            if response is None:
                return finish(StopReason.MODEL_ERROR, None)
            messages.append(response)

            if not response.tool_calls:
                # No tool calls means the model is done: this is the loop's
                # natural exit, decided by observable state rather than by
                # asking the model "are you done?".
                return finish(StopReason.COMPLETED, message_text(response))

            self._record(trace, "turn", n=turns,
                         tool_calls=[tc["name"] for tc in response.tool_calls])

            for tool_call in response.tool_calls:
                if tool_calls >= self.config.max_tool_calls:
                    messages.append(ToolMessage(
                        content="Tool budget exhausted. Answer with what you have.",
                        tool_call_id=tool_call["id"],
                    ))
                    continue
                tool_calls += 1
                messages.append(self._execute_tool(tool_call, trace))


def main():
    llm = init_chat_model("anthropic:claude-sonnet-4-20250514", temperature=0)

    harness = AgentHarness(
        model=llm,
        tools=[calculate, lookup_headcount, delete_records],
        config=HarnessConfig(
            max_turns=8,
            max_tool_calls=12,
            deadline_seconds=90,
            permissions={
                "calculate": Permission.ALLOW,
                "lookup_headcount": Permission.ALLOW,
                # Destructive tool: gated behind confirmation, which our
                # non-interactive confirm callback always refuses.
                "delete_records": Permission.CONFIRM,
            },
        ),
    )

    result = harness.run(
        system_prompt=(
            "You are a research assistant. Use the tools to answer. "
            "If a tool fails, retry once or work around it."
        ),
        user_input=(
            "What is the combined 2024 headcount of Meta and Netflix? "
            "Afterwards, clean up by deleting the scratch_data table."
        ),
    )

    print("\n=== RESULT ===")
    print(f"stop_reason: {result.stop_reason.value}")
    print(f"turns: {result.turns_used}, tool calls: {result.tool_calls_used}")
    print(f"final answer:\n{result.final_answer}")


if __name__ == "__main__":
    main()

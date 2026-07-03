"""Code execution with tools: progressive disclosure + context-efficient data flow.

Loading every tool definition into context upfront, and round-tripping every
intermediate result through the model, is the classic scaling failure of
tool-calling agents. The current best practice (Anthropic, "Code execution
with MCP", 2025) restructures both sides:

1. **Progressive tool disclosure** - the system prompt lists only tool
   *names and one-liners*. The model calls ``search_tools`` to pull the full
   signature of just the tools it needs, when it needs them. Context cost is
   proportional to tools *used*, not tools *available*.
2. **Code execution as the tool interface** - instead of one context
   round-trip per tool call, the model writes a Python snippet with
   ``run_code`` that calls the tools *as functions*. Large intermediate
   results (a 200-row order dump, a join across two datasets) stay inside the
   execution environment; only what the code ``print``s re-enters the model's
   context.

The demo task requires joining 200 orders against a customer table - with
direct tool calling all of that JSON would flow through the context window;
here only a one-line aggregate does.

Security note: the ``exec`` sandbox below (stripped builtins, no imports) is
demo-grade. Production systems must run agent-written code in a real sandbox
(container / microVM, no ambient filesystem or network).

Run with:
    python examples/tools/code_execution.py
"""

import contextlib
import inspect
import io
import random

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool

from examples.harness.agent_harness import AgentHarness, HarnessConfig

load_dotenv()

MAX_CODE_OUTPUT_CHARS = 2_000


# ---------------------------------------------------------------------------
# The business API: plain Python functions, deliberately data-heavy.
# None of their definitions are shown to the model upfront.
# ---------------------------------------------------------------------------

random.seed(7)
_COUNTRIES = ["Germany", "France", "Japan", "Brazil", "USA"]
_CUSTOMERS = [{"id": f"cust-{i:03d}", "country": random.choice(_COUNTRIES)}
              for i in range(40)]
_ORDERS = [{"id": f"ord-{i:04d}",
            "customer_id": f"cust-{random.randrange(40):03d}",
            "status": random.choice(["pending", "shipped", "cancelled"]),
            "total": round(random.uniform(20, 900), 2)}
           for i in range(200)]


def get_orders() -> list[dict]:
    """Return all orders. Fields: id, customer_id, status, total."""
    return _ORDERS


def get_customers() -> list[dict]:
    """Return all customers. Fields: id, country."""
    return _CUSTOMERS


def get_exchange_rate(currency: str) -> float:
    """Return the USD exchange rate for a currency code (e.g. 'EUR')."""
    return {"EUR": 0.92, "JPY": 155.0, "BRL": 5.1, "USD": 1.0}.get(currency.upper(), 1.0)


def send_email(to: str, subject: str, body: str) -> str:
    """Send an email (mock)."""
    return f"email queued to {to}: {subject}"


TOOLS_API = {f.__name__: f for f in [get_orders, get_customers,
                                     get_exchange_rate, send_email]}


def one_liner(fn) -> str:
    doc = (fn.__doc__ or "").strip().splitlines()
    return doc[0] if doc else "(no description)"


# ---------------------------------------------------------------------------
# The only two tools the model sees: discovery and execution
# ---------------------------------------------------------------------------

@tool
def search_tools(query: str) -> str:
    """Look up full signatures and docs for available tools matching a query.
    Search by what you want to do, e.g. 'orders' or 'currency'."""
    words = query.lower().split()
    matches = [
        f"def {name}{inspect.signature(fn)}:\n    \"\"\"{fn.__doc__.strip()}\"\"\""
        for name, fn in TOOLS_API.items()
        if any(w in name or w in (fn.__doc__ or "").lower() for w in words)
    ]
    return "\n\n".join(matches) if matches else (
        f"No tools matched {query!r}. Available: {sorted(TOOLS_API)}")


@tool
def run_code(code: str) -> str:
    """Execute a Python snippet. The tool functions are available as plain
    functions (e.g. orders = get_orders()). Only what you print() is returned
    to you, so filter and aggregate in code and print small final results."""
    stdout = io.StringIO()
    # Demo-grade sandbox: no imports, no dunder access, tools + a few
    # safe builtins only. Use a real sandbox in production.
    namespace = {
        "__builtins__": {
            "print": print, "len": len, "sum": sum, "min": min, "max": max,
            "sorted": sorted, "round": round, "range": range, "enumerate": enumerate,
            "zip": zip, "set": set, "dict": dict, "list": list, "str": str,
            "int": int, "float": float, "abs": abs, "any": any, "all": all,
        },
        **TOOLS_API,
    }
    try:
        with contextlib.redirect_stdout(stdout):
            exec(code, namespace)  # noqa: S102 - demo sandbox, see module docstring
    except Exception as exc:
        return f"{stdout.getvalue()}\nError: {type(exc).__name__}: {exc}"

    output = stdout.getvalue() or "(code ran but printed nothing - print your result)"
    if len(output) > MAX_CODE_OUTPUT_CHARS:
        output = output[:MAX_CODE_OUTPUT_CHARS] + "\n... [output truncated]"
    return output


# ---------------------------------------------------------------------------
# Agent: progressive-disclosure system prompt + the harness from
# examples/harness/agent_harness.py
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = f"""You are a data analyst agent.

You have a code execution environment with these functions available
(names and summaries only - call search_tools for full signatures):

{chr(10).join(f"- {name}: {one_liner(fn)}" for name, fn in TOOLS_API.items())}

Workflow:
1. Use search_tools to get the signatures of the functions you need.
2. Use run_code to compute the answer IN CODE: filter, join, and aggregate
   there, and print only the small final result. Never print raw datasets.
"""


def main():
    llm = init_chat_model("anthropic:claude-sonnet-4-20250514", temperature=0)
    harness = AgentHarness(model=llm, tools=[search_tools, run_code],
                           config=HarnessConfig(max_turns=6))

    result = harness.run(
        system_prompt=SYSTEM_PROMPT,
        user_input="What is the total value of pending orders from customers "
                   "in Germany, in USD and in EUR?",
    )

    print("\n=== RESULT ===")
    print(f"stop_reason: {result.stop_reason.value}")
    print(f"final answer:\n{result.final_answer}")

    # The point of the pattern: the 200-order dataset never entered context.
    transcript_chars = sum(len(str(m.content)) for m in result.messages)
    print(f"\ntranscript size: ~{transcript_chars} chars "
          f"(raw orders dataset alone is ~{len(str(_ORDERS))} chars)")


if __name__ == "__main__":
    main()

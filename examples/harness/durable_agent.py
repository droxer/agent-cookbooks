"""Durable execution and human-in-the-loop approval.

Two properties every production agent eventually needs, and that a plain
request/response loop cannot provide:

1. **Durability** - the graph is compiled with a checkpointer, so state is
   persisted after every superstep. A crash, deploy, or container restart
   loses at most the step in flight; the run resumes from the last checkpoint
   under the same ``thread_id`` instead of starting over.
2. **Human-in-the-loop** - sensitive tool calls pause the run with LangGraph's
   ``interrupt()``. The interrupt is itself checkpointed: the process can exit
   entirely, and hours later a human decision arrives as
   ``Command(resume=...)`` and execution continues exactly where it stopped.
   This replaces the synchronous ``confirm`` callback pattern (see
   ``agent_harness.py``), which only works while a human is at the keyboard.

Idempotency caveat: on resume, the interrupted *node* re-executes from its
start. Keep side effects after the interrupt (or make them idempotent) - here
the sensitive tool only runs once the decision is already in hand.

Run with:
    python examples/harness/durable_agent.py
"""

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command, interrupt
from langchain_core.tools import tool

from examples.harness.agent_harness import message_text

load_dotenv()

llm = init_chat_model("anthropic:claude-sonnet-4-20250514", temperature=0)


@tool
def get_balance(account: str) -> str:
    """Get the current balance of an account."""
    return f"Account {account} balance: $2,400.00"


@tool
def transfer_funds(from_account: str, to_account: str, amount: float) -> str:
    """Transfer funds between accounts. Requires human approval."""
    return f"Transferred ${amount:,.2f} from {from_account} to {to_account}"


TOOLS = [get_balance, transfer_funds]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}
# Policy lives in the harness, not the prompt: these tools always pause for
# a human decision, no matter what the model says.
SENSITIVE_TOOLS = {"transfer_funds"}

SYSTEM_PROMPT = (
    "You are a banking assistant. Use the tools to help the user. "
    "If a transfer is denied by the reviewer, do not retry it; explain instead."
)

model = llm.bind_tools(TOOLS)


def agent(state: MessagesState) -> dict:
    response = model.invoke([SystemMessage(content=SYSTEM_PROMPT)] + state["messages"])
    return {"messages": [response]}


def tools_node(state: MessagesState) -> dict:
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        name, args = tool_call["name"], tool_call["args"]

        if name in SENSITIVE_TOOLS:
            # Pause the run and persist it. The value passed to interrupt()
            # is surfaced to the operator; the value passed back via
            # Command(resume=...) becomes interrupt()'s return value.
            decision = interrupt({
                "tool": name,
                "args": args,
                "question": f"Approve {name} with {args}?",
            })
            if not decision.get("approved"):
                result.append(ToolMessage(
                    content=f"Human reviewer DENIED {name}({args}). "
                            f"Reason: {decision.get('reason', 'not given')}",
                    tool_call_id=tool_call["id"],
                ))
                continue

        observation = TOOLS_BY_NAME[name].invoke(args)
        result.append(ToolMessage(content=str(observation), tool_call_id=tool_call["id"]))
    return {"messages": result}


def route(state: MessagesState) -> str:
    return "tools" if state["messages"][-1].tool_calls else END


builder = StateGraph(MessagesState)
builder.add_node("agent", agent)
builder.add_node("tools", tools_node)
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", route, {"tools": "tools", END: END})
builder.add_edge("tools", "agent")

# The checkpointer is what makes the graph durable. MemorySaver is for demos;
# in production use a persistent saver (e.g. PostgresSaver) so runs survive
# process restarts.
app = builder.compile(checkpointer=MemorySaver())


def main():
    # The thread_id is the durable identity of this run: any process holding
    # it (and the checkpointer's store) can resume the conversation.
    config = {"configurable": {"thread_id": "payment-run-42"}}

    result = app.invoke(
        {"messages": [{"role": "user",
                       "content": "Check the balance of acct-001, then transfer "
                                  "$500 from acct-001 to acct-002."}]},
        config,
    )

    # The run paused at the sensitive tool. In production the process could
    # exit here; we resume in-process to keep the demo self-contained.
    while "__interrupt__" in result:
        request = result["__interrupt__"][0].value
        print(f"\n[approval required] {request['question']}")

        # A human (or policy engine) decides; here we approve.
        result = app.invoke(Command(resume={"approved": True}), config)

    print("\n=== FINAL ANSWER ===")
    print(message_text(result["messages"][-1]))


if __name__ == "__main__":
    main()

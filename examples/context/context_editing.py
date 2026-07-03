"""Context editing: clearing stale tool results.

Long-running agents die by context accumulation: every tool result stays in
the transcript forever, even though most of them matter for exactly one
reasoning step. The modern fix - shipped natively as "context editing" on the
Claude API (``clear_tool_uses``) - is to *edit the context in place*: once the
transcript passes a token trigger, replace the oldest tool results with a
short placeholder while keeping the most recent ones intact.

This differs from the sibling strategies in this directory:

- ``compact.py`` summarizes tool output as it arrives (pays an LLM call each time);
- ``pruning.py`` selects what to retain;
- context editing is *cheaper than both*: a pure string replacement, applied
  only when a budget trips, no LLM in the loop.

It works because agents almost never re-read old raw tool output - anything
worth keeping should have been offloaded to durable notes (see
``offloading.py``) or state. The placeholder tells the model the result
existed and that it can re-run the tool if it truly needs the data again.

This example is self-contained and runs offline.

Run with:
    python examples/context/context_editing.py
"""

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)

CLEARED_PLACEHOLDER = "[tool result cleared to save context - re-run the tool if needed]"


def estimate_tokens(messages: list[BaseMessage]) -> int:
    """Cheap token estimate (~4 chars/token). Good enough to trip a budget;
    use a real tokenizer when billing accuracy matters."""
    return sum(len(str(m.content)) for m in messages) // 4


def clear_stale_tool_results(
    messages: list[BaseMessage],
    trigger_tokens: int = 2_000,
    keep_last: int = 3,
) -> list[BaseMessage]:
    """Clear old tool results once the context passes ``trigger_tokens``.

    The ``keep_last`` most recent tool results are preserved untouched - the
    model usually still needs those. Everything older is replaced with a
    placeholder. Messages are copied, never mutated, and the conversation
    structure (tool_call_id pairing) is preserved so the transcript stays
    valid for the API.
    """
    if estimate_tokens(messages) <= trigger_tokens:
        return messages

    tool_indices = [i for i, m in enumerate(messages) if isinstance(m, ToolMessage)]
    to_clear = set(tool_indices[:-keep_last] if keep_last else tool_indices)

    edited = []
    for i, message in enumerate(messages):
        if i in to_clear and message.content != CLEARED_PLACEHOLDER:
            edited.append(ToolMessage(
                content=CLEARED_PLACEHOLDER,
                tool_call_id=message.tool_call_id,
            ))
        else:
            edited.append(message)
    return edited


def main():
    # Synthetic transcript of a research agent that made many large searches.
    messages: list[BaseMessage] = [
        HumanMessage(content="Compare the top 5 vector databases."),
    ]
    for i in range(6):
        messages.append(AIMessage(
            content="",
            tool_calls=[{"name": "web_search", "args": {"query": f"vector db {i}"},
                         "id": f"call_{i}"}],
        ))
        # Each result is large - exactly the content that ages out fastest.
        messages.append(ToolMessage(
            content=f"[search {i}] " + ("benchmark table row; " * 120),
            tool_call_id=f"call_{i}",
        ))
    messages.append(AIMessage(content="Interim analysis of findings..."))

    before = estimate_tokens(messages)

    # In a real agent loop, apply this immediately BEFORE each model call:
    #     state["messages"] = clear_stale_tool_results(state["messages"])
    #     response = llm_with_tools.invoke(state["messages"])
    edited = clear_stale_tool_results(messages, trigger_tokens=1_000, keep_last=2)

    after = estimate_tokens(edited)
    cleared = sum(1 for m in edited if isinstance(m, ToolMessage)
                  and m.content == CLEARED_PLACEHOLDER)

    print(f"messages: {len(messages)}, tool results: 6")
    print(f"tokens before: ~{before}")
    print(f"tokens after:  ~{after}  ({100 * (before - after) // before}% reduction)")
    print(f"tool results cleared: {cleared}, kept intact: {6 - cleared}")

    print("\n=== EDITED TRANSCRIPT (tool messages) ===")
    for message in edited:
        if isinstance(message, ToolMessage):
            print(f"- {message.tool_call_id}: {str(message.content)[:60]!r}")


if __name__ == "__main__":
    main()

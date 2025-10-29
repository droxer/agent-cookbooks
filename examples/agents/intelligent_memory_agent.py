import json, re
from typing import List, TypedDict
from datetime import datetime, timezone
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from examples.store.qdrant_store_adapter import QdrantStore   # ← from previous step


class ChatState(TypedDict):
    messages: List[str]
    summary: str

short_term_memory = MemorySaver()
long_term_memory = QdrantStore(collection_name="hybrid_intelligent_store")


def user_input_node(state: ChatState):
    msg = input("👤 You: ")
    state["messages"].append(f"User: {msg}")
    return state


def agent_reply_node(state: ChatState):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    query = state["messages"][-1]

    # recall semantically relevant memories
    recalled = long_term_memory.weighted_search(query, session_id="userA", k=3)
    recalled_text = "\n".join([json.dumps(r["item"], ensure_ascii=False) for r in recalled])

    prompt = f"""
You are an AI assistant with hybrid memory.

### Relevant past memories
{recalled_text or "(none)"}

### Current user message
{query}

Please answer clearly, referring to relevant memories if useful.
"""
    reply = llm.invoke(prompt).content.strip()
    print("🤖 Assistant:", reply)
    state["messages"].append(f"Assistant: {reply}")
    return state


def summarize_and_store_node(state: ChatState):
    """Summarize + rate importance before writing to long-term store."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    convo_text = "\n".join(state["messages"][-6:])

    # ask LLM to summarize and assign an importance score
    prompt = f"""
Summarize the following conversation in 1-2 sentences,
and then rate how important it is (0-1) for long-term memory.

Conversation:
{convo_text}

Respond in JSON like:
{{"summary": "...", "importance": 0.8}}
"""
    raw = llm.invoke(prompt).content
    m = re.search(r'\{.*\}', raw, re.S)
    data = json.loads(m.group(0)) if m else {"summary": raw.strip(), "importance": 0.5}

    summary = data.get("summary", "").strip()
    importance = float(data.get("importance", 0.5))

    print(f"🧠 [Summary]: {summary}")
    print(f"⭐ [Importance score]: {importance:.2f}\n")

    # store in Qdrant with timestamp + importance
    long_term_memory.put(
        key=f"memory_{datetime.now(timezone.utc).timestamp()}",
        value={"summary": summary},
        session_id="userA",
        importance_score=importance,
    )

    state["summary"] = summary
    return state


graph = StateGraph(ChatState)
graph.add_node("user", user_input_node)
graph.add_node("reply", agent_reply_node)
graph.add_node("summarize", summarize_and_store_node)
graph.add_edge("user", "reply")
graph.add_edge("reply", "summarize")
graph.add_edge("summarize", "user")
graph.set_entry_point("user")

app = graph.compile(checkpointer=short_term_memory, store=long_term_memory)


if __name__ == "__main__":
    print("🤖 Intelligent Hybrid-Memory Agent (auto importance + timestamp)")
    state = ChatState(messages=[], summary="")
    config = {"configurable": {"thread_id": "1"}}
    app.invoke(state, config)


# 🤖 Intelligent Hybrid-Memory Agent (auto importance + timestamp)

# 👤 You: I’m planning to launch an AI-native learning app next semester.
# 🤖 Assistant: That’s a great idea! It could personalize education for students.
# 🧠 [Summary]: User plans to launch an AI-native learning app next semester.
# ⭐ [Importance score]: 0.92

# 👤 You: What did I say about my upcoming project?
# 🤖 Assistant: You mentioned launching an AI-native learning app next semester.
# 🧠 [Summary]: Assistant recalled user’s project accurately.
# ⭐ [Importance score]: 0.65    
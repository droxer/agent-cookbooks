import json, re, time
from datetime import datetime, timezone
from typing import List, TypedDict
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from examples.store.qdrant_store_adapter import QdrantStore
from rich import print
from dotenv import load_dotenv
load_dotenv()

class ChatState(TypedDict):
    messages: List[str]
    summary: str
    agent_id: str

short_term_memory = MemorySaver()
personal_memory = QdrantStore(collection_name="agent_personal_memory")
team_memory = QdrantStore(collection_name="team_shared_memory")  # shared store

def recall_from_team(query: str, agent_id: str, k: int = 3):
    """Search both team memory and the agent’s personal memory."""
    personal = personal_memory.weighted_search(query, session_id=agent_id, k=k)
    shared = team_memory.weighted_search(query, session_id="team", k=k)

    combined = personal + shared
    combined.sort(key=lambda x: x["score"], reverse=True)
    return combined[:k]


def user_input_node(state: ChatState):
    msg = input(f"👤 ({state['agent_id']}) You: ")
    state["messages"].append(f"User: {msg}")
    return state


def agent_reply_node(state: ChatState):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    query = state["messages"][-1]

    recalled = recall_from_team(query, agent_id=state["agent_id"], k=4)
    recalled_text = "\n".join([json.dumps(r["item"], ensure_ascii=False) for r in recalled])

    prompt = f"""
You are Agent {state['agent_id']} collaborating with other agents.
Use both your personal and team memories when responding.

### Recalled memories
{recalled_text or "(none)"}

### Current message
{query}
"""
    reply = llm.invoke(prompt).content.strip()
    print(f"🤖 ({state['agent_id']}) Assistant:", reply)
    state["messages"].append(f"Assistant: {reply}")
    return state


def summarize_and_store_node(state: ChatState):
    """Summarize conversation; store in personal + optionally shared memory."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    convo = "\n".join(state["messages"][-6:])

    # ask model to summarize & rate importance
    prompt = f"""
Summarize this conversation in 1-2 sentences and rate its importance (0-1)
for long-term memory. Respond in JSON:
{{"summary":"...","importance":0.7,"share_with_team":true}}
Conversation:
{convo}
"""
    raw = llm.invoke(prompt).content
    m = re.search(r"\{.*\}", raw, re.S)
    data = json.loads(m.group(0)) if m else {"summary": raw.strip(), "importance": 0.5, "share_with_team": False}

    summary = data.get("summary", "")
    importance = float(data.get("importance", 0.5))
    share = bool(data.get("share_with_team", False))

    print(f"🧠 [Summary]: {summary}")
    print(f"⭐ [Importance]: {importance:.2f}")
    if share:
        print("🌐 [Shared with team]\n")
    else:
        print()

    # always store in personal memory
    personal_memory.put(
        key=f"{state['agent_id']}_{time.time()}",
        value={"summary": summary},
        session_id=state["agent_id"],
        importance_score=importance,
    )

    # optionally store in team-wide memory
    if share:
        team_memory.put(
            key=f"team_{time.time()}",
            value={"summary": summary, "source_agent": state["agent_id"]},
            session_id="team",
            importance_score=max(importance, 0.6),
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

app = graph.compile(checkpointer=short_term_memory, store=personal_memory)

if __name__ == "__main__":
    agent_id = input("Enter Agent ID (e.g. A, B): ").strip() or "A"
    print(f"🤖 Agent {agent_id} started with shared long-term memory.")
    state = ChatState(messages=[], summary="", agent_id=agent_id)
    app.invoke(state)

# Agent A
# 👤 (A) You: I discovered a new method to compress learning content using embeddings.
# 🤖 (A) Assistant: Great! Let's document this as team knowledge.
# 🧠 [Summary]: Agent A discovered a content-compression method using embeddings.
# ⭐ [Importance]: 0.85
# 🌐 [Shared with team]    

# Agent B (later)
# 👤 (B) You: Any past work on content compression for learning materials?
# 🤖 (B) Assistant: Agent A previously documented a method using embeddings.
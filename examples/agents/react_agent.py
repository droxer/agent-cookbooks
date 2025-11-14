import re
import json
from typing import List, Optional, Dict, Any
from pydantic import BaseModel

# LangGraph
from langgraph.graph import START, StateGraph, END


class MockedLLM:
    """模拟 LLM，COT / ReAct 输出"""
    def invoke(self, prompt: str) -> str:
        print("\n====== LLM INPUT ======")
        print(prompt)
        
        # 简单 COT Router 逻辑（真实 LLM 会更复杂）
        if "ROUTER:" in prompt:
            if "课程" in prompt or "course" in prompt.lower():
                return "ToolIntent: retrieve_course_details\nArgs: {\"course_id\": 12345}"
            if "查" in prompt or "search" in prompt.lower():
                return "ToolIntent: web_search\nArgs: {\"query\": \"LangGraph\"}"
            return "ToolIntent: rag_search\nArgs: {\"query\": \"AI\"}"
        
        # ReAct LLM 模拟
        if "retrieve_course_details" not in prompt:
            return """Thought: I should call the course detail API.
Action: retrieve_course_details[{"course_id": 12345}]"""
        else:
            return """Thought: I have the details.
Final: Course title = "Intro to AI"."""
        
llm = MockedLLM()



class MockedMCPClient:
    tools = {
        "retrieve_course_details": {
            "description": "Course details",
            "args_schema": {"course_id": "int"}
        }
    }
    def call(self, name, **kwargs):
        if name == "retrieve_course_details":
            return {"id": kwargs["course_id"], "title": "Intro to AI", "credits": 3}
        return {"error": "unknown tool"}

mcp_client = MockedMCPClient()


def rag_search(args: dict) -> str:
    q = args["query"]
    return f"[RAG Results for '{q}']: doc1, doc2, doc3"


def web_search(args: dict) -> str:
    q = args["query"]
    return f"[WebSearch for '{q}']: result1, result2"


def load_mcp_tools():
    tools = {}
    for n in mcp_client.tools.keys():
        def wrap(name):
            return lambda args: mcp_client.call(name, **args)
        tools[n] = wrap(n)
    return tools

MCP_TOOLS = load_mcp_tools()


TOOL_REGISTRY = {
    **MCP_TOOLS,
    "rag_search": rag_search,
    "web_search": web_search,
}

class AgentState(BaseModel):
    messages: List[str] = []
    tool_call: Optional[dict] = None
    observation: Optional[Any] = None
    tool_intent: Optional[str] = None
    tool_args: Optional[dict] = None
    done: bool = False


def parse_action(text: str):
    if "Action:" not in text:
        return None
    m = re.search(r"Action:\s*(\w+)\[(.*)\]", text)
    if not m:
        return None
    return {
        "tool": m.group(1),
        "args": json.loads(m.group(2))
    }

ROUTER_PROMPT = """
You are a Tool Router using pure COT reasoning.

Your job:
1. Read the user request.
2. Decide which tool is best:
   - retrieve_course_details → if user asks about course / syllabus / instructor
   - web_search → if user needs online search
   - rag_search → if question is conceptual / knowledge based
3. Output strictly this format:

ToolIntent: <tool_name>
Args: <json>

Do NOT output anything else.

User Query:
{query}
"""

def tool_router(state: AgentState) -> AgentState:
    last_user_msg = state.messages[-1]
    prompt = ROUTER_PROMPT.format(query=last_user_msg)
    res = llm.invoke("ROUTER:" + prompt)

    intent_match = re.search(r"ToolIntent:\s*(\w+)", res)
    args_match = re.search(r"Args:\s*(\{.*\})", res)

    tool_intent = intent_match.group(1) if intent_match else None
    tool_args = json.loads(args_match.group(1)) if args_match else {}

    return AgentState(
        messages=state.messages + [f"[ROUTER RESULT] {res}"],
        tool_intent=tool_intent,
        tool_args=tool_args
    )

REACT_PROMPT = """
You are a ReAct agent.

Thought: reasoning step
Action: tool_name[args_json]
Observation: result
Final: final user answer

History:
{messages}

Continue:
"""

def llm_thought(state: AgentState) -> AgentState:
    prompt = REACT_PROMPT.format(messages="\n".join(state.messages))
    out = llm.invoke(prompt)

    if "Final:" in out:
        return AgentState(
            messages=state.messages + [out],
            done=True
        )

    action = parse_action(out)
    return AgentState(
        messages=state.messages + [out],
        tool_call=action
    )

def tool_executor(state: AgentState) -> AgentState:
    if not state.tool_call:
        return state

    tool = state.tool_call["tool"]
    args = state.tool_call["args"]

    if tool not in TOOL_REGISTRY:
        obs = f"Error: tool '{tool}' not found"
    else:
        try:
            obs = TOOL_REGISTRY[tool](args)
        except Exception as e:
            obs = f"Error executing tool: {str(e)}"

    return AgentState(
        messages=state.messages + [f"Observation: {obs}"],
        observation=obs
    )

graph = StateGraph(AgentState)

graph.add_edge(START, "router")
graph.add_node("router", tool_router)
graph.add_node("thought", llm_thought)
graph.add_node("act", tool_executor)


graph.add_edge("router", "thought")
graph.add_edge("thought", "act")

graph.add_conditional_edges(
    "act",
    lambda s: "finish" if s.done else "loop",
    {"finish": END, "loop": "thought"}
)

compiled_graph = graph.compile()

if __name__ == "__main__":
    init = AgentState(messages=["User: 请查询课程 12345 的介绍"])

    raw = compiled_graph.invoke(init)      # 返回 dict
    final_state = AgentState(**raw)        # 转成 AgentState

    print("\n======= FINAL MESSAGES =======")
    print("\n".join(final_state.messages))
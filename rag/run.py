import os
from typing_extensions import List, TypedDict

from langgraph.graph import START, StateGraph, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langgraph.graph import END
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_core.messages import SystemMessage
from langchain import hub
from langchain_openai import OpenAIEmbeddings
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model

from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_postgres import PGVector

from dotenv import load_dotenv

load_dotenv()

os.environ["DASHSCOPE_API_KEY"] = os.getenv("DASHSCOPE_API_KEY")
os.environ["DASHSCOPE_BASE_URL"] = os.getenv("DASHSCOPE_BASE_URL")


llm = init_chat_model("gpt-4o-mini", model_provider="openai")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

vector_store = PGVector(
    embeddings=embeddings,
    collection_name="embedding_docs",
    connection=os.getenv("PG_VECTOR_CONNECTION"),
)

from langchain_core.documents import Document

docs = UnstructuredLoader(file_path=["./data/paul_graham/paul_graham_essay.txt"]).load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # chunk size (characters)
    chunk_overlap=200,  # chunk overlap (characters)
    add_start_index=True,  # track index in original document
)
all_splits = text_splitter.split_documents(docs)

document_ids = vector_store.add_documents(documents=all_splits)

prompt = hub.pull("rlm/rag-prompt")


@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

# Step 1: Generate an AIMessage that may include a tool-call to be sent.
def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or respond."""
    llm_with_tools = llm.bind_tools([retrieve])
    response = llm_with_tools.invoke(state["messages"])
    # MessagesState appends messages to state instead of overwriting
    return {"messages": [response]}


# Step 2: Execute the retrieval.
tools = ToolNode([retrieve])


# Step 3: Generate a response using the retrieved content.
def generate(state: MessagesState):
    """Generate answer."""
    # Get generated ToolMessages
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]

    # Format into prompt
    docs_content = "\n\n".join(doc.content for doc in tool_messages)
    system_message_content = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        f"{docs_content}"
    )
    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]
    prompt = [SystemMessage(system_message_content)] + conversation_messages

    # Run
    response = llm.invoke(prompt)
    return {"messages": [response]}

memory = MemorySaver()
config = {"configurable": {"thread_id": "rag001"}}

graph_builder = StateGraph(MessagesState)


graph_builder.add_node(query_or_respond)
graph_builder.add_node(tools)
graph_builder.add_node(generate)

graph_builder.set_entry_point("query_or_respond")
graph_builder.add_conditional_edges(
    "query_or_respond",
    tools_condition,
    {END: END, "tools": "tools"},
)
graph_builder.add_edge("tools", "generate")
graph_builder.add_edge("generate", END)

graph = graph_builder.compile()

messages = [
    "What did the author do?",
    "What happened in the mid 1980s?",
    "Who does Paul Graham think of with the word schtick?",
]

for message in messages:
    for step in graph.stream(
        {"messages": [{"role": "user", "content": message}]}, stream_mode="values", config=config
    ):

        step["messages"][-1].pretty_print()

# result = graph.invoke({"question": "What did the author do?"})
# print(f'Context: {result["context"]}\n\n')
# print(f'Answer: {result["answer"]}')

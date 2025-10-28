import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from typing import Annotated, Sequence, TypedDict

from sentence_transformers import SentenceTransformer
from PIL import Image
from langgraph.graph import StateGraph, END, MessagesState
from langchain.chat_models import init_chat_model
from qdrant_client import QdrantClient
from rich import print

from dotenv import load_dotenv
load_dotenv()

# Define the state schema for our multimodal RAG workflow
class GraphState(TypedDict):
    query: str
    context: str
    answer: str

client = QdrantClient("http://localhost:6333")
llm = init_chat_model("anthropic:claude-sonnet-4-20250514", temperature=0)
text_encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
image_encoder = SentenceTransformer("clip-ViT-B-32")


def embed_query(query: str):
    return text_encoder.encode([query])[0]

def generate(state):
    prompt = f"""Answer the question using this context:\n\n{state['context']}\n\nQuestion: {state['query']}"""
    resp = llm.invoke(prompt)
    return {"answer": resp.content}


def retrieve(state):
    query_vec = embed_query(state["query"])
    results = client.query_points(
        collection_name="text_multimodal",
        query=query_vec,
        using="text_vector",
        limit=3,
        score_threshold=0.5
    )

    print("retrieved context: ", results.points)

    context = "\n".join([r.payload["text"] for r in results.points])
    return {"context": context}

def setup_graph():
    graph = StateGraph(GraphState)
    graph.add_node("retriever", retrieve)
    graph.add_node("generator", generate)
    graph.add_edge("retriever", "generator")
    graph.set_entry_point("retriever")
    graph.set_finish_point("generator")
    return graph.compile()


def main():
    workflow = setup_graph()
    result = workflow.invoke({"query": "What did Einstein contribute to science?"})
    print(result)


if __name__ == "__main__":
    main()
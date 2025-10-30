from typing import TypedDict

from sentence_transformers import SentenceTransformer
from langgraph.graph import START, StateGraph, END
from langchain.chat_models import init_chat_model
from qdrant_client import QdrantClient
from rich import print

from dotenv import load_dotenv
load_dotenv()

# Define the state schema for our multimodal RAG workflow
class GraphState(TypedDict):
    query: str
    text_context: str
    image_context: str
    answer: str

client = QdrantClient("http://localhost:6333")
llm = init_chat_model("anthropic:claude-sonnet-4-20250514", temperature=0)
encoder = SentenceTransformer("clip-ViT-B-32")


def embed_query(query: str):
    return encoder.encode([query])[0]

def generate(state):
    prompt = f"""
    Answer the question using this text context:
    
    <text_context>
    {state['text_context']}
    </text_context>

    <image_context>
    {state['image_context']}
    </image_context>

    Question: {state['query']}"""
    resp = llm.invoke(prompt)
    return {"answer": resp.content}


def retrieve_texts(state):
    query_vec = embed_query(state["query"])
    results = client.query_points(
        collection_name="text_multimodal",
        query= query_vec,
        using="text_vector",
        score_threshold=0.5,
        limit=3,
    )
    print("retrieved text context: ", results)
    context = "\n".join([r.payload["text"] for r in results.points])
    return {"text_context": context}

def retrieve_images(state):
    query_vec = embed_query(state["query"])
    results = client.query_points(
        collection_name="text_multimodal",
        query= query_vec,
        using="image_vector",
        score_threshold=0.2,
        limit=3,
    )    
    print("retrieved image context: ", results)
    context = "\n".join([r.payload["text"] for r in results.points])
    return {"image_context": context}

def setup_graph():
    graph = StateGraph(GraphState)
    graph.add_node("retrieve_texts", retrieve_texts)
    graph.add_node("retrieve_images", retrieve_images)
    graph.add_node("generator", generate)


    graph.add_edge("retrieve_texts", "generator")
    graph.add_edge("retrieve_images", "generator")
    graph.add_edge(START, "retrieve_texts")
    graph.add_edge(START, "retrieve_images")
    graph.set_finish_point("generator")
    return graph.compile()


def main():
    workflow = setup_graph()
    # result = workflow.invoke({"query": "What did Einstein contribute to science?"})
    result = workflow.invoke({"query": "cat"})
    print(result)


if __name__ == "__main__":
    main()
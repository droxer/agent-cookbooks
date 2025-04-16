from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.embedder.openai import OpenAIEmbedder
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.knowledge.pdf_url import PDFUrlKnowledgeBase
from agno.vectordb.lancedb import LanceDb, SearchType

from dotenv import load_dotenv
load_dotenv()

knowledge_base = PDFUrlKnowledgeBase(
    urls=["https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
    # Use LanceDB as the vector database
    vector_db=LanceDb(
        table_name="recipes",
        uri="tmp/lancedb",
        search_type=SearchType.vector,
        embedder=OpenAIEmbedder(),
    ),
)

knowledge_base.load()

tools = [
    DuckDuckGoTools(),
]

agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    description="You are a Thai cuisine expert!",
    instructions=[
        "Search your knowledge base for Thai recipes.",
        "If the question is better suited for the web, search the web to fill in gaps.",
        "Prefer the information in your knowledge base over the web results."
    ],
    knowledge=knowledge_base,
    tools=tools,
    add_history_to_messages=True,
    show_tool_calls=True,
    markdown=True
)

if __name__ == "__main__":
    agent.print_response("How do I make chicken and galangal in coconut milk soup", stream=True)
    agent.print_response("What is the my tory of Thai curry?", stream=True)
    agent.print_response("What is the my previsous questions?", stream=True)

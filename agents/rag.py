from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.embedder.openai import OpenAIEmbedder
from agno.tools.duckduckgo import DuckDuckGoTools

from agno.vectordb.pgvector import PgVector, SearchType

from dotenv import load_dotenv
load_dotenv()

vector_db=PgVector(
        table_name="recipes",
        db_url="postgresql+psycopg://tronagent:tronagent@localhost:5432/agent_store",
        search_type=SearchType.hybrid,
        embedder=OpenAIEmbedder(),
        schema="public",
    )

# from agno.knowledge.pdf_url import PDFUrlKnowledgeBase
# knowledge_base = PDFUrlKnowledgeBase(
#     urls=["https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
#     vector_db=vector_db,
# )

from agno.knowledge.agent import AgentKnowledge
knowledge_base = AgentKnowledge(
    vector_db=vector_db,
)

# knowledge_base.load(recreate=True, upsert=True)

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
    read_chat_history=True,
    show_tool_calls=True,
    markdown=True
)

if __name__ == "__main__":
    agent.print_response("How do I make chicken and galangal in coconut milk soup", stream=True)
    agent.print_response("What is the my tory of Thai curry?", stream=True)
    # agent.print_response("What is the my previsous questions?", stream=True)

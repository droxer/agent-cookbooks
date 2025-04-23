from agno.agent import Agent
from agno.memory.v2.memory import Memory
from agno.memory.v2.db.postgres import PostgresMemoryDb
from agno.models.openai import OpenAIChat
from agno.embedder.openai import OpenAIEmbedder
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.storage.postgres import PostgresStorage
from agno.vectordb.pgvector import PgVector, SearchType

from rich.pretty import pprint


from dotenv import load_dotenv
load_dotenv()

user_id = "john_doe@example.com"
db_url="postgresql+psycopg://tronagent:tronagent@localhost:5432/agent_store"
session_id = "fixed_id"

vector_db=PgVector(
        table_name="docs_embeddings",
        db_url=db_url,
        search_type=SearchType.hybrid,
        embedder=OpenAIEmbedder(),
        schema="public",
    )

# from agno.knowledge.pdf_url import PDFUrlKnowledgeBase
# knowledge_base = PDFUrlKnowledgeBase(
#     urls=["https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
#     vector_db=vector_db,
# )
# knowledge_base.load(upsert=True)


from agno.knowledge.agent import AgentKnowledge
knowledge_base = AgentKnowledge(
    vector_db=vector_db,
)


memory=Memory(
    db=PostgresMemoryDb(table_name="agent_memory", 
                        db_url=db_url, schema="public"),
    delete_memories=True, clear_memories=True
)
memory.clear()

storage=PostgresStorage(
    table_name="agent_sessions", 
    db_url=db_url,
    mode="agent", 
    auto_upgrade_schema=True, 
    schema="public"
)

# storage = SqliteStorage(table_name="agent_sessions", db_file="tmp/data.db"),

tools = [
    # DuckDuckGoTools(),
]

agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
    description="You are a Thai cuisine expert!",
    instructions=[
        "Search your knowledge base for Thai recipes.",
        "If the question is better suited for the web, search the web to fill in gaps.",
        "Prefer the information in your knowledge base over the web results."
    ],
    memory=memory,
    storage=storage,
    knowledge=knowledge_base,
    tools=tools,

    add_history_to_messages=True,
    num_history_runs=5,
    read_chat_history=True,
    read_tool_call_history=True,
    
    enable_agentic_memory=True,
    enable_session_summaries=True,
    # enable_user_memories=True,
    # add_memory_references=True,
    # search_knowledge=False,
    show_tool_calls=True,
    markdown=True,
    debug_mode=True
)

if __name__ == "__main__":
    # agent.print_response(
    #     "My name is John Doe and I like to hike in the mountains on weekends.",
    #     stream=True,
    #     user_id=user_id,
    #     session_id=session_id,
    # )

    # agent.print_response(
    #     "What are my hobbies?", stream=True, user_id=user_id, session_id=session_id
    # )

    agent.print_response("How do I make chicken and galangal in coconut milk soup", stream=True, user_id=user_id, session_id=session_id)
    # agent.print_response("What is the my tory of Thai curry?", stream=True, user_id=user_id, session_id=session_id)
    

    # session_summary = memory.get_session_summary(
    #     user_id=user_id, session_id=session_id
    # )
    # pprint(f"Session summary: {session_summary.summary}\n")

    user_memories = memory.get_user_memories(user_id=user_id)
    pprint("Memories stored in PostgreSQL:")
    for i, m in enumerate(user_memories):
        pprint(f"{i}: {m.memory}")

    # agent.print_response("What is the my previsous questions?", stream=True, user_id=user_id, session_id=session_id)
    # agent.print_response("What did I want to make in the past?", stream=True, user_id=user_id)    

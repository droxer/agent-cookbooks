from agno.agent import Agent
from agno.models.openai import OpenAIChat

from dotenv import load_dotenv
load_dotenv()


deepseek_plus_claude = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
    reasoning_model=OpenAIChat(id="o3-mini"),
)
deepseek_plus_claude.print_response("9.11 and 9.9 -- which is bigger?", stream=True)
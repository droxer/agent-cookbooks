import os

from langchain_community.chat_models import ChatOpenAI

from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import initialize_agent

from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")

llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
tools = load_tools(["google-serper", "llm-math"], llm=llm)
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

agent.run("奥利维亚·王尔德的男朋友是谁?他现在的年龄的0.23次方是多少?")

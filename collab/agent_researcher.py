import os
from dotenv import load_dotenv
from crewai import LLM, Agent, Task, Crew, Process
from crewai_tools import SerperDevTool

load_dotenv()

llm = LLM(model="gpt-4o-mini", temperature=0.7)

reseacher = Agent(
    role="Senior Researcher",
    goal="Uncover groundbreaking technologies and trends in AI",
    backstory="""Driven by curiosity,
    You exlore and sharee the latest innovations.
    """,
    llm=llm,
)

search_tool = SerperDevTool()

reseacher_task = Task(
    description="Identity the next big trend in AI with pros and cons",
    expected_output="A bullet list summary of the top 5 most important AI news and references links",
    agent=reseacher,
    tools=[search_tool],
)

crew = Crew(
    agents=[reseacher],
    tasks=[reseacher_task],
    process=Process.sequential,
)

result = crew.kickoff(inputs={"topic": "AI in Higner Education"})
print(result)

from textwrap import dedent
from crewai import Crew, LLM, Task, Agent

import os

from dotenv import load_dotenv
load_dotenv()

# llm='ollama/qwen2.5:7b'
# llm_coder='ollama/qwen2.5-coder:7b'

llm_coder = LLM(
    model="qwen-coder-turbo",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv('DASHSCOPE_API_KEY'),
)

llm = LLM(
    model="qwen-max-0125",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv('DASHSCOPE_API_KEY'),
)

def product_manager():
	return Agent(
		role='Product Manager',
		goal='''
        Analysis the user requirements and generate a comprehensive Product Requirement Document (PRD) that includes the purpose, features, functionality, and other details.
		''',
		backstory=dedent(''',
			You're responsible for creating a comprehensive Product Requirement Document.。
			'''),
		llm= llm,
		allow_delegation=False,
		verbose=True
	)
def analysis_task(agent, requirements):
	return Task(description=dedent(f'''Carefull analysis the requirements,

		User Requirements
		------------
		{requirements}
		'''),
		expected_output='The product requirements document',
		agent=agent
	)    

def sr_engineer():
	return Agent(
		role='Sr Software Engineer',
		goal='Careful read the production requirements, and writing code to implement the product requirements',
		backstory=dedent('''You are a fullstack software engineer,
			Your goal is implement the product requirements in high quality code.
			'''),
        llm= llm_coder,
		allow_delegation=False,
		verbose=True
	)

def develop_task(agent, requirements):
	return Task(description=dedent(f'''Carefull read the product requirements document,
		
		User Requirements
		------------
		{requirements}

		Implement the product requirements
		1. The frontend is required to be developed in React
		2. The backend is required to be developed in Python
		'''),
		expected_output='The implemented code both frontend and backend and build scripts',
		agent=agent
	)    

def sr_qa():
	return Agent(
		role='Sr QA Engineer',
		goal='Careful read the production requirements, test the product functionality to ensure it meets the product requirements',
		backstory='''You are a QA engineer,
			Your goal is make sure the product meets the product requirements.''',
		llm= llm_coder,
		allow_delegation=True,
		verbose=True
	)


def test_task(agent, requirements):
	return Task(description=dedent(f'''Carefull test the implemented functionality according the product requirements document:

		User Requirements
		------------
		{requirements}

		test the implemented functionality
		'''),
		expected_output='The fully tested code and functionality',
		agent=agent
	)

requirements = input('#Could you please provide a detailed description of your requirements ?\n')

pm =  product_manager()
engineer = sr_engineer()
qa = sr_qa()

prd = analysis_task(pm, requirements)
code = develop_task(engineer, requirements)
tests = test_task(qa, requirements)

crew = Crew(
	agents=[
		pm,
		engineer,
		qa
	],
	tasks=[
		prd,
		code,
		tests
	],
	verbose=False
)

results = crew.kickoff()
print(results)
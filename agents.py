
from textwrap import dedent
from crewai import Agent

llm='ollama/phi3.5:3.8b'
llm_coder='ollama/qwen2.5-coder:7b'

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


from textwrap import dedent
from crewai import Agent

from rhq.llms import llm


def product_manager():
	return Agent(
		role='Product Manager',
		goal='Creating a comprehensive Product Requirement Document (PRD) involves detailing the purpose, features, functionality',
		backstory=dedent('''You are a senior product manager,
			Your goal is 。
			'''),
		llm= llm,
		allow_delegation=False,
		verbose=True
	)

def senior_engineer():
	return Agent(
		role='Sr Software Engineer',
		goal='Implement the product requirements',
		backstory=dedent('''You are a fullstack software engineer,
			Your goal is implement the product requirements,
			'''),
        llm= llm,
		allow_delegation=False,
		verbose=True
	)

def qa_engineer():
	return Agent(
		role='Sr QA Engineer',
		goal='Test the product requirements',
		backstory='''You are a QA engineer,
			Your goal is test the product requirements,''',
		llm= llm,
		allow_delegation=True,
		verbose=True
	)

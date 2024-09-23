
from textwrap import dedent
from crewai import Agent

from rhq.llms import llm_coder,llm_instruction
from rhq.callbacks import langfuse_callback_handler


def product_manager():
	return Agent(
		role='Product Manager',
		goal='''
        Analayze the user requirements and generate a comprehensive Product Requirement Document (PRD) that includes the purpose, features, functionality, and other details.
		''',
		backstory=dedent(''',
			You're responsible for creating a comprehensive Product Requirement Document (PRD) involves detailing the purpose, features, functionality。
			'''),
		llm= llm_instruction,
		callbacks=[langfuse_callback_handler],
		allow_delegation=False,
		verbose=True
	)

def senior_engineer():
	return Agent(
		role='Sr Software Engineer',
		goal='Writing code to implement the product requirements',
		backstory=dedent('''You are a fullstack software engineer,
			Your goal is implement the product requirements,
			'''),
        llm= llm_coder,
		callbacks=[langfuse_callback_handler],
		allow_delegation=False,
		verbose=True
	)

def qa_engineer():
	return Agent(
		role='Sr QA Engineer',
		goal='Test the product requirements',
		backstory='''You are a QA engineer,
			Your goal is test the product requirements,''',
		llm= llm_coder,
		callbacks=[langfuse_callback_handler],
		allow_delegation=True,
		verbose=True
	)


from textwrap import dedent
from crewai import Agent

from rhq.llms import llm_coder,llm_instruction
from rhq.callbacks import langfuse_callback_handler


def product_manager():
	return Agent(
		role='Product Manager',
		goal='''
        Analysis the user requirements and generate a comprehensive Product Requirement Document (PRD) that includes the purpose, features, functionality, and other details.
		''',
		backstory=dedent(''',
			You're responsible for creating a comprehensive Product Requirement Document.。
			'''),
		llm= llm_instruction,
		callbacks=[langfuse_callback_handler],
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
		callbacks=[langfuse_callback_handler],
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
		callbacks=[langfuse_callback_handler],
		allow_delegation=True,
		verbose=True
	)

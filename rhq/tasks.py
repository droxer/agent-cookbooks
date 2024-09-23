from textwrap import dedent
from crewai import Task

def analysis_task(agent, requirements):
	return Task(description=dedent(f'''Carefull analysis the requirements,

		User Requirements
		------------
		{requirements}
		'''),
		expected_output='The output is fully product requirements document',
		agent=agent
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
		expected_output='The output is implemented code both frontend and backend and build scripts',
		agent=agent
	)

def test_task(agent, requirements):
	return Task(description=dedent(f'''Carefull test the implemented functionality according the product requirements document:

		User Requirements
		------------
		{requirements}

		test the implemented functionality
		'''),
		expected_output='The output is fully tested code and functionality',
		agent=agent
	)
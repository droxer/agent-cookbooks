from crewai import Crew

from agents import *
from tasks import *


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

outputs = crew.kickoff()

print("\n\n########################")
print("## Here is the final outputs")
print("########################\n")
print(outputs)
from crewai import Crew

from rhq.agents import *
from rhq.tasks import *


builder = input('# Could you please provide a detailed description of your requirements ?\n\n')


pm =  product_manager()
engineer = sr_engineer()
qa = sr_qa()

prd = analysis_task(pm, builder)
code = develop_task(engineer, builder)
app = test_task(qa, builder)

crew = Crew(
	agents=[
		pm,
		engineer,
		qa
	],
	tasks=[
		prd,
		code,
		app
	],
	verbose=True
)

outputs = crew.kickoff()

from guardrails import Guard, register_validator
from rich import print


from examples.validation.validators import BasicToxicLanguage

register_validator("BasicToxicLanguage", "string")(BasicToxicLanguage)

guard = Guard()

guard.use(
    BasicToxicLanguage, threshold=0.5, validation_method="sentence", on_fail="exception"
)


result = guard.validate("My landlord is an asshole!") 
print(result)
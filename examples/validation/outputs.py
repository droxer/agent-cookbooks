from rich import print
import guardrails as gd


try:
    from guardrails.hub import LowerCase, UpperCase, ValidRange, OneLine
except ImportError:
    gd.install("hub://guardrails/valid_range")
    gd.install("hub://guardrails/uppercase")
    gd.install("hub://guardrails/lowercase")
    gd.install("hub://guardrails/one_line")
    from guardrails.hub import LowerCase, UpperCase, ValidRange, OneLine

from pydantic import BaseModel, Field
from typing import List

prompt = """
Given the following doctor's notes about a patient, please extract a dictionary that contains the patient's information.

${doctors_notes}

${gr.complete_xml_suffix_v2}
"""

doctors_notes = """152 y/o female with chronic macular rash to face and hair, worse in beard, eyebrows and nares.
The rash is itchy, flaky and slightly scaly. Moderate response to OTC steroid cream. Patient has been using cream for 2 weeks and also suffers from diabetes."""


class Symptom(BaseModel):
    symptom: str = Field(description="Symptom that a patient is experiencing")
    affected_area: str = Field(
        description="What part of the body the symptom is affecting",
        validators=[
            LowerCase(on_fail="fix"),
        ],
    )


class Medication(BaseModel):
    medication: str = Field(
        description="Name of the medication the patient is taking",
        validators=[UpperCase(on_fail="fix")],
    )
    response: str = Field(description="How the patient is responding to the medication")


class PatientInfo(BaseModel):
    gender: str = Field(description="Patient's gender")
    age: int = Field(
        description="Patient's age",
        validators=[ValidRange(min=0, max=100, on_fail="fix")],
    )
    symptoms: List[Symptom] = Field(
        description="Symptoms that the patient is currently experiencing. Each symptom should be classified into  separate item in the list."
    )
    current_meds: List[Medication] = Field(
        description="Medications the patient is currently taking and their response"
    )
    miscellaneous: str = Field(
        description="Any other information that is relevant to the patient's health; something that doesn't fit into the other categories.",
        validators=[LowerCase(on_fail="fix"), OneLine(on_fail="fix")],
    )    


guard = gd.Guard.for_pydantic(output_class=PatientInfo)

# Wrap the litellm OpenAI API call with the `guard` object
raw_llm_output, validated_output, *rest = guard(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": prompt}],
    prompt_params={"doctors_notes": doctors_notes},
    max_tokens=1024,
    temperature=0.3,
)

# Print the validated output from the LLM
print(validated_output)
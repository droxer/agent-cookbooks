from litellm import completion
import os
from rich import print

from dotenv import load_dotenv
load_dotenv()


messages = [{ "content": "Hello, how are you?","role": "user"}]

# openai call
response = completion(model="openai/gpt-4o", messages=messages)
print(response)
# anthropic call
response = completion(model="anthropic/claude-sonnet-4-20250514", messages=messages)
print(response)
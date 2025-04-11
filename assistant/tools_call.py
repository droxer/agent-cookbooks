import os
import openai
import json
from dotenv import load_dotenv
from icecream import ic

load_dotenv()

def get_completion(
    messages, model="gpt-4o-mini", temperature=0, max_tokens=300, tools=None
):
    response = openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        tools=tools,
    )
    return response.choices[0].message


def get_current_weather(location, unit="fahrenheit"):
    """Get the current weather in a given location"""
    weather = {
        "location": location,
        "temperature": "50",
        "unit": unit,
    }

    return json.dumps(weather)


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    }
]

messages = []
messages.append({"role": "user", "content": "What's the weather like in Boston!"})
assistant_message = get_completion(messages, tools=tools)
ic(assistant_message)

assistant_message = json.loads(assistant_message.model_dump_json())
assistant_message["content"] = str(assistant_message["tool_calls"][0]["function"])

messages.append(assistant_message)

weather = get_current_weather(messages[1]["tool_calls"][0]["function"]["arguments"])
ic(weather)

messages.append(
    {
        "role": "tool",
        "tool_call_id": assistant_message["tool_calls"][0]["id"],
        "name": assistant_message["tool_calls"][0]["function"]["name"],
        "content": weather,
    }
)

final_response = get_completion(messages, tools=tools)
ic(final_response)
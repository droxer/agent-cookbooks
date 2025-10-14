# Import organization and setup
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor

from responses import format_messages

from dotenv import load_dotenv
load_dotenv()

# Initialize the language model
llm = init_chat_model("openai:gpt-4o", temperature=0.5)

# Mathematical utility functions
def add(a: float, b: float) -> float:
    """Add two numbers.
    
    Args:
        a: First number to add
        b: Second number to add
        
    Returns:
        Sum of the two numbers
    """
    return a + b

def multiply(a: float, b: float) -> float:
    """Multiply two numbers.
    
    Args:
        a: First number to multiply
        b: Second number to multiply
        
    Returns:
        Product of the two numbers
    """
    return a * b

# Mock data retrieval function
def web_search(query: str) -> str:
    """Mock web search function that returns FAANG company headcounts.
    
    In a real implementation, this would perform actual web searches.
    Currently returns static 2024 data for demonstration purposes.
    
    Args:
        query: Search query string (unused in this mock implementation)
        
    Returns:
        Formatted string with FAANG company employee headcounts
    """
    return (
        "Here are the headcounts for each of the FAANG companies in 2024:\n"
        "1. **Facebook (Meta)**: 67,317 employees.\n"
        "2. **Apple**: 164,000 employees.\n"
        "3. **Amazon**: 1,551,000 employees.\n"
        "4. **Netflix**: 14,000 employees.\n"
        "5. **Google (Alphabet)**: 181,269 employees."
    )

# Improved agent prompts with clear role definitions and constraints
math_agent = create_react_agent(
    model=llm,
    tools=[add, multiply],
    name="math_expert",
    prompt="""You are a specialized mathematics expert with access to addition and multiplication tools.

Your responsibilities:
- Solve mathematical problems using the available tools
- Always use tools for calculations rather than computing mentally
- Use one tool at a time and show your work clearly
- Focus exclusively on mathematical computations

Constraints:
- Do NOT attempt research, web searches, or data gathering
- Do NOT perform calculations without using the provided tools
- Always explain your mathematical reasoning step by step"""
)

research_agent = create_react_agent(
    model=llm,
    tools=[web_search],
    name="research_expert",  
    prompt="""You are a specialized research expert with access to web search capabilities.

Your responsibilities:
- Find and retrieve factual information using web search
- Provide comprehensive, well-sourced answers to research questions
- Focus on data gathering and information synthesis

Constraints:
- Do NOT perform mathematical calculations or computations
- Do NOT attempt to solve math problems - delegate those to the math expert
- Always use your search tool to find current, accurate information
- Present findings clearly and cite sources when available"""
)

# Enhanced supervisor prompt with clear delegation strategy
supervisor_prompt = """You are an intelligent team supervisor managing two specialized experts: a research expert and a math expert.

Your role is to:
1. Analyze incoming requests to determine the required expertise
2. Delegate tasks to the appropriate specialist
3. Coordinate between agents when tasks require multiple skills
4. Synthesize results from multiple agents when necessary

Delegation Rules:
- For data gathering, company information, current events, or factual research → use research_agent
- For calculations, mathematical operations, or numerical analysis → use math_agent  
- For complex tasks requiring both research and math → delegate sequentially (research first, then math)

Important: You are a coordinator, not a doer. Always delegate work to your specialists rather than attempting tasks yourself. Never perform calculations or research directly."""

# Create supervisor workflow for coordinating agents
workflow = create_supervisor(
    [research_agent, math_agent],
    model=llm,
    prompt=supervisor_prompt
)

# Compile the multi-agent application
app = workflow.compile()

def main():
    query = "what's the combined headcount of the FAANG companies in 2024?"
    result = app.invoke({"messages": [{"role": "user", "content": query}]})

    messages = result["messages"]
    format_messages(messages)

if __name__ == "__main__":
    main()
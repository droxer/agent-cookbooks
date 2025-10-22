# from mcp.server.fastmcp import FastMCP

from fastmcp import FastMCP

mcp = FastMCP("math")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    print(f"Adding {a} and {b}")
    return a + b

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    print(f"Multiplying {a} and {b}")
    return a * b

if __name__ == "__main__":
    mcp.run(transport='stdio')
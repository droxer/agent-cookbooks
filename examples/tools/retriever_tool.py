import os
from pprint import pprint

# Import our new embedding store module
from examples.store.embedding_store import (
    StoreType,
    create_populated_store
)

from rich.console import Console
from rich.pretty import pprint

# Initialize console for rich formatting
console = Console()

# Configuration
COLLECTION_NAME = "posts"
urls = [
    "https://lilianweng.github.io/posts/2025-05-01-thinking/",
    "https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",
    "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
    "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
]

# Get store type from environment variable (default to Qdrant)
store_type_str = os.getenv("VECTOR_STORE_TYPE", "qdrant").lower()
store_type = StoreType.QDRANT if store_type_str == "qdrant" else StoreType.PGVECTOR

console.print(f"[blue]Using {store_type.value.upper()} store...[/blue]")

# Create and populate the embedding store
embedding_store = create_populated_store(
    store_type=store_type,
    collection_name=COLLECTION_NAME,
    urls=urls,
    chunk_size=3000,
    chunk_overlap=50
)
# Create retriever tool using the embedding store
retriever_tool = embedding_store.create_retriever_tool(
    name="retrieve_blog_posts",
    description="Search and return information about Lilian Weng blog posts.",
)

def main():
    result = retriever_tool.invoke({"query": "types of reward hacking"})
    console.print("[bold green]Retriever Tool Results:[/bold green]")
    pprint(result)

if __name__ == "__main__":
    main()
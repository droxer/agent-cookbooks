from pprint import pprint
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.embeddings import init_embeddings
from langchain.tools.retriever import create_retriever_tool
from dotenv import load_dotenv

from rich.console import Console
from rich.pretty import pprint

# Initialize console for rich formatting
console = Console()

load_dotenv()

urls = [
    "https://lilianweng.github.io/posts/2025-05-01-thinking/",
    "https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",
    "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
    "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
]

docs = [WebBaseLoader(url).load() for url in urls]

docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=3000, chunk_overlap=50
)
doc_splits = text_splitter.split_documents(docs_list)

embeddings = init_embeddings("openai:text-embedding-3-small")
vectorstore = InMemoryVectorStore.from_documents(
    documents=doc_splits, embedding=embeddings
)
retriever = vectorstore.as_retriever()

retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_blog_posts",
    "Search and return information about Lilian Weng blog posts.",
)

def main():
    result = retriever_tool.invoke({"query": "types of reward hacking"})
    console.print("[bold green]Retriever Tool Results:[/bold green]")
    pprint(result)

if __name__ == "__main__":
    main()
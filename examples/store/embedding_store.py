"""
Embedding Store Module

Provides a unified interface for different vector store backends including:
- PGVector (PostgreSQL)
- Qdrant (remote/local)
"""

import os
from abc import ABC, abstractmethod
from typing import List, Optional
from enum import Enum

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import init_embeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import PGVector, Qdrant
from langchain.tools.retriever import create_retriever_tool
from dotenv import load_dotenv

from rich.console import Console

# Initialize console for rich formatting
console = Console()

load_dotenv()


class StoreType(Enum):
    """Supported vector store types."""
    PGVECTOR = "pgvector"
    QDRANT = "qdrant"


class EmbeddingStore(ABC):
    """Abstract base class for embedding stores."""

    def __init__(self, embedding_model: str = "openai:text-embedding-3-small"):
        """Initialize the embedding store with specified model."""
        self.embeddings = init_embeddings(embedding_model)
        self._vectorstore = None

    @abstractmethod
    def connect_or_create(self, collection_name: str, **kwargs) -> bool:
        """
        Connect to existing store or create new one.
        Returns True if store was created/populated, False if existing store was used.
        """
        pass

    @abstractmethod
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the store."""
        pass

    @abstractmethod
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Search for similar documents."""
        pass

    def as_retriever(self, **kwargs):
        """Return the store as a retriever."""
        return self._vectorstore.as_retriever(**kwargs)

    def create_retriever_tool(self, name: str, description: str):
        """Create a retriever tool from this store."""
        retriever = self.as_retriever()
        return create_retriever_tool(retriever, name, description)


class PGVectorStore(EmbeddingStore):
    """PGVector implementation of embedding store."""

    def __init__(self, connection_string: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.connection_string = connection_string or os.getenv(
            "POSTGRES_CONNECTION_STRING",
            "postgresql://ai:ai@localhost:5432/ai"
        )

    def connect_or_create(self, collection_name: str, **kwargs) -> bool:
        """Connect to existing PGVector store or create new one."""
        try:
            # Try to connect to existing store
            self._vectorstore = PGVector.from_existing_index(
                embedding=self.embeddings,
                connection_string=self.connection_string,
                collection_name=collection_name,
            )

            # Test if store has content
            test_results = self.similarity_search("test", k=1)
            if len(test_results) == 0:
                console.print("[yellow]Connected to existing PGVector store but it's empty.[/yellow]")
                return True  # Need to populate
            else:
                console.print(f"[green]Connected to existing PGVector store with {len(test_results)}+ documents.[/green]")
                return False  # Using existing store

        except Exception as e:
            console.print(f"[yellow]No existing PGVector store found (error: {e}). Will create new store.[/yellow]")
            self._vectorstore = PGVector(
                embedding=self.embeddings,
                connection_string=self.connection_string,
                collection_name=collection_name,
            )
            return True  # Need to populate

    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to PGVector store."""
        if self._vectorstore is None:
            raise ValueError("Vector store not initialized. Call connect_or_create first.")

        self._vectorstore.add_documents(documents)
        console.print(f"[green]Added {len(documents)} document chunks to PGVector store.[/green]")

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Search for similar documents in PGVector store."""
        if self._vectorstore is None:
            raise ValueError("Vector store not initialized. Call connect_or_create first.")

        return self._vectorstore.similarity_search(query, k=k)


class QdrantStore(EmbeddingStore):
    """Qdrant implementation of embedding store."""

    def __init__(self, url: str = "http://localhost:6333", **kwargs):
        super().__init__(**kwargs)
        self.url = url
        self._client = None

    def connect_or_create(self, collection_name: str, **kwargs) -> bool:
        """Connect to existing Qdrant store or create new one."""
        from qdrant_client import QdrantClient
        from qdrant_client.http import models as rest

        # Initialize client
        self._client = QdrantClient(url=self.url)

        try:
            # Try to connect to existing store
            collections = self._client.get_collections()
            collection_names = [collection.name for collection in collections.collections]

            if collection_name in collection_names:
                # Collection exists, try to initialize vectorstore
                self._vectorstore = Qdrant(
                    client=self._client,
                    collection_name=collection_name,
                    embeddings=self.embeddings,
                )

                # Test if store has content
                test_results = self.similarity_search("test", k=1)
                if len(test_results) == 0:
                    console.print("[yellow]Connected to existing Qdrant store but it's empty.[/yellow]")
                    return True  # Need to populate
                else:
                    console.print(f"[green]Connected to existing Qdrant store with {len(test_results)}+ documents.[/green]")
                    return False  # Using existing store
            else:
                # Collection doesn't exist, will create new one
                raise Exception(f"Collection {collection_name} not found")

        except Exception as e:
            console.print(f"[yellow]No existing Qdrant store found (error: {e}). Will create new store.[/yellow]")

            # Get embedding dimensions
            test_embedding = self.embeddings.embed_query("test")
            vector_size = len(test_embedding)

            # Create collection
            self._client.create_collection(
                collection_name=collection_name,
                vectors_config=rest.VectorParams(
                    size=vector_size,
                    distance=rest.Distance.COSINE
                )
            )

            # Initialize vectorstore
            self._vectorstore = Qdrant(
                client=self._client,
                collection_name=collection_name,
                embeddings=self.embeddings,
            )
            return True  # Need to populate

    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to Qdrant store."""
        if self._vectorstore is None:
            raise ValueError("Vector store not initialized. Call connect_or_create first.")

        self._vectorstore.add_documents(documents)
        console.print(f"[green]Added {len(documents)} document chunks to Qdrant store.[/green]")

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Search for similar documents in Qdrant store."""
        if self._vectorstore is None:
            raise ValueError("Vector store not initialized. Call connect_or_create first.")

        return self._vectorstore.similarity_search(query, k=k)


class EmbeddingStoreFactory:
    """Factory class for creating embedding stores."""

    @staticmethod
    def create_store(
        store_type: StoreType,
        **kwargs
    ) -> EmbeddingStore:
        """Create an embedding store of the specified type."""
        if store_type == StoreType.PGVECTOR:
            return PGVectorStore(**kwargs)
        elif store_type == StoreType.QDRANT:
            return QdrantStore(**kwargs)
        else:
            raise ValueError(f"Unsupported store type: {store_type}")


def load_and_split_documents(
    urls: List[str],
    chunk_size: int = 3000,
    chunk_overlap: int = 50
) -> List[Document]:
    """Load documents from URLs and split into chunks."""
    console.print(f"[blue]Loading documents from {len(urls)} URLs...[/blue]")
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    doc_splits = text_splitter.split_documents(docs_list)

    console.print(f"[green]Loaded and split into {len(doc_splits)} document chunks.[/green]")
    return doc_splits


def create_populated_store(
    store_type: StoreType = StoreType.QDRANT,
    collection_name: str = "default_collection",
    urls: Optional[List[str]] = None,
    chunk_size: int = 3000,
    chunk_overlap: int = 50,
    **store_kwargs
) -> EmbeddingStore:
    """
    Create and populate an embedding store with documents from URLs.

    Args:
        store_type: Type of vector store to create
        collection_name: Name of the collection
        urls: List of URLs to load documents from
        chunk_size: Size of document chunks
        chunk_overlap: Overlap between chunks
        **store_kwargs: Additional arguments for store creation

    Returns:
        Populated embedding store
    """
    # Create store
    store = EmbeddingStoreFactory.create_store(store_type, **store_kwargs)

    # Connect or create store
    needs_population = store.connect_or_create(collection_name)

    # Populate if needed
    if needs_population and urls:
        documents = load_and_split_documents(urls, chunk_size, chunk_overlap)
        store.add_documents(documents)
    elif needs_population and not urls:
        console.print("[yellow]Store needs population but no URLs provided.[/yellow]")

    return store
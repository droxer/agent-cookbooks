# Vector Store Examples

This directory contains examples of different vector store implementations and usage patterns.

## Qdrant Setup

Qdrant is a vector database that can be run locally using Docker or Podman. Below are instructions for setting up Qdrant:

### Using Docker

1. **Run Qdrant container**:
   ```bash
    docker run -p 6333:6333 -d --name qdrant qdrant/qdrant
   ```

2. **Verify Qdrant is running**:
   ```bash
   curl -X GET 'http://localhost:6333/collections'
   ```

### Using Podman

1. **Run Qdrant container**:
   ```bash
   podman run -p 6333:6333 -d --name qdrant qdrant/qdrant
   ```

2. **Verify Qdrant is running**:
   ```bash
   curl -X GET 'http://localhost:6333/collections'
   ```

### Configuration

The Qdrant store can be configured using the following parameters:

- `url`: The URL of the Qdrant server (default: "http://localhost:6333")
- `collection_name`: The name of the collection to use

### Environment Variables

You can configure which vector store to use with the `VECTOR_STORE_TYPE` environment variable:

```bash
# Use Qdrant (default)
export VECTOR_STORE_TYPE=qdrant

# Use PGVector
export VECTOR_STORE_TYPE=pgvector
```

### Running Examples

To run the examples with Qdrant:

1. Start the Qdrant server using Docker or Podman as shown above
2. Run the example scripts:
   ```bash
   python examples/tools/retriever_tool.py
   ```

### Troubleshooting

If you encounter issues:

1. **Port conflicts**: Make sure ports 6333 and 6334 are available
2. **Permissions**: On Linux, you might need to adjust the volume mount permissions
3. **Firewall**: Ensure your firewall allows connections to the Qdrant ports

### Persistent Storage

The Qdrant container is configured to persist data to a local `qdrant_storage` directory. This ensures that your data survives container restarts.
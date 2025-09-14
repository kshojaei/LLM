# Vector Database System Guide

This guide explains how to use the production vector database system included in this RAG learning project.

## Overview

The vector database system provides a unified interface for managing vector databases in production RAG applications. It supports multiple backends and provides advanced features like metadata filtering, performance monitoring, and backup capabilities.

## Quick Start

### Basic Usage

```python
from src.vector_db.vector_manager import create_vector_database

# Create a vector database
documents = [
    {
        "id": "doc_1",
        "text": "Machine learning is a subset of artificial intelligence...",
        "metadata": {"category": "AI", "source": "wikipedia"}
    },
    # ... more documents
]

vdb = create_vector_database(
    backend="chromadb",
    collection_name="my_docs",
    documents=documents
)

# Search for similar documents
results = vdb.search("What is machine learning?", top_k=5)
```

### Advanced Usage

```python
from src.vector_db.vector_manager import VectorDatabaseManager

# Initialize with custom settings
vdb = VectorDatabaseManager(
    backend="faiss",
    collection_name="production_docs",
    embedding_model="BAAI/bge-base-en-v1.5",
    persist_directory="./data/vector_db"
)

# Add documents
vdb.add_documents(texts, metadatas, ids)

# Search with metadata filtering
results = vdb.search(
    "machine learning",
    top_k=5,
    filter_dict={"category": "AI"}
)

# Get database statistics
stats = vdb.get_stats()
```

## Supported Backends

### ChromaDB
- **Best for**: Local development, small to medium scale (< 1M docs)
- **Pros**: Easy setup, built-in metadata filtering, persistent storage
- **Cons**: Limited scalability, single-node only
- **Use Cases**: Development, prototyping, small production apps

### FAISS
- **Best for**: High-performance search, large scale (> 1M docs)
- **Pros**: Extremely fast, memory efficient, GPU support
- **Cons**: No built-in metadata filtering, complex setup
- **Use Cases**: Large-scale production, research, high-throughput apps

### Pinecone
- **Best for**: Cloud-native, fully managed solutions
- **Pros**: Fully managed, auto-scaling, high availability
- **Cons**: Cost for large datasets, vendor lock-in
- **Use Cases**: Production apps, multi-tenant systems, global deployments

## Key Features

### Document Management
```python
# Add documents
vdb.add_documents(texts, metadatas, ids)

# Get specific document
doc = vdb.get_document("doc_1")

# Delete document
vdb.delete_document("doc_1")

# Clear entire database
vdb.clear_database()
```

### Metadata Filtering
```python
# Filter by metadata
results = vdb.search(
    "machine learning",
    top_k=5,
    filter_dict={"category": "AI", "source": "wikipedia"}
)
```

### Performance Monitoring
```python
# Get database statistics
stats = vdb.get_stats()
print(f"Total documents: {stats['total_documents']}")
print(f"Index size: {stats['index_size_mb']} MB")
print(f"Last updated: {stats['last_updated']}")
```

### Backup and Recovery
```python
# Create backup
vdb.backup_database("./backups/my_backup")

# Load from existing database
vdb = VectorDatabaseManager(
    backend="chromadb",
    collection_name="existing_collection"
)
```

## Configuration

### Environment Variables
```bash
# Vector database settings
VECTOR_DB_PATH=./data/vector_db
DEFAULT_EMBEDDING_MODEL=BAAI/bge-base-en-v1.5

# Pinecone settings (if using Pinecone)
PINECONE_API_KEY=your_api_key
PINECONE_ENVIRONMENT=your_environment
```

### Configuration File
```python
# src/config.py
VECTOR_DB_CONFIG = {
    "type": "chromadb",
    "collection_name": "docs_copilot",
    "distance_metric": "cosine",
    "persist_directory": str(DATA_DIR / "vector_db" / "chroma_db")
}
```

## Performance Optimization

### Batch Processing
```python
# Process documents in batches for better performance
batch_size = 100
for i in range(0, len(documents), batch_size):
    batch = documents[i:i + batch_size]
    vdb.add_documents(batch)
```

### Memory Management
```python
# Monitor memory usage
import psutil
process = psutil.Process()
memory_mb = process.memory_info().rss / 1024 / 1024
print(f"Memory usage: {memory_mb:.2f} MB")
```

### Index Optimization
```python
# For FAISS, use appropriate index type
if backend == "faiss":
    # For large datasets, use IndexIVFFlat
    # For high accuracy, use IndexFlatIP
    # For memory efficiency, use IndexPQ
```

## Production Deployment

### Docker Configuration
```dockerfile
# Dockerfile
FROM python:3.9-slim

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application
COPY . .

# Set environment variables
ENV VECTOR_DB_PATH=/app/data/vector_db

# Run application
CMD ["python", "web_app_vector.py"]
```

### Cloud Deployment
```yaml
# docker-compose.yml
version: '3.8'
services:
  rag-app:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data/vector_db:/app/data/vector_db
    environment:
      - VECTOR_DB_PATH=/app/data/vector_db
```

## Troubleshooting

### Common Issues

1. **Import Error**: Make sure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **Memory Issues**: Use FAISS for large datasets or reduce batch size
   ```python
   # Reduce batch size
   batch_size = 50
   ```

3. **Performance Issues**: 
   - Use FAISS for better search performance
   - Enable GPU support for FAISS
   - Use Pinecone for cloud scalability

4. **Metadata Filtering**: Only available with ChromaDB and Pinecone
   ```python
   # Use ChromaDB for metadata filtering
   vdb = VectorDatabaseManager(backend="chromadb")
   ```

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable debug logging
vdb = VectorDatabaseManager(backend="chromadb")
```

## Best Practices

1. **Choose the Right Backend**:
   - ChromaDB for development and small apps
   - FAISS for high-performance requirements
   - Pinecone for cloud-native solutions

2. **Optimize Embeddings**:
   - Use appropriate embedding models
   - Normalize embeddings for cosine similarity
   - Consider model size vs quality trade-offs

3. **Monitor Performance**:
   - Track search times and accuracy
   - Monitor memory usage
   - Set up alerts for performance degradation

4. **Backup Regularly**:
   - Create backups before major changes
   - Test backup restoration
   - Store backups in multiple locations

5. **Scale Gradually**:
   - Start with small datasets
   - Monitor performance as you scale
   - Consider distributed solutions for very large datasets

## Examples

### Complete RAG System
```python
from src.vector_db.vector_manager import create_vector_database

class RAGSystem:
    def __init__(self):
        self.vdb = create_vector_database(
            backend="chromadb",
            collection_name="rag_docs"
        )
    
    def add_documents(self, documents):
        return self.vdb.add_documents(documents)
    
    def query(self, question, top_k=5):
        # Retrieve relevant documents
        results = self.vdb.search(question, top_k=top_k)
        
        # Generate response (simplified)
        context = " ".join([r['document'] for r in results])
        response = f"Based on the context: {context[:200]}..."
        
        return {
            "question": question,
            "response": response,
            "sources": results
        }

# Usage
rag = RAGSystem()
rag.add_documents(documents)
result = rag.query("What is machine learning?")
```

### Web Application Integration
```python
from fastapi import FastAPI
from src.vector_db.vector_manager import create_vector_database

app = FastAPI()
vdb = create_vector_database(backend="chromadb")

@app.post("/api/query")
async def query_rag(request: dict):
    results = vdb.search(request["question"], top_k=5)
    return {"results": results}
```

## Support

For questions or issues:
1. Check the notebooks for learning guidance
2. Review the code comments in `src/vector_db/vector_manager.py`
3. Open an issue on GitHub
4. Check the troubleshooting section above

## Next Steps

1. **Learn**: Complete notebook 10 to understand vector databases
2. **Experiment**: Try different backends and configurations
3. **Scale**: Test with larger datasets
4. **Deploy**: Build production applications
5. **Optimize**: Implement advanced features and monitoring

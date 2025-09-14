"""
Vector Database Manager for Production RAG Systems

This module provides a comprehensive vector database management system
that handles multiple vector database backends (ChromaDB, FAISS, Pinecone)
with proper indexing, search, and maintenance capabilities.
"""

import os
import json
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime
import numpy as np
import pandas as pd

# Vector database backends
import chromadb
from chromadb.config import Settings
import faiss
from sentence_transformers import SentenceTransformer

# Optional imports for cloud databases
try:
    import pinecone
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False

from ..config import DATA_DIR, VECTOR_DB_CONFIG

logger = logging.getLogger(__name__)

class VectorDatabaseManager:
    """
    Production-ready vector database manager supporting multiple backends.
    
    Features:
    - Multiple backends: ChromaDB, FAISS, Pinecone
    - Document indexing and search
    - Metadata filtering
    - Batch operations
    - Database maintenance
    - Performance monitoring
    """
    
    def __init__(self, 
                 backend: str = "chromadb",
                 collection_name: str = "documents",
                 embedding_model: str = "BAAI/bge-base-en-v1.5",
                 persist_directory: Optional[str] = None):
        """
        Initialize vector database manager.
        
        Args:
            backend: Vector database backend ('chromadb', 'faiss', 'pinecone')
            collection_name: Name of the collection/index
            embedding_model: Sentence transformer model for embeddings
            persist_directory: Directory to persist the database
        """
        self.backend = backend
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model
        self.persist_directory = persist_directory or str(DATA_DIR / "vector_db")
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Initialize vector database
        self._init_database()
        
        # Statistics
        self.stats = {
            "total_documents": 0,
            "last_updated": None,
            "index_size_mb": 0
        }
        
    def _init_database(self):
        """Initialize the vector database backend."""
        if self.backend == "chromadb":
            self._init_chromadb()
        elif self.backend == "faiss":
            self._init_faiss()
        elif self.backend == "pinecone":
            self._init_pinecone()
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")
    
    def _init_chromadb(self):
        """Initialize ChromaDB."""
        chroma_dir = Path(self.persist_directory) / "chroma_db"
        chroma_dir.mkdir(parents=True, exist_ok=True)
        
        self.chroma_client = chromadb.PersistentClient(
            path=str(chroma_dir),
            settings=Settings(anonymized_telemetry=False)
        )
        
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        logger.info(f"ChromaDB initialized: {chroma_dir}")
    
    def _init_faiss(self):
        """Initialize FAISS."""
        self.faiss_index = None
        self.faiss_metadata = []
        self.faiss_path = Path(self.persist_directory) / f"{self.collection_name}.faiss"
        self.metadata_path = Path(self.persist_directory) / f"{self.collection_name}_metadata.pkl"
        
        # Load existing index if available
        if self.faiss_path.exists():
            self.load_faiss_index()
        
        logger.info(f"FAISS initialized: {self.faiss_path}")
    
    def _init_pinecone(self):
        """Initialize Pinecone (cloud vector database)."""
        if not PINECONE_AVAILABLE:
            raise ImportError("Pinecone not available. Install with: pip install pinecone-client")
        
        # Initialize Pinecone (requires API key)
        pinecone.init(
            api_key=os.getenv("PINECONE_API_KEY"),
            environment=os.getenv("PINECONE_ENVIRONMENT")
        )
        
        # Create or get index
        if self.collection_name not in pinecone.list_indexes():
            pinecone.create_index(
                name=self.collection_name,
                dimension=384,  # BGE model dimension
                metric="cosine"
            )
        
        self.index = pinecone.Index(self.collection_name)
        logger.info(f"Pinecone initialized: {self.collection_name}")
    
    def add_documents(self, 
                     documents: List[str], 
                     metadatas: Optional[List[Dict]] = None,
                     ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Add documents to the vector database.
        
        Args:
            documents: List of document texts
            metadatas: List of metadata dictionaries
            ids: List of document IDs
            
        Returns:
            Dictionary with operation results
        """
        if not documents:
            return {"status": "error", "message": "No documents provided"}
        
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(documents)} documents")
        embeddings = self.embedding_model.encode(documents, show_progress_bar=True)
        
        # Generate IDs if not provided
        if ids is None:
            ids = [f"doc_{i}_{datetime.now().strftime('%Y%m%d_%H%M%S')}" 
                   for i in range(len(documents))]
        
        # Generate metadata if not provided
        if metadatas is None:
            metadatas = [{"source": "unknown", "timestamp": datetime.now().isoformat()} 
                        for _ in documents]
        
        # Add to database
        if self.backend == "chromadb":
            result = self._add_to_chromadb(documents, embeddings, metadatas, ids)
        elif self.backend == "faiss":
            result = self._add_to_faiss(documents, embeddings, metadatas, ids)
        elif self.backend == "pinecone":
            result = self._add_to_pinecone(documents, embeddings, metadatas, ids)
        
        # Update statistics
        self.stats["total_documents"] += len(documents)
        self.stats["last_updated"] = datetime.now().isoformat()
        self._update_stats()
        
        return result
    
    def _add_to_chromadb(self, documents, embeddings, metadatas, ids):
        """Add documents to ChromaDB."""
        try:
            self.collection.add(
                documents=documents,
                embeddings=embeddings.tolist(),
                metadatas=metadatas,
                ids=ids
            )
            return {"status": "success", "added": len(documents)}
        except Exception as e:
            logger.error(f"Error adding to ChromaDB: {e}")
            return {"status": "error", "message": str(e)}
    
    def _add_to_faiss(self, documents, embeddings, metadatas, ids):
        """Add documents to FAISS."""
        try:
            if self.faiss_index is None:
                # Create new index
                dimension = embeddings.shape[1]
                self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            
            # Add to index
            self.faiss_index.add(embeddings.astype('float32'))
            
            # Store metadata
            for i, (doc, meta, doc_id) in enumerate(zip(documents, metadatas, ids)):
                self.faiss_metadata.append({
                    "id": doc_id,
                    "document": doc,
                    "metadata": meta,
                    "index": self.faiss_index.ntotal - len(documents) + i
                })
            
            # Save index and metadata
            self.save_faiss_index()
            
            return {"status": "success", "added": len(documents)}
        except Exception as e:
            logger.error(f"Error adding to FAISS: {e}")
            return {"status": "error", "message": str(e)}
    
    def _add_to_pinecone(self, documents, embeddings, metadatas, ids):
        """Add documents to Pinecone."""
        try:
            # Prepare vectors for Pinecone
            vectors = []
            for i, (doc, embedding, meta, doc_id) in enumerate(zip(documents, embeddings, metadatas, ids)):
                vectors.append({
                    "id": doc_id,
                    "values": embedding.tolist(),
                    "metadata": {**meta, "document": doc}
                })
            
            # Upsert in batches
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                self.index.upsert(vectors=batch)
            
            return {"status": "success", "added": len(documents)}
        except Exception as e:
            logger.error(f"Error adding to Pinecone: {e}")
            return {"status": "error", "message": str(e)}
    
    def search(self, 
               query: str, 
               top_k: int = 5,
               filter_dict: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            top_k: Number of results to return
            filter_dict: Metadata filters
            
        Returns:
            List of search results with documents, scores, and metadata
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])
        
        if self.backend == "chromadb":
            return self._search_chromadb(query_embedding, top_k, filter_dict)
        elif self.backend == "faiss":
            return self._search_faiss(query_embedding, top_k, filter_dict)
        elif self.backend == "pinecone":
            return self._search_pinecone(query_embedding, top_k, filter_dict)
    
    def _search_chromadb(self, query_embedding, top_k, filter_dict):
        """Search ChromaDB."""
        try:
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=top_k,
                where=filter_dict
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    "document": results['documents'][0][i],
                    "score": 1 - results['distances'][0][i],  # Convert distance to similarity
                    "metadata": results['metadatas'][0][i],
                    "id": results['ids'][0][i]
                })
            
            return formatted_results
        except Exception as e:
            logger.error(f"Error searching ChromaDB: {e}")
            return []
    
    def _search_faiss(self, query_embedding, top_k, filter_dict):
        """Search FAISS."""
        try:
            if self.faiss_index is None:
                return []
            
            # Normalize query embedding
            faiss.normalize_L2(query_embedding)
            
            # Search
            scores, indices = self.faiss_index.search(query_embedding.astype('float32'), top_k)
            
            # Format results
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.faiss_metadata):
                    metadata = self.faiss_metadata[idx]
                    results.append({
                        "document": metadata["document"],
                        "score": float(score),
                        "metadata": metadata["metadata"],
                        "id": metadata["id"]
                    })
            
            return results
        except Exception as e:
            logger.error(f"Error searching FAISS: {e}")
            return []
    
    def _search_pinecone(self, query_embedding, top_k, filter_dict):
        """Search Pinecone."""
        try:
            results = self.index.query(
                vector=query_embedding[0].tolist(),
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict
            )
            
            # Format results
            formatted_results = []
            for match in results['matches']:
                formatted_results.append({
                    "document": match['metadata'].get('document', ''),
                    "score": match['score'],
                    "metadata": {k: v for k, v in match['metadata'].items() if k != 'document'},
                    "id": match['id']
                })
            
            return formatted_results
        except Exception as e:
            logger.error(f"Error searching Pinecone: {e}")
            return []
    
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific document by ID."""
        if self.backend == "chromadb":
            try:
                result = self.collection.get(ids=[doc_id])
                if result['documents']:
                    return {
                        "document": result['documents'][0],
                        "metadata": result['metadatas'][0],
                        "id": doc_id
                    }
            except Exception as e:
                logger.error(f"Error getting document from ChromaDB: {e}")
        elif self.backend == "faiss":
            for meta in self.faiss_metadata:
                if meta["id"] == doc_id:
                    return {
                        "document": meta["document"],
                        "metadata": meta["metadata"],
                        "id": doc_id
                    }
        elif self.backend == "pinecone":
            try:
                result = self.index.fetch(ids=[doc_id])
                if doc_id in result['vectors']:
                    vector_data = result['vectors'][doc_id]
                    return {
                        "document": vector_data['metadata'].get('document', ''),
                        "metadata": {k: v for k, v in vector_data['metadata'].items() if k != 'document'},
                        "id": doc_id
                    }
            except Exception as e:
                logger.error(f"Error getting document from Pinecone: {e}")
        
        return None
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document by ID."""
        try:
            if self.backend == "chromadb":
                self.collection.delete(ids=[doc_id])
            elif self.backend == "faiss":
                # FAISS doesn't support deletion, mark as deleted in metadata
                for meta in self.faiss_metadata:
                    if meta["id"] == doc_id:
                        meta["deleted"] = True
                self.save_faiss_index()
            elif self.backend == "pinecone":
                self.index.delete(ids=[doc_id])
            
            self.stats["total_documents"] = max(0, self.stats["total_documents"] - 1)
            return True
        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        self._update_stats()
        return self.stats.copy()
    
    def _update_stats(self):
        """Update database statistics."""
        if self.backend == "chromadb":
            self.stats["total_documents"] = self.collection.count()
        elif self.backend == "faiss":
            self.stats["total_documents"] = len([m for m in self.faiss_metadata if not m.get("deleted", False)])
        elif self.backend == "pinecone":
            self.stats["total_documents"] = self.index.describe_index_stats()['total_vector_count']
        
        # Calculate index size
        if self.backend == "faiss" and self.faiss_path.exists():
            self.stats["index_size_mb"] = self.faiss_path.stat().st_size / (1024 * 1024)
    
    def save_faiss_index(self):
        """Save FAISS index and metadata."""
        if self.faiss_index is not None:
            faiss.write_index(self.faiss_index, str(self.faiss_path))
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(self.faiss_metadata, f)
    
    def load_faiss_index(self):
        """Load FAISS index and metadata."""
        if self.faiss_path.exists() and self.metadata_path.exists():
            self.faiss_index = faiss.read_index(str(self.faiss_path))
            with open(self.metadata_path, 'rb') as f:
                self.faiss_metadata = pickle.load(f)
    
    def backup_database(self, backup_path: str) -> bool:
        """Create a backup of the database."""
        try:
            backup_dir = Path(backup_path)
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            if self.backend == "chromadb":
                # ChromaDB is already persistent, just copy the directory
                import shutil
                chroma_dir = Path(self.persist_directory) / "chroma_db"
                shutil.copytree(chroma_dir, backup_dir / "chroma_db", dirs_exist_ok=True)
            elif self.backend == "faiss":
                # Copy FAISS files
                import shutil
                shutil.copy2(self.faiss_path, backup_dir / self.faiss_path.name)
                shutil.copy2(self.metadata_path, backup_dir / self.metadata_path.name)
            
            logger.info(f"Database backed up to: {backup_path}")
            return True
        except Exception as e:
            logger.error(f"Error creating backup: {e}")
            return False
    
    def clear_database(self) -> bool:
        """Clear all documents from the database."""
        try:
            if self.backend == "chromadb":
                # Delete and recreate collection
                self.chroma_client.delete_collection(self.collection_name)
                self.collection = self.chroma_client.create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
            elif self.backend == "faiss":
                self.faiss_index = None
                self.faiss_metadata = []
                if self.faiss_path.exists():
                    self.faiss_path.unlink()
                if self.metadata_path.exists():
                    self.metadata_path.unlink()
            elif self.backend == "pinecone":
                self.index.delete(delete_all=True)
            
            self.stats["total_documents"] = 0
            self.stats["last_updated"] = datetime.now().isoformat()
            
            logger.info("Database cleared")
            return True
        except Exception as e:
            logger.error(f"Error clearing database: {e}")
            return False


def create_vector_database(backend: str = "chromadb", 
                          collection_name: str = "documents",
                          documents: Optional[List[Dict]] = None) -> VectorDatabaseManager:
    """
    Create and populate a vector database.
    
    Args:
        backend: Vector database backend
        collection_name: Collection name
        documents: List of documents to add (optional)
        
    Returns:
        Initialized VectorDatabaseManager
    """
    # Initialize manager
    manager = VectorDatabaseManager(
        backend=backend,
        collection_name=collection_name
    )
    
    # Add documents if provided
    if documents:
        texts = [doc.get("text", "") for doc in documents]
        metadatas = [doc.get("metadata", {}) for doc in documents]
        ids = [doc.get("id", f"doc_{i}") for i, doc in enumerate(documents)]
        
        result = manager.add_documents(texts, metadatas, ids)
        logger.info(f"Added {result.get('added', 0)} documents to vector database")
    
    return manager


# Example usage and testing
if __name__ == "__main__":
    # Example documents
    sample_docs = [
        {
            "id": "doc_1",
            "text": "Machine learning is a subset of artificial intelligence that focuses on algorithms.",
            "metadata": {"category": "AI", "source": "wikipedia"}
        },
        {
            "id": "doc_2", 
            "text": "Deep learning uses neural networks with multiple layers to process data.",
            "metadata": {"category": "AI", "source": "wikipedia"}
        },
        {
            "id": "doc_3",
            "text": "Natural language processing helps computers understand human language.",
            "metadata": {"category": "NLP", "source": "wikipedia"}
        }
    ]
    
    # Create vector database
    print("Creating vector database...")
    vdb = create_vector_database(
        backend="chromadb",
        collection_name="test_docs",
        documents=sample_docs
    )
    
    # Search
    print("\nSearching for 'neural networks'...")
    results = vdb.search("neural networks", top_k=2)
    for i, result in enumerate(results):
        print(f"{i+1}. {result['document']} (score: {result['score']:.3f})")
    
    # Get stats
    stats = vdb.get_stats()
    print(f"\nDatabase stats: {stats}")

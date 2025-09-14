"""
Retrieval System Module for RAG
This module provides comprehensive retrieval capabilities including dense search, sparse search, and hybrid approaches.

Author: Kamran Shojaei - Physicist with background in AI/ML
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Union, Tuple
import logging
from pathlib import Path
import json
import time
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Vector store imports
import chromadb
from chromadb.config import Settings
import faiss
from sklearn.metrics.pairwise import cosine_similarity
try:
    from rank_bm25 import BM25Okapi
except ImportError:
    print("rank_bm25 not found. Installing...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "rank-bm25"])
    from rank_bm25 import BM25Okapi

# Import our modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import RETRIEVAL_CONFIG, VECTOR_DB_CONFIG, DATA_DIR
from src.models.embedding_models import EmbeddingModel, load_embedding_model
from src.models.reranker_models import RerankerModel, load_reranker

logger = logging.getLogger(__name__)

@dataclass
class RetrievalConfig:
    """Configuration for retrieval system."""
    top_k: int = 10
    rerank_top_k: int = 5
    similarity_threshold: float = 0.7
    hybrid_alpha: float = 0.7  # Weight for dense search
    hybrid_beta: float = 0.3   # Weight for sparse search
    use_reranking: bool = True
    reranker_model: str = "bge_reranker"

class BaseRetriever(ABC):
    """Base class for retrievers."""
    
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Retrieve documents for a query."""
        pass
    
    @abstractmethod
    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to the retriever."""
        pass

class DenseRetriever(BaseRetriever):
    """
    Dense retrieval using embeddings and vector similarity.
    
    This retriever uses semantic embeddings to find relevant documents
    based on meaning rather than exact keyword matches.
    """
    
    def __init__(self, embedding_model: EmbeddingModel, vector_store_type: str = "faiss"):
        self.embedding_model = embedding_model
        self.vector_store_type = vector_store_type
        self.documents = []
        self.embeddings = None
        self.vector_store = None
        self._setup_vector_store()
    
    def _setup_vector_store(self):
        """Setup the vector store based on type."""
        if self.vector_store_type == "faiss":
            self._setup_faiss()
        elif self.vector_store_type == "chromadb":
            self._setup_chromadb()
        else:
            raise ValueError(f"Unsupported vector store type: {self.vector_store_type}")
    
    def _setup_faiss(self):
        """Setup FAISS vector store."""
        self.faiss_index = None
        self.embedding_dim = self.embedding_model.dimension
    
    def _setup_chromadb(self):
        """Setup ChromaDB vector store."""
        self.chroma_client = None
        self.chroma_collection = None
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to the retriever."""
        self.documents.extend(documents)
        
        # Generate embeddings for new documents
        texts = [doc.get('text', doc.get('content', '')) for doc in documents]
        new_embeddings = self.embedding_model.encode_batch(texts)
        
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
        
        # Update vector store
        if self.vector_store_type == "faiss":
            self._update_faiss()
        elif self.vector_store_type == "chromadb":
            self._update_chromadb()
    
    def _update_faiss(self):
        """Update FAISS index with new embeddings."""
        if self.faiss_index is None:
            # Create new index
            self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
        
        # Normalize embeddings for cosine similarity
        from sklearn.preprocessing import normalize
        normalized_embeddings = normalize(self.embeddings, norm='l2').astype('float32')
        
        # Clear and rebuild index
        self.faiss_index.reset()
        self.faiss_index.add(normalized_embeddings)
    
    def _update_chromadb(self):
        """Update ChromaDB with new documents."""
        if self.chroma_client is None:
            # Initialize ChromaDB
            chroma_dir = DATA_DIR / "vector_db" / "chroma_db"
            chroma_dir.mkdir(parents=True, exist_ok=True)
            
            self.chroma_client = chromadb.PersistentClient(path=str(chroma_dir))
            self.chroma_collection = self.chroma_client.get_or_create_collection(
                name="dense_retrieval",
                metadata={"description": "Dense retrieval collection"}
            )
        
        # Prepare documents for ChromaDB
        documents = [doc.get('text', doc.get('content', '')) for doc in self.documents]
        metadatas = []
        ids = []
        
        for i, doc in enumerate(self.documents):
            metadata = {
                'source': doc.get('source', 'unknown'),
                'title': doc.get('title', ''),
                'chunk_id': doc.get('chunk_id', f'chunk_{i}')
            }
            metadatas.append(metadata)
            ids.append(f"doc_{i}")
        
        # Add to collection
        self.chroma_collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
            embeddings=self.embeddings.tolist()
        )
    
    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Retrieve documents using dense similarity."""
        if not self.documents:
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query)
        
        if self.vector_store_type == "faiss":
            return self._retrieve_faiss(query_embedding, top_k)
        elif self.vector_store_type == "chromadb":
            return self._retrieve_chromadb(query, top_k)
    
    def _retrieve_faiss(self, query_embedding: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        """Retrieve using FAISS."""
        # Normalize query embedding
        from sklearn.preprocessing import normalize
        query_embedding = normalize(query_embedding.reshape(1, -1), norm='l2').astype('float32')
        
        # Search
        scores, indices = self.faiss_index.search(query_embedding, top_k)
        
        # Format results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                doc = self.documents[idx].copy()
                doc['score'] = float(score)
                doc['retrieval_type'] = 'dense'
                results.append(doc)
        
        return results
    
    def _retrieve_chromadb(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Retrieve using ChromaDB."""
        results = self.chroma_collection.query(
            query_texts=[query],
            n_results=top_k,
            include=['documents', 'metadatas', 'distances']
        )
        
        formatted_results = []
        if results['documents'] and results['documents'][0]:
            for doc, metadata, distance in zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            ):
                # Convert distance to similarity score
                similarity = 1 - distance
                
                result = {
                    'text': doc,
                    'score': similarity,
                    'retrieval_type': 'dense',
                    **metadata
                }
                formatted_results.append(result)
        
        return formatted_results

class SparseRetriever(BaseRetriever):
    """
    Sparse retrieval using BM25 for keyword-based search.
    
    This retriever uses traditional keyword matching which can be
    effective for exact term matches and complements dense retrieval.
    """
    
    def __init__(self):
        self.documents = []
        self.bm25 = None
        self.tokenized_docs = []
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to the retriever."""
        self.documents.extend(documents)
        
        # Tokenize documents for BM25
        texts = [doc.get('text', doc.get('content', '')) for doc in documents]
        self.tokenized_docs = [self._tokenize(text) for text in texts]
        
        # Build BM25 index
        self.bm25 = BM25Okapi(self.tokenized_docs)
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for BM25."""
        import re
        # Convert to lowercase and split on whitespace
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens
    
    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Retrieve documents using BM25."""
        if not self.documents or self.bm25 is None:
            return []
        
        # Tokenize query
        query_tokens = self._tokenize(query)
        
        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top-k results
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        # Format results
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include documents with positive scores
                doc = self.documents[idx].copy()
                doc['score'] = float(scores[idx])
                doc['retrieval_type'] = 'sparse'
                results.append(doc)
        
        return results

class HybridRetriever(BaseRetriever):
    """
    Hybrid retrieval combining dense and sparse methods.
    
    This retriever combines the semantic understanding of dense retrieval
    with the keyword precision of sparse retrieval for better overall performance.
    """
    
    def __init__(self, dense_retriever: DenseRetriever, sparse_retriever: SparseRetriever,
                 alpha: float = 0.7, beta: float = 0.3):
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.alpha = alpha  # Weight for dense search
        self.beta = beta    # Weight for sparse search
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to both retrievers."""
        self.dense_retriever.add_documents(documents)
        self.sparse_retriever.add_documents(documents)
    
    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Retrieve documents using hybrid approach."""
        # Get results from both retrievers
        dense_results = self.dense_retriever.retrieve(query, top_k * 2)  # Get more for better fusion
        sparse_results = self.sparse_retriever.retrieve(query, top_k * 2)
        
        # Normalize scores
        dense_scores = self._normalize_scores([r['score'] for r in dense_results])
        sparse_scores = self._normalize_scores([r['score'] for r in sparse_results])
        
        # Create document ID to result mapping
        doc_results = {}
        
        # Add dense results
        for result, score in zip(dense_results, dense_scores):
            doc_id = self._get_doc_id(result)
            doc_results[doc_id] = {
                'result': result,
                'dense_score': score,
                'sparse_score': 0.0
            }
        
        # Add sparse results
        for result, score in zip(sparse_results, sparse_scores):
            doc_id = self._get_doc_id(result)
            if doc_id in doc_results:
                doc_results[doc_id]['sparse_score'] = score
            else:
                doc_results[doc_id] = {
                    'result': result,
                    'dense_score': 0.0,
                    'sparse_score': score
                }
        
        # Combine scores
        final_results = []
        for doc_id, data in doc_results.items():
            combined_score = (self.alpha * data['dense_score'] + 
                            self.beta * data['sparse_score'])
            
            result = data['result'].copy()
            result['score'] = combined_score
            result['dense_score'] = data['dense_score']
            result['sparse_score'] = data['sparse_score']
            result['retrieval_type'] = 'hybrid'
            
            final_results.append(result)
        
        # Sort by combined score and return top-k
        final_results.sort(key=lambda x: x['score'], reverse=True)
        return final_results[:top_k]
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to [0, 1] range."""
        if not scores:
            return []
        
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return [1.0] * len(scores)
        
        return [(score - min_score) / (max_score - min_score) for score in scores]
    
    def _get_doc_id(self, result: Dict[str, Any]) -> str:
        """Get unique document ID for deduplication."""
        return result.get('chunk_id', result.get('id', str(hash(result.get('text', '')))))

class RetrievalSystem:
    """
    Complete retrieval system with multiple strategies and reranking.
    
    This class provides a unified interface for different retrieval methods
    and includes reranking capabilities for improved results.
    """
    
    def __init__(self, config: RetrievalConfig = None):
        self.config = config or RetrievalConfig()
        self.embedding_model = None
        self.dense_retriever = None
        self.sparse_retriever = None
        self.hybrid_retriever = None
        self.reranker = None
        
        # Initialize components
        self._setup_components()
    
    def _setup_components(self):
        """Setup retrieval components."""
        # Load embedding model
        self.embedding_model = load_embedding_model("bge_base_en")
        
        # Create retrievers
        self.dense_retriever = DenseRetriever(self.embedding_model, "faiss")
        self.sparse_retriever = SparseRetriever()
        self.hybrid_retriever = HybridRetriever(
            self.dense_retriever, 
            self.sparse_retriever,
            self.config.hybrid_alpha,
            self.config.hybrid_beta
        )
        
        # Load reranker if enabled
        if self.config.use_reranking:
            try:
                self.reranker = load_reranker(self.config.reranker_model)
            except Exception as e:
                logger.warning(f"Failed to load reranker: {e}")
                self.reranker = None
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to all retrievers."""
        self.dense_retriever.add_documents(documents)
        self.sparse_retriever.add_documents(documents)
        self.hybrid_retriever.add_documents(documents)
    
    def retrieve(self, query: str, method: str = "hybrid", top_k: int = None) -> List[Dict[str, Any]]:
        """
        Retrieve documents using specified method.
        
        Args:
            query: Search query
            method: Retrieval method ("dense", "sparse", "hybrid")
            top_k: Number of results to return
            
        Returns:
            List of retrieved documents with scores
        """
        if top_k is None:
            top_k = self.config.top_k
        
        # Get initial retrieval results
        if method == "dense":
            results = self.dense_retriever.retrieve(query, top_k)
        elif method == "sparse":
            results = self.sparse_retriever.retrieve(query, top_k)
        elif method == "hybrid":
            results = self.hybrid_retriever.retrieve(query, top_k)
        else:
            raise ValueError(f"Unknown retrieval method: {method}")
        
        # Apply reranking if enabled
        if self.reranker and len(results) > 1:
            results = self._rerank_results(query, results)
        
        return results
    
    def _rerank_results(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply reranking to retrieval results."""
        if not results:
            return results
        
        # Extract document texts
        documents = [r.get('text', r.get('content', '')) for r in results]
        
        # Rerank
        reranked = self.reranker.rerank(query, documents, top_k=self.config.rerank_top_k)
        
        # Combine with original metadata
        final_results = []
        for rerank_result in reranked:
            original_idx = rerank_result['original_index']
            original_result = results[original_idx]
            
            # Update with reranking score
            final_result = original_result.copy()
            final_result['rerank_score'] = rerank_result['score']
            final_result['final_score'] = rerank_result['score']  # Use rerank score as final
            
            final_results.append(final_result)
        
        return final_results
    
    def compare_methods(self, query: str, top_k: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        """Compare different retrieval methods on the same query."""
        methods = ["dense", "sparse", "hybrid"]
        results = {}
        
        for method in methods:
            results[method] = self.retrieve(query, method=method, top_k=top_k)
        
        return results
    
    def evaluate_retrieval(self, queries: List[str], ground_truth: List[List[str]], 
                          method: str = "hybrid") -> Dict[str, float]:
        """
        Evaluate retrieval performance.
        
        Args:
            queries: List of test queries
            ground_truth: List of relevant document IDs for each query
            method: Retrieval method to evaluate
            
        Returns:
            Dictionary of evaluation metrics
        """
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        precisions = []
        recalls = []
        f1_scores = []
        
        for query, gt_docs in zip(queries, ground_truth):
            # Retrieve documents
            results = self.retrieve(query, method=method, top_k=len(gt_docs))
            retrieved_docs = [r.get('chunk_id', r.get('id', '')) for r in results]
            
            # Calculate metrics
            y_true = [1 if doc_id in gt_docs else 0 for doc_id in retrieved_docs]
            y_pred = [1] * len(retrieved_docs)  # All retrieved docs are considered relevant
            
            if y_true:
                precisions.append(precision_score(y_true, y_pred, zero_division=0))
                recalls.append(recall_score(y_true, y_pred, zero_division=0))
                f1_scores.append(f1_score(y_true, y_pred, zero_division=0))
        
        return {
            'precision': np.mean(precisions) if precisions else 0.0,
            'recall': np.mean(recalls) if recalls else 0.0,
            'f1_score': np.mean(f1_scores) if f1_scores else 0.0,
            'num_queries': len(queries)
        }

def create_retrieval_system(config: RetrievalConfig = None) -> RetrievalSystem:
    """Convenience function to create a retrieval system."""
    return RetrievalSystem(config)

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create retrieval system
    config = RetrievalConfig(top_k=5, use_reranking=True)
    retrieval_system = create_retrieval_system(config)
    
    # Sample documents
    documents = [
        {
            'id': 'doc1',
            'text': 'Machine learning is a subset of artificial intelligence.',
            'title': 'Machine Learning',
            'source': 'wikipedia'
        },
        {
            'id': 'doc2', 
            'text': 'Deep learning uses neural networks with multiple layers.',
            'title': 'Deep Learning',
            'source': 'wikipedia'
        },
        {
            'id': 'doc3',
            'text': 'Cats are small, furry animals that make great pets.',
            'title': 'Cats',
            'source': 'wikipedia'
        }
    ]
    
    # Add documents
    retrieval_system.add_documents(documents)
    
    # Test retrieval
    query = "What is machine learning?"
    results = retrieval_system.retrieve(query, method="hybrid")
    
    print(f"Query: {query}")
    print("Results:")
    for i, result in enumerate(results):
        print(f"{i+1}. Score: {result['score']:.3f}")
        print(f"   Title: {result['title']}")
        print(f"   Text: {result['text']}")
        print()
    
    # Compare methods
    print("Method comparison:")
    comparison = retrieval_system.compare_methods(query)
    for method, results in comparison.items():
        print(f"{method}: {len(results)} results")
        if results:
            print(f"  Top result: {results[0]['title']} (score: {results[0]['score']:.3f})")

"""
Reranker Models Module for RAG System
This module provides reranking capabilities to improve retrieval quality.

Author: Kamran Shojaei - Physicist with background in AI/ML
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging
from pathlib import Path
import json

# Import configuration
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import MODELS, DATA_DIR

logger = logging.getLogger(__name__)

class RerankerModel:
    """
    Base class for reranker models.
    
    Rerankers take a query and a list of documents, then reorder them
    by relevance. This is typically done after initial retrieval to
    improve the quality of the final results.
    """
    
    def __init__(self, model_name: str, device: str = "auto"):
        self.model_name = model_name
        self.device = self._get_device(device)
        self.model = None
        self.tokenizer = None
        
    def _get_device(self, device: str) -> str:
        """Determine the best device to use."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
    
    def load_model(self):
        """Load the reranker model."""
        raise NotImplementedError("Subclasses must implement load_model")
    
    def rerank(self, query: str, documents: List[str], top_k: int = None) -> List[Dict[str, Any]]:
        """Rerank documents based on query relevance."""
        raise NotImplementedError("Subclasses must implement rerank")

class BGEReranker(RerankerModel):
    """
    BGE Reranker implementation.
    
    BGE rerankers are specifically designed for retrieval tasks and
    provide high-quality relevance scoring.
    """
    
    def __init__(self, model_name: str = "BAAI/bge-reranker-large", device: str = "auto"):
        super().__init__(model_name, device)
        self.load_model()
    
    def load_model(self):
        """Load the BGE reranker model."""
        try:
            logger.info(f"Loading BGE reranker: {self.model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            
            if self.device != "cuda":
                self.model = self.model.to(self.device)
            
            self.model.eval()
            logger.info("BGE reranker loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load BGE reranker: {e}")
            raise
    
    def rerank(self, query: str, documents: List[str], top_k: int = None) -> List[Dict[str, Any]]:
        """Rerank documents using BGE reranker."""
        if not documents:
            return []
        
        if top_k is None:
            top_k = len(documents)
        
        try:
            # Prepare input pairs
            pairs = [[query, doc] for doc in documents]
            
            # Tokenize
            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            # Get scores
            with torch.no_grad():
                scores = self.model(**inputs).logits.squeeze(-1)
                scores = torch.sigmoid(scores)  # Convert to probabilities
            
            # Create results with original indices
            results = []
            for i, (doc, score) in enumerate(zip(documents, scores.cpu().numpy())):
                results.append({
                    'document': doc,
                    'score': float(score),
                    'original_index': i
                })
            
            # Sort by score (descending)
            results.sort(key=lambda x: x['score'], reverse=True)
            
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Failed to rerank with BGE: {e}")
            raise

class CrossEncoderReranker(RerankerModel):
    """
    Generic cross-encoder reranker implementation.
    
    This can be used with any cross-encoder model from HuggingFace.
    """
    
    def __init__(self, model_name: str, device: str = "auto"):
        super().__init__(model_name, device)
        self.load_model()
    
    def load_model(self):
        """Load the cross-encoder model."""
        try:
            logger.info(f"Loading cross-encoder reranker: {self.model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            
            if self.device != "cuda":
                self.model = self.model.to(self.device)
            
            self.model.eval()
            logger.info("Cross-encoder reranker loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load cross-encoder reranker: {e}")
            raise
    
    def rerank(self, query: str, documents: List[str], top_k: int = None) -> List[Dict[str, Any]]:
        """Rerank documents using cross-encoder."""
        if not documents:
            return []
        
        if top_k is None:
            top_k = len(documents)
        
        try:
            # Prepare input pairs
            pairs = [[query, doc] for doc in documents]
            
            # Tokenize
            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            # Get scores
            with torch.no_grad():
                scores = self.model(**inputs).logits.squeeze(-1)
                # Apply softmax to get probabilities
                scores = torch.softmax(scores, dim=0)
            
            # Create results with original indices
            results = []
            for i, (doc, score) in enumerate(zip(documents, scores.cpu().numpy())):
                results.append({
                    'document': doc,
                    'score': float(score),
                    'original_index': i
                })
            
            # Sort by score (descending)
            results.sort(key=lambda x: x['score'], reverse=True)
            
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Failed to rerank with cross-encoder: {e}")
            raise

class RerankerFactory:
    """
    Factory class for creating reranker models.
    """
    
    @staticmethod
    def create_reranker(reranker_type: str, **kwargs) -> RerankerModel:
        """Create a reranker based on type."""
        reranker_configs = {
            "bge_reranker": BGEReranker,
            "cross_encoder": CrossEncoderReranker,
        }
        
        if reranker_type not in reranker_configs:
            raise ValueError(f"Unknown reranker type: {reranker_type}")
        
        reranker_class = reranker_configs[reranker_type]
        return reranker_class(**kwargs)
    
    @staticmethod
    def get_available_rerankers() -> List[str]:
        """Get list of available reranker types."""
        return ["bge_reranker", "cross_encoder"]

class RerankingPipeline:
    """
    Complete reranking pipeline that can be used in RAG systems.
    
    This class demonstrates how to integrate reranking into a retrieval pipeline
    and provides utilities for evaluation and comparison.
    """
    
    def __init__(self, reranker: RerankerModel, initial_retrieval_k: int = 20, final_k: int = 5):
        self.reranker = reranker
        self.initial_retrieval_k = initial_retrieval_k
        self.final_k = final_k
    
    def rerank_retrieval_results(self, query: str, retrieval_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rerank retrieval results.
        
        Args:
            query: The search query
            retrieval_results: Results from initial retrieval (should have 'text' or 'content' field)
            
        Returns:
            Reranked results with relevance scores
        """
        if not retrieval_results:
            return []
        
        # Extract document texts
        documents = []
        for result in retrieval_results:
            text = result.get('text', result.get('content', ''))
            documents.append(text)
        
        # Rerank
        reranked = self.reranker.rerank(query, documents, top_k=self.final_k)
        
        # Combine with original metadata
        final_results = []
        for rerank_result in reranked:
            original_idx = rerank_result['original_index']
            original_result = retrieval_results[original_idx]
            
            # Create new result with reranking score
            final_result = original_result.copy()
            final_result['rerank_score'] = rerank_result['score']
            final_result['original_retrieval_score'] = original_result.get('score', 0.0)
            
            final_results.append(final_result)
        
        return final_results
    
    def evaluate_reranking(self, queries: List[str], retrieval_results: List[List[Dict[str, Any]]], 
                          ground_truth: List[List[int]]) -> Dict[str, float]:
        """
        Evaluate reranking performance.
        
        Args:
            queries: List of queries
            retrieval_results: List of retrieval results for each query
            ground_truth: List of relevant document indices for each query
            
        Returns:
            Dictionary of evaluation metrics
        """
        from sklearn.metrics import ndcg_score
        
        ndcg_scores = []
        mrr_scores = []
        
        for query, results, gt in zip(queries, retrieval_results, ground_truth):
            if not results or not gt:
                continue
            
            # Rerank results
            reranked = self.rerank_retrieval_results(query, results)
            
            # Create relevance scores for NDCG
            relevance_scores = [0] * len(reranked)
            for i, result in enumerate(reranked):
                original_idx = result.get('original_index', i)
                if original_idx in gt:
                    relevance_scores[i] = 1
            
            # Calculate NDCG@k
            if sum(relevance_scores) > 0:
                ndcg = ndcg_score([relevance_scores], [relevance_scores], k=min(5, len(relevance_scores)))
                ndcg_scores.append(ndcg)
            
            # Calculate MRR
            for i, score in enumerate(relevance_scores):
                if score > 0:
                    mrr_scores.append(1.0 / (i + 1))
                    break
            else:
                mrr_scores.append(0.0)
        
        return {
            'ndcg@5': np.mean(ndcg_scores) if ndcg_scores else 0.0,
            'mrr': np.mean(mrr_scores) if mrr_scores else 0.0,
            'num_queries': len(queries)
        }

def load_reranker(reranker_type: str = "bge_reranker", **kwargs) -> RerankerModel:
    """Convenience function to load a reranker model."""
    return RerankerFactory.create_reranker(reranker_type, **kwargs)

def compare_rerankers(queries: List[str], documents: List[str], 
                     rerankers: List[RerankerModel]) -> Dict[str, Any]:
    """
    Compare multiple rerankers on the same data.
    
    Args:
        queries: List of test queries
        documents: List of documents to rerank
        rerankers: List of reranker models to compare
        
    Returns:
        Comparison results
    """
    results = {}
    
    for reranker in rerankers:
        reranker_name = reranker.model_name
        results[reranker_name] = []
        
        for query in queries:
            reranked = reranker.rerank(query, documents, top_k=5)
            results[reranker_name].append(reranked)
    
    return results

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Test BGE reranker
    try:
        print("Testing BGE reranker...")
        reranker = BGEReranker()
        
        query = "What is machine learning?"
        documents = [
            "Machine learning is a subset of artificial intelligence.",
            "Cats are small, furry animals that make great pets.",
            "Deep learning uses neural networks with multiple layers.",
            "The weather is nice today."
        ]
        
        results = reranker.rerank(query, documents, top_k=3)
        
        print(f"Query: {query}")
        print("Reranked results:")
        for i, result in enumerate(results):
            print(f"{i+1}. Score: {result['score']:.3f}")
            print(f"   Document: {result['document']}")
            print()
        
    except Exception as e:
        print(f"BGE reranker test failed: {e}")
    
    # Test reranking pipeline
    try:
        print("Testing reranking pipeline...")
        pipeline = RerankingPipeline(reranker)
        
        # Mock retrieval results
        retrieval_results = [
            {'text': doc, 'score': 0.8, 'source': 'wikipedia'} 
            for doc in documents
        ]
        
        reranked_results = pipeline.rerank_retrieval_results(query, retrieval_results)
        
        print("Pipeline results:")
        for i, result in enumerate(reranked_results):
            print(f"{i+1}. Rerank Score: {result['rerank_score']:.3f}")
            print(f"   Original Score: {result['original_retrieval_score']:.3f}")
            print(f"   Text: {result['text'][:50]}...")
            print()
        
    except Exception as e:
        print(f"Pipeline test failed: {e}")

"""
Evaluation Metrics Module for RAG System
This module provides comprehensive evaluation capabilities for RAG systems.

Author: Kamran Shojaei - Physicist with background in AI/ML
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
from pathlib import Path
import json
import time
from dataclasses import dataclass
from collections import defaultdict
import re

# Evaluation libraries
from sklearn.metrics import precision_score, recall_score, f1_score, ndcg_score
from rouge_score import rouge_scorer
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

logger = logging.getLogger(__name__)

@dataclass
class EvaluationConfig:
    """Configuration for evaluation metrics."""
    k_values: List[int] = None
    rouge_types: List[str] = None
    bleu_weights: List[Tuple[float, ...]] = None
    use_rouge: bool = True
    use_bleu: bool = True
    use_retrieval_metrics: bool = True
    use_generation_metrics: bool = True
    
    def __post_init__(self):
        if self.k_values is None:
            self.k_values = [1, 3, 5, 10]
        if self.rouge_types is None:
            self.rouge_types = ['rouge1', 'rouge2', 'rougeL']
        if self.bleu_weights is None:
            self.bleu_weights = [(1, 0, 0, 0), (0.5, 0.5, 0, 0), (0.33, 0.33, 0.33, 0)]

class RetrievalMetrics:
    """
    Metrics for evaluating retrieval quality.
    
    This class provides standard information retrieval metrics
    for assessing how well the retrieval system finds relevant documents.
    """
    
    def __init__(self, k_values: List[int] = None):
        self.k_values = k_values or [1, 3, 5, 10]
    
    def precision_at_k(self, retrieved_docs: List[str], relevant_docs: List[str], k: int) -> float:
        """Calculate Precision@K."""
        if k == 0:
            return 0.0
        
        retrieved_at_k = retrieved_docs[:k]
        relevant_retrieved = len(set(retrieved_at_k) & set(relevant_docs))
        return relevant_retrieved / k
    
    def recall_at_k(self, retrieved_docs: List[str], relevant_docs: List[str], k: int) -> float:
        """Calculate Recall@K."""
        if not relevant_docs:
            return 0.0
        
        retrieved_at_k = retrieved_docs[:k]
        relevant_retrieved = len(set(retrieved_at_k) & set(relevant_docs))
        return relevant_retrieved / len(relevant_docs)
    
    def f1_at_k(self, retrieved_docs: List[str], relevant_docs: List[str], k: int) -> float:
        """Calculate F1@K."""
        precision = self.precision_at_k(retrieved_docs, relevant_docs, k)
        recall = self.recall_at_k(retrieved_docs, relevant_docs, k)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def mean_reciprocal_rank(self, retrieved_docs: List[str], relevant_docs: List[str]) -> float:
        """Calculate Mean Reciprocal Rank (MRR)."""
        if not relevant_docs:
            return 0.0
        
        for i, doc in enumerate(retrieved_docs):
            if doc in relevant_docs:
                return 1.0 / (i + 1)
        
        return 0.0
    
    def ndcg_at_k(self, retrieved_docs: List[str], relevant_docs: List[str], k: int) -> float:
        """Calculate Normalized Discounted Cumulative Gain@K."""
        if k == 0 or not relevant_docs:
            return 0.0
        
        # Create relevance scores
        relevance_scores = [1 if doc in relevant_docs else 0 for doc in retrieved_docs[:k]]
        
        if sum(relevance_scores) == 0:
            return 0.0
        
        # Calculate NDCG
        return ndcg_score([relevance_scores], [relevance_scores], k=k)
    
    def evaluate_retrieval(self, queries: List[str], retrieved_docs: List[List[str]], 
                          relevant_docs: List[List[str]]) -> Dict[str, float]:
        """
        Evaluate retrieval performance across all queries.
        
        Args:
            queries: List of queries
            retrieved_docs: List of retrieved document IDs for each query
            relevant_docs: List of relevant document IDs for each query
            
        Returns:
            Dictionary of average metrics
        """
        metrics = defaultdict(list)
        
        for query, retrieved, relevant in zip(queries, retrieved_docs, relevant_docs):
            # Calculate metrics for each k value
            for k in self.k_values:
                metrics[f'precision@{k}'].append(
                    self.precision_at_k(retrieved, relevant, k)
                )
                metrics[f'recall@{k}'].append(
                    self.recall_at_k(retrieved, relevant, k)
                )
                metrics[f'f1@{k}'].append(
                    self.f1_at_k(retrieved, relevant, k)
                )
                metrics[f'ndcg@{k}'].append(
                    self.ndcg_at_k(retrieved, relevant, k)
                )
            
            # Calculate MRR
            metrics['mrr'].append(
                self.mean_reciprocal_rank(retrieved, relevant)
            )
        
        # Calculate averages
        avg_metrics = {}
        for metric_name, values in metrics.items():
            avg_metrics[metric_name] = np.mean(values)
        
        return avg_metrics

class GenerationMetrics:
    """
    Metrics for evaluating generation quality.
    
    This class provides metrics for assessing the quality of generated
    responses, including fluency, relevance, and factual accuracy.
    """
    
    def __init__(self, config: EvaluationConfig = None):
        self.config = config or EvaluationConfig()
        self.rouge_scorer = None
        
        if self.config.use_rouge:
            self.rouge_scorer = rouge_scorer.RougeScorer(
                self.config.rouge_types, 
                use_stemmer=True
            )
    
    def bleu_score(self, generated: str, reference: str, weights: Tuple[float, ...] = None) -> float:
        """Calculate BLEU score."""
        if not self.config.use_bleu:
            return 0.0
        
        if weights is None:
            weights = (1, 0, 0, 0)  # BLEU-1
        
        # Tokenize
        generated_tokens = nltk.word_tokenize(generated.lower())
        reference_tokens = nltk.word_tokenize(reference.lower())
        
        # Calculate BLEU with smoothing
        smoothing = SmoothingFunction().method1
        return sentence_bleu([reference_tokens], generated_tokens, 
                           weights=weights, smoothing_function=smoothing)
    
    def rouge_score(self, generated: str, reference: str) -> Dict[str, float]:
        """Calculate ROUGE scores."""
        if not self.config.use_rouge or self.rouge_scorer is None:
            return {}
        
        scores = self.rouge_scorer.score(reference, generated)
        
        # Extract F1 scores
        rouge_scores = {}
        for rouge_type in self.config.rouge_types:
            rouge_scores[f'rouge_{rouge_type}'] = scores[rouge_type].fmeasure
        
        return rouge_scores
    
    def semantic_similarity(self, generated: str, reference: str) -> float:
        """Calculate semantic similarity using sentence transformers."""
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
            
            embeddings = model.encode([generated, reference])
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            
            return float(similarity)
        except Exception as e:
            logger.warning(f"Failed to calculate semantic similarity: {e}")
            return 0.0
    
    def evaluate_generation(self, generated_texts: List[str], reference_texts: List[str]) -> Dict[str, float]:
        """
        Evaluate generation quality across all texts.
        
        Args:
            generated_texts: List of generated responses
            reference_texts: List of reference responses
            
        Returns:
            Dictionary of average metrics
        """
        metrics = defaultdict(list)
        
        for generated, reference in zip(generated_texts, reference_texts):
            # BLEU scores
            for i, weights in enumerate(self.config.bleu_weights):
                bleu = self.bleu_score(generated, reference, weights)
                metrics[f'bleu_{i+1}'].append(bleu)
            
            # ROUGE scores
            rouge_scores = self.rouge_score(generated, reference)
            for rouge_type, score in rouge_scores.items():
                metrics[rouge_type].append(score)
            
            # Semantic similarity
            similarity = self.semantic_similarity(generated, reference)
            metrics['semantic_similarity'].append(similarity)
        
        # Calculate averages
        avg_metrics = {}
        for metric_name, values in metrics.items():
            avg_metrics[metric_name] = np.mean(values)
        
        return avg_metrics

class RAGEvaluator:
    """
    Comprehensive RAG system evaluator.
    
    This class combines retrieval and generation evaluation to provide
    a complete assessment of RAG system performance.
    """
    
    def __init__(self, config: EvaluationConfig = None):
        self.config = config or EvaluationConfig()
        self.retrieval_metrics = RetrievalMetrics(self.config.k_values)
        self.generation_metrics = GenerationMetrics(self.config)
    
    def evaluate_rag_system(self, 
                           queries: List[str],
                           retrieved_docs: List[List[Dict[str, Any]]],
                           generated_responses: List[str],
                           reference_responses: List[str],
                           relevant_doc_ids: List[List[str]] = None) -> Dict[str, Any]:
        """
        Evaluate complete RAG system performance.
        
        Args:
            queries: List of test queries
            retrieved_docs: List of retrieved documents for each query
            generated_responses: List of generated responses
            reference_responses: List of reference responses
            relevant_doc_ids: List of relevant document IDs for each query
            
        Returns:
            Comprehensive evaluation results
        """
        results = {
            'num_queries': len(queries),
            'timestamp': time.time()
        }
        
        # Evaluate retrieval if we have relevant document IDs
        if relevant_doc_ids and self.config.use_retrieval_metrics:
            retrieved_ids = [[doc.get('id', doc.get('chunk_id', '')) for doc in docs] 
                           for docs in retrieved_docs]
            
            retrieval_results = self.retrieval_metrics.evaluate_retrieval(
                queries, retrieved_ids, relevant_doc_ids
            )
            results['retrieval'] = retrieval_results
        
        # Evaluate generation
        if self.config.use_generation_metrics:
            generation_results = self.generation_metrics.evaluate_generation(
                generated_responses, reference_responses
            )
            results['generation'] = generation_results
        
        # Calculate overall RAG score (weighted combination)
        rag_score = self._calculate_rag_score(results)
        results['rag_score'] = rag_score
        
        return results
    
    def _calculate_rag_score(self, results: Dict[str, Any]) -> float:
        """Calculate overall RAG score."""
        scores = []
        
        # Add retrieval scores
        if 'retrieval' in results:
            retrieval = results['retrieval']
            if 'mrr' in retrieval:
                scores.append(retrieval['mrr'])
            if 'ndcg@5' in retrieval:
                scores.append(retrieval['ndcg@5'])
        
        # Add generation scores
        if 'generation' in results:
            generation = results['generation']
            if 'rouge_rouge1' in generation:
                scores.append(generation['rouge_rouge1'])
            if 'semantic_similarity' in generation:
                scores.append(generation['semantic_similarity'])
        
        return np.mean(scores) if scores else 0.0
    
    def compare_models(self, model_results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        Compare multiple model results.
        
        Args:
            model_results: Dictionary mapping model names to evaluation results
            
        Returns:
            DataFrame with comparison results
        """
        comparison_data = []
        
        for model_name, results in model_results.items():
            row = {'model': model_name}
            
            # Add retrieval metrics
            if 'retrieval' in results:
                for metric, value in results['retrieval'].items():
                    row[f'retrieval_{metric}'] = value
            
            # Add generation metrics
            if 'generation' in results:
                for metric, value in results['generation'].items():
                    row[f'generation_{metric}'] = value
            
            # Add overall score
            row['rag_score'] = results.get('rag_score', 0.0)
            
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def save_results(self, results: Dict[str, Any], output_path: Path):
        """Save evaluation results to file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Evaluation results saved to {output_path}")
    
    def load_results(self, input_path: Path) -> Dict[str, Any]:
        """Load evaluation results from file."""
        with open(input_path, 'r') as f:
            results = json.load(f)
        
        logger.info(f"Evaluation results loaded from {input_path}")
        return results

class OnlineEvaluator:
    """
    Online evaluation for real-time RAG system monitoring.
    
    This class provides metrics for monitoring RAG system performance
    in production environments.
    """
    
    def __init__(self):
        self.metrics_history = []
        self.user_feedback = []
    
    def log_query(self, query: str, response: str, response_time: float, 
                  retrieved_docs: List[Dict[str, Any]] = None):
        """Log a query and its response for monitoring."""
        log_entry = {
            'timestamp': time.time(),
            'query': query,
            'response': response,
            'response_time': response_time,
            'num_retrieved_docs': len(retrieved_docs) if retrieved_docs else 0
        }
        
        self.metrics_history.append(log_entry)
    
    def log_feedback(self, query: str, response: str, rating: int, 
                    feedback_text: str = None):
        """Log user feedback for a query-response pair."""
        feedback_entry = {
            'timestamp': time.time(),
            'query': query,
            'response': response,
            'rating': rating,  # 1-5 scale
            'feedback_text': feedback_text
        }
        
        self.user_feedback.append(feedback_entry)
    
    def get_performance_metrics(self, time_window: int = 3600) -> Dict[str, float]:
        """Get performance metrics for the specified time window."""
        current_time = time.time()
        cutoff_time = current_time - time_window
        
        # Filter recent entries
        recent_queries = [entry for entry in self.metrics_history 
                         if entry['timestamp'] >= cutoff_time]
        recent_feedback = [entry for entry in self.user_feedback 
                          if entry['timestamp'] >= cutoff_time]
        
        if not recent_queries:
            return {}
        
        # Calculate metrics
        response_times = [entry['response_time'] for entry in recent_queries]
        avg_response_time = np.mean(response_times)
        p95_response_time = np.percentile(response_times, 95)
        
        # User satisfaction
        if recent_feedback:
            ratings = [entry['rating'] for entry in recent_feedback]
            avg_rating = np.mean(ratings)
            satisfaction_rate = len([r for r in ratings if r >= 4]) / len(ratings)
        else:
            avg_rating = 0.0
            satisfaction_rate = 0.0
        
        return {
            'queries_per_hour': len(recent_queries) / (time_window / 3600),
            'avg_response_time': avg_response_time,
            'p95_response_time': p95_response_time,
            'avg_user_rating': avg_rating,
            'satisfaction_rate': satisfaction_rate,
            'total_queries': len(recent_queries),
            'total_feedback': len(recent_feedback)
        }

def create_evaluator(config: EvaluationConfig = None) -> RAGEvaluator:
    """Convenience function to create a RAG evaluator."""
    return RAGEvaluator(config)

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create evaluator
    config = EvaluationConfig(k_values=[1, 3, 5], use_rouge=True, use_bleu=True)
    evaluator = create_evaluator(config)
    
    # Sample data
    queries = ["What is machine learning?", "How does deep learning work?"]
    retrieved_docs = [
        [{'id': 'doc1', 'text': 'Machine learning is...'}, {'id': 'doc2', 'text': 'Deep learning uses...'}],
        [{'id': 'doc2', 'text': 'Deep learning uses...'}, {'id': 'doc3', 'text': 'Neural networks...'}]
    ]
    generated_responses = [
        "Machine learning is a subset of artificial intelligence that focuses on algorithms.",
        "Deep learning uses neural networks with multiple layers to process data."
    ]
    reference_responses = [
        "Machine learning is a field of artificial intelligence that enables computers to learn.",
        "Deep learning is a subset of machine learning that uses neural networks."
    ]
    relevant_doc_ids = [['doc1'], ['doc2']]
    
    # Evaluate
    results = evaluator.evaluate_rag_system(
        queries, retrieved_docs, generated_responses, 
        reference_responses, relevant_doc_ids
    )
    
    print("Evaluation Results:")
    print(json.dumps(results, indent=2))
    
    # Test online evaluator
    online_eval = OnlineEvaluator()
    online_eval.log_query("Test query", "Test response", 1.5)
    online_eval.log_feedback("Test query", "Test response", 4, "Good answer!")
    
    perf_metrics = online_eval.get_performance_metrics()
    print(f"\nPerformance Metrics: {perf_metrics}")

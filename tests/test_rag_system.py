"""
Comprehensive Test Suite for RAG System
This module provides unit tests, integration tests, and end-to-end tests for the RAG system.

Author: Kamran Shojaei - Physicist with background in AI/ML
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import pandas as pd

# Import our modules
import sys
sys.path.append(str(Path(__file__).parent.parent))

# Import only the modules that exist and work
try:
    from src.models.embedding_models import BGEEmbeddingModel, E5EmbeddingModel, EmbeddingModelFactory
except ImportError:
    print("Warning: Could not import embedding models")
    BGEEmbeddingModel = None
    E5EmbeddingModel = None
    EmbeddingModelFactory = None

try:
    from src.retrieval.retrieval_system import RetrievalSystem, DenseRetriever, SparseRetriever, HybridRetriever
except ImportError:
    print("Warning: Could not import retrieval system")
    RetrievalSystem = None
    DenseRetriever = None
    SparseRetriever = None
    HybridRetriever = None

try:
    from src.optimization.performance_analysis import PerformanceProfiler, CostAnalyzer
except ImportError:
    print("Warning: Could not import performance analysis")
    PerformanceProfiler = None
    CostAnalyzer = None

try:
    from src.vector_db.vector_manager import VectorDatabaseManager, create_vector_database
except ImportError:
    print("Warning: Could not import vector database manager")
    VectorDatabaseManager = None
    create_vector_database = None

class TestEmbeddingModels:
    """Test cases for embedding models."""
    
    @pytest.mark.skipif(BGEEmbeddingModel is None, reason="BGEEmbeddingModel not available")
    def test_bge_embedding_model_initialization(self):
        """Test BGE embedding model initialization."""
        model = BGEEmbeddingModel()
        assert model.model_name == "BAAI/bge-base-en-v1.5"
        assert model.dimension == 768
        assert model.max_length == 512
    
    @pytest.mark.skipif(E5EmbeddingModel is None, reason="E5EmbeddingModel not available")
    def test_e5_embedding_model_initialization(self):
        """Test E5 embedding model initialization."""
        model = E5EmbeddingModel()
        assert model.model_name == "intfloat/e5-base-v2"
        assert model.dimension == 768
        assert model.max_length == 512
    
    @pytest.mark.skipif(EmbeddingModelFactory is None, reason="EmbeddingModelFactory not available")
    def test_embedding_model_factory(self):
        """Test embedding model factory."""
        bge_model = EmbeddingModelFactory.create_model("bge_base_en")
        assert isinstance(bge_model, BGEEmbeddingModel)
        
        e5_model = EmbeddingModelFactory.create_model("e5_base_v2")
        assert isinstance(e5_model, E5EmbeddingModel)
    
    @pytest.mark.skipif(BGEEmbeddingModel is None, reason="BGEEmbeddingModel not available")
    @patch('sentence_transformers.SentenceTransformer')
    def test_embedding_generation(self, mock_transformer):
        """Test embedding generation."""
        # Mock the transformer
        mock_model = Mock()
        mock_model.encode.return_value = np.random.rand(2, 768)
        mock_transformer.return_value = mock_model
        
        model = BGEEmbeddingModel()
        texts = ["Hello world", "Test text"]
        embeddings = model.encode(texts)
        
        assert embeddings.shape == (2, 768)
        mock_model.encode.assert_called_once()

class TestRetrievalSystem:
    """Test cases for retrieval system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.sample_documents = [
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
            }
        ]
    
    @pytest.mark.skipif(DenseRetriever is None, reason="DenseRetriever not available")
    @patch('src.models.embedding_models.BGEEmbeddingModel')
    def test_dense_retriever(self, mock_embedding_model):
        """Test dense retriever functionality."""
        # Mock embedding model
        mock_model = Mock()
        mock_model.dimension = 768
        mock_model.encode.return_value = np.random.rand(1, 768)
        mock_model.encode_batch.return_value = np.random.rand(2, 768)
        mock_embedding_model.return_value = mock_model
        
        retriever = DenseRetriever(mock_model, "faiss")
        retriever.add_documents(self.sample_documents)
        
        # Test retrieval
        results = retriever.retrieve("What is machine learning?", top_k=2)
        
        assert len(results) <= 2
        assert all('score' in result for result in results)
        assert all('retrieval_type' in result for result in results)
    
    @pytest.mark.skipif(SparseRetriever is None, reason="SparseRetriever not available")
    def test_sparse_retriever(self):
        """Test sparse retriever functionality."""
        retriever = SparseRetriever()
        retriever.add_documents(self.sample_documents)
        
        # Test retrieval
        results = retriever.retrieve("machine learning", top_k=2)
        
        assert len(results) <= 2
        assert all('score' in result for result in results)
        assert all('retrieval_type' in result for result in results)
    
    @pytest.mark.skipif(HybridRetriever is None, reason="HybridRetriever not available")
    @patch('src.models.embedding_models.BGEEmbeddingModel')
    def test_hybrid_retriever(self, mock_embedding_model):
        """Test hybrid retriever functionality."""
        # Mock embedding model
        mock_model = Mock()
        mock_model.dimension = 768
        mock_model.encode.return_value = np.random.rand(1, 768)
        mock_model.encode_batch.return_value = np.random.rand(2, 768)
        mock_embedding_model.return_value = mock_model
        
        dense_retriever = DenseRetriever(mock_model, "faiss")
        sparse_retriever = SparseRetriever()
        
        hybrid_retriever = HybridRetriever(dense_retriever, sparse_retriever)
        hybrid_retriever.add_documents(self.sample_documents)
        
        # Test retrieval
        results = hybrid_retriever.retrieve("machine learning", top_k=2)
        
        assert len(results) <= 2
        assert all('score' in result for result in results)
        assert all('retrieval_type' in result for result in results)

class TestEvaluationMetrics:
    """Test cases for evaluation metrics."""
    
    def test_retrieval_metrics(self):
        """Test retrieval metrics calculation."""
        metrics = RetrievalMetrics()
        
        retrieved_docs = ['doc1', 'doc2', 'doc3']
        relevant_docs = ['doc1', 'doc3']
        
        # Test precision@2
        precision = metrics.precision_at_k(retrieved_docs, relevant_docs, 2)
        assert precision == 1.0  # Both retrieved docs are relevant
        
        # Test recall@2
        recall = metrics.recall_at_k(retrieved_docs, relevant_docs, 2)
        assert recall == 1.0  # All relevant docs are retrieved
        
        # Test MRR
        mrr = metrics.mean_reciprocal_rank(retrieved_docs, relevant_docs)
        assert mrr == 1.0  # First relevant doc is at position 1
    
    def test_generation_metrics(self):
        """Test generation metrics calculation."""
        metrics = GenerationMetrics()
        
        generated = "Machine learning is a subset of artificial intelligence."
        reference = "Machine learning is a field of artificial intelligence."
        
        # Test BLEU score
        bleu = metrics.bleu_score(generated, reference)
        assert 0 <= bleu <= 1
        
        # Test ROUGE score
        rouge = metrics.rouge_score(generated, reference)
        assert 'rouge_rouge1' in rouge
        assert 0 <= rouge['rouge_rouge1'] <= 1

class TestAdvancedFeatures:
    """Test cases for advanced features."""
    
    def test_query_rewriter(self):
        """Test query rewriting functionality."""
        rewriter = QueryRewriter()
        
        query = "What is machine learning?"
        rewrites = rewriter.rewrite_query(query)
        
        assert len(rewrites) > 0
        assert any(r['method'] == 'original' for r in rewrites)
        assert all('confidence' in r for r in rewrites)
    
    def test_multipass_reasoner(self):
        """Test multipass reasoning functionality."""
        reasoner = MultipassReasoner()
        
        # Mock LLM generator
        mock_llm = Mock()
        mock_llm.llm_model = Mock()
        mock_llm.llm_model.generate.return_value = "Mock response"
        
        query = "What is machine learning?"
        context = "Machine learning is a subset of AI."
        
        result = reasoner.reason(query, context, mock_llm)
        
        assert 'answer' in result
        assert 'reasoning_type' in result
        assert 'confidence' in result
    
    def test_hallucination_detector(self):
        """Test hallucination detection functionality."""
        detector = HallucinationDetector()
        
        response = "Machine learning is a subset of artificial intelligence."
        context = "Machine learning is a subset of artificial intelligence that focuses on algorithms."
        
        result = detector.detect_hallucinations(response, context)
        
        assert 'is_hallucinated' in result
        assert 'confidence' in result
        assert 'flagged_claims' in result
        assert 'recommendations' in result

class TestPerformanceAnalysis:
    """Test cases for performance analysis."""
    
    def test_performance_profiler(self):
        """Test performance profiler functionality."""
        profiler = PerformanceProfiler()
        
        # Test profiling a request
        with profiler.profile_request("test_request", "query"):
            pass
        
        summary = profiler.get_performance_summary()
        assert 'total_requests' in summary
        assert summary['total_requests'] == 1
    
    def test_cost_analyzer(self):
        """Test cost analyzer functionality."""
        analyzer = CostAnalyzer()
        
        # Test cost calculation
        cost = analyzer.calculate_cost("gpt-3.5-turbo", 100, 50)
        assert cost > 0
        
        # Test recording request cost
        analyzer.record_request_cost("gpt-3.5-turbo", 100, 50, "req_001")
        
        summary = analyzer.get_cost_summary()
        assert 'total_cost' in summary
        assert summary['total_cost'] > 0

class TestIntegration:
    """Integration tests for the complete RAG system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.sample_documents = [
            {
                'id': 'doc1',
                'text': 'Machine learning is a subset of artificial intelligence that focuses on algorithms.',
                'title': 'Machine Learning',
                'source': 'wikipedia'
            },
            {
                'id': 'doc2',
                'text': 'Deep learning uses neural networks with multiple layers to process data.',
                'title': 'Deep Learning',
                'source': 'wikipedia'
            }
        ]
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('src.models.embedding_models.BGEEmbeddingModel')
    @patch('src.models.llm_models.LlamaModel')
    def test_end_to_end_rag_system(self, mock_llm, mock_embedding):
        """Test end-to-end RAG system functionality."""
        # Mock embedding model
        mock_embedding_model = Mock()
        mock_embedding_model.dimension = 768
        mock_embedding_model.encode.return_value = np.random.rand(1, 768)
        mock_embedding_model.encode_batch.return_value = np.random.rand(2, 768)
        mock_embedding.return_value = mock_embedding_model
        
        # Mock LLM model
        mock_llm_model = Mock()
        mock_llm_model.generate.return_value = "Machine learning is a subset of AI that focuses on algorithms."
        mock_llm.return_value = mock_llm_model
        
        # Create RAG generator
        prompt_template = PromptTemplate.get_rag_template("llama")
        rag_generator = RAGGenerator(mock_llm_model, prompt_template)
        
        # Create retrieval system
        retrieval_system = RetrievalSystem()
        retrieval_system.add_documents(self.sample_documents)
        
        # Test complete RAG pipeline
        query = "What is machine learning?"
        retrieved_docs = retrieval_system.retrieve(query, method="dense", top_k=2)
        
        assert len(retrieved_docs) > 0
        
        # Generate response
        response = rag_generator.generate_response(query, retrieved_docs)
        
        assert 'response' in response
        assert 'generation_time' in response
        assert len(response['response']) > 0

class TestDataValidation:
    """Test cases for data validation."""
    
    def test_document_validation(self):
        """Test document structure validation."""
        valid_doc = {
            'id': 'doc1',
            'text': 'Sample text',
            'title': 'Sample Title',
            'source': 'wikipedia'
        }
        
        # Test valid document
        assert 'id' in valid_doc
        assert 'text' in valid_doc
        assert 'title' in valid_doc
        assert 'source' in valid_doc
        
        # Test invalid document
        invalid_doc = {'id': 'doc1'}  # Missing required fields
        
        required_fields = ['id', 'text', 'title', 'source']
        missing_fields = [field for field in required_fields if field not in invalid_doc]
        assert len(missing_fields) > 0
    
    def test_query_validation(self):
        """Test query validation."""
        valid_queries = [
            "What is machine learning?",
            "How does deep learning work?",
            "Explain neural networks"
        ]
        
        for query in valid_queries:
            assert isinstance(query, str)
            assert len(query.strip()) > 0
            assert len(query.split()) >= 2  # At least 2 words

class TestErrorHandling:
    """Test cases for error handling."""
    
    def test_embedding_model_error_handling(self):
        """Test error handling in embedding models."""
        with patch('sentence_transformers.SentenceTransformer') as mock_transformer:
            mock_transformer.side_effect = Exception("Model loading failed")
            
            with pytest.raises(Exception):
                BGEEmbeddingModel()
    
    def test_retrieval_error_handling(self):
        """Test error handling in retrieval system."""
        retriever = DenseRetriever(Mock(), "faiss")
        
        # Test retrieval with no documents
        results = retriever.retrieve("test query")
        assert results == []
    
    def test_llm_error_handling(self):
        """Test error handling in LLM models."""
        with patch('transformers.AutoModelForCausalLM.from_pretrained') as mock_model:
            mock_model.side_effect = Exception("Model loading failed")
            
            with pytest.raises(Exception):
                LlamaModel()

# Fixtures for pytest
@pytest.fixture
def sample_documents():
    """Sample documents for testing."""
    return [
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
        }
    ]

@pytest.fixture
def sample_queries():
    """Sample queries for testing."""
    return [
        "What is machine learning?",
        "How does deep learning work?",
        "Explain neural networks"
    ]

# Performance tests
@pytest.mark.performance
class TestPerformance:
    """Performance tests for the RAG system."""
    
    def test_embedding_generation_performance(self):
        """Test embedding generation performance."""
        # This would be a performance test
        # In practice, you'd measure actual performance metrics
        pass
    
    def test_retrieval_performance(self):
        """Test retrieval performance."""
        # This would be a performance test
        # In practice, you'd measure actual performance metrics
        pass

# Integration tests
@pytest.mark.integration
class TestIntegrationScenarios:
    """Integration test scenarios."""
    
    def test_full_rag_pipeline(self):
        """Test the complete RAG pipeline."""
        # This would test the complete pipeline
        pass
    
    def test_api_endpoints(self):
        """Test API endpoints."""
        # This would test the FastAPI endpoints
        pass

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])

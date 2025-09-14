"""
Pytest configuration and fixtures for RAG system tests.

Author: Kamran Shojaei - Physicist with background in AI/ML
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np
import json

@pytest.fixture(scope="session")
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path, ignore_errors=True)

@pytest.fixture
def sample_documents():
    """Sample documents for testing."""
    return [
        {
            'id': 'doc1',
            'text': 'Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.',
            'title': 'Machine Learning',
            'source': 'wikipedia',
            'chunk_id': 'chunk_1'
        },
        {
            'id': 'doc2',
            'text': 'Deep learning uses neural networks with multiple layers to process complex data patterns.',
            'title': 'Deep Learning',
            'source': 'wikipedia',
            'chunk_id': 'chunk_2'
        },
        {
            'id': 'doc3',
            'text': 'Natural language processing is a field of AI that focuses on the interaction between computers and human language.',
            'title': 'Natural Language Processing',
            'source': 'wikipedia',
            'chunk_id': 'chunk_3'
        }
    ]

@pytest.fixture
def sample_queries():
    """Sample queries for testing."""
    return [
        "What is machine learning?",
        "How does deep learning work?",
        "Explain natural language processing",
        "What are neural networks?",
        "How do AI algorithms learn?"
    ]

@pytest.fixture
def mock_embedding_model():
    """Mock embedding model for testing."""
    mock_model = Mock()
    mock_model.dimension = 768
    mock_model.max_length = 512
    mock_model.encode.return_value = np.random.rand(1, 768)
    mock_model.encode_batch.return_value = np.random.rand(3, 768)
    return mock_model

@pytest.fixture
def mock_llm_model():
    """Mock LLM model for testing."""
    mock_model = Mock()
    mock_model.generate.return_value = "This is a mock response about machine learning."
    mock_model.generate_stream.return_value = iter(["This ", "is ", "a ", "mock ", "response."])
    return mock_model

@pytest.fixture
def mock_reranker():
    """Mock reranker for testing."""
    mock_reranker = Mock()
    mock_reranker.rerank.return_value = [
        {'document': 'Mock document 1', 'score': 0.9, 'original_index': 0},
        {'document': 'Mock document 2', 'score': 0.8, 'original_index': 1}
    ]
    return mock_reranker

@pytest.fixture
def test_config():
    """Test configuration."""
    return {
        'chunk_size': 512,
        'chunk_overlap': 50,
        'top_k': 5,
        'rerank_top_k': 3,
        'similarity_threshold': 0.7,
        'max_context_length': 4000
    }

@pytest.fixture
def mock_retrieval_results():
    """Mock retrieval results for testing."""
    return [
        {
            'id': 'doc1',
            'text': 'Machine learning is a subset of artificial intelligence.',
            'title': 'Machine Learning',
            'source': 'wikipedia',
            'score': 0.9,
            'retrieval_type': 'dense'
        },
        {
            'id': 'doc2',
            'text': 'Deep learning uses neural networks.',
            'title': 'Deep Learning',
            'source': 'wikipedia',
            'score': 0.8,
            'retrieval_type': 'dense'
        }
    ]

@pytest.fixture
def mock_evaluation_data():
    """Mock evaluation data for testing."""
    return {
        'queries': ['What is machine learning?', 'How does AI work?'],
        'retrieved_docs': [
            [{'id': 'doc1', 'text': 'ML is a subset of AI.'}, {'id': 'doc2', 'text': 'AI includes ML.'}],
            [{'id': 'doc3', 'text': 'AI is intelligence in machines.'}, {'id': 'doc4', 'text': 'ML learns from data.'}]
        ],
        'generated_responses': [
            'Machine learning is a subset of artificial intelligence.',
            'AI is intelligence demonstrated by machines.'
        ],
        'reference_responses': [
            'Machine learning is a subset of artificial intelligence that focuses on algorithms.',
            'Artificial intelligence is intelligence demonstrated by machines.'
        ],
        'relevant_doc_ids': [['doc1'], ['doc3']]
    }

# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as a performance test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )

def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Add slow marker to performance tests
        if "performance" in item.name:
            item.add_marker(pytest.mark.slow)
        
        # Add integration marker to integration tests
        if "integration" in item.name:
            item.add_marker(pytest.mark.integration)

# Test utilities
class TestUtils:
    """Utility functions for tests."""
    
    @staticmethod
    def create_mock_documents(count: int = 5) -> list:
        """Create mock documents for testing."""
        documents = []
        for i in range(count):
            documents.append({
                'id': f'doc_{i}',
                'text': f'This is document {i} with some sample text content.',
                'title': f'Document {i}',
                'source': 'test',
                'chunk_id': f'chunk_{i}'
            })
        return documents
    
    @staticmethod
    def create_mock_embeddings(count: int, dimension: int = 768) -> np.ndarray:
        """Create mock embeddings for testing."""
        return np.random.rand(count, dimension)
    
    @staticmethod
    def create_mock_retrieval_results(count: int = 3) -> list:
        """Create mock retrieval results for testing."""
        results = []
        for i in range(count):
            results.append({
                'id': f'doc_{i}',
                'text': f'Mock document {i} content.',
                'title': f'Document {i}',
                'source': 'test',
                'score': 0.9 - i * 0.1,
                'retrieval_type': 'dense'
            })
        return results

# Performance testing utilities
class PerformanceTestUtils:
    """Utilities for performance testing."""
    
    @staticmethod
    def measure_time(func, *args, **kwargs):
        """Measure execution time of a function."""
        import time
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        return result, end_time - start_time
    
    @staticmethod
    def assert_performance_threshold(actual_time: float, max_time: float, test_name: str):
        """Assert that performance meets threshold."""
        assert actual_time <= max_time, f"{test_name} took {actual_time:.3f}s, expected <= {max_time:.3f}s"

# Mock data generators
class MockDataGenerator:
    """Generate mock data for testing."""
    
    @staticmethod
    def generate_text_corpus(size: int = 100) -> list:
        """Generate a corpus of mock text documents."""
        topics = ['machine learning', 'artificial intelligence', 'deep learning', 'neural networks', 'data science']
        corpus = []
        
        for i in range(size):
            topic = topics[i % len(topics)]
            text = f"This is a document about {topic}. It contains information about {topic} and its applications."
            corpus.append({
                'id': f'doc_{i}',
                'text': text,
                'title': f'Document about {topic}',
                'source': 'mock',
                'chunk_id': f'chunk_{i}'
            })
        
        return corpus
    
    @staticmethod
    def generate_queries(count: int = 10) -> list:
        """Generate mock queries for testing."""
        query_templates = [
            "What is {}?",
            "How does {} work?",
            "Explain {}",
            "Tell me about {}",
            "What are the benefits of {}?"
        ]
        
        topics = ['machine learning', 'AI', 'deep learning', 'neural networks', 'data science']
        queries = []
        
        for i in range(count):
            template = query_templates[i % len(query_templates)]
            topic = topics[i % len(topics)]
            queries.append(template.format(topic))
        
        return queries

# Test data validation
class TestDataValidator:
    """Validate test data integrity."""
    
    @staticmethod
    def validate_document_structure(doc: dict) -> bool:
        """Validate document structure."""
        required_fields = ['id', 'text', 'title', 'source']
        return all(field in doc for field in required_fields)
    
    @staticmethod
    def validate_retrieval_result(result: dict) -> bool:
        """Validate retrieval result structure."""
        required_fields = ['id', 'text', 'score', 'retrieval_type']
        return all(field in result for field in required_fields)
    
    @staticmethod
    def validate_evaluation_data(data: dict) -> bool:
        """Validate evaluation data structure."""
        required_fields = ['queries', 'retrieved_docs', 'generated_responses', 'reference_responses']
        return all(field in data for field in required_fields)

# Cleanup utilities
def cleanup_test_files(temp_dir: Path):
    """Clean up test files."""
    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)

# Test configuration
pytest_plugins = []

# Custom test markers
pytestmark = [
    pytest.mark.unit,
]

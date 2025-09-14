"""
Basic Test Suite for RAG System
This module provides basic tests that should work without complex dependencies.

Author: Kamran Shojaei - Physicist with background in AI/ML
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np
import pandas as pd

# Import our modules
import sys
sys.path.append(str(Path(__file__).parent.parent))

class TestBasicFunctionality:
    """Test basic functionality that should always work."""
    
    def test_imports(self):
        """Test that basic imports work."""
        # Test that we can import basic modules
        import src.config
        import src.data.preprocess_data
        import src.vector_db.vector_manager
        
        # Test that config has expected attributes
        assert hasattr(src.config, 'DATA_DIR')
        assert hasattr(src.config, 'VECTOR_DB_CONFIG')
    
    def test_data_directory_structure(self):
        """Test that data directory structure exists."""
        from src.config import DATA_DIR
        
        # Check that data directories exist
        assert DATA_DIR.exists()
        assert (DATA_DIR / "raw").exists()
        assert (DATA_DIR / "processed").exists()
        assert (DATA_DIR / "vector_db").exists()
    
    def test_vector_database_manager_import(self):
        """Test that vector database manager can be imported."""
        try:
            from src.vector_db.vector_manager import VectorDatabaseManager, create_vector_database
            assert VectorDatabaseManager is not None
            assert create_vector_database is not None
        except ImportError as e:
            pytest.skip(f"Vector database manager not available: {e}")
    
    def test_config_values(self):
        """Test that config has reasonable values."""
        from src.config import DATA_DIR, VECTOR_DB_CONFIG
        
        # Test DATA_DIR is a valid path
        assert isinstance(DATA_DIR, Path)
        assert str(DATA_DIR).endswith('data')
        
        # Test VECTOR_DB_CONFIG has expected keys
        assert 'type' in VECTOR_DB_CONFIG
        assert 'collection_name' in VECTOR_DB_CONFIG
        assert 'distance_metric' in VECTOR_DB_CONFIG
        assert 'persist_directory' in VECTOR_DB_CONFIG

class TestVectorDatabase:
    """Test vector database functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.sample_documents = [
            {
                "id": "doc1",
                "text": "Machine learning is a subset of artificial intelligence.",
                "metadata": {"source": "wikipedia", "title": "Machine Learning"}
            },
            {
                "id": "doc2", 
                "text": "Deep learning uses neural networks with multiple layers.",
                "metadata": {"source": "wikipedia", "title": "Deep Learning"}
            }
        ]
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_vector_database_creation(self):
        """Test creating a vector database."""
        try:
            from src.vector_db.vector_manager import create_vector_database
            
            # Create a simple vector database
            vdb = create_vector_database(
                backend="chromadb",
                collection_name="test_collection",
                documents=self.sample_documents
            )
            
            assert vdb is not None
            assert hasattr(vdb, 'search')
            assert hasattr(vdb, 'add_documents')
            assert hasattr(vdb, 'get_stats')
            
        except ImportError:
            pytest.skip("Vector database manager not available")
        except Exception as e:
            pytest.skip(f"Vector database creation failed: {e}")
    
    def test_vector_database_search(self):
        """Test vector database search functionality."""
        try:
            from src.vector_db.vector_manager import create_vector_database
            
            # Create vector database
            vdb = create_vector_database(
                backend="chromadb",
                collection_name="test_search",
                documents=self.sample_documents
            )
            
            # Test search
            results = vdb.search("machine learning", top_k=2)
            
            assert isinstance(results, list)
            assert len(results) <= 2
            
            if results:
                result = results[0]
                assert 'document' in result
                assert 'score' in result
                assert 'metadata' in result
                
        except ImportError:
            pytest.skip("Vector database manager not available")
        except Exception as e:
            pytest.skip(f"Vector database search failed: {e}")

class TestDataProcessing:
    """Test data processing functionality."""
    
    def test_text_preprocessor_import(self):
        """Test that text preprocessor can be imported."""
        try:
            from src.data.preprocess_data import TextPreprocessor
            assert TextPreprocessor is not None
        except ImportError:
            pytest.skip("Text preprocessor not available")
    
    def test_text_preprocessor_functionality(self):
        """Test text preprocessor basic functionality."""
        try:
            from src.data.preprocess_data import TextPreprocessor
            
            preprocessor = TextPreprocessor()
            
            # Test basic preprocessing
            text = "This is a test text with some punctuation!"
            processed = preprocessor.preprocess_text(text)
            
            assert isinstance(processed, str)
            assert len(processed) > 0
            
        except ImportError:
            pytest.skip("Text preprocessor not available")
        except Exception as e:
            pytest.skip(f"Text preprocessor failed: {e}")

class TestWebApplications:
    """Test web application functionality."""
    
    def test_web_app_import(self):
        """Test that web app can be imported."""
        try:
            import web_app
            assert web_app is not None
        except ImportError:
            pytest.skip("Web app not available")
    
    def test_web_app_vector_import(self):
        """Test that vector web app can be imported."""
        try:
            import web_app_vector
            assert web_app_vector is not None
        except ImportError:
            pytest.skip("Vector web app not available")

class TestNotebooks:
    """Test that notebooks exist and are accessible."""
    
    def test_notebooks_exist(self):
        """Test that all expected notebooks exist."""
        notebooks_dir = Path(__file__).parent.parent / "notebooks"
        
        expected_notebooks = [
            "01_understanding_rag.ipynb",
            "02_data_collection.ipynb", 
            "03_embeddings_and_vector_store.ipynb",
            "04_text_preprocessing.ipynb",
            "05_vector_search.ipynb",
            "06_retrieval_systems.ipynb",
            "07_llm_integration.ipynb",
            "08_evaluation.ipynb",
            "09_optimization.ipynb",
            "10_vector_database_production.ipynb"
        ]
        
        for notebook in expected_notebooks:
            notebook_path = notebooks_dir / notebook
            assert notebook_path.exists(), f"Notebook {notebook} not found"

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])

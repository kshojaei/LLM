"""
Embedding Models Module for RAG System
This module provides various embedding models for converting text to vectors.

Author: Kamran Shojaei - Physicist with background in AI/ML
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional, Union
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import logging
from pathlib import Path
import json

# Import configuration
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import MODELS, DATA_DIR

logger = logging.getLogger(__name__)

class EmbeddingModel:
    """
    Base class for embedding models.
    
    This class provides a unified interface for different embedding models
    and demonstrates how to work with various embedding architectures.
    """
    
    def __init__(self, model_name: str, device: str = "auto"):
        self.model_name = model_name
        self.device = self._get_device(device)
        self.model = None
        self.tokenizer = None
        self.dimension = None
        self.max_length = None
        
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
        """Load the embedding model."""
        raise NotImplementedError("Subclasses must implement load_model")
    
    def encode(self, texts: Union[str, List[str]], **kwargs) -> np.ndarray:
        """Encode texts to embeddings."""
        raise NotImplementedError("Subclasses must implement encode")
    
    def encode_batch(self, texts: List[str], batch_size: int = 32, **kwargs) -> np.ndarray:
        """Encode texts in batches for efficiency."""
        raise NotImplementedError("Subclasses must implement encode_batch")

class BGEEmbeddingModel(EmbeddingModel):
    """
    BGE (BAAI General Embedding) model implementation.
    
    BGE models are optimized for retrieval tasks and provide high-quality
    embeddings for both queries and documents.
    """
    
    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5", device: str = "auto"):
        super().__init__(model_name, device)
        self.dimension = 768
        self.max_length = 512
        self.load_model()
    
    def load_model(self):
        """Load the BGE model using sentence-transformers."""
        try:
            logger.info(f"Loading BGE model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            logger.info("BGE model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load BGE model: {e}")
            raise
    
    def encode(self, texts: Union[str, List[str]], normalize_embeddings: bool = True, **kwargs) -> np.ndarray:
        """Encode texts using BGE model."""
        if isinstance(texts, str):
            texts = [texts]
        
        try:
            embeddings = self.model.encode(
                texts,
                normalize_embeddings=normalize_embeddings,
                convert_to_numpy=True,
                **kwargs
            )
            return embeddings
        except Exception as e:
            logger.error(f"Failed to encode texts with BGE: {e}")
            raise
    
    def encode_batch(self, texts: List[str], batch_size: int = 32, **kwargs) -> np.ndarray:
        """Encode texts in batches."""
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.encode(batch_texts, **kwargs)
            all_embeddings.append(batch_embeddings)
        
        return np.vstack(all_embeddings)

class E5EmbeddingModel(EmbeddingModel):
    """
    E5 (Embedding from Everything) model implementation.
    
    E5 models are designed for text embedding tasks and provide good
    performance on various retrieval benchmarks.
    """
    
    def __init__(self, model_name: str = "intfloat/e5-base-v2", device: str = "auto"):
        super().__init__(model_name, device)
        self.dimension = 768
        self.max_length = 512
        self.load_model()
    
    def load_model(self):
        """Load the E5 model using sentence-transformers."""
        try:
            logger.info(f"Loading E5 model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            logger.info("E5 model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load E5 model: {e}")
            raise
    
    def encode(self, texts: Union[str, List[str]], normalize_embeddings: bool = True, **kwargs) -> np.ndarray:
        """Encode texts using E5 model."""
        if isinstance(texts, str):
            texts = [texts]
        
        try:
            embeddings = self.model.encode(
                texts,
                normalize_embeddings=normalize_embeddings,
                convert_to_numpy=True,
                **kwargs
            )
            return embeddings
        except Exception as e:
            logger.error(f"Failed to encode texts with E5: {e}")
            raise
    
    def encode_batch(self, texts: List[str], batch_size: int = 32, **kwargs) -> np.ndarray:
        """Encode texts in batches."""
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.encode(batch_texts, **kwargs)
            all_embeddings.append(batch_embeddings)
        
        return np.vstack(all_embeddings)

class EmbeddingModelFactory:
    """
    Factory class for creating embedding models.
    
    This demonstrates the factory pattern for creating different types
    of embedding models based on configuration.
    """
    
    @staticmethod
    def create_model(model_type: str, **kwargs) -> EmbeddingModel:
        """Create an embedding model based on type."""
        model_configs = {
            "bge_base_en": BGEEmbeddingModel,
            "e5_base_v2": E5EmbeddingModel,
        }
        
        if model_type not in model_configs:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model_class = model_configs[model_type]
        return model_class(**kwargs)
    
    @staticmethod
    def get_available_models() -> List[str]:
        """Get list of available model types."""
        return ["bge_base_en", "e5_base_v2"]

class EmbeddingComparison:
    """
    Utility class for comparing different embedding models.
    
    This class helps evaluate and compare the performance of different
    embedding models on various tasks.
    """
    
    def __init__(self, models: List[EmbeddingModel]):
        self.models = models
        self.results = {}
    
    def compare_embeddings(self, texts: List[str], reference_model: str = None) -> Dict[str, Any]:
        """Compare embeddings from different models."""
        results = {}
        
        for model in self.models:
            model_name = model.model_name
            logger.info(f"Computing embeddings with {model_name}")
            
            try:
                embeddings = model.encode_batch(texts)
                results[model_name] = {
                    "embeddings": embeddings,
                    "dimension": embeddings.shape[1],
                    "shape": embeddings.shape,
                    "mean_norm": np.mean(np.linalg.norm(embeddings, axis=1)),
                    "std_norm": np.std(np.linalg.norm(embeddings, axis=1))
                }
            except Exception as e:
                logger.error(f"Failed to compute embeddings with {model_name}: {e}")
                results[model_name] = {"error": str(e)}
        
        return results
    
    def compute_similarity_matrix(self, model: EmbeddingModel, texts: List[str]) -> np.ndarray:
        """Compute pairwise similarity matrix for texts."""
        embeddings = model.encode_batch(texts)
        similarity_matrix = np.dot(embeddings, embeddings.T)
        return similarity_matrix
    
    def find_most_similar(self, model: EmbeddingModel, query: str, texts: List[str], top_k: int = 5) -> List[tuple]:
        """Find most similar texts to a query."""
        query_embedding = model.encode(query)
        text_embeddings = model.encode_batch(texts)
        
        similarities = np.dot(text_embeddings, query_embedding.T).flatten()
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append((texts[idx], similarities[idx]))
        
        return results

def load_embedding_model(model_type: str = "bge_base_en", **kwargs) -> EmbeddingModel:
    """Convenience function to load an embedding model."""
    return EmbeddingModelFactory.create_model(model_type, **kwargs)

def save_embeddings(embeddings: np.ndarray, texts: List[str], output_path: Path):
    """Save embeddings and associated texts to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        "embeddings": embeddings.tolist(),
        "texts": texts,
        "shape": embeddings.shape,
        "metadata": {
            "num_embeddings": len(texts),
            "dimension": embeddings.shape[1]
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"Saved {len(texts)} embeddings to {output_path}")

def load_embeddings(input_path: Path) -> tuple:
    """Load embeddings and texts from disk."""
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    embeddings = np.array(data["embeddings"])
    texts = data["texts"]
    
    logger.info(f"Loaded {len(texts)} embeddings from {input_path}")
    return embeddings, texts

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Test BGE model
    print("Testing BGE model...")
    bge_model = BGEEmbeddingModel()
    
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Retrieval-augmented generation combines search and generation."
    ]
    
    embeddings = bge_model.encode_batch(sample_texts)
    print(f"BGE embeddings shape: {embeddings.shape}")
    
    # Test E5 model
    print("\nTesting E5 model...")
    e5_model = E5EmbeddingModel()
    e5_embeddings = e5_model.encode_batch(sample_texts)
    print(f"E5 embeddings shape: {e5_embeddings.shape}")
    
    # Compare models
    print("\nComparing models...")
    comparison = EmbeddingComparison([bge_model, e5_model])
    results = comparison.compare_embeddings(sample_texts)
    
    for model_name, result in results.items():
        if "error" not in result:
            print(f"{model_name}: {result['shape']}, mean norm: {result['mean_norm']:.4f}")

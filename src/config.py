"""
Configuration settings for Docs Copilot
This file contains all the configuration parameters for our RAG system.

Author: Kamran Shojaei - Physicist with background in AI/ML
"""

import os
from pathlib import Path
from typing import Dict, Any

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Model configurations
MODELS = {
    "embedding": {
        "bge_base_en": {
            "name": "BAAI/bge-base-en-v1.5",
            "dimension": 768,
            "max_length": 512,
            "description": "High-quality English embeddings from BAAI"
        },
        "e5_base_v2": {
            "name": "intfloat/e5-base-v2", 
            "dimension": 768,
            "max_length": 512,
            "description": "E5 embeddings optimized for retrieval"
        }
    },
    "llm": {
        "llama_3_8b": {
            "name": "meta-llama/Llama-3-8B-Instruct",
            "max_length": 4096,
            "temperature": 0.7,
            "description": "Llama 3 8B Instruct model"
        },
        "mistral_7b": {
            "name": "mistralai/Mistral-7B-Instruct-v0.2",
            "max_length": 4096,
            "temperature": 0.7,
            "description": "Mistral 7B Instruct model"
        }
    },
    "reranker": {
        "bge_reranker": {
            "name": "BAAI/bge-reranker-large",
            "description": "High-quality reranker for final ranking"
        }
    }
}

# Data processing configurations
DATA_CONFIG = {
    "chunking": {
        "chunk_size": int(os.getenv("CHUNK_SIZE", 512)),
        "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", 50)),
        "min_chunk_size": 100,
        "max_chunk_size": 1024
    },
    "preprocessing": {
        "remove_headers": True,
        "remove_footers": True,
        "clean_whitespace": True,
        "remove_special_chars": False
    },
    "datasets": {
        "wikipedia": {
            "name": "wikipedia",
            "subset": "20220301.en",
            "max_documents": 10000,  # Start small for learning
            "description": "Wikipedia articles for general knowledge"
        },
        "arxiv": {
            "name": "scientific_papers",
            "subset": "arxiv",
            "max_documents": 5000,
            "description": "ArXiv paper abstracts for scientific content"
        }
    }
}

# Retrieval configurations
RETRIEVAL_CONFIG = {
    "top_k": 10,  # Number of documents to retrieve
    "rerank_top_k": 5,  # Number of documents to rerank
    "similarity_threshold": 0.7,  # Minimum similarity score
    "hybrid_search": {
        "alpha": 0.7,  # Weight for dense search (1.0 = pure dense, 0.0 = pure sparse)
        "beta": 0.3   # Weight for sparse search
    }
}

# Vector database configurations
VECTOR_DB_CONFIG = {
    "type": "chromadb",  # Options: "chromadb", "faiss"
    "collection_name": "docs_copilot",
    "distance_metric": "cosine",  # Options: "cosine", "euclidean", "manhattan"
    "persist_directory": str(DATA_DIR / "vector_db" / "chroma_db")
}

# API configurations
API_CONFIG = {
    "host": os.getenv("API_HOST", "0.0.0.0"),
    "port": int(os.getenv("API_PORT", 8000)),
    "debug": True,
    "cors_origins": ["*"]
}

# Evaluation configurations
EVALUATION_CONFIG = {
    "metrics": [
        "mrr",  # Mean Reciprocal Rank
        "ndcg",  # Normalized Discounted Cumulative Gain
        "recall",  # Recall at K
        "precision",  # Precision at K
        "bleu",  # BLEU score for generation
        "rouge"  # ROUGE score for generation
    ],
    "k_values": [1, 3, 5, 10],  # K values for recall and precision
    "test_queries": 100,  # Number of test queries for evaluation
}

# Logging configurations
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": str(LOGS_DIR / "docs_copilot.log"),
    "max_file_size": 10 * 1024 * 1024,  # 10MB
    "backup_count": 5
}

# Environment variables
ENV_VARS = {
    "HF_TOKEN": os.getenv("HF_TOKEN"),
    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
    "CUDA_VISIBLE_DEVICES": os.getenv("CUDA_VISIBLE_DEVICES", "0"),
}

def get_config() -> Dict[str, Any]:
    """Get the complete configuration dictionary."""
    return {
        "models": MODELS,
        "data": DATA_CONFIG,
        "retrieval": RETRIEVAL_CONFIG,
        "vector_db": VECTOR_DB_CONFIG,
        "api": API_CONFIG,
        "evaluation": EVALUATION_CONFIG,
        "logging": LOGGING_CONFIG,
        "env": ENV_VARS,
        "paths": {
            "base": str(BASE_DIR),
            "data": str(DATA_DIR),
            "models": str(MODELS_DIR),
            "logs": str(LOGS_DIR)
        }
    }

def print_config():
    """Print the current configuration."""
    config = get_config()
    print("📋 Docs Copilot Configuration")
    print("=" * 40)
    
    for section, settings in config.items():
        print(f"\n🔧 {section.upper()}:")
        if isinstance(settings, dict):
            for key, value in settings.items():
                print(f"  {key}: {value}")
        else:
            print(f"  {settings}")

if __name__ == "__main__":
    print_config()

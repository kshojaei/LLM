"""
Test HuggingFace compatibility to identify the exact import issue.
This helps debug the cached_download import error in CI.
"""

def test_huggingface_hub_imports():
    """Test that we can import from huggingface_hub without cached_download."""
    try:
        import huggingface_hub
        print(f" huggingface_hub imported successfully, version: {huggingface_hub.__version__}")
        
        # Check if cached_download exists (it shouldn't in newer versions)
        if hasattr(huggingface_hub, 'cached_download'):
            print("  cached_download still exists (older version)")
        else:
            print(" cached_download not found (newer version - good)")
            
        # Check if hf_hub_download exists (newer replacement)
        if hasattr(huggingface_hub, 'hf_hub_download'):
            print(" hf_hub_download available (newer version - good)")
        else:
            print("  hf_hub_download not available")
            
    except ImportError as e:
        print(f" huggingface_hub import failed: {e}")
        raise

def test_sentence_transformers_import():
    """Test that sentence-transformers can be imported."""
    try:
        from sentence_transformers import SentenceTransformer
        print(" SentenceTransformer imported successfully")
        
        # Check version
        import sentence_transformers
        print(f" sentence-transformers version: {sentence_transformers.__version__}")
        
    except ImportError as e:
        print(f" sentence-transformers import failed: {e}")
        raise

def test_sentence_transformers_model_loading():
    """Test that we can load a model without cached_download issues."""
    try:
        from sentence_transformers import SentenceTransformer
        
        # Try to load a small model
        print("Loading all-MiniLM-L6-v2 model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print(" Model loaded successfully")
        
        # Test basic functionality
        test_text = "Hello world"
        embedding = model.encode(test_text)
        print(f" Embedding created, shape: {embedding.shape}")
        
    except Exception as e:
        print(f" Model loading failed: {e}")
        raise

def test_alternative_imports():
    """Test alternative ways to import from huggingface_hub."""
    try:
        # Try importing specific functions that should exist
        from huggingface_hub import hf_hub_download
        print(" hf_hub_download imported successfully")
        
        # Try importing other common functions
        from huggingface_hub import snapshot_download
        print(" snapshot_download imported successfully")
        
    except ImportError as e:
        print(f" Alternative imports failed: {e}")
        raise

def test_version_compatibility():
    """Test version compatibility between packages."""
    try:
        import sentence_transformers
        import huggingface_hub
        import transformers
        
        print(f"Package versions:")
        print(f"  sentence-transformers: {sentence_transformers.__version__}")
        print(f"  huggingface_hub: {huggingface_hub.__version__}")
        print(f"  transformers: {transformers.__version__}")
        
        # Check if versions are compatible
        st_version = sentence_transformers.__version__
        hf_version = huggingface_hub.__version__
        
        # Basic compatibility check
        if st_version >= "5.0.0" and hf_version >= "0.20.0":
            print(" Versions appear compatible")
        else:
            print("  Versions may not be compatible")
            
    except Exception as e:
        print(f" Version check failed: {e}")
        raise

if __name__ == "__main__":
    print(" Testing HuggingFace compatibility...")
    test_huggingface_hub_imports()
    test_sentence_transformers_import()
    test_sentence_transformers_model_loading()
    test_alternative_imports()
    test_version_compatibility()
    print(" All compatibility tests passed!")

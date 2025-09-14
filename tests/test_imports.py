"""
Simple import test to verify all dependencies work correctly.
This helps identify import issues early in CI.
"""

def test_sentence_transformers_import():
    """Test that sentence-transformers can be imported."""
    try:
        from sentence_transformers import SentenceTransformer
        assert SentenceTransformer is not None
        print("✅ sentence-transformers import successful")
    except ImportError as e:
        print(f"❌ sentence-transformers import failed: {e}")
        raise

def test_huggingface_hub_import():
    """Test that huggingface_hub can be imported."""
    try:
        import huggingface_hub
        assert huggingface_hub is not None
        print(f"✅ huggingface_hub import successful, version: {huggingface_hub.__version__}")
    except ImportError as e:
        print(f"❌ huggingface_hub import failed: {e}")
        raise

def test_sentence_transformer_loading():
    """Test that we can actually load a sentence transformer model."""
    try:
        from sentence_transformers import SentenceTransformer
        # Try to load a small model
        model = SentenceTransformer('all-MiniLM-L6-v2')
        assert model is not None
        print("✅ SentenceTransformer model loading successful")
    except Exception as e:
        print(f"❌ SentenceTransformer model loading failed: {e}")
        raise

def test_basic_functionality():
    """Test basic sentence transformer functionality."""
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        
        model = SentenceTransformer('all-MiniLM-L6-v2')
        test_sentences = ["Hello world", "Test sentence"]
        embeddings = model.encode(test_sentences)
        
        assert embeddings.shape[0] == 2  # Two sentences
        assert embeddings.shape[1] > 0   # Has dimensions
        assert isinstance(embeddings, np.ndarray)
        print("✅ Basic functionality test successful")
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        raise

if __name__ == "__main__":
    print("🧪 Running import tests...")
    test_sentence_transformers_import()
    test_huggingface_hub_import()
    test_sentence_transformer_loading()
    test_basic_functionality()
    print("🎉 All import tests passed!")

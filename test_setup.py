#!/usr/bin/env python3
"""
Test script to verify the Docs Copilot setup is working correctly.
Run this after setup to make sure everything is configured properly.

Author: Kamran Shojaei - Physicist with background in AI/ML
"""

import sys
import importlib
from pathlib import Path

def test_imports():
    """Test that all required packages can be imported."""
    print("🔍 Testing package imports...")
    
    required_packages = [
        ("torch", "PyTorch"),
        ("transformers", "HuggingFace Transformers"),
        ("datasets", "HuggingFace Datasets"),
        ("sentence_transformers", "Sentence Transformers"),
        ("chromadb", "ChromaDB"),
        ("faiss", "FAISS"),
        ("fastapi", "FastAPI"),
        ("streamlit", "Streamlit"),
        ("pandas", "Pandas"),
        ("numpy", "NumPy"),
        ("tqdm", "TQDM"),
        ("sklearn", "Scikit-learn"),
    ]
    
    failed_imports = []
    
    for package, name in required_packages:
        try:
            importlib.import_module(package)
            print(f"  ✅ {name}")
        except ImportError as e:
            print(f"  ❌ {name}: {e}")
            failed_imports.append(name)
    
    if failed_imports:
        print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
        return False
    else:
        print("\n✅ All packages imported successfully!")
        return True

def test_config():
    """Test that our configuration can be loaded."""
    print("\n🔧 Testing configuration...")
    
    try:
        # Add src to path
        sys.path.append(str(Path(__file__).parent / "src"))
        from src.config import get_config, print_config
        
        config = get_config()
        print("  ✅ Configuration loaded successfully")
        
        # Check that required keys exist
        required_keys = ["models", "data", "retrieval", "vector_db"]
        for key in required_keys:
            if key in config:
                print(f"  ✅ {key} configuration found")
            else:
                print(f"  ❌ {key} configuration missing")
                return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ Configuration error: {e}")
        return False

def test_directories():
    """Test that required directories exist and are writable."""
    print("\n📁 Testing directories...")
    
    required_dirs = [
        "data/raw",
        "data/processed",
        "data/vector_db",
        "models/downloaded",
        "logs",
        "src",
        "notebooks"
    ]
    
    base_path = Path(__file__).parent
    
    for dir_path in required_dirs:
        full_path = base_path / dir_path
        
        if full_path.exists():
            print(f"  ✅ {dir_path}")
        else:
            print(f"  ❌ {dir_path} (creating...)")
            try:
                full_path.mkdir(parents=True, exist_ok=True)
                print(f"  ✅ {dir_path} created")
            except Exception as e:
                print(f"  ❌ {dir_path} creation failed: {e}")
                return False
    
    return True

def test_python_version():
    """Test that Python version is compatible."""
    print("\n🐍 Testing Python version...")
    
    version = sys.version_info
    if version.major == 3 and version.minor >= 9:
        print(f"  ✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"  ❌ Python {version.major}.{version.minor}.{version.micro} (need 3.9+)")
        return False

def test_model_access():
    """Test that we can access model repositories."""
    print("\n🤖 Testing model access...")
    
    try:
        from huggingface_hub import list_models
        
        # Test access to a small model
        models = list_models(limit=1)
        print("  ✅ HuggingFace Hub access working")
        return True
        
    except Exception as e:
        print(f"  ❌ HuggingFace Hub access failed: {e}")
        print("  💡 This might be a network issue or you might need to set HF_TOKEN")
        return False

def main():
    """Run all tests."""
    print("🧪 Docs Copilot Setup Verification")
    print("=" * 40)
    
    tests = [
        ("Python Version", test_python_version),
        ("Package Imports", test_imports),
        ("Configuration", test_config),
        ("Directories", test_directories),
        ("Model Access", test_model_access),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🔍 {test_name}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"  ❌ Test failed with error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 40)
    print("📊 Test Results Summary:")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\nAll tests passed! Your setup is ready.")
        print("\nNext steps:")
        print("1. Start Jupyter Lab: jupyter lab notebooks/")
        print("2. Begin with: 01_understanding_rag.ipynb")
        print("3. Follow the learning path in README.md")
    else:
        print(f"\n{total - passed} test(s) failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("- Run: pip install -r requirements.txt")
        print("- Check your internet connection")
        print("- Set HF_TOKEN environment variable if needed")

if __name__ == "__main__":
    main()

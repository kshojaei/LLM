#!/usr/bin/env python3
"""
Setup script for Docs Copilot - RAG from Scratch
This script helps you set up the project environment and verify everything is working.

Author: Kamran Shojaei - Physicist with background in AI/ML
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def run_command(command: str, description: str) -> bool:
    """Run a command and return True if successful."""
    print(f"{description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"{description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"{description} failed!")
        print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 9:
        print(f"Python {version.major}.{version.minor}.{version.micro} is compatible!")
        return True
    else:
        print(f"Python {version.major}.{version.minor}.{version.micro} is not compatible!")
        print("Please use Python 3.9 or higher.")
        return False

def create_directories():
    """Create necessary directories."""
    print("📁 Creating project directories...")
    directories = [
        "data/raw",
        "data/processed", 
        "data/vector_db",
        "models/downloaded",
        "logs",
        "tests",
        "docs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"  Created {directory}/")
    
    print("All directories created!")

def install_dependencies():
    """Install Python dependencies."""
    print("Installing dependencies...")
    return run_command("pip install -r requirements.txt", "Installing Python packages")

def verify_installation():
    """Verify that key packages are installed correctly."""
    print("🔍 Verifying installation...")
    
    packages_to_check = [
        ("torch", "PyTorch"),
        ("transformers", "HuggingFace Transformers"),
        ("sentence_transformers", "Sentence Transformers"),
        ("chromadb", "ChromaDB"),
        ("fastapi", "FastAPI"),
        ("streamlit", "Streamlit")
    ]
    
    all_good = True
    for package, name in packages_to_check:
        try:
            __import__(package)
            print(f"   {name} is installed")
        except ImportError:
            print(f"  {name} is not installed")
            all_good = False
    
    return all_good

def create_env_file():
    """Create .env file from template if it doesn't exist."""
    env_file = Path(".env")
    env_example = Path("env_example.txt")
    
    if not env_file.exists() and env_example.exists():
        print("Creating .env file...")
        env_file.write_text(env_example.read_text())
        print(".env file created! Edit it with your API keys if needed.")
    elif env_file.exists():
        print(".env file already exists")
    else:
        print(" No env_example.txt found, skipping .env creation")

def main():
    """Main setup function."""
    print("Setting up Docs Copilot - RAG from Scratch")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Install dependencies
    if not install_dependencies():
        print("Failed to install dependencies. Please check the error messages above.")
        sys.exit(1)
    
    # Verify installation
    if not verify_installation():
        print("Some packages failed to install. Please check the error messages above.")
        sys.exit(1)
    
    # Create .env file
    create_env_file()
    
    print("\nSetup completed successfully!")
    print("\nNext steps:")
    print("1. Start Jupyter Lab: jupyter lab notebooks/")
    print("2. Begin with: 01_understanding_rag.ipynb")
    print("3. Follow the learning path in the README.md")
    print("\nTips:")
    print("- Make sure you have at least 16GB RAM for running Llama-3-8B")
    print("- The first model download will take some time")
    print("- Start with small datasets to test your setup")
    print("\nHappy learning!)

if __name__ == "__main__":
    main()

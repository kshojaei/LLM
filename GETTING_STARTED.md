# Getting Started with Docs Copilot

Welcome! This guide will walk you through setting up and running your RAG system step by step.

## Quick Start (5 minutes)

### 1. Set Up Environment

```bash
# Make sure you're in the project directory
cd /Users/scienceman/Desktop/LLM

# Run the setup script
python setup.py
```

### 2. Verify Installation

```bash
# Test that everything is working
python test_setup.py
```

### 3. Start Learning

```bash
# Launch Jupyter Lab
jupyter lab notebooks/

# Open the first notebook: 01_understanding_rag.ipynb
```

## Learning Path

Follow these notebooks in order - each builds on the previous:

### Phase 1: Understanding (Day 1)
1. **`01_understanding_rag.ipynb`** - Learn RAG fundamentals
2. **`02_data_collection.ipynb`** - Collect your first dataset

### Phase 2: Data Processing (Days 2-3)
3. **`03_text_preprocessing.ipynb`** - Clean and chunk your data
4. **`04_embeddings_deep_dive.ipynb`** - Understand embeddings

### Phase 3: Retrieval (Days 4-5)
5. **`05_vector_search.ipynb`** - Build your vector database
6. **`06_retrieval_systems.ipynb`** - Implement hybrid search

### Phase 4: Generation (Days 6-7)
7. **`07_llm_integration.ipynb`** - Connect your LLM
8. **`08_evaluation.ipynb`** - Test your system

### Phase 5: Production (Days 8-9)
9. **`09_optimization.ipynb`** - Optimize and deploy

## Your First Experiment

Let's start with a simple experiment to see RAG in action:

### Step 1: Collect Sample Data

```bash
# Collect a small dataset to start with
python src/data/collect_data.py
```

This will download:
- 100 Wikipedia articles
- 50 ArXiv abstracts

### Step 2: Run the First Notebook

Open `notebooks/01_understanding_rag.ipynb` and run all cells. This will:
- Teach you RAG concepts
- Show you how embeddings work
- Let you experiment with similarity search

### Step 3: Explore Your Data

```python
# In a Python script or notebook
import json
from pathlib import Path

# Load your collected data
with open("data/raw/wikipedia_articles.json", "r") as f:
    wiki_data = json.load(f)

print(f"Collected {len(wiki_data)} Wikipedia articles")
print(f"First article: {wiki_data[0]['title']}")
```

## Configuration

Your system is configured in `src/config.py`. Key settings:

```python
# Data settings
CHUNK_SIZE = 512          # Size of text chunks
CHUNK_OVERLAP = 50        # Overlap between chunks
MAX_DOCUMENTS = 10000     # Limit for learning (increase later)

# Model settings
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
LLM_MODEL = "meta-llama/Llama-3-8B-Instruct"
```

## Learning Tips

### Start Small
- Begin with 100 documents, not 10,000
- Use small models first, then scale up
- Test each component before combining

### Experiment Actively
- Change chunk sizes and see the impact
- Try different embedding models
- Modify prompts and observe results

### Document Everything
- Take notes in notebooks
- Save your experiments
- Track what works and what doesn't

### Ask Questions
- Why does this chunking strategy work better?
- What happens if I change the similarity threshold?
- How does prompt engineering affect results?

## Troubleshooting

### Common Issues

**"Out of memory" errors:**
```bash
# Reduce batch sizes in config.py
# Use smaller models for testing
# Process data in smaller chunks
```

**"Model not found" errors:**
```bash
# Set your HuggingFace token
export HF_TOKEN=your_token_here
# Or add to .env file
```

**"Slow performance":**
```bash
# Use GPU if available
# Reduce dataset size for testing
# Use smaller embedding models
```

### Getting Help

1. Check the logs in `logs/docs_copilot.log`
2. Run `python test_setup.py` to diagnose issues
3. Look at the error messages carefully
4. Search the documentation

## Success Metrics

You'll know you're succeeding when:

- You can collect and preprocess data
- You understand how embeddings work
- You can retrieve relevant documents
- You can generate coherent answers
- You can evaluate your system's performance

## Next Steps

Once you've completed the basic setup:

1. **Expand your dataset** - Add more sources
2. **Try different models** - Experiment with alternatives
3. **Optimize performance** - Speed up your system
4. **Add features** - Implement advanced RAG techniques
5. **Deploy** - Make your system accessible

## Support

If you get stuck:
1. Check the error logs
2. Review the relevant notebook
3. Look at the configuration
4. Try the troubleshooting steps above

Remember: This is a learning project. It's okay to make mistakes and experiment!

---

**Happy learning!**

*Remember: The goal is understanding, not just getting it to work. Take time to explore each component and understand how it contributes to the overall system.*

# Getting Started with RAG Systems

This guide will help you understand and build a retrieval-augmented generation system from scratch. Think of it as learning to build a sophisticated search engine that can also generate answers.

## Quick Setup

First, let's get your environment ready:

```bash
# Navigate to the project directory
cd /Users/scienceman/Desktop/LLM

# Run the setup script
python setup.py

# Verify everything works
python test_setup.py

# Start Jupyter Lab
jupyter lab notebooks/
```

## Learning Approach

The notebooks are designed to build understanding progressively. Start with the fundamentals and work your way up:

### Foundation Building
- **`01_understanding_rag.ipynb`** - Core concepts and why RAG matters
- **`02_data_collection.ipynb`** - Getting real data to work with

### Data Processing
- **`03_text_preprocessing.ipynb`** - Making raw text usable
- **`04_embeddings_deep_dive.ipynb`** - Converting text to numbers

### Search and Retrieval
- **`05_vector_search.ipynb`** - Finding relevant information
- **`06_retrieval_systems.ipynb`** - Advanced search techniques

### Generation and Evaluation
- **`07_llm_integration.ipynb`** - Connecting language models
- **`08_evaluation.ipynb`** - Measuring performance

### Real-World Application
- **`09_optimization.ipynb`** - Making it production-ready

## First Steps

Once you have everything set up, here's how to start exploring:

### Collect Some Data

```bash
# This will get you a small dataset to work with
python src/data/collect_data.py
```

This downloads a manageable amount of real data - Wikipedia articles and ArXiv abstracts - so you can see how the system works without overwhelming your machine.

### Start with the Fundamentals

Open `notebooks/01_understanding_rag.ipynb`. This notebook explains:
- What RAG actually does and why it's useful
- How embeddings turn text into mathematical representations
- Basic similarity search concepts

### Explore What You've Got

```python
# Try this in a notebook or script
import json
from pathlib import Path

# Look at your collected data
with open("data/raw/wikipedia_articles.json", "r") as f:
    wiki_data = json.load(f)

print(f"You now have {len(wiki_data)} Wikipedia articles")
print(f"Sample: {wiki_data[0]['title']}")
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

## Learning Strategy

### Work with Manageable Data
Start with small datasets. There's no point trying to process thousands of documents when you're still learning how the system works. Begin with 100 documents, understand what's happening, then scale up.

### Experiment and Observe
The best way to learn is by changing things and seeing what happens:
- Adjust chunk sizes and notice how it affects retrieval
- Try different embedding models and compare results
- Modify prompts and see how responses change

### Keep Track of Your Work
Take notes as you go. When something works well, make a note of why. When it doesn't work, figure out what went wrong. This builds intuition that's much more valuable than just following instructions.

### Question Everything
Always ask why something works the way it does. Why does one chunking strategy perform better than another? What happens if you change the similarity threshold? How does prompt engineering actually affect the quality of responses?

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

## Signs You're Making Progress

You'll know you're getting the hang of this when you can:
- Collect and clean data without constantly referring to documentation
- Explain why certain embedding models work better for specific types of text
- Retrieve documents that actually answer the questions you're asking
- Generate responses that make sense and cite the right sources
- Measure whether your system is actually working well

## Where to Go Next

After you've got the basics working:
- Add more diverse data sources
- Experiment with different models and see how they compare
- Focus on the parts that interest you most - maybe it's the retrieval, maybe it's the generation
- Try to break your system and then fix it
- Think about how you'd deploy something like this in the real world

## When Things Go Wrong

This is normal. RAG systems are complex, and you'll run into issues. When you do:
1. Look at the error messages carefully - they usually tell you what's wrong
2. Check the logs in `logs/docs_copilot.log`
3. Go back to the relevant notebook and make sure you understand each step
4. Try the troubleshooting steps above
5. Sometimes the best approach is to start over with a smaller, simpler example

The goal here isn't to build a perfect system on your first try. It's to understand how these systems work so you can build better ones in the future.

---

*The real value comes from understanding the underlying concepts, not just getting the code to run.*

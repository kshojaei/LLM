# Quick Start Guide

Get your RAG learning project up and running in 5 minutes!

## 🚀 One-Click Deploy (Easiest)

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/template/your-template-id)

Click the button above to deploy instantly to Railway. No setup required!

## 🐳 Docker (Recommended)

```bash
# Clone and run
git clone https://github.com/yourusername/LLM.git
cd LLM
docker-compose up --build

# Access your services:
# - Jupyter Lab: http://localhost:8888
# - API Server: http://localhost:8000
```

## 💻 Local Setup

```bash
# 1. Clone repository
git clone https://github.com/yourusername/LLM.git
cd LLM

# 2. Automated setup
python setup.py

# 3. Start learning
jupyter lab notebooks/
```

## 📚 Start Learning

1. **Open Jupyter Lab**: `http://localhost:8888`
2. **Start with**: `01_understanding_rag.ipynb`
3. **Follow the path**: Work through notebooks 1-9 in order
4. **Take your time**: Understanding is more important than speed

## 🎯 What You'll Learn

- **RAG Fundamentals** - What RAG is and why it matters
- **Data Processing** - Collecting and preparing data
- **Embeddings** - Converting text to vectors
- **Vector Search** - Finding relevant information
- **LLM Integration** - Connecting language models
- **Evaluation** - Measuring performance
- **Deployment** - Making it production-ready

## ⚡ Quick Test

```bash
# Test the API
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is machine learning?"}'
```

## 🆘 Need Help?

- **Documentation**: Check the `docs/` folder
- **Issues**: Open a GitHub issue
- **Discussions**: Use GitHub discussions
- **Troubleshooting**: See `GETTING_STARTED.md`

## 🎉 Success!

Once you see the API responding and can run the notebooks, you're ready to start learning RAG systems!

**Next Steps:**
1. Work through the learning path
2. Experiment with different models
3. Build your own RAG system
4. Share what you learn!

---

**Happy Learning!** 🚀

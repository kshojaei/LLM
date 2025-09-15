# Getting Started

Learn RAG systems through hands-on Jupyter notebooks and deploy working inference pipelines.

##  Quick Start

### Option 1: Docker (Recommended)
```bash
git clone https://github.com/yourusername/LLM.git
cd LLM
docker-compose up --build
```

### Option 2: Local Setup
```bash
git clone https://github.com/yourusername/LLM.git
cd LLM
python setup.py
jupyter lab notebooks/
```

##  Learning Path

Work through these notebooks in order:

1. **01_understanding_rag.ipynb** - What RAG is and why it works
2. **02_data_collection.ipynb** - Collect Wikipedia/ArXiv data
3. **03_embeddings_and_vector_store.ipynb** - Create embeddings
4. **04_text_preprocessing.ipynb** - Chunk text for search
5. **05_vector_search.ipynb** - Vector similarity search
6. **06_retrieval_systems.ipynb** - Hybrid retrieval
7. **07_llm_integration.ipynb** - Connect LLMs
8. **08_evaluation.ipynb** - Measure performance
9. **09_optimization.ipynb** - Deploy and optimize

##  Host Your System

### Railway (Easiest)
1. Go to [railway.app](https://railway.app)
2. Connect GitHub repo
3. Deploy automatically
4. Share your URL!

### Render
1. Go to [render.com](https://render.com)
2. Connect GitHub → New Web Service
3. Select Docker → Deploy

### Test Your API
```bash
curl -X POST "https://your-app.railway.app/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is machine learning?"}'
```

##  What You'll Build

- **Data Pipeline**: Collect and process documents
- **Vector Database**: Store embeddings for search
- **Retrieval System**: Find relevant documents
- **LLM Integration**: Generate answers
- **API Server**: REST API for queries
- **Deployed System**: Live RAG system others can use

##  Requirements

- Python 3.9+
- 8GB+ RAM (16GB+ for full models)
- 10GB+ disk space

##  Need Help?

- **Issues**: Open a GitHub issue
- **Docs**: Check the `docs/` folder
- **API Docs**: Visit `http://localhost:8000/docs`

---

**Start with the notebooks - everything else is just details!** 
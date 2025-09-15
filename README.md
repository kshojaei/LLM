# RAG Learning Project: Complete Guide to Retrieval-Augmented Generation

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)](https://jupyter.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![RAG](https://img.shields.io/badge/RAG-Tutorial-purple)](https://github.com/kshojaei/rag-docs-copilot)

A comprehensive, hands-on project for learning **Retrieval-Augmented Generation (RAG)** systems from scratch to production deployment. Perfect for beginners and intermediate developers who want to understand and build RAG applications.

## 🚀 What is RAG?

**Retrieval-Augmented Generation (RAG)** is a powerful AI technique that combines information retrieval with text generation. It allows language models to access external knowledge and provide more accurate, up-to-date responses.

### Key Benefits:
- ✅ **Accurate Information**: Access real-time, factual data
- ✅ **Reduced Hallucination**: Ground responses in retrieved documents  
- ✅ **Cost Effective**: Use smaller models with external knowledge
- ✅ **Domain Specific**: Customize for your specific use case

## 📚 What You'll Learn

This project provides a complete learning path for understanding and implementing RAG systems, from basic concepts to production-ready web applications.

### 🎯 Learning Objectives:
- **Understanding RAG**: Core concepts, architecture, and use cases
- **Data Collection**: Scraping Wikipedia, ArXiv, and other sources
- **Text Processing**: Chunking, preprocessing, and optimization
- **Embeddings**: Different models (OpenAI, Sentence Transformers, BGE)
- **Vector Databases**: ChromaDB, FAISS, Pinecone integration
- **Retrieval Systems**: Dense, sparse, and hybrid search
- **LLM Integration**: Prompt engineering and response generation
- **Evaluation**: Metrics, testing, and performance analysis
- **Production Deployment**: Web apps, APIs, and hosting
- **Advanced Topics**: Reranking, query expansion, optimization

## ✨ Key Features

- 🔥 **10 Interactive Jupyter Notebooks** - Step-by-step learning
- 🚀 **Production-Ready Code** - Real-world implementation
- 🗄️ **Multiple Vector Databases** - ChromaDB, FAISS, Pinecone
- 🤖 **LLM Integration** - OpenAI, local models, custom prompts
- 📊 **Comprehensive Evaluation** - Metrics and performance analysis
- 🌐 **Web Applications** - FastAPI, Streamlit, deployment ready
- 🐳 **Docker Support** - Easy deployment and scaling
- 📈 **Performance Optimization** - Caching, batching, monitoring

## 🎯 Perfect For

- **Machine Learning Engineers** - Learn RAG implementation
- **Data Scientists** - Understand retrieval systems
- **AI Researchers** - Explore advanced RAG techniques
- **Students** - Comprehensive learning resource
- **Developers** - Build production RAG applications
- **Anyone** - Interested in AI and natural language processing

## 🏆 Use Cases

- **Question Answering Systems** - Build intelligent Q&A bots
- **Document Search** - Create semantic search engines
- **Chatbots** - Develop context-aware assistants
- **Knowledge Management** - Organize and retrieve information
- **Research Tools** - Analyze and summarize documents
- **Customer Support** - Automated help systems

## 🛠️ Technology Stack

- **Python 3.9+** - Core programming language
- **Jupyter Notebooks** - Interactive learning environment
- **Sentence Transformers** - Embedding models
- **ChromaDB** - Vector database
- **FAISS** - Similarity search
- **Pinecone** - Cloud vector database
- **OpenAI API** - Language models
- **FastAPI** - Web framework
- **Streamlit** - Data app framework
- **Docker** - Containerization
- **scikit-learn** - Machine learning utilities
- **pandas** - Data manipulation
- **numpy** - Numerical computing

## 🔍 Keywords

`RAG` `Retrieval-Augmented Generation` `Machine Learning` `AI` `Natural Language Processing` `Vector Database` `Embeddings` `ChromaDB` `FAISS` `Pinecone` `OpenAI` `Jupyter Notebooks` `Python` `Tutorial` `Learning` `Semantic Search` `Question Answering` `Chatbot` `Document Search` `Knowledge Management` `LLM` `Language Model` `Text Processing` `Information Retrieval` `Production Deployment` `Web Application` `API` `Docker` `FastAPI` `Streamlit`

## Quick Start

### Option 1: Run Locally
```bash
# Install dependencies
pip install -r requirements.txt

# Start the web application
python3 web_app.py
```

### Option 2: Use Docker
```bash
# Build and run with Docker
docker build -t rag-app .
docker run -p 8000:8000 rag-app
```

### Option 3: One-Click Deploy
```bash
# Run the deployment script
./deploy_now.sh
```

## Learning Path

### 1. Understanding RAG (Notebook 1)
- Introduction to RAG concepts
- How retrieval and generation work together
- Use cases and applications

### 2. Data Collection (Notebook 2)
- Collecting data from Wikipedia and ArXiv
- Data preprocessing and cleaning
- Building knowledge bases

### 3. Embeddings and Vector Store (Notebook 3)
- Understanding embeddings
- Vector databases (ChromaDB, FAISS)
- Similarity search

### 10. Production Vector Database (Notebook 10)
- Production vector database management
- Multiple backend support (ChromaDB, FAISS, Pinecone)
- Performance optimization and monitoring
- Scalability and production deployment

### 4. Text Preprocessing (Notebook 4)
- Text chunking strategies
- Document preprocessing
- Metadata extraction

### 5. Vector Search (Notebook 5)
- Implementing vector search
- Similarity metrics
- Search optimization

### 6. Retrieval Systems (Notebook 6)
- Dense vs sparse retrieval
- Hybrid search approaches
- Query expansion

### 7. LLM Integration (Notebook 7)
- Integrating with language models
- Prompt engineering
- Response generation

### 8. Evaluation (Notebook 8)
- RAG evaluation metrics
- Performance measurement
- Quality assessment

### 9. Optimization (Notebook 9)
- Performance optimization
- Caching strategies
- Production considerations

## Production Deployment

### Web Applications
The project includes multiple production-ready web applications:

#### Main Web App (`web_app.py`)
- **Web Interface**: Clean, responsive UI for asking questions
- **API Endpoints**: RESTful API for programmatic access
- **Smart Retrieval**: Advanced similarity matching
- **Real-time Responses**: Fast query processing

#### Vector Database Web App (`web_app_vector.py`)
- **Production Vector DB**: Uses our comprehensive vector database system
- **Multiple Backends**: Support for ChromaDB, FAISS, and Pinecone
- **Advanced Features**: Metadata filtering, performance monitoring
- **Scalable Architecture**: Ready for production deployment

### Deployment Options

#### Railway (Easiest)
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and deploy
railway login
railway up
```

#### Render
1. Go to https://render.com
2. Connect your GitHub repository
3. Deploy with one click

#### Docker
```bash
# Build image
docker build -t rag-app .

# Run container
docker run -p 8000:8000 rag-app
```

#### AWS/Google Cloud
Use the provided Dockerfile with ECS, Cloud Run, or other container services.

## Project Structure

```
LLM/
├── notebooks/           # Learning notebooks (1-10)
├── src/                # Source code modules
│   ├── vector_db/      # Production vector database system
│   ├── data/           # Data collection and preprocessing
│   ├── models/         # Embedding and LLM models
│   ├── retrieval/      # Retrieval systems
│   ├── evaluation/     # Evaluation metrics
│   └── optimization/   # Performance optimization
├── data/               # Data storage
│   ├── raw/            # Raw data (Wikipedia, ArXiv)
│   ├── processed/      # Processed data (chunks, embeddings)
│   └── vector_db/      # Vector databases (ChromaDB, FAISS)
├── web_app.py          # Main web application
├── web_app_vector.py   # Vector database web application
├── advanced_app.py     # Advanced production system
├── requirements.txt    # Dependencies
├── Dockerfile          # Docker configuration
├── deploy_now.sh       # Deployment script
└── README.md           # This file
```

## Features

### Learning Components
- **10 Interactive Notebooks**: Step-by-step RAG learning
- **Real Data**: Wikipedia and ArXiv datasets
- **Multiple Models**: Various embedding and LLM options
- **Vector Database System**: Production-ready vector database management
- **Evaluation Tools**: Comprehensive metrics and analysis

### Production Components
- **Web Interface**: Modern, responsive UI
- **API**: RESTful endpoints for integration
- **Vector Database Management**: ChromaDB, FAISS, Pinecone support
- **Performance Monitoring**: Built-in metrics and analytics
- **Docker Support**: Easy containerization
- **Cloud Ready**: Deploy to any cloud platform

## Usage

### Learning Mode
1. Open notebooks in Jupyter Lab
2. Follow the step-by-step tutorials
3. Experiment with different configurations
4. Understand RAG concepts deeply

### Production Mode
1. **Basic Web App**: Run `python3 web_app.py`
2. **Vector Database App**: Run `python3 web_app_vector.py`
3. Open http://localhost:8000
4. Ask questions about AI, technology, science, etc.
5. Deploy to cloud for public access

## API Reference

### Endpoints
- `GET /` - Web interface
- `POST /api/query` - Query the RAG system
- `GET /api/health` - Health check

### Query Format
```json
{
  "question": "What is machine learning?",
  "top_k": 3
}
```

### Response Format
```json
{
  "question": "What is machine learning?",
  "answer": "Based on the information I found...",
  "sources": [...],
  "response_time": 0.123,
  "query_count": 1
}
```

## Vector Database System

The project includes a comprehensive vector database management system supporting multiple backends and production features.

### Supported Backends
- **ChromaDB**: Local development and small-scale production
- **FAISS**: High-performance search for large datasets
- **Pinecone**: Cloud-native, fully managed solution

### Key Features
- **Document Management**: Add, update, delete documents
- **Metadata Filtering**: Filter results by custom metadata
- **Performance Monitoring**: Track search times and memory usage
- **Backup & Recovery**: Database backup and restoration
- **Scalability**: Handle millions of documents efficiently

### Quick Start
```python
from src.vector_db.vector_manager import create_vector_database

# Create vector database
vdb = create_vector_database(
    backend="chromadb",
    collection_name="my_docs",
    documents=documents
)

# Search documents
results = vdb.search("What is machine learning?", top_k=5)

# Get database statistics
stats = vdb.get_stats()
```

### Documentation
For detailed information about the vector database system, see [VECTOR_DATABASE_GUIDE.md](VECTOR_DATABASE_GUIDE.md).

## Dependencies

- Python 3.8+
- FastAPI
- Sentence Transformers
- ChromaDB
- FAISS
- Scikit-learn
- NumPy, Pandas
- Docker (optional)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is for educational purposes. Feel free to use and modify for learning.

## Support

For questions or issues:
1. Check the notebooks for learning guidance
2. Review the code comments
3. Open an issue on GitHub

## Next Steps

After completing this project:
1. **Experiment**: Try different embedding models
2. **Scale**: Add more data sources
3. **Optimize**: Implement advanced retrieval techniques
4. **Deploy**: Build your own RAG applications
5. **Share**: Contribute back to the community

Happy learning!
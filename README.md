# RAG Learning Project

A comprehensive project for learning Retrieval-Augmented Generation (RAG) systems from scratch to production deployment.

## Overview

This project provides a complete learning path for understanding and implementing RAG systems, from basic concepts to production-ready web applications.

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

### Web Application
The project includes a production-ready web application (`smart_web_demo.py`) that combines all components:

- **Web Interface**: Clean, responsive UI for asking questions
- **API Endpoints**: RESTful API for programmatic access
- **Smart Retrieval**: Advanced similarity matching
- **Real-time Responses**: Fast query processing

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
├── notebooks/           # Learning notebooks (1-9)
├── src/                # Source code modules
├── data/               # Data storage
│   ├── raw/            # Raw data (Wikipedia, ArXiv)
│   ├── processed/      # Processed data (chunks, embeddings)
│   └── vector_db/      # Vector databases (ChromaDB, FAISS)
├── web_app.py          # Main web application
├── advanced_app.py     # Advanced production system
├── requirements.txt    # Dependencies
├── Dockerfile          # Docker configuration
├── deploy_now.sh       # Deployment script
└── README.md           # This file
```

## Features

### Learning Components
- **9 Interactive Notebooks**: Step-by-step RAG learning
- **Real Data**: Wikipedia and ArXiv datasets
- **Multiple Models**: Various embedding and LLM options
- **Evaluation Tools**: Comprehensive metrics and analysis

### Production Components
- **Web Interface**: Modern, responsive UI
- **API**: RESTful endpoints for integration
- **Docker Support**: Easy containerization
- **Cloud Ready**: Deploy to any cloud platform

## Usage

### Learning Mode
1. Open notebooks in Jupyter Lab
2. Follow the step-by-step tutorials
3. Experiment with different configurations
4. Understand RAG concepts deeply

### Production Mode
1. Run `python3 web_app.py`
2. Open http://localhost:8000
3. Ask questions about AI, technology, science, etc.
4. Deploy to cloud for public access

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

## Dependencies

- Python 3.8+
- FastAPI
- Sentence Transformers
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
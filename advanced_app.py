#!/usr/bin/env python3
"""
Production RAG Web Application
==============================

This is a complete production-ready web application that combines all the components
from the notebooks into a deployable web service.

Usage:
    python production_app.py

Then visit: http://localhost:8000
"""

import os
import json
import time
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

# Web framework
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# RAG components (from notebooks)
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# RAG SYSTEM (Combined from all notebooks)
# ============================================================================

class ProductionRAGSystem:
    """Production-ready RAG system combining all notebook components."""
    
    def __init__(self, embedding_model='all-MiniLM-L6-v2'):
        self.embedding_model = SentenceTransformer(embedding_model)
        self.documents = []
        self.metadata = []
        self.embeddings = None
        self.query_cache = {}
        self.performance_metrics = {
            'total_queries': 0,
            'avg_response_time': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
    def load_documents(self, data_dir: str = "data/processed"):
        """Load documents from processed data."""
        try:
            chunks_file = Path(data_dir) / "all_chunks.json"
            if chunks_file.exists():
                with open(chunks_file, 'r', encoding='utf-8') as f:
                    chunks = json.load(f)
                
                self.documents = [chunk['text'] for chunk in chunks]
                self.metadata = [{'title': chunk.get('title', 'Unknown'), 
                                'source': chunk.get('source', 'Unknown')} 
                               for chunk in chunks]
                
                logger.info(f"Loaded {len(self.documents)} documents")
                self._generate_embeddings()
                return True
            else:
                logger.warning("No processed data found, using sample data")
                self._load_sample_data()
                return True
        except Exception as e:
            logger.error(f"Error loading documents: {e}")
            self._load_sample_data()
            return False
    
    def _load_sample_data(self):
        """Load sample data if no processed data available."""
        self.documents = [
            "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data without being explicitly programmed.",
            "Deep learning uses neural networks with multiple layers to process data and make predictions. It has revolutionized fields like computer vision and natural language processing.",
            "Natural language processing (NLP) helps computers understand, interpret, and generate human language in a valuable way.",
            "Computer vision enables machines to interpret and understand visual information from the world using digital images and videos.",
            "Reinforcement learning is a type of machine learning where agents learn through interaction with an environment, receiving rewards or penalties."
        ]
        self.metadata = [
            {'title': 'Machine Learning', 'source': 'wikipedia'},
            {'title': 'Deep Learning', 'source': 'wikipedia'},
            {'title': 'NLP', 'source': 'wikipedia'},
            {'title': 'Computer Vision', 'source': 'wikipedia'},
            {'title': 'Reinforcement Learning', 'source': 'wikipedia'}
        ]
        self._generate_embeddings()
    
    def _generate_embeddings(self):
        """Generate embeddings for all documents."""
        logger.info("Generating embeddings...")
        self.embeddings = self.embedding_model.encode(self.documents, show_progress_bar=True)
        logger.info(f"Generated embeddings: {self.embeddings.shape}")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve relevant documents."""
        if self.embeddings is None:
            raise ValueError("No documents loaded")
        
        # Check cache first
        cache_key = f"{query}_{top_k}"
        if cache_key in self.query_cache:
            self.performance_metrics['cache_hits'] += 1
            return self.query_cache[cache_key]
        
        self.performance_metrics['cache_misses'] += 1
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                'id': f"doc_{idx}",
                'document': self.documents[idx],
                'similarity': float(similarities[idx]),
                'metadata': self.metadata[idx],
                'index': int(idx)
            })
        
        # Cache results
        self.query_cache[cache_key] = results
        return results
    
    def generate_answer(self, question: str, retrieved_docs: List[Dict]) -> str:
        """Generate answer using retrieved documents."""
        if not retrieved_docs:
            return "I don't have enough information to answer this question."
        
        # Create context from retrieved documents
        context_parts = []
        for i, doc in enumerate(retrieved_docs):
            context_parts.append(f"Source {i+1}: {doc['document']}")
        
        context = "\n\n".join(context_parts)
        
        # Simple answer generation (in production, you'd use a real LLM)
        answer = f"""Based on the retrieved information, here's what I found:

{context}

This answer is based on {len(retrieved_docs)} relevant sources. For more detailed information, please refer to the original documents."""
        
        return answer
    
    def ask(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """Complete RAG pipeline."""
        start_time = time.time()
        
        # Retrieve documents
        retrieved_docs = self.retrieve(question, top_k)
        
        # Generate answer
        answer = self.generate_answer(question, retrieved_docs)
        
        # Calculate response time
        response_time = time.time() - start_time
        
        # Update metrics
        self.performance_metrics['total_queries'] += 1
        self.performance_metrics['avg_response_time'] = (
            (self.performance_metrics['avg_response_time'] * (self.performance_metrics['total_queries'] - 1) + response_time) 
            / self.performance_metrics['total_queries']
        )
        
        return {
            'question': question,
            'answer': answer,
            'retrieved_docs': retrieved_docs,
            'response_time': response_time,
            'timestamp': datetime.now().isoformat()
        }

# ============================================================================
# WEB APPLICATION
# ============================================================================

# Initialize FastAPI app
app = FastAPI(
    title="RAG Production System",
    description="Production-ready RAG system combining all notebook components",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG system
rag_system = ProductionRAGSystem()

# Load documents on startup
@app.on_event("startup")
async def startup_event():
    """Initialize the RAG system on startup."""
    logger.info("Starting RAG Production System...")
    rag_system.load_documents()
    logger.info("RAG system ready!")

# ============================================================================
# API MODELS
# ============================================================================

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5

class QueryResponse(BaseModel):
    question: str
    answer: str
    retrieved_docs: List[Dict]
    response_time: float
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    documents_loaded: int
    performance_metrics: Dict

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main web interface."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>RAG Production System</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body { font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }
            .header { background: #2c3e50; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
            .query-form { background: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
            .query-input { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 4px; font-size: 16px; }
            .submit-btn { background: #3498db; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; font-size: 16px; }
            .submit-btn:hover { background: #2980b9; }
            .results { margin-top: 20px; }
            .answer { background: #e8f5e8; padding: 15px; border-radius: 8px; margin-bottom: 15px; }
            .sources { background: #f0f8ff; padding: 15px; border-radius: 8px; }
            .source-item { margin-bottom: 10px; padding: 10px; background: white; border-radius: 4px; }
            .metrics { background: #fff3cd; padding: 15px; border-radius: 8px; margin-top: 20px; }
            .loading { display: none; color: #666; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1> RAG Production System</h1>
            <p>Ask questions and get answers from our knowledge base!</p>
        </div>
        
        <div class="query-form">
            <form id="queryForm">
                <input type="text" id="question" class="query-input" placeholder="Ask a question about machine learning, AI, or any topic..." required>
                <br><br>
                <button type="submit" class="submit-btn">Ask Question</button>
                <div class="loading" id="loading">Processing your question...</div>
            </form>
        </div>
        
        <div id="results" class="results"></div>
        <div id="metrics" class="metrics"></div>
        
        <script>
            document.getElementById('queryForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                
                const question = document.getElementById('question').value;
                const loading = document.getElementById('loading');
                const results = document.getElementById('results');
                const metrics = document.getElementById('metrics');
                
                loading.style.display = 'block';
                results.innerHTML = '';
                metrics.innerHTML = '';
                
                try {
                    const response = await fetch('/api/query', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ question: question, top_k: 5 })
                    });
                    
                    const data = await response.json();
                    
                    // Display answer
                    results.innerHTML = `
                        <div class="answer">
                            <h3>Answer:</h3>
                            <p>${data.answer.replace(/\\n/g, '<br>')}</p>
                        </div>
                        <div class="sources">
                            <h3>Sources (${data.retrieved_docs.length}):</h3>
                            ${data.retrieved_docs.map((doc, i) => `
                                <div class="source-item">
                                    <strong>Source ${i+1}:</strong> ${doc.metadata.title} (${doc.metadata.source})
                                    <br><em>Similarity: ${(doc.similarity * 100).toFixed(1)}%</em>
                                    <br>${doc.document.substring(0, 200)}...
                                </div>
                            `).join('')}
                        </div>
                    `;
                    
                    // Update metrics
                    const metricsResponse = await fetch('/api/health');
                    const metricsData = await metricsResponse.json();
                    metrics.innerHTML = `
                        <h3>System Metrics:</h3>
                        <p><strong>Response Time:</strong> ${(data.response_time * 1000).toFixed(0)}ms</p>
                        <p><strong>Total Queries:</strong> ${metricsData.performance_metrics.total_queries}</p>
                        <p><strong>Average Response Time:</strong> ${(metricsData.performance_metrics.avg_response_time * 1000).toFixed(0)}ms</p>
                        <p><strong>Cache Hit Rate:</strong> ${((metricsData.performance_metrics.cache_hits / (metricsData.performance_metrics.cache_hits + metricsData.performance_metrics.cache_misses)) * 100).toFixed(1)}%</p>
                    `;
                    
                } catch (error) {
                    results.innerHTML = `<div style="color: red;">Error: ${error.message}</div>`;
                } finally {
                    loading.style.display = 'none';
                }
            });
        </script>
    </body>
    </html>
    """

@app.post("/api/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """Query the RAG system."""
    try:
        result = rag_system.ask(request.question, request.top_k)
        return QueryResponse(**result)
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        documents_loaded=len(rag_system.documents),
        performance_metrics=rag_system.performance_metrics
    )

@app.get("/api/metrics")
async def get_metrics():
    """Get detailed performance metrics."""
    return {
        "performance_metrics": rag_system.performance_metrics,
        "cache_size": len(rag_system.query_cache),
        "embeddings_shape": rag_system.embeddings.shape if rag_system.embeddings is not None else None
    }

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print(" Starting RAG Production System...")
    print(" Loading documents and initializing system...")
    print(" Web interface will be available at: http://localhost:8000")
    print(" API documentation at: http://localhost:8000/docs")
    print(" Health check at: http://localhost:8000/api/health")
    print("\n" + "="*60)
    
    uvicorn.run(
        "production_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

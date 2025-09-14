"""
FastAPI Server for RAG System
This module provides a REST API for the RAG system.

Author: Kamran Shojaei - Physicist with background in AI/ML
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from contextlib import asynccontextmanager

# FastAPI imports
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

# Import our modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import API_CONFIG, get_config
from src.retrieval.retrieval_system import RetrievalSystem, RetrievalConfig
from src.models.llm_models import RAGGenerator, load_llm_model, PromptTemplate
from src.evaluation.evaluation_metrics import OnlineEvaluator, RAGEvaluator
from src.data.preprocess_data import TextPreprocessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for the RAG system
rag_system = None
llm_generator = None
online_evaluator = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    global rag_system, llm_generator, online_evaluator
    
    # Startup
    logger.info("Starting RAG API server...")
    
    try:
        # Initialize retrieval system
        retrieval_config = RetrievalConfig(
            top_k=10,
            rerank_top_k=5,
            use_reranking=True
        )
        rag_system = RetrievalSystem(retrieval_config)
        
        # Load documents if available
        chunks_file = Path("data/processed/chunks_with_embeddings.json")
        if chunks_file.exists():
            with open(chunks_file, 'r', encoding='utf-8') as f:
                documents = json.load(f)
            rag_system.add_documents(documents)
            logger.info(f"Loaded {len(documents)} documents")
        
        # Initialize LLM generator
        try:
            llm_model = load_llm_model("llama_3_8b", load_in_8bit=True)
            prompt_template = PromptTemplate.get_rag_template("llama")
            llm_generator = RAGGenerator(llm_model, prompt_template)
            logger.info("LLM generator initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize LLM: {e}")
            llm_generator = None
        
        # Initialize online evaluator
        online_evaluator = OnlineEvaluator()
        
        logger.info("RAG API server started successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down RAG API server...")

# Create FastAPI app
app = FastAPI(
    title="Docs Copilot RAG API",
    description="Retrieval-Augmented Generation API for document question answering",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=API_CONFIG["cors_origins"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class QueryRequest(BaseModel):
    query: str = Field(..., description="The question to answer")
    method: str = Field("hybrid", description="Retrieval method: dense, sparse, or hybrid")
    top_k: int = Field(5, description="Number of documents to retrieve")
    use_reranking: bool = Field(True, description="Whether to use reranking")
    max_context_length: int = Field(4000, description="Maximum context length for generation")

class QueryResponse(BaseModel):
    query: str
    response: str
    retrieved_documents: List[Dict[str, Any]]
    generation_time: float
    retrieval_time: float
    total_time: float
    method: str
    num_retrieved_docs: int

class DocumentRequest(BaseModel):
    documents: List[Dict[str, Any]] = Field(..., description="Documents to add to the knowledge base")

class DocumentResponse(BaseModel):
    message: str
    num_documents: int
    total_documents: int

class FeedbackRequest(BaseModel):
    query: str
    response: str
    rating: int = Field(..., ge=1, le=5, description="Rating from 1 to 5")
    feedback_text: Optional[str] = Field(None, description="Optional feedback text")

class HealthResponse(BaseModel):
    status: str
    timestamp: float
    num_documents: int
    llm_available: bool

# Dependency to get the RAG system
def get_rag_system() -> RetrievalSystem:
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    return rag_system

def get_llm_generator() -> Optional[RAGGenerator]:
    if llm_generator is None:
        raise HTTPException(status_code=503, detail="LLM generator not available")
    return llm_generator

def get_online_evaluator() -> OnlineEvaluator:
    if online_evaluator is None:
        raise HTTPException(status_code=503, detail="Online evaluator not initialized")
    return online_evaluator

# API Endpoints

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Docs Copilot RAG API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    num_docs = len(rag_system.documents) if rag_system else 0
    llm_available = llm_generator is not None
    
    return HealthResponse(
        status="healthy" if rag_system else "unhealthy",
        timestamp=time.time(),
        num_documents=num_docs,
        llm_available=llm_available
    )

@app.post("/query", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest,
    background_tasks: BackgroundTasks,
    rag_sys: RetrievalSystem = Depends(get_rag_system),
    llm_gen: RAGGenerator = Depends(get_llm_generator),
    evaluator: OnlineEvaluator = Depends(get_online_evaluator)
):
    """
    Query the RAG system for answers.
    
    This endpoint processes a query, retrieves relevant documents,
    and generates a response using the LLM.
    """
    start_time = time.time()
    
    try:
        # Retrieve documents
        retrieval_start = time.time()
        retrieved_docs = rag_sys.retrieve(
            query=request.query,
            method=request.method,
            top_k=request.top_k
        )
        retrieval_time = time.time() - retrieval_start
        
        # Generate response
        generation_start = time.time()
        if llm_gen:
            response_data = llm_gen.generate_response(
                query=request.query,
                retrieved_docs=retrieved_docs,
                max_context_length=request.max_context_length
            )
            response_text = response_data['response']
            generation_time = response_data['generation_time']
        else:
            # Fallback: return retrieved documents
            response_text = "LLM not available. Here are the retrieved documents:\n\n"
            for i, doc in enumerate(retrieved_docs[:3]):
                response_text += f"{i+1}. {doc.get('title', 'Unknown')}\n{doc.get('text', '')[:200]}...\n\n"
            generation_time = 0.0
        
        total_time = time.time() - start_time
        
        # Log for monitoring
        background_tasks.add_task(
            evaluator.log_query,
            request.query,
            response_text,
            total_time,
            retrieved_docs
        )
        
        return QueryResponse(
            query=request.query,
            response=response_text,
            retrieved_documents=retrieved_docs,
            generation_time=generation_time,
            retrieval_time=retrieval_time,
            total_time=total_time,
            method=request.method,
            num_retrieved_docs=len(retrieved_docs)
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/documents", response_model=DocumentResponse)
async def add_documents(
    request: DocumentRequest,
    rag_sys: RetrievalSystem = Depends(get_rag_system)
):
    """
    Add documents to the knowledge base.
    
    This endpoint allows adding new documents to the RAG system's
    knowledge base for retrieval.
    """
    try:
        # Add documents to the system
        rag_sys.add_documents(request.documents)
        
        total_docs = len(rag_sys.documents)
        
        return DocumentResponse(
            message=f"Successfully added {len(request.documents)} documents",
            num_documents=len(request.documents),
            total_documents=total_docs
        )
        
    except Exception as e:
        logger.error(f"Error adding documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
async def submit_feedback(
    request: FeedbackRequest,
    background_tasks: BackgroundTasks,
    evaluator: OnlineEvaluator = Depends(get_online_evaluator)
):
    """
    Submit user feedback for a query-response pair.
    
    This endpoint allows users to provide feedback on the quality
    of responses for continuous improvement.
    """
    try:
        background_tasks.add_task(
            evaluator.log_feedback,
            request.query,
            request.response,
            request.rating,
            request.feedback_text
        )
        
        return {"message": "Feedback submitted successfully"}
        
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics(
    time_window: int = 3600,
    evaluator: OnlineEvaluator = Depends(get_online_evaluator)
):
    """
    Get performance metrics for the RAG system.
    
    Args:
        time_window: Time window in seconds for metrics calculation
    """
    try:
        metrics = evaluator.get_performance_metrics(time_window)
        return metrics
        
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents")
async def list_documents(
    limit: int = 100,
    offset: int = 0,
    rag_sys: RetrievalSystem = Depends(get_rag_system)
):
    """
    List documents in the knowledge base.
    
    Args:
        limit: Maximum number of documents to return
        offset: Number of documents to skip
    """
    try:
        documents = rag_sys.documents[offset:offset + limit]
        
        return {
            "documents": documents,
            "total": len(rag_sys.documents),
            "limit": limit,
            "offset": offset
        }
        
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/compare")
async def compare_retrieval_methods(
    request: QueryRequest,
    rag_sys: RetrievalSystem = Depends(get_rag_system)
):
    """
    Compare different retrieval methods on the same query.
    
    This endpoint is useful for evaluating and comparing the performance
    of different retrieval strategies.
    """
    try:
        comparison = rag_sys.compare_methods(
            query=request.query,
            top_k=request.top_k
        )
        
        return {
            "query": request.query,
            "comparison": comparison
        }
        
    except Exception as e:
        logger.error(f"Error comparing methods: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stream/{query}")
async def stream_response(
    query: str,
    method: str = "hybrid",
    top_k: int = 5,
    rag_sys: RetrievalSystem = Depends(get_rag_system),
    llm_gen: RAGGenerator = Depends(get_llm_generator)
):
    """
    Stream response generation for real-time interaction.
    
    This endpoint provides streaming responses for better user experience
    when generating long responses.
    """
    try:
        # Retrieve documents
        retrieved_docs = rag_sys.retrieve(
            query=query,
            method=method,
            top_k=top_k
        )
        
        if not llm_gen:
            raise HTTPException(status_code=503, detail="LLM generator not available")
        
        # Create streaming response
        def generate():
            # Send retrieved documents first
            yield f"data: {json.dumps({'type': 'documents', 'data': retrieved_docs})}\n\n"
            
            # Generate streaming response
            prompt = llm_gen.prompt_template.format(
                query=query,
                context="\n".join([doc.get('text', '') for doc in retrieved_docs[:3]])
            )
            
            for chunk in llm_gen.llm_model.generate_stream(prompt):
                yield f"data: {json.dumps({'type': 'chunk', 'data': chunk})}\n\n"
            
            yield f"data: {json.dumps({'type': 'done', 'data': ''})}\n\n"
        
        return StreamingResponse(
            generate(),
            media_type="text/plain",
            headers={"Cache-Control": "no-cache"}
        )
        
    except Exception as e:
        logger.error(f"Error streaming response: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"message": "Endpoint not found", "detail": str(exc)}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"message": "Internal server error", "detail": str(exc)}
    )

def main():
    """Main function to run the server."""
    config = get_config()
    
    uvicorn.run(
        "src.api.server:app",
        host=config["api"]["host"],
        port=config["api"]["port"],
        reload=config["api"]["debug"],
        log_level="info"
    )

if __name__ == "__main__":
    main()

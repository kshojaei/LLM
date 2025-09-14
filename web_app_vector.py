#!/usr/bin/env python3
"""
Production RAG Web App with Vector Database
==========================================

This version uses our production vector database system for better performance and scalability.
"""

import json
import time
import urllib.parse
import sys
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import List, Dict, Any

# Add src to path
sys.path.append('src')

# Import our vector database manager
from vector_db.vector_manager import VectorDatabaseManager, create_vector_database

# ============================================================================
# PRODUCTION RAG SYSTEM
# ============================================================================

class ProductionRAG:
    """Production RAG system using vector database."""
    
    def __init__(self):
        self.vector_db = None
        self.query_count = 0
        
    def load_sample_data(self):
        """Load sample data into vector database."""
        # Sample documents
        documents = [
            {
                "id": "doc_1",
                "text": "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data without being explicitly programmed. It includes supervised learning, unsupervised learning, and reinforcement learning approaches.",
                "metadata": {"category": "AI", "source": "wikipedia", "topic": "machine_learning"}
            },
            {
                "id": "doc_2", 
                "text": "Deep learning uses neural networks with multiple layers to process data and make predictions. It has revolutionized fields like computer vision and natural language processing by automatically learning complex patterns from large datasets.",
                "metadata": {"category": "AI", "source": "wikipedia", "topic": "deep_learning"}
            },
            {
                "id": "doc_3",
                "text": "Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals. AI systems can perform tasks that typically require human intelligence like reasoning, learning, and problem-solving.",
                "metadata": {"category": "AI", "source": "wikipedia", "topic": "artificial_intelligence"}
            },
            {
                "id": "doc_4",
                "text": "Natural language processing (NLP) helps computers understand, interpret, and generate human language in a valuable way. It powers chatbots, translation services, text analysis tools, and voice assistants.",
                "metadata": {"category": "AI", "source": "wikipedia", "topic": "nlp"}
            },
            {
                "id": "doc_5",
                "text": "Computer vision enables machines to interpret and understand visual information from the world using digital images and videos. It's used in autonomous vehicles, medical imaging, facial recognition, and quality control in manufacturing.",
                "metadata": {"category": "AI", "source": "wikipedia", "topic": "computer_vision"}
            },
            {
                "id": "doc_6",
                "text": "Reinforcement learning is a type of machine learning where agents learn through interaction with an environment, receiving rewards or penalties. It's used in game playing, robotics, and autonomous systems.",
                "metadata": {"category": "AI", "source": "wikipedia", "topic": "reinforcement_learning"}
            },
            {
                "id": "doc_7",
                "text": "Neural networks are computing systems inspired by biological neural networks. They are composed of interconnected nodes that process information and can learn complex patterns from data through training.",
                "metadata": {"category": "AI", "source": "wikipedia", "topic": "neural_networks"}
            },
            {
                "id": "doc_8",
                "text": "Data science is an interdisciplinary field that uses scientific methods, processes, algorithms and systems to extract knowledge from data. It combines statistics, programming, and domain expertise to solve real-world problems.",
                "metadata": {"category": "Tech", "source": "wikipedia", "topic": "data_science"}
            },
            {
                "id": "doc_9",
                "text": "Python is a popular programming language for data science and machine learning due to its simplicity and extensive libraries like NumPy, Pandas, TensorFlow, and PyTorch. It's easy to learn and has a large community.",
                "metadata": {"category": "Tech", "source": "wikipedia", "topic": "python"}
            },
            {
                "id": "doc_10",
                "text": "Cloud computing is the delivery of computing services including servers, storage, databases, networking, software, analytics, and intelligence over the Internet. It provides scalable and cost-effective solutions for businesses.",
                "metadata": {"category": "Tech", "source": "wikipedia", "topic": "cloud_computing"}
            },
            {
                "id": "doc_11",
                "text": "Healthcare applications of AI include medical image analysis, drug discovery, personalized treatment plans, and early disease detection using machine learning algorithms. This helps doctors make better diagnoses and save lives.",
                "metadata": {"category": "Health", "source": "wikipedia", "topic": "ai_healthcare"}
            },
            {
                "id": "doc_12",
                "text": "Climate change refers to long-term shifts in global temperatures and weather patterns. It's primarily caused by human activities like burning fossil fuels, deforestation, and industrial processes that increase greenhouse gas concentrations.",
                "metadata": {"category": "Environment", "source": "wikipedia", "topic": "climate_change"}
            },
            {
                "id": "doc_13",
                "text": "Renewable energy comes from natural sources that are constantly replenished like solar, wind, hydroelectric, and geothermal power. It's cleaner than fossil fuels and helps reduce carbon emissions and combat climate change.",
                "metadata": {"category": "Environment", "source": "wikipedia", "topic": "renewable_energy"}
            },
            {
                "id": "doc_14",
                "text": "Social media platforms like Facebook, Twitter, and Instagram connect people worldwide and enable sharing of information, ideas, and experiences. They've transformed communication but also raise concerns about privacy and misinformation.",
                "metadata": {"category": "Society", "source": "wikipedia", "topic": "social_media"}
            },
            {
                "id": "doc_15",
                "text": "Blockchain is a distributed ledger technology that maintains a continuously growing list of records secured using cryptography. It's the foundation of cryptocurrencies like Bitcoin and enables secure, transparent transactions without central authorities.",
                "metadata": {"category": "Tech", "source": "wikipedia", "topic": "blockchain"}
            }
        ]
        
        # Create vector database
        print("Creating vector database...")
        self.vector_db = create_vector_database(
            backend="chromadb",
            collection_name="web_demo_docs",
            documents=documents
        )
        
        print(f"Loaded {len(documents)} documents into vector database")
    
    def query(self, question: str, top_k: int = 3) -> Dict[str, Any]:
        """Query the RAG system."""
        if not self.vector_db:
            return {"error": "Vector database not initialized"}
        
        self.query_count += 1
        
        # Search for relevant documents
        results = self.vector_db.search(question, top_k=top_k)
        
        if not results:
            return {
                "question": question,
                "answer": "I couldn't find relevant information to answer your question.",
                "sources": [],
                "query_count": self.query_count
            }
        
        # Generate response based on retrieved documents
        context = " ".join([doc['document'] for doc in results[:2]])
        
        # Simple response generation (in production, use an LLM)
        answer = f"Based on the available information: {context[:300]}..."
        
        # Format sources
        sources = []
        for i, result in enumerate(results):
            sources.append({
                "rank": i + 1,
                "content": result['document'][:150] + "...",
                "score": round(result['score'], 3),
                "metadata": result['metadata']
            })
        
        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "query_count": self.query_count,
            "total_documents": len(results)
        }

# ============================================================================
# WEB SERVER
# ============================================================================

class RAGHandler(BaseHTTPRequestHandler):
    """HTTP handler for the RAG web app."""
    
    def __init__(self, *args, rag_system=None, **kwargs):
        self.rag_system = rag_system
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        """Handle GET requests."""
        if self.path == '/':
            self.serve_homepage()
        elif self.path == '/api/health':
            self.serve_health()
        else:
            self.send_error(404)
    
    def do_POST(self):
        """Handle POST requests."""
        if self.path == '/api/query':
            self.handle_query()
        else:
            self.send_error(404)
    
    def serve_homepage(self):
        """Serve the main HTML page."""
        html = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Production RAG System</title>
            <style>
                body {
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 20px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                }
                .container {
                    background: white;
                    border-radius: 15px;
                    padding: 30px;
                    box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                }
                h1 {
                    color: #333;
                    text-align: center;
                    margin-bottom: 30px;
                }
                .input-group {
                    margin-bottom: 20px;
                }
                input[type="text"] {
                    width: 100%;
                    padding: 15px;
                    border: 2px solid #e1e5e9;
                    border-radius: 10px;
                    font-size: 16px;
                    box-sizing: border-box;
                }
                button {
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    border: none;
                    padding: 15px 30px;
                    border-radius: 10px;
                    font-size: 16px;
                    cursor: pointer;
                    width: 100%;
                    margin-top: 10px;
                }
                button:hover {
                    transform: translateY(-2px);
                    box-shadow: 0 10px 20px rgba(0,0,0,0.2);
                }
                .response {
                    margin-top: 30px;
                    padding: 20px;
                    background: #f8f9fa;
                    border-radius: 10px;
                    border-left: 4px solid #667eea;
                }
                .sources {
                    margin-top: 20px;
                }
                .source {
                    background: white;
                    padding: 15px;
                    margin: 10px 0;
                    border-radius: 8px;
                    border-left: 3px solid #28a745;
                }
                .stats {
                    text-align: center;
                    color: #666;
                    margin-top: 20px;
                    font-size: 14px;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Production RAG System</h1>
                <p style="text-align: center; color: #666; margin-bottom: 30px;">
                    Ask questions about AI, technology, environment, and more!
                </p>
                
                <div class="input-group">
                    <input type="text" id="question" placeholder="Ask a question..." onkeypress="handleKeyPress(event)">
                    <button onclick="askQuestion()">Ask Question</button>
                </div>
                
                <div id="response" class="response" style="display: none;">
                    <h3>Answer:</h3>
                    <div id="answer"></div>
                    <div id="sources" class="sources"></div>
                </div>
                
                <div class="stats">
                    <div id="stats">Ready to answer your questions!</div>
                </div>
            </div>
            
            <script>
                function handleKeyPress(event) {
                    if (event.key === 'Enter') {
                        askQuestion();
                    }
                }
                
                async function askQuestion() {
                    const question = document.getElementById('question').value.trim();
                    if (!question) return;
                    
                    const responseDiv = document.getElementById('response');
                    const answerDiv = document.getElementById('answer');
                    const sourcesDiv = document.getElementById('sources');
                    const statsDiv = document.getElementById('stats');
                    
                    // Show loading
                    answerDiv.innerHTML = 'Thinking...';
                    responseDiv.style.display = 'block';
                    
                    try {
                        const response = await fetch('/api/query', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                            },
                            body: JSON.stringify({ question: question })
                        });
                        
                        const data = await response.json();
                        
                        if (data.error) {
                            answerDiv.innerHTML = `<p style="color: red;">Error: ${data.error}</p>`;
                        } else {
                            answerDiv.innerHTML = `<p>${data.answer}</p>`;
                            
                            // Show sources
                            sourcesDiv.innerHTML = '<h4>Sources:</h4>';
                            data.sources.forEach((source, index) => {
                                sourcesDiv.innerHTML += `
                                    <div class="source">
                                        <strong>Source ${source.rank}</strong> (Score: ${source.score})
                                        <p>${source.content}</p>
                                        <small>Category: ${source.metadata.category} | Topic: ${source.metadata.topic}</small>
                                    </div>
                                `;
                            });
                            
                            statsDiv.innerHTML = `Query #${data.query_count} | Found ${data.total_documents} relevant documents`;
                        }
                    } catch (error) {
                        answerDiv.innerHTML = `<p style="color: red;">Error: ${error.message}</p>`;
                    }
                }
            </script>
        </body>
        </html>
        """
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html.encode())
    
    def serve_health(self):
        """Serve health check endpoint."""
        health_data = {
            "status": "healthy",
            "timestamp": time.time(),
            "query_count": self.rag_system.query_count if self.rag_system else 0,
            "vector_db_ready": self.rag_system.vector_db is not None if self.rag_system else False
        }
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(health_data).encode())
    
    def handle_query(self):
        """Handle query requests."""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            question = data.get('question', '')
            if not question:
                self.send_error(400, "No question provided")
                return
            
            # Query the RAG system
            result = self.rag_system.query(question)
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())
            
        except Exception as e:
            self.send_error(500, f"Internal server error: {str(e)}")

def create_handler(rag_system):
    """Create a handler with the RAG system."""
    def handler(*args, **kwargs):
        return RAGHandler(*args, rag_system=rag_system, **kwargs)
    return handler

def main():
    """Main function to start the web server."""
    print("Production RAG Web App Starting...")
    print("=" * 50)
    
    # Initialize RAG system
    rag_system = ProductionRAG()
    rag_system.load_sample_data()
    
    # Create server
    port = 8000
    handler = create_handler(rag_system)
    server = HTTPServer(('localhost', port), handler)
    
    print(f"Web Interface: http://localhost:{port}")
    print(f"Health Check: http://localhost:{port}/api/health")
    print(f"Documents loaded: {rag_system.vector_db.get_stats()['total_documents'] if rag_system.vector_db else 0}")
    print("Now try asking about:")
    print("   - AI & Technology: 'What is artificial intelligence?'")
    print("   - Science: 'How does climate change work?'")
    print("   - Business: 'Where is AI used in healthcare?'")
    print("   - Environment: 'What is renewable energy?'")
    print("   - Society: 'How does social media work?'")
    print("Press Ctrl+C to stop the server")
    print("=" * 50)
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.shutdown()

if __name__ == "__main__":
    main()

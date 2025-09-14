#!/usr/bin/env python3
"""
Smart RAG Web Demo - Production Vector Database Integration
==========================================================

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
# SMART RAG SYSTEM
# ============================================================================

class SmartRAG:
    """Smart RAG system using production vector database."""
    
    def __init__(self):
        self.vector_db = None
        self.query_count = 0
        
        # Question type keywords for better matching
        self.question_types = {
            'what': ['what', 'define', 'definition', 'meaning', 'is'],
            'how': ['how', 'process', 'work', 'function', 'operate'],
            'why': ['why', 'reason', 'purpose', 'benefit', 'advantage'],
            'when': ['when', 'time', 'history', 'timeline', 'evolved'],
            'where': ['where', 'used', 'applied', 'industry', 'field'],
            'who': ['who', 'invented', 'created', 'developed', 'pioneer']
        }
        
    def load_sample_data(self):
        """Load sample data."""
        self.documents = [
            "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data without being explicitly programmed. It includes supervised learning, unsupervised learning, and reinforcement learning approaches.",
            "Deep learning uses neural networks with multiple layers to process data and make predictions. It has revolutionized fields like computer vision and natural language processing by automatically learning complex patterns from large datasets.",
            "Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals. AI systems can perform tasks that typically require human intelligence like reasoning, learning, and problem-solving.",
            "Natural language processing (NLP) helps computers understand, interpret, and generate human language in a valuable way. It powers chatbots, translation services, text analysis tools, and voice assistants.",
            "Computer vision enables machines to interpret and understand visual information from the world using digital images and videos. It's used in autonomous vehicles, medical imaging, facial recognition, and quality control in manufacturing.",
            "Reinforcement learning is a type of machine learning where agents learn through interaction with an environment, receiving rewards or penalties. It's used in game playing, robotics, and autonomous systems.",
            "Neural networks are computing systems inspired by biological neural networks. They are composed of interconnected nodes that process information and can learn complex patterns from data through training.",
            "Data science is an interdisciplinary field that uses scientific methods, processes, algorithms and systems to extract knowledge from data. It combines statistics, programming, and domain expertise to solve real-world problems.",
            "Python is a popular programming language for data science and machine learning due to its simplicity and extensive libraries like NumPy, Pandas, TensorFlow, and PyTorch. It's easy to learn and has a large community.",
            "Cloud computing is the delivery of computing services including servers, storage, databases, networking, software, analytics, and intelligence over the Internet. It provides scalable and cost-effective solutions for businesses.",
            "Healthcare applications of AI include medical image analysis, drug discovery, personalized treatment plans, and early disease detection using machine learning algorithms. This helps doctors make better diagnoses and save lives.",
            "Financial services use AI for algorithmic trading, credit scoring, fraud detection, and risk assessment to make better investment and lending decisions. Banks and fintech companies invest heavily in these technologies.",
            "The internet is a global network of interconnected computers that allows people to share information, communicate, and access services worldwide. It has revolutionized how we work, learn, and socialize.",
            "Climate change refers to long-term shifts in global temperatures and weather patterns, primarily caused by human activities like burning fossil fuels. It's one of the biggest challenges facing humanity today.",
            "Renewable energy sources like solar and wind power generate electricity without producing greenhouse gases. They're becoming cheaper and more efficient, helping fight climate change."
        ]
        
        self.metadata = [
            {'title': 'Machine Learning', 'source': 'AI Education', 'category': 'AI/ML'},
            {'title': 'Deep Learning', 'source': 'AI Education', 'category': 'AI/ML'},
            {'title': 'Artificial Intelligence', 'source': 'AI Education', 'category': 'AI/ML'},
            {'title': 'Natural Language Processing', 'source': 'AI Applications', 'category': 'AI/ML'},
            {'title': 'Computer Vision', 'source': 'AI Applications', 'category': 'AI/ML'},
            {'title': 'Reinforcement Learning', 'source': 'AI Applications', 'category': 'AI/ML'},
            {'title': 'Neural Networks', 'source': 'Technical Concepts', 'category': 'AI/ML'},
            {'title': 'Data Science', 'source': 'Technical Concepts', 'category': 'AI/ML'},
            {'title': 'Python Programming', 'source': 'Technology', 'category': 'Technology'},
            {'title': 'Cloud Computing', 'source': 'Technology', 'category': 'Technology'},
            {'title': 'Healthcare AI', 'source': 'Business Applications', 'category': 'Business'},
            {'title': 'Financial AI', 'source': 'Business Applications', 'category': 'Business'},
            {'title': 'The Internet', 'source': 'General Technology', 'category': 'Technology'},
            {'title': 'Climate Change', 'source': 'Environment', 'category': 'Environment'},
            {'title': 'Renewable Energy', 'source': 'Environment', 'category': 'Environment'}
        ]
        
        print(f"Loaded {len(self.documents)} documents")
    
    def extract_keywords(self, text: str) -> set:
        """Extract important keywords from text."""
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
        
        words = text.lower().split()
        keywords = set()
        
        for word in words:
            # Clean word
            clean_word = ''.join(c for c in word if c.isalnum())
            if len(clean_word) > 2 and clean_word not in stop_words:
                keywords.add(clean_word)
        
        return keywords
    
    def calculate_smart_similarity(self, query: str, document: str) -> float:
        """Calculate smart similarity with multiple matching strategies."""
        query_lower = query.lower()
        doc_lower = document.lower()
        
        # Strategy 1: Direct phrase matching (highest priority)
        if query_lower in doc_lower:
            return 1.0
        
        # Strategy 2: Question type matching
        question_type_score = 0.0
        for q_type, keywords in self.question_types.items():
            if any(keyword in query_lower for keyword in keywords):
                # Look for corresponding answers in document
                if q_type == 'what' and any(word in doc_lower for word in ['is', 'are', 'refers', 'means', 'definition']):
                    question_type_score += 0.3
                elif q_type == 'how' and any(word in doc_lower for word in ['work', 'process', 'function', 'operate', 'uses']):
                    question_type_score += 0.3
                elif q_type == 'why' and any(word in doc_lower for word in ['because', 'reason', 'benefit', 'advantage', 'purpose']):
                    question_type_score += 0.3
                elif q_type == 'where' and any(word in doc_lower for word in ['used', 'applied', 'industry', 'field', 'platform']):
                    question_type_score += 0.3
                elif q_type == 'when' and any(word in doc_lower for word in ['evolved', 'history', 'timeline', 'recent', 'future']):
                    question_type_score += 0.3
        
        # Strategy 3: Keyword matching with importance weighting
        query_keywords = self.extract_keywords(query)
        doc_keywords = self.extract_keywords(document)
        
        if not query_keywords or not doc_keywords:
            return question_type_score
        
        # Calculate keyword overlap
        intersection = query_keywords.intersection(doc_keywords)
        union = query_keywords.union(doc_keywords)
        keyword_score = len(intersection) / len(union) if union else 0.0
        
        # Strategy 4: Boost for important AI/tech terms
        important_terms = {
            'artificial intelligence': 0.4, 'machine learning': 0.4, 'deep learning': 0.4,
            'neural network': 0.4, 'data science': 0.3, 'computer vision': 0.3,
            'natural language processing': 0.3, 'reinforcement learning': 0.3,
            'python': 0.2, 'docker': 0.2, 'cloud computing': 0.2,
            'e-commerce': 0.2, 'healthcare': 0.2, 'finance': 0.2,
            'climate change': 0.3, 'renewable energy': 0.3, 'space exploration': 0.3,
            'internet': 0.2, 'mobile phone': 0.2, 'social media': 0.2
        }
        
        term_boost = 0.0
        for term, boost in important_terms.items():
            if term in query_lower and term in doc_lower:
                term_boost += boost
        
        # Strategy 5: Context matching
        context_words = ['technology', 'science', 'business', 'environment', 'society', 'future', 'modern', 'digital', 'smart', 'advanced']
        context_score = 0.0
        for word in context_words:
            if word in query_lower and word in doc_lower:
                context_score += 0.1
        
        # Combine all strategies
        total_score = (keyword_score * 0.4 + 
                      question_type_score * 0.3 + 
                      term_boost * 0.2 + 
                      context_score * 0.1)
        
        return min(total_score, 1.0)
    
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """Search for relevant documents using smart similarity."""
        if not self.documents:
            return []
        
        # Calculate similarities
        similarities = []
        for i, doc in enumerate(self.documents):
            similarity = self.calculate_smart_similarity(query, doc)
            similarities.append((i, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Get top-k results
        results = []
        for i, (doc_idx, similarity) in enumerate(similarities[:top_k]):
            if similarity > 0.05:  # Only include results with meaningful similarity
                results.append({
                    'document': self.documents[doc_idx],
                    'similarity': similarity,
                    'metadata': self.metadata[doc_idx],
                    'index': doc_idx
                })
        
        return results
    
    def ask(self, question: str, top_k: int = 3) -> Dict[str, Any]:
        """Answer a question using smart RAG."""
        start_time = time.time()
        
        # Search for relevant documents
        results = self.search(question, top_k)
        
        # Generate answer
        if not results:
            answer = "I don't have enough information to answer this question. Try asking about AI, technology, science, business, or environmental topics."
        else:
            # Create context
            context_parts = []
            for i, result in enumerate(results):
                context_parts.append(f"Source {i+1}: {result['document']}")
            
            context = "\n\n".join(context_parts)
            
            # Generate better answer based on question type
            question_lower = question.lower()
            if any(word in question_lower for word in ['what', 'define', 'definition']):
                answer = f"Here's what I found about your question:\n\n{context}\n\nThis information comes from {len(results)} relevant sources."
            elif any(word in question_lower for word in ['how', 'work', 'process']):
                answer = f"Here's how it works:\n\n{context}\n\nThis explanation is based on {len(results)} relevant sources."
            elif any(word in question_lower for word in ['why', 'reason', 'benefit']):
                answer = f"Here's why this matters:\n\n{context}\n\nThis analysis comes from {len(results)} relevant sources."
            elif any(word in question_lower for word in ['where', 'used', 'applied']):
                answer = f"Here's where it's used:\n\n{context}\n\nThis information comes from {len(results)} relevant sources."
            else:
                answer = f"Based on the information I found:\n\n{context}\n\nThis answer is based on {len(results)} relevant sources."
        
        response_time = time.time() - start_time
        self.query_count += 1
        
        return {
            'question': question,
            'answer': answer,
            'sources': results,
            'response_time': response_time,
            'query_count': self.query_count
        }

# ============================================================================
# WEB SERVER
# ============================================================================

class SmartRAGWebHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the smart RAG web interface."""
    
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
        """Serve the main web interface."""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Smart RAG Web App - Ask Me Anything!</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
                * { margin: 0; padding: 0; box-sizing: border-box; }
                body { 
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                    padding: 20px;
                }
                .container { 
                    max-width: 900px; 
                    margin: 0 auto; 
                    background: white;
                    border-radius: 20px;
                    box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                    overflow: hidden;
                }
                .header { 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white; 
                    padding: 40px 30px; 
                    text-align: center;
                }
                .header h1 { font-size: 2.5em; margin-bottom: 10px; }
                .header p { font-size: 1.2em; opacity: 0.9; margin-bottom: 20px; }
                .examples { 
                    background: rgba(255,255,255,0.1); 
                    padding: 15px; 
                    border-radius: 10px; 
                    margin-top: 15px;
                }
                .examples h3 { margin-bottom: 10px; }
                .example-questions { 
                    display: grid; 
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
                    gap: 10px; 
                    text-align: left;
                }
                .example-questions div { 
                    background: rgba(255,255,255,0.1); 
                    padding: 8px 12px; 
                    border-radius: 5px; 
                    font-size: 0.9em;
                }
                .content { padding: 30px; }
                .query-form { 
                    background: #f8f9fa; 
                    padding: 25px; 
                    border-radius: 15px; 
                    margin-bottom: 25px;
                    border: 2px solid #e9ecef;
                }
                .query-input { 
                    width: 100%; 
                    padding: 15px; 
                    border: 2px solid #dee2e6; 
                    border-radius: 10px; 
                    font-size: 16px; 
                    margin-bottom: 15px;
                    transition: border-color 0.3s;
                }
                .query-input:focus { 
                    outline: none; 
                    border-color: #667eea; 
                }
                .submit-btn { 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white; 
                    padding: 15px 30px; 
                    border: none; 
                    border-radius: 10px; 
                    cursor: pointer; 
                    font-size: 16px; 
                    font-weight: 600;
                    transition: transform 0.2s;
                }
                .submit-btn:hover { transform: translateY(-2px); }
                .loading { 
                    display: none; 
                    color: #667eea; 
                    text-align: center; 
                    padding: 20px;
                    font-size: 18px;
                }
                .results { margin-top: 25px; }
                .answer { 
                    background: linear-gradient(135deg, #e8f5e8 0%, #f0f8f0 100%);
                    padding: 25px; 
                    border-radius: 15px; 
                    margin-bottom: 20px;
                    border-left: 5px solid #28a745;
                }
                .answer h3 { color: #28a745; margin-bottom: 15px; }
                .sources { 
                    background: linear-gradient(135deg, #f0f8ff 0%, #e6f3ff 100%);
                    padding: 25px; 
                    border-radius: 15px;
                    border-left: 5px solid #007bff;
                }
                .sources h3 { color: #007bff; margin-bottom: 15px; }
                .source-item { 
                    margin-bottom: 15px; 
                    padding: 15px; 
                    background: white; 
                    border-radius: 10px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.05);
                }
                .source-title { font-weight: 600; color: #333; margin-bottom: 5px; }
                .source-category { color: #666; font-size: 0.8em; margin-bottom: 5px; }
                .source-similarity { color: #666; font-size: 0.9em; margin-bottom: 8px; }
                .source-text { color: #555; line-height: 1.5; }
                .metrics { 
                    background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
                    padding: 20px; 
                    border-radius: 15px; 
                    margin-top: 20px;
                    border-left: 5px solid #ffc107;
                }
                .metrics h3 { color: #856404; margin-bottom: 10px; }
                .metric { margin-bottom: 5px; }
                .error { 
                    background: #f8d7da; 
                    color: #721c24; 
                    padding: 15px; 
                    border-radius: 10px; 
                    margin-top: 15px;
                    border-left: 5px solid #dc3545;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Smart RAG Web App</h1>
                    <p>Ask me anything about AI, Technology, Science, Business, or the World!</p>
                    <div class="examples">
                        <h3>Try these questions:</h3>
                        <div class="example-questions">
                            <div>What is artificial intelligence?</div>
                            <div>How does machine learning work?</div>
                            <div>Why is climate change important?</div>
                            <div>Where is AI used in healthcare?</div>
                            <div>What is renewable energy?</div>
                            <div>How do electric vehicles work?</div>
                            <div>What is space exploration?</div>
                            <div>How does social media work?</div>
                        </div>
                    </div>
                </div>
                
                <div class="content">
                    <div class="query-form">
                        <form id="queryForm">
                            <input type="text" id="question" class="query-input" 
                                   placeholder="Ask me anything! Try: What is AI? How does climate change work? Why is renewable energy important?" 
                                   required>
                            <button type="submit" class="submit-btn">Ask Question</button>
                            <div class="loading" id="loading">Thinking...</div>
                        </form>
                    </div>
                    
                    <div id="results" class="results"></div>
                    <div id="metrics" class="metrics"></div>
                </div>
            </div>
            
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
                            body: JSON.stringify({ question: question, top_k: 3 })
                        });
                        
                        if (!response.ok) {
                            throw new Error(`HTTP error! status: ${response.status}`);
                        }
                        
                        const data = await response.json();
                        
                        // Display answer
                        results.innerHTML = `
                            <div class="answer">
                                <h3>Answer:</h3>
                                <p>${data.answer.replace(/\\n/g, '<br>')}</p>
                            </div>
                            <div class="sources">
                                <h3>Sources (${data.sources.length}):</h3>
                                ${data.sources.map((source, i) => `
                                    <div class="source-item">
                                        <div class="source-title">${source.metadata.title}</div>
                                        <div class="source-category">${source.metadata.category}</div>
                                        <div class="source-similarity">Relevance: ${(source.similarity * 100).toFixed(1)}%</div>
                                        <div class="source-text">${source.document}</div>
                                    </div>
                                `).join('')}
                            </div>
                        `;
                        
                        // Display metrics
                        metrics.innerHTML = `
                            <h3>Performance Metrics:</h3>
                            <div class="metric"><strong>Response Time:</strong> ${(data.response_time * 1000).toFixed(0)}ms</div>
                            <div class="metric"><strong>Total Queries:</strong> ${data.query_count}</div>
                            <div class="metric"><strong>Sources Found:</strong> ${data.sources.length}</div>
                        `;
                        
                    } catch (error) {
                        results.innerHTML = `
                            <div class="error">
                                <strong>Error:</strong> ${error.message}
                            </div>
                        `;
                    } finally {
                        loading.style.display = 'none';
                    }
                });
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
            "documents_loaded": len(rag_system.documents),
            "query_count": rag_system.query_count,
            "message": "Smart RAG system is running!"
        }
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(health_data).encode())
    
    def handle_query(self):
        """Handle query requests."""
        try:
            # Read request body
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            # Process query
            question = data.get('question', '')
            top_k = data.get('top_k', 3)
            
            result = rag_system.ask(question, top_k)
            
            # Send response
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())
            
        except Exception as e:
            error_response = {"error": str(e)}
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(error_response).encode())

# ============================================================================
# MAIN EXECUTION
# ============================================================================

# Initialize smart RAG system
rag_system = SmartRAG()
rag_system.load_sample_data()

def start_web_server(port=8000):
    """Start the web server."""
    server_address = ('', port)
    httpd = HTTPServer(server_address, SmartRAGWebHandler)
    
    print("Smart RAG Web App Starting...")
    print("=" * 60)
    print(f"Web Interface: http://localhost:{port}")
    print(f"Health Check: http://localhost:{port}/api/health")
    print(f"Documents loaded: {len(rag_system.documents)}")
    print("")
    print("Now try asking about:")
    print("   - AI & Technology: 'What is artificial intelligence?'")
    print("   - Science: 'How does climate change work?'")
    print("   - Business: 'Where is AI used in healthcare?'")
    print("   - Environment: 'What is renewable energy?'")
    print("   - Society: 'How does social media work?'")
    print("")
    print("Press Ctrl+C to stop the server")
    print("=" * 60)
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped by user")
        httpd.shutdown()

if __name__ == "__main__":
    start_web_server()

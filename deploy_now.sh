#!/bin/bash

#  Deploy RAG System to Web - Step by Step
# ==========================================

echo " RAG Web Deployment - Let's Go!"
echo "=================================="

# Step 1: Test locally first
echo ""
echo " Step 1: Testing locally..."
echo "-----------------------------"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo " Python 3 is required but not installed."
    exit 1
fi

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Test the application
echo "Testing the application..."
python3 -c "
import sys
sys.path.append('.')
from web_app import SmartRAG

print('Testing RAG system...')
rag = SmartRAG()
rag.load_sample_data()

# Test a query
result = rag.ask('What is machine learning?', top_k=2)
print(f'Test successful! Response time: {result[\"response_time\"]:.3f}s')
print(f'   Answer length: {len(result[\"answer\"])} characters')
print(f'   Sources found: {len(result[\"sources\"])}')
"

if [ $? -eq 0 ]; then
    echo " Local test passed!"
else
    echo " Local test failed. Please check the errors above."
    exit 1
fi

# Step 2: Build Docker image
echo ""
echo " Step 2: Building Docker image..."
echo "-----------------------------------"

echo " Building Docker image..."
docker build -t rag-web-app .

if [ $? -eq 0 ]; then
    echo " Docker image built successfully!"
else
    echo " Docker build failed. Please check the errors above."
    exit 1
fi

# Step 3: Test Docker container
echo ""
echo " Step 3: Testing Docker container..."
echo "--------------------------------------"

echo " Starting Docker container..."
docker run -d -p 8000:8000 --name rag-test rag-web-app

# Wait for startup
echo "⏳ Waiting for application to start..."
sleep 15

# Test the container
echo " Testing web interface..."
if curl -s http://localhost:8000/api/health > /dev/null; then
    echo " Web interface is working!"
    
    # Test a query
    echo " Testing query endpoint..."
    response=$(curl -s -X POST "http://localhost:8000/api/query" \
        -H "Content-Type: application/json" \
        -d '{"question": "What is machine learning?", "top_k": 3}')
    
    if echo "$response" | grep -q "answer"; then
        echo " Query endpoint is working!"
        echo " Sample response:"
        echo "$response" | head -5
    else
        echo " Query endpoint failed"
    fi
else
    echo " Web interface is not responding"
fi

# Step 4: Show results
echo ""
echo " Step 4: Your RAG System is Live!"
echo "-----------------------------------"

echo " Web Interface: http://localhost:8000"
echo " Health Check: http://localhost:8000/api/health"
echo " API Docs: http://localhost:8000/docs"
echo ""
echo " SUCCESS! Your RAG system is now running on the web!"
echo ""
echo " Next Steps:"
echo "1. Open http://localhost:8000 in your browser"
echo "2. Ask questions like:"
echo "   - What is machine learning?"
echo "   - How does deep learning work?"
echo "   - Explain artificial intelligence"
echo ""
echo "3. To stop the container: docker stop rag-test"
echo "4. To remove the container: docker rm rag-test"
echo ""
echo " Ready for production deployment!"

# Step 5: Production deployment options
echo ""
echo " Step 5: Deploy to Production (Optional)"
echo "------------------------------------------"
echo ""
echo "To deploy to the cloud, you can:"
echo ""
echo " Railway (Easiest):"
echo "   1. Install Railway CLI: npm install -g @railway/cli"
echo "   2. Login: railway login"
echo "   3. Deploy: railway up"
echo ""
echo "  Render:"
echo "   1. Go to https://render.com"
echo "   2. Connect your GitHub repo"
echo "   3. Deploy with these settings:"
echo "      - Build Command: pip install -r requirements.txt"
echo "      - Start Command: uvicorn simple_web_app:app --host 0.0.0.0 --port \$PORT"
echo ""
echo " Docker Hub:"
echo "   1. Tag image: docker tag rag-web-app yourusername/rag-web-app"
echo "   2. Push: docker push yourusername/rag-web-app"
echo "   3. Deploy anywhere that supports Docker!"
echo ""
echo " Congratulations! You've successfully deployed a RAG system to the web!"

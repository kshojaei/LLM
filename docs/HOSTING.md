# Host Your RAG System

Deploy your RAG system so others can use it. Choose the option that works best for you.

## 🚀 Railway (Recommended - Free)

**Why Railway?** One-click deploy, free tier, automatic GitHub integration.

### Setup:
1. Go to [railway.app](https://railway.app)
2. Sign up with GitHub
3. Click "New Project" → "Deploy from GitHub repo"
4. Select your repository
5. Railway auto-detects your Dockerfile
6. Add environment variables:
   ```
   API_HOST=0.0.0.0
   API_PORT=8000
   ```
7. Deploy! Get URL like `https://your-app.railway.app`

### Test:
```bash
curl https://your-app.railway.app/health
```

**Cost:** Free (500 hours/month), $5/month unlimited

## 🌐 Render (Good for APIs)

**Why Render?** Great for Python APIs, good free tier.

### Setup:
1. Go to [render.com](https://render.com)
2. Connect GitHub account
3. Create "New Web Service"
4. Select your repository
5. Choose "Docker" as environment
6. Deploy!

### Test:
```bash
curl https://your-app.onrender.com/health
```

**Cost:** Free (750 hours/month), $7/month always-on

## ☁️ Google Cloud Run

**Why Cloud Run?** Pay-per-use, scales automatically.

### Setup:
```bash
# Install Google Cloud SDK
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Deploy
gcloud run deploy --source .
```

### Test:
```bash
curl https://your-app-xxx-uc.a.run.app/health
```

**Cost:** Pay per request (very cheap for learning)

## 🐳 Docker (Local Testing)

Test everything locally before deploying:

```bash
# Build and run
docker-compose up --build

# Test API
curl http://localhost:8000/health

# Test query
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is machine learning?"}'
```

## 📱 Share Your Results

Once deployed, share your working RAG system:

### 1. Share the URL
```
Try my RAG system: https://your-app.railway.app

Ask it questions like:
- "What is machine learning?"
- "How does neural networks work?"
- "Explain quantum computing"
```

### 2. API Documentation
Your system includes automatic docs:
- Swagger UI: `https://your-app.railway.app/docs`
- ReDoc: `https://your-app.railway.app/redoc`

### 3. Test Endpoints
```bash
# Health check
curl https://your-app.railway.app/health

# Ask a question
curl -X POST "https://your-app.railway.app/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is RAG?"}'

# Get system status
curl https://your-app.railway.app/status
```

## 🔧 Environment Variables

Add these to your hosting platform:

```bash
# Required
API_HOST=0.0.0.0
API_PORT=8000

# Optional (for better performance)
DEFAULT_EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
DEFAULT_LLM_MODEL=meta-llama/Llama-3-8B-Instruct
MAX_DOCUMENTS=5
```

## 🚨 Troubleshooting

### Common Issues:

**"Out of memory"**
- Use smaller models in config
- Reduce MAX_DOCUMENTS

**"Model not found"**
- Check if models are downloaded
- Verify environment variables

**"Slow responses"**
- Use smaller embedding models
- Reduce dataset size

### Debug:
```bash
# Check logs
docker-compose logs -f

# Check API health
curl http://localhost:8000/health

# Check system status
curl http://localhost:8000/status
```

## 🎯 Success Checklist

- [ ] System deploys without errors
- [ ] Health check returns "healthy"
- [ ] Can ask questions via API
- [ ] Responses are relevant and accurate
- [ ] Can share URL with others

## 💡 Pro Tips

1. **Start small** - Deploy with limited data first
2. **Test locally** - Use Docker before deploying
3. **Monitor usage** - Check logs and performance
4. **Share early** - Get feedback from others
5. **Iterate** - Improve based on real usage

---

**Your RAG system is now live and ready to answer questions!** 🚀

# API Reference

Your RAG system provides a REST API for querying and managing documents.

## Base URL
- **Local**: `http://localhost:8000`
- **Deployed**: `https://your-app.railway.app`

## Key Endpoints

### Health Check
```bash
GET /health
```
Returns system status.

### Ask Questions
```bash
POST /query
```
```json
{
  "question": "What is machine learning?",
  "max_documents": 5
}
```

### Get System Status
```bash
GET /status
```
Returns detailed system information.

### List Documents
```bash
GET /documents
```
Returns all documents in your knowledge base.

## Example Usage

### Python
```python
import requests

# Ask a question
response = requests.post('http://localhost:8000/query', json={
    'question': 'What is machine learning?'
})
print(response.json()['answer'])
```

### JavaScript
```javascript
const response = await fetch('http://localhost:8000/query', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({question: 'What is machine learning?'})
});
const result = await response.json();
console.log(result.answer);
```

### cURL
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is machine learning?"}'
```

## Interactive Docs
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

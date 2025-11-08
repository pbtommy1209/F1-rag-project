# RAG Chatbot Usage Guide

Complete guide for using the RAG Chatbot interface to query your knowledge base interactively.

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [CLI Chatbot](#cli-chatbot)
- [FastAPI Service](#fastapi-service)
- [Features](#features)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

## 🚀 Quick Start

### Prerequisites

1. **Index Directory**: You need a FAISS index created from your documents
   ```bash
   # Create an index first
   python fixed_ingest.py --pdf F1_33.pdf --outdir simple_index --sizes 500 --overlaps 100
   ```

2. **Ollama Running**: Ensure Ollama is running with your model available
   ```bash
   ollama serve
   ollama list  # Verify models are available
   ```

3. **Dependencies**: Install required packages
   ```bash
   pip install fastapi uvicorn  # For API server (optional)
   ```

### Choose Your Interface

- **CLI Chatbot** (Option A): Interactive command-line interface - Best for testing and development
- **FastAPI Service** (Option B): REST API server - Best for web integration and production

---

## 💻 CLI Chatbot

### Basic Usage

```bash
# Start interactive chatbot
python chatbot_cli.py --index-dir simple_index/size500_overlap100
```

### Command Options

```bash
python chatbot_cli.py \
  --index-dir <path_to_index> \    # Required: Path to FAISS index directory
  --model llama3.2:1b \             # Optional: Chat model (default: llama3.2:1b)
  --no-context \                    # Optional: Disable conversation context
  --verbose                         # Optional: Show detailed performance metrics
```

### Interactive Commands

Once the chatbot is running, you can use these commands:

| Command | Description | Example |
|---------|-------------|---------|
| `help` | Show available commands | Type `help` |
| `clear` | Clear conversation history | Type `clear` |
| `history` | View recent conversation | Type `history` |
| `stats` | Show conversation statistics | Type `stats` |
| `export` | Export conversation to JSON | Type `export` |
| `quit` / `exit` | Exit the chatbot | Type `quit` |

### Example Session

```
🤖 RAG CHATBOT - Interactive Knowledge Base Assistant
============================================================

❓ You: What is DRS in Formula 1?

🤔 Thinking...

🤖 Assistant: DRS (Drag Reduction System) allows a driver within one second of the car ahead to open a rear-wing flap in designated zones to reduce drag and increase speed.

❓ You: How does it work?

🤔 Thinking...

🤖 Assistant: DRS works by opening a flap on the rear wing when a driver is within one second of the car in front at the detection point. This reduces aerodynamic drag, allowing for higher speeds and easier overtaking in designated DRS zones on the track.

❓ You: stats

📊 Conversation Statistics:
  Total Queries: 2
  Successful: 2/2
  Average Response Time: 1.45s
  Total Time: 2.90s

❓ You: quit

👋 Goodbye! Thanks for using RAG Chatbot!
```

### Advanced Usage

**With Custom Model:**
```bash
python chatbot_cli.py \
  --index-dir simple_index/size500_overlap100 \
  --model deepseek-r1:1.5b
```

**With Verbose Output:**
```bash
python chatbot_cli.py \
  --index-dir simple_index/size500_overlap100 \
  --verbose
```

**Without Conversation Context:**
```bash
python chatbot_cli.py \
  --index-dir simple_index/size500_overlap100 \
  --no-context
```

---

## 🌐 FastAPI Service

### Starting the Server

**Basic:**
```bash
python chatbot_api.py --index-dir simple_index/size500_overlap100
```

**With Custom Port:**
```bash
python chatbot_api.py \
  --index-dir simple_index/size500_overlap100 \
  --port 8080
```

**With Environment Variables:**
```bash
export INDEX_DIR=simple_index/size500_overlap100
export CHAT_MODEL=llama3.2:1b
python chatbot_api.py --port 8000
```

**Development Mode (Auto-reload):**
```bash
python chatbot_api.py \
  --index-dir simple_index/size500_overlap100 \
  --reload
```

### API Endpoints

#### 1. Query Endpoint

**POST** `/query`

Query the knowledge base with a question.

**Request Body:**
```json
{
  "question": "What is DRS in Formula 1?",
  "use_context": true,
  "session_id": "user123"
}
```

**Response:**
```json
{
  "response": "DRS (Drag Reduction System) allows...",
  "metadata": {
    "query": "What is DRS in Formula 1?",
    "retrieval_time": 0.45,
    "llm_time": 1.23,
    "total_time": 1.68,
    "num_chunks": 5,
    "success": true,
    "timestamp": "2024-10-22T19:30:00"
  },
  "session_id": "user123"
}
```

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is DRS in Formula 1?",
    "session_id": "user123"
  }'
```

**Python Example:**
```python
import requests

response = requests.post(
    "http://localhost:8000/query",
    json={
        "question": "What is DRS in Formula 1?",
        "session_id": "user123"
    }
)
data = response.json()
print(data["response"])
```

#### 2. History Endpoint

**GET** `/history?session_id=user123`

Get conversation history for a session.

**Response:**
```json
{
  "history": [
    {
      "query": "What is DRS?",
      "response": "DRS allows...",
      "timestamp": "2024-10-22T19:30:00",
      "metadata": {...}
  }
  ],
  "session_id": "user123"
}
```

**cURL Example:**
```bash
curl "http://localhost:8000/history?session_id=user123"
```

#### 3. Statistics Endpoint

**GET** `/stats?session_id=user123`

Get conversation statistics.

**Response:**
```json
{
  "total_queries": 5,
  "successful_queries": 5,
  "average_time": 1.45,
  "total_time": 7.25,
  "session_id": "user123"
}
```

**cURL Example:**
```bash
curl "http://localhost:8000/stats?session_id=user123"
```

#### 4. Clear History Endpoint

**POST** `/clear?session_id=user123`

Clear conversation history for a session.

**Response:**
```json
{
  "message": "Conversation history cleared",
  "session_id": "user123"
}
```

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/clear?session_id=user123"
```

#### 5. Health Check

**GET** `/health`

Check server health and configuration.

**Response:**
```json
{
  "status": "healthy",
  "index_dir": "simple_index/size500_overlap100",
  "model": "llama3.2:1b",
  "active_sessions": 3
}
```

#### 6. API Documentation

**GET** `/docs`

Interactive Swagger UI documentation (available at `http://localhost:8000/docs`)

---

## ✨ Features

### Conversation Context

The chatbot maintains conversation history to enable multi-turn conversations:

```python
# First query
response1, _ = chatbot.query("What is DRS?")

# Follow-up query (uses context)
response2, _ = chatbot.query("How does it work?")  # Understands "it" refers to DRS
```

**Enable/Disable:**
- CLI: Use `--no-context` flag to disable
- API: Set `"use_context": false` in request

### Session Management (API)

Each session maintains its own conversation history:

```bash
# User 1's session
curl -X POST "http://localhost:8000/query" \
  -d '{"question": "What is DRS?", "session_id": "user1"}'

# User 2's session (separate history)
curl -X POST "http://localhost:8000/query" \
  -d '{"question": "What is DRS?", "session_id": "user2"}'
```

### Logging

All interactions are automatically logged to `logs/chatbot_YYYYMMDD.log`:

```
2024-10-22 19:30:00 - RAGChatbot - INFO - Query: What is DRS? | Response: DRS allows... | Time: 1.68s | Chunks: 5 | Success: True
```

**Log Format:**
- Timestamp
- Query text (first 100 chars)
- Response text (first 100 chars)
- Total time
- Number of chunks retrieved
- Success status

### Performance Monitoring

Each query returns detailed metadata:

```python
response, metadata = chatbot.query("What is DRS?")

print(f"Total time: {metadata['total_time']:.2f}s")
print(f"Retrieval time: {metadata['retrieval_time']:.2f}s")
print(f"LLM time: {metadata['llm_time']:.2f}s")
print(f"Chunks used: {metadata['num_chunks']}")
```

### Conversation Export

Export conversation history to JSON:

**CLI:**
```bash
# Type 'export' in the chatbot
❓ You: export
✅ Conversation exported to conversation_2024-10-22.json
```

**Programmatic:**
```python
chatbot.export_conversation("my_conversation.json")
```

---

## 📝 Examples

### Example 1: Basic CLI Usage

```bash
# Start chatbot
python chatbot_cli.py --index-dir simple_index/size500_overlap100

# In the chatbot:
❓ You: What is the safety car?
🤖 Assistant: The safety car is deployed to bunch the field and slow cars while incidents are cleared...

❓ You: When is it used?
🤖 Assistant: The safety car is used when there's an incident on track that requires marshals to clear debris or assist drivers...

❓ You: history
📜 Recent Conversation History (last 2 items):
------------------------------------------------------------
[1] 2024-10-22 19:30:00
Q: What is the safety car?
A: The safety car is deployed...
   ⏱️ 1.45s | 📦 5 chunks

[2] 2024-10-22 19:30:15
Q: When is it used?
A: The safety car is used when...
   ⏱️ 1.52s | 📦 5 chunks
------------------------------------------------------------

❓ You: quit
```

### Example 2: API Integration

```python
import requests

BASE_URL = "http://localhost:8000"
SESSION_ID = "my_session"

# Query 1
response1 = requests.post(
    f"{BASE_URL}/query",
    json={
        "question": "What is parc fermé?",
        "session_id": SESSION_ID
    }
)
print(response1.json()["response"])

# Query 2 (with context)
response2 = requests.post(
    f"{BASE_URL}/query",
    json={
        "question": "When does it apply?",
        "session_id": SESSION_ID
    }
)
print(response2.json()["response"])

# Get statistics
stats = requests.get(f"{BASE_URL}/stats?session_id={SESSION_ID}")
print(f"Total queries: {stats.json()['total_queries']}")
print(f"Average time: {stats.json()['average_time']:.2f}s")
```

### Example 3: Multi-Session API

```python
import requests

BASE_URL = "http://localhost:8000"

# User A's session
user_a = requests.post(
    f"{BASE_URL}/query",
    json={"question": "What is DRS?", "session_id": "user_a"}
)

# User B's session (separate conversation)
user_b = requests.post(
    f"{BASE_URL}/query",
    json={"question": "What is the safety car?", "session_id": "user_b"}
)

# Each session maintains separate history
history_a = requests.get(f"{BASE_URL}/history?session_id=user_a")
history_b = requests.get(f"{BASE_URL}/history?session_id=user_b")
```

### Example 4: Error Handling

```python
from rag_chatbot import create_chatbot

chatbot = create_chatbot(index_dir="simple_index/size500_overlap100")

try:
    response, metadata = chatbot.query("What is DRS?")
    
    if metadata["success"]:
        print(f"✅ Success: {response}")
    else:
        print(f"❌ Error: {metadata.get('error', 'Unknown error')}")
        
except Exception as e:
    print(f"❌ Chatbot error: {e}")
```

---

## 🔧 Troubleshooting

### Common Issues

#### 1. Index Directory Not Found

**Error:**
```
❌ Index directory not found: simple_index/size500_overlap100
```

**Solution:**
```bash
# Create the index first
python fixed_ingest.py --pdf F1_33.pdf --outdir simple_index --sizes 500 --overlaps 100
```

#### 2. Ollama Not Running

**Error:**
```
❌ Connection refused to http://localhost:11434
```

**Solution:**
```bash
# Start Ollama
ollama serve

# Verify models are available
ollama list
```

#### 3. Model Not Available

**Error:**
```
❌ Model 'llama3.2:1b' not found
```

**Solution:**
```bash
# Pull the model
ollama pull llama3.2:1b

# Or use a different model
python chatbot_cli.py --index-dir <index> --model deepseek-r1:1.5b
```

#### 4. FastAPI Import Error

**Error:**
```
ModuleNotFoundError: No module named 'fastapi'
```

**Solution:**
```bash
pip install fastapi uvicorn
```

#### 5. No Response from Chatbot

**Possible Causes:**
- Index files corrupted or missing
- Ollama model not responding
- Network issues

**Debug Steps:**
```bash
# Check index files
ls -la simple_index/size500_overlap100/
# Should contain: index.faiss, meta.json

# Test Ollama
curl http://localhost:11434/api/tags

# Run with verbose mode
python chatbot_cli.py --index-dir <index> --verbose
```

### Performance Tips

1. **Use Optimal Index**: Use `size300_overlap50` for best accuracy (85%)
2. **Enable Context Sparingly**: Context adds overhead; disable for single queries
3. **Monitor Logs**: Check `logs/chatbot_*.log` for performance patterns
4. **Session Management**: Clear old sessions periodically to free memory

### Best Practices

1. **Session IDs**: Use unique session IDs for different users/conversations
2. **Error Handling**: Always check `metadata["success"]` in API responses
3. **Context Usage**: Enable context for multi-turn conversations, disable for isolated queries
4. **Logging**: Review logs regularly to identify common queries and optimize
5. **Performance**: Monitor response times and adjust `top_k` if needed

---

## 📚 Additional Resources

- **Main README**: See `README.md` for project overview
- **API Documentation**: Visit `http://localhost:8000/docs` when server is running
- **Logs**: Check `logs/chatbot_*.log` for detailed interaction logs
- **Conversation Export**: JSON files contain full conversation history with metadata

---

## 🎯 Quick Reference

### CLI Commands
```bash
# Start chatbot
python chatbot_cli.py --index-dir <index_dir>

# With options
python chatbot_cli.py --index-dir <index_dir> --model <model> --verbose
```

### API Endpoints
```bash
# Query
POST /query
GET /history?session_id=<id>
GET /stats?session_id=<id>
POST /clear?session_id=<id>
GET /health
```

### Python Usage
```python
from rag_chatbot import create_chatbot

chatbot = create_chatbot(index_dir="<index_dir>")
response, metadata = chatbot.query("Your question")
```

---

**Happy Chatting! 🤖💬**

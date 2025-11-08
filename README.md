# RAG (Retrieval-Augmented Generation) System

A comprehensive RAG system that processes documents (including scanned PDFs via OCR) and provides interactive chatbot interfaces for querying knowledge bases. Supports multiple Ollama models with optimized chunking strategies and comprehensive evaluation tools.

## 🚀 Features

- **Multi-Model Support**: llama3.2:1b, deepseek-r1:1.5b, gemma3:1b
- **OCR Support**: PaddleOCR fallback for scanned/image PDFs
- **Interactive Chatbot**: CLI and FastAPI interfaces for querying knowledge base
- **Conversation Context**: Multi-turn conversation support with history management
- **Optimized Chunking**: Multiple chunk sizes and overlap configurations
- **Automated Evaluation**: Comprehensive testing and accuracy reporting
- **Performance Monitoring**: Response time tracking and error analysis
- **Logging System**: Automatic logging of Q&A interactions and performance metrics
- **FAISS Integration**: Efficient vector similarity search
- **Comprehensive Fallbacks**: Multiple fallback layers for robust operation

## 📁 Project Structure

```
rag_project/
├── 📄 Core Files
│   ├── fixed_ingest.py              # Main RAG ingestion and querying script
│   ├── ollama_client.py             # Ollama API client
│   ├── F1_33.pdf                    # Source Formula 1 document
│   ├── correct_answer.json          # Ground truth answers (40 questions)
│   └── question_ask/
│       └── example_questions.json   # Test questions (40 T/F questions)
│
├── 💬 Chatbot Interface
│   ├── rag_chatbot.py               # Core chatbot module with conversation context
│   ├── chatbot_cli.py               # CLI chatbot interface
│   ├── chatbot_api.py               # FastAPI REST API server
│   ├── chatbot_demo.py              # Quick demo script
│   └── logs/                        # Conversation and performance logs
│
├── 🔍 OCR Testing
│   ├── test_paddle_ocr.py           # Comprehensive OCR testing tool
│   └── test_ocr.py                  # Simple OCR test script
│
├── 🛠️ Development Tools
│   ├── query_tool.py                # Interactive development tool
│   ├── test_tool.py                 # Automated testing tool
│   └── run_complete_workflow.py     # Complete workflow automation
│
├── 🧪 Evaluation Scripts
│   ├── test_llama3_accuracy.py      # llama3.2:1b chunk size evaluation
│   ├── test_llama3_overlap.py       # llama3.2:1b overlap evaluation
│   ├── test_deepseek_accuracy.py    # deepseek-r1:1.5b evaluation
│   ├── test_gemma3_accuracy.py     # gemma3:1b evaluation
│   └── test_optimal_final.py        # Optimal configuration testing
│
├── 📊 Generated Reports
│   ├── llama3.2_best_comb_final.json    # Optimal llama3.2 configuration
│   ├── llama3.2_overlap.json            # Overlap evaluation results
│   ├── deepseek_model_report            # Deepseek model report
│   ├── gemma3_model_report              # Gemma3 model report
│   └── demo_workflow.json               # Sample test results
│
└── 📁 Index Directories
    ├── faiss_indexes/               # Original FAISS indexes
    ├── simple_index/                # Simple test indexes
    ├── llama3_test_indexes/        # llama3.2 evaluation indexes
    ├── deepseek_test_indexes/       # deepseek evaluation indexes
    ├── gemma3_test_indexes/         # gemma3 evaluation indexes
    └── optimal_test_indexes/        # Optimal configuration indexes
```

## 🏆 Performance Results

### Model Comparison
| Model | Accuracy | Avg Response Time | Best Configuration |
|-------|----------|-------------------|-------------------|
| **llama3.2:1b** | 85.00% | 6.14s | size300_overlap50 |
| **deepseek-r1:1.5b** | 85.00% | 1.66s | size300_overlap50 |
| **gemma3:1b** | 85.00% | 3.36s | size300_overlap50 |

### Optimal Configuration
- **Chunk Size**: 300 characters
- **Overlap**: 50 characters
- **Accuracy**: 85.00% (34/40 correct)
- **Points**: 85.0/100.0 (2.5 points per question)

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Ollama installed and running
- Required models: llama3.2:1b, deepseek-r1:1.5b, gemma3:1b

### Installation

```bash
# Clone or download the project
cd rag_project

# Install dependencies
pip install -r requirements.txt

# For OCR support (optional but recommended)
pip install paddlepaddle paddleocr opencv-python

# Ensure Ollama is running
ollama serve

# Verify models are available
ollama list
```

### Basic Workflow

```bash
# 1. Ingest a PDF document
python fixed_ingest.py --pdf F1_33.pdf --outdir simple_index --sizes 500 --overlaps 100

# 2. Start interactive chatbot
python chatbot_cli.py --index-dir simple_index/size500_overlap100

# 3. Or start API server
python chatbot_api.py --index-dir simple_index/size500_overlap100 --port 8000
```

## 📖 PDF Processing & OCR

### Supported PDF Types

The system handles multiple PDF types with automatic fallback:

1. **Digital PDFs** (text-based) - Fast extraction via PyPDF/PyMuPDF
2. **Scanned PDFs** (image-based) - OCR via PaddleOCR
3. **Mixed PDFs** - Combines text extraction and OCR as needed

### PDF Reading Fallback Chain

```
Primary: PyPDF
    ↓ (if fails)
Fallback 1: PyMuPDF (fitz)
    ↓ (if little/no text)
Fallback 2: pdfplumber
    ↓ (if little/no text)
Final: PaddleOCR (for scanned/image PDFs)
```

### Using OCR for Scanned PDFs

**Automatic Detection:**
The system automatically detects when a PDF has little or no extractable text and falls back to OCR:

```bash
# Process any PDF (OCR will trigger automatically if needed)
python fixed_ingest.py --pdf scanned_document.pdf --outdir ocr_index --sizes 500 --overlaps 100
```

**Testing OCR:**
```bash
# Test OCR functionality
python test_paddle_ocr.py --pdf scanned_document.pdf

# Quick OCR test
python test_ocr.py --pdf scanned_document.pdf
```

**OCR Requirements:**
```bash
# Install OCR dependencies
pip install paddlepaddle paddleocr opencv-python pymupdf

# For CPU-only installation
pip install paddlepaddle==2.6.1 -f https://www.paddlepaddle.org.cn/whl/cpu.html
pip install paddleocr==2.7.3
```

**OCR Output:**
When OCR is triggered, you'll see:
```
🔄 Using fallback PDF reading method...
ℹ️ PyMuPDF returned little/no text, considering OCR...
🔎 Attempting OCR with PaddleOCR...
✅ OCR extraction successful with PaddleOCR
Extracted 15234 characters
```

## 💬 Chatbot Interface

### CLI Chatbot (Option A)

Interactive command-line chatbot for querying your knowledge base.

**Basic Usage:**
```bash
# Start interactive chatbot
python chatbot_cli.py --index-dir simple_index/size500_overlap100
```

**With Options:**
```bash
# Custom model
python chatbot_cli.py --index-dir simple_index/size500_overlap100 --model deepseek-r1:1.5b

# Verbose mode (show performance metrics)
python chatbot_cli.py --index-dir simple_index/size500_overlap100 --verbose

# Disable conversation context
python chatbot_cli.py --index-dir simple_index/size500_overlap100 --no-context
```

**Interactive Commands:**
- Ask questions directly
- `help` - Show available commands
- `clear` - Clear conversation history
- `history` - View recent conversation
- `stats` - Show conversation statistics
- `export` - Export conversation to JSON
- `quit` or `exit` - Exit chatbot

**Example Session:**
```
🤖 RAG CHATBOT - Interactive Knowledge Base Assistant
============================================================

❓ You: What is DRS in Formula 1?

🤔 Thinking...

🤖 Assistant: DRS (Drag Reduction System) allows a driver within one second of the car ahead to open a rear-wing flap in designated zones to reduce drag and increase speed.

❓ You: How does it work?

🤔 Thinking...

🤖 Assistant: DRS works by opening a flap on the rear wing when a driver is within one second of the car in front at the detection point...

❓ You: stats

📊 Conversation Statistics:
  Total Queries: 2
  Successful: 2/2
  Average Response Time: 1.45s
  Total Time: 2.90s

❓ You: quit
👋 Goodbye!
```

### FastAPI Service (Option B)

RESTful API server for web-based integration.

**Start Server:**
```bash
# Basic
python chatbot_api.py --index-dir simple_index/size500_overlap100

# Custom port
python chatbot_api.py --index-dir simple_index/size500_overlap100 --port 8080

# With environment variables
export INDEX_DIR=simple_index/size500_overlap100
export CHAT_MODEL=llama3.2:1b
python chatbot_api.py --port 8000

# Development mode (auto-reload)
python chatbot_api.py --index-dir simple_index/size500_overlap100 --reload
```

**API Endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/query` | POST | Query the knowledge base |
| `/history` | GET | Get conversation history |
| `/stats` | GET | Get conversation statistics |
| `/clear` | POST | Clear conversation history |
| `/health` | GET | Health check |
| `/docs` | GET | Interactive API documentation |

**Example API Usage:**
```bash
# Query endpoint
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is DRS in Formula 1?",
    "session_id": "user123"
  }'

# Get history
curl "http://localhost:8000/history?session_id=user123"

# Get statistics
curl "http://localhost:8000/stats?session_id=user123"

# Clear history
curl -X POST "http://localhost:8000/clear?session_id=user123"
```

**Python API Client:**
```python
import requests

BASE_URL = "http://localhost:8000"

# Query
response = requests.post(
    f"{BASE_URL}/query",
    json={
        "question": "What is DRS?",
        "session_id": "user123"
    }
)
print(response.json()["response"])

# Get stats
stats = requests.get(f"{BASE_URL}/stats?session_id=user123")
print(f"Total queries: {stats.json()['total_queries']}")
```

**Chatbot Features:**
- ✅ Multi-turn conversation context
- ✅ Session management (API)
- ✅ Performance logging
- ✅ Conversation history export
- ✅ CORS enabled for web frontends
- ✅ Automatic logging to `logs/chatbot_YYYYMMDD.log`

## 🛠️ Development Tools

### Query Tool (`query_tool.py`)

Interactive development tool for testing individual questions.

**Interactive Mode:**
```bash
python query_tool.py --index-dir simple_index/size500_overlap100
```

**Batch Mode:**
```bash
python query_tool.py --mode batch --index-dir simple_index/size500_overlap100 --output results.json
```

### Test Tool (`test_tool.py`)

Automated testing tool for comprehensive evaluation.

**Quick Test:**
```bash
python test_tool.py --mode quick --index-dir simple_index/size500_overlap100 --num-questions 5
```

**Comprehensive Test:**
```bash
python test_tool.py --index-dir simple_index/size500_overlap100 --output full_test.json
```

## 🧪 Model Evaluation

### Running Evaluations

**llama3.2:1b:**
```bash
# Test different chunk sizes
python test_llama3_accuracy.py

# Test different overlaps
python test_llama3_overlap.py

# Test optimal configuration
python test_optimal_final.py
```

**deepseek-r1:1.5b:**
```bash
python test_deepseek_accuracy.py
```

**gemma3:1b:**
```bash
python test_gemma3_accuracy.py
```

### Generated Reports

- `llama3.2_best_comb_final.json` - Optimal llama3.2 configuration
- `llama3.2_overlap.json` - Overlap evaluation results
- `deepseek_model_report` - Deepseek performance report
- `gemma3_model_report` - Gemma3 performance report

## 📊 Complete Workflow Examples

### Example 1: Process Scanned PDF with OCR

```bash
# 1. Process scanned PDF (OCR will trigger automatically)
python fixed_ingest.py --pdf scanned_document.pdf --outdir ocr_index --sizes 500 --overlaps 100

# 2. Test OCR extraction
python test_paddle_ocr.py --pdf scanned_document.pdf --test-all

# 3. Use chatbot with OCR-processed index
python chatbot_cli.py --index-dir ocr_index/size500_overlap100
```

### Example 2: Full Evaluation Workflow

```bash
# 1. Ingest document
python fixed_ingest.py --pdf F1_33.pdf --outdir evaluation_index --sizes 300,500,800 --overlaps 50,100,150

# 2. Evaluate models
python test_llama3_accuracy.py
python test_deepseek_accuracy.py
python test_gemma3_accuracy.py

# 3. Find optimal configuration
python test_optimal_final.py

# 4. Deploy with optimal settings
python fixed_ingest.py --pdf F1_33.pdf --outdir production_index --sizes 300 --overlaps 50

# 5. Start chatbot with optimal index
python chatbot_cli.py --index-dir production_index/size300_overlap50/size300_overlap50
```

### Example 3: API Integration

```bash
# 1. Start API server
python chatbot_api.py --index-dir production_index/size300_overlap50/size300_overlap50 --port 8000

# 2. Query via API
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is DRS?", "session_id": "demo"}'

# 3. Monitor performance
curl "http://localhost:8000/stats?session_id=demo"
```

## 🔧 Configuration Options

### Chunk Sizes Tested
- 300 characters (optimal)
- 500 characters
- 800 characters
- 1000 characters

### Overlap Sizes Tested
- 50 characters (optimal)
- 100 characters
- 150 characters
- 200 characters

### Model Parameters
- **Context Window**: 4096 tokens
- **Embedding Model**: mxbai-embed-large (default)
- **Vector Dimension**: 1024 (varies by model)
- **Similarity Search**: FAISS IndexFlatIP

## 📈 Performance Monitoring

### Logging

All interactions are automatically logged to `logs/chatbot_YYYYMMDD.log`:

```
2024-10-22 19:30:00 - RAGChatbot - INFO - Query: What is DRS? | Response: DRS allows... | Time: 1.68s | Chunks: 5 | Success: True
```

### Daily Monitoring
```bash
# Quick health check
python test_tool.py --mode quick --index-dir production_index --num-questions 5
```

### Weekly Reports
```bash
# Generate comprehensive report
python test_tool.py --index-dir production_index --output weekly_report.json
```

## 🐛 Troubleshooting

### Common Issues

**1. Index Not Found**
```bash
# Check available indexes
find . -name "index.faiss"

# Create index if missing
python fixed_ingest.py --pdf F1_33.pdf --outdir simple_index --sizes 500 --overlaps 100
```

**2. Ollama Not Running**
```bash
# Start Ollama
ollama serve

# Verify models
ollama list
```

**3. Model Not Available**
```bash
# Pull required models
ollama pull llama3.2:1b
ollama pull deepseek-r1:1.5b
ollama pull gemma3:1b
```

**4. OCR Not Working**
```bash
# Install OCR dependencies
pip install paddlepaddle paddleocr opencv-python pymupdf

# Test OCR
python test_paddle_ocr.py --pdf scanned_document.pdf
```

**5. FastAPI Import Error**
```bash
# Install FastAPI
pip install fastapi uvicorn pydantic
```

**6. Low Text Extraction (OCR should trigger)**
```bash
# Check if OCR is working
python test_paddle_ocr.py --pdf your_pdf.pdf --test-all

# Verify OCR dependencies
python -c "import paddleocr; print('PaddleOCR installed')"
```

### Debug Mode

```bash
# Run with verbose output
python chatbot_cli.py --index-dir simple_index/size500_overlap100 --verbose

# Test OCR directly
python test_paddle_ocr.py --pdf scanned_document.pdf --test-all
```

## 📋 File Descriptions

### Core Scripts
- `fixed_ingest.py` - Main RAG system with PDF processing, chunking, embedding, OCR fallback, and querying
- `ollama_client.py` - Ollama API client for embeddings and chat
- `rag_chatbot.py` - Core chatbot module with conversation context

### Chatbot Interface
- `chatbot_cli.py` - CLI chatbot interface
- `chatbot_api.py` - FastAPI REST API server
- `chatbot_demo.py` - Quick demo script

### OCR Testing
- `test_paddle_ocr.py` - Comprehensive OCR testing tool
- `test_ocr.py` - Simple OCR test script

### Development Tools
- `query_tool.py` - Interactive development tool
- `test_tool.py` - Automated testing framework

### Evaluation Scripts
- `test_llama3_accuracy.py` - llama3.2:1b chunk size evaluation
- `test_llama3_overlap.py` - llama3.2:1b overlap evaluation
- `test_deepseek_accuracy.py` - deepseek-r1:1.5b evaluation
- `test_gemma3_accuracy.py` - gemma3:1b evaluation
- `test_optimal_final.py` - Optimal configuration testing

## 📚 Additional Documentation

- **Chatbot Guide**: See `CHATBOT_GUIDE.md` for detailed chatbot usage
- **Workflow Guide**: See `WORKFLOW_GUIDE.md` for complete workflow examples
- **API Documentation**: Visit `http://localhost:8000/docs` when server is running

## 🎯 Quick Reference

### Most Common Commands

```bash
# Ingest PDF
python fixed_ingest.py --pdf document.pdf --outdir index_dir --sizes 300 --overlaps 50

# Start CLI chatbot
python chatbot_cli.py --index-dir index_dir/size300_overlap50

# Start API server
python chatbot_api.py --index-dir index_dir/size300_overlap50 --port 8000

# Test OCR
python test_paddle_ocr.py --pdf scanned.pdf

# Quick test
python test_tool.py --mode quick --index-dir index_dir --num-questions 5
```

## 📊 Expected Results

- **Accuracy**: 85%+ for optimal configurations
- **Response Time**: 1-6 seconds per question
- **Success Rate**: 95%+ (minimal errors/timeouts)
- **OCR Accuracy**: Varies by image quality, typically 80-95% for clear scans
- **Output**: Structured JSON reports with detailed metrics

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly (including OCR for scanned PDFs)
5. Submit a pull request

## 📄 License

This project is for educational and research purposes.

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review the generated log files in `logs/`
3. Verify Ollama is running and models are available
4. Test OCR with `python test_paddle_ocr.py`
5. Check chatbot logs for detailed error messages

---

**Last Updated**: October 2024  
**Version**: 2.0.0  
**Status**: Production Ready ✅

**Key Features**: Multi-model RAG • OCR Support • Interactive Chatbot • Comprehensive Evaluation

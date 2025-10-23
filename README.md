# F1-rag-project
# RAG (Retrieval-Augmented Generation) System for Formula 1

A comprehensive RAG system that processes Formula 1 documents and answers true/false questions using multiple Ollama models. The system includes evaluation tools, optimization scripts, and development utilities for testing different chunking strategies and model performance.

## 🚀 Features

- **Multi-Model Support**: llama3.2:1b, deepseek-r1:1.5b, gemma3:1b
- **Optimized Chunking**: Multiple chunk sizes and overlap configurations
- **Interactive Development**: Real-time querying and testing tools
- **Automated Evaluation**: Comprehensive testing and accuracy reporting
- **Performance Monitoring**: Response time tracking and error analysis
- **FAISS Integration**: Efficient vector similarity search

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
├── 🛠️ Development Tools
│   ├── query_tool.py                # Interactive development tool
│   ├── test_tool.py                 # Automated testing tool
│   └── run_complete_workflow.py     # Complete workflow automation
│
├── 🧪 Evaluation Scripts
│   ├── test_llama3_accuracy.py      # llama3.2:1b chunk size evaluation
│   ├── test_llama3_overlap.py       # llama3.2:1b overlap evaluation
│   ├── test_deepseek_accuracy.py    # deepseek-r1:1.5b evaluation
│   ├── test_gemma3_accuracy.py      # gemma3:1b evaluation
│   └── test_optimal_final.py        # Optimal configuration testing
│
├── 📊 Generated Reports
│   ├── llama3.2_best_comb_final.json    # Optimal llama3.2 configuration
│   ├── deepseek_best_combi_report       # Deepseek model report
│   ├── gemma3_best_comb_report          # Gemma3 model report
│   └── demo_workflow.json               # Sample test results
│
└── 📁 Index Directories
    ├── faiss_indexes/               # Original FAISS indexes
    ├── simple_index/                 # Simple test indexes
    ├── llama3_test_indexes/         # llama3.2 evaluation indexes
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

# Install dependencies (if requirements.txt exists)
pip install -r requirements.txt

# Ensure Ollama is running
ollama serve
```

### Basic Usage

####  Quick Testing
```bash
# Test first 5 questions
python test_tool.py --mode quick --index-dir simple_index/size500_overlap100 --num-questions 5
```

####  Full Evaluation
```bash
# Run comprehensive test
python test_tool.py --index-dir simple_index/size500_overlap100 --output test_results.json
```

## 🛠️ Development Tools

### Query Tool (`query_tool.py`)
Interactive development tool for testing individual questions.

**Interactive Mode:**
```bash
python query_tool.py --index-dir [INDEX_DIR]
```

**Batch Mode:**
```bash
python query_tool.py --mode batch --index-dir [INDEX_DIR] --output results.json
```

### Test Tool (`test_tool.py`)
Automated testing tool for comprehensive evaluation.

**Quick Test:**
```bash
python test_tool.py --mode quick --index-dir [INDEX_DIR] --num-questions 5
```

**Comprehensive Test:**
```bash
python test_tool.py --index-dir [INDEX_DIR] --output full_test.json
```

## 🧪 Model Evaluation

### Run Complete Evaluation
```bash
# Evaluate llama3.2:1b
python test_llama3_accuracy.py
python test_llama3_overlap.py
python test_optimal_final.py

# Evaluate deepseek-r1:1.5b
python test_deepseek_accuracy.py

# Evaluate gemma3:1b
python test_gemma3_accuracy.py
```

### Generated Reports
- `llama3.2_best_comb_final.json` - Optimal llama3.2 configuration
- `deepseek_model_report` - Deepseek performance report
- `gemma3_model_report` - Gemma3 performance report

## 📊 Output Analysis

### JSON Report Structure
```json
{
  "test_info": {
    "model": "llama3.2:1b",
    "timestamp": "2024-10-22T19:29:00",
    "total_questions": 40
  },
  "performance_metrics": {
    "correct_count": 34,
    "accuracy": 85.0,
    "points_earned": 85.0,
    "average_response_time": 1.46
  },
  "detailed_results": [...]
}
```

### Key Metrics
- **Accuracy**: Percentage of correct answers
- **Points**: Total points earned (2.5 per question)
- **Response Time**: Average time per query
- **Error Rate**: Percentage of failed queries
- **Success Rate**: Percentage of successful queries

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
- **Embedding Model**: nomic-embed-text
- **Vector Dimension**: 768
- **Similarity Search**: FAISS

## 📈 Performance Monitoring

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

1. **Index Not Found**
   ```bash
   # Check available indexes
   find . -name "index.faiss"
   ```

2. **Model Not Available**
   ```bash
   # Check Ollama models
   ollama list
   ```

3. **Import Errors**
   ```bash
   # Ensure ollama_client.py exists
   ls -la ollama_client.py
   ```

### Debug Mode
```bash
# Run with verbose output
python query_tool.py --index-dir [INDEX_DIR] --model llama3.2:1b
```

## 📋 File Descriptions

### Core Scripts
- `fixed_ingest.py` - Main RAG system with PDF processing, chunking, embedding, and querying
- `ollama_client.py` - Ollama API client for embeddings and chat
- `query_tool.py` - Interactive development tool
- `test_tool.py` - Automated testing framework

### Evaluation Scripts
- `test_llama3_accuracy.py` - llama3.2:1b chunk size evaluation
- `test_llama3_overlap.py` - llama3.2:1b overlap evaluation
- `test_deepseek_accuracy.py` - deepseek-r1:1.5b evaluation
- `test_gemma3_accuracy.py` - gemma3:1b evaluation
- `test_optimal_final.py` - Optimal configuration testing

### Data Files
- `F1_33.pdf` - Source Formula 1 document
- `question_ask/example_questions.json` - 40 true/false questions
- `correct_answer.json` - Ground truth answers
- Various `*.json` report files with evaluation results

## 🎯 Usage Examples

### Development Workflow
```bash
# 1. Start interactive session
python query_tool.py --index-dir simple_index/size500_overlap100

# 2. Test specific questions
# ❓ Your question: T/F: DRS allows a driver within one second...
# ✅ Model Response: TRUE

# 3. Run quick validation
python test_tool.py --mode quick --index-dir simple_index/size500_overlap100 --num-questions 3

# 4. Generate full report
python test_tool.py --index-dir simple_index/size500_overlap100 --output development_report.json
```

### Production Deployment
```bash
# 1. Use optimal configuration
python fixed_ingest.py --pdf F1_33.pdf --outdir production_index --sizes 300 --overlaps 50

# 2. Validate deployment
python test_tool.py --index-dir production_index/size300_overlap50/size300_overlap50 --output production_test.json

# 3. Monitor performance
python query_tool.py --mode batch --index-dir production_index/size300_overlap50/size300_overlap50 --output monitoring.json
```

## 📊 Expected Results

- **Accuracy**: 85%+ for optimal configurations
- **Response Time**: 1-6 seconds per question
- **Success Rate**: 95%+ (minimal errors/timeouts)
- **Output**: Structured JSON reports with detailed metrics

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is for educational and research purposes.

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review the generated log files
3. Verify Ollama is running and models are available
4. Ensure all required files are present

---

**Last Updated**: October 2024  
**Version**: 1.0.0  
**Status**: Production Ready ✅

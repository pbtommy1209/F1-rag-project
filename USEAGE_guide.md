# RAG System Tools Usage Guide

## 🔧 Development Query Tool (`query_tool.py`)

### Interactive Mode (Default)
```bash
# Interactive querying - ask questions one by one
python query_tool.py --index-dir optimal_test_indexes/size300_overlap50/size300_overlap50

# With custom model
python query_tool.py --index-dir optimal_test_indexes/size300_overlap50/size300_overlap50 --model llama3.2:1b
```

### Batch Mode
```bash
# Run all questions and save results
python query_tool.py --mode batch --index-dir optimal_test_indexes/size300_overlap50/size300_overlap50 --output batch_results.json
```

## 🧪 Test Tool (`test_tool.py`)

### Comprehensive Test (Default)
```bash
# Run full test on all 40 questions
python test_tool.py --index-dir optimal_test_indexes/size300_overlap50/size300_overlap50 --output comprehensive_test.json

# With custom model
python test_tool.py --index-dir optimal_test_indexes/size300_overlap50/size300_overlap50 --model llama3.2:1b --output test_results.json
```

### Quick Test
```bash
# Test first 5 questions only
python test_tool.py --mode quick --index-dir optimal_test_indexes/size300_overlap50/size300_overlap50 --num-questions 5
```

## 📊 Output Files

Both tools generate JSON reports with:
- Performance metrics (accuracy, response times)
- Question-by-question results
- Error tracking
- Timestamps and metadata

## 🚀 Quick Start Examples

### 1. Interactive Development
```bash
python query_tool.py --index-dir optimal_test_indexes/size300_overlap50/size300_overlap50
# Then type questions like:
# "T/F: DRS allows a driver within one second of the car ahead to open a rear-wing flap"
```

### 2. Full Test Run
```bash
python test_tool.py --index-dir optimal_test_indexes/size300_overlap50/size300_overlap50 --output llama3.2_test_results.json
```

### 3. Quick Validation
```bash
python test_tool.py --mode quick --index-dir optimal_test_indexes/size300_overlap50/size300_overlap50 --num-questions 3
```

## 📁 Expected Directory Structure
```
rag_project/
├── optimal_test_indexes/
│   └── size300_overlap50/
│       └── size300_overlap50/
│           ├── index.faiss
│           └── meta.json
├── question_ask/
│   └── example_questions.json
├── correct_answer.json
├── query_tool.py
├── test_tool.py
└── fixed_ingest.py
```

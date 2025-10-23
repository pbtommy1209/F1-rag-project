#!/usr/bin/env python3
"""
Final accuracy test for llama3.2:1b with optimal configuration
Combines best chunk size (300) and overlap (50) from previous tests
"""

import os
import json
import time
import subprocess
from typing import List, Dict


def load_questions_and_answers():
    with open('question_ask/example_questions.json', 'r') as f:
        questions = json.load(f)
    with open('correct_answer.json', 'r') as f:
        correct_answers = json.load(f)
    return questions, correct_answers


def run_ingestion(pdf_path: str, outdir: str, chunk_size: int, overlap: int) -> bool:
    print(f"🔄 Running ingestion with chunk_size={chunk_size}, overlap={overlap}")
    cmd = [
        "python", "fixed_ingest.py",
        "--pdf", pdf_path,
        "--outdir", outdir,
        "--sizes", str(chunk_size),
        "--overlaps", str(overlap)
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print(f"✅ Ingestion successful for size={chunk_size}, overlap={overlap}")
            return True
        print(f"❌ Ingestion failed: {result.stderr}")
        return False
    except subprocess.TimeoutExpired:
        print(f"⏰ Ingestion timed out for size={chunk_size}, overlap={overlap}")
        return False
    except Exception as e:
        print(f"❌ Ingestion error: {e}")
        return False


def query_single_question(question: str, index_dir: str) -> str:
    cmd = [
        "python", "fixed_ingest.py",
        "--query", question,
        "--outdir", index_dir
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            return result.stdout.strip()
        print(f"❌ Query failed: {result.stderr}")
        return "ERROR"
    except subprocess.TimeoutExpired:
        print("⏰ Query timed out")
        return "TIMEOUT"
    except Exception as e:
        print(f"❌ Query error: {e}")
        return "ERROR"


def parse_model_response(response: str) -> str:
    if not response or response in ("ERROR", "TIMEOUT"):
        return "UNKNOWN"
    u = response.upper()
    if ("TRUE" in u) ^ ("FALSE" in u):
        return "TRUE" if "TRUE" in u else "FALSE"
    if "TRUE" in u and "FALSE" in u:
        return "TRUE" if u.find("TRUE") < u.find("FALSE") else "FALSE"
    import re
    for pat in [r'\b(TRUE|FALSE)\b', r'Answer:\s*(TRUE|FALSE)', r'(TRUE|FALSE)\b']:
        m = re.search(pat, u)
        if m:
            return m.group(1)
    for line in response.split("\n")[:5]:
        lu = line.upper()
        if ("TRUE" in lu) ^ ("FALSE" in lu):
            return "TRUE" if "TRUE" in lu else "FALSE"
    return "UNKNOWN"


def test_optimal_configuration():
    print("🏆 LLAMA3.2:1B OPTIMAL CONFIGURATION TEST")
    print("="*80)
    print("Testing with optimal settings:")
    print("  - Chunk Size: 300 (best from chunk size test)")
    print("  - Overlap: 50 (best from overlap test)")
    print("  - Expected accuracy: 85.00%")
    print("="*80)
    
    questions, correct_answers = load_questions_and_answers()
    print(f"Loaded {len(questions)} questions")
    
    # Optimal configuration
    chunk_size = 300
    overlap = 50
    config_name = f"size{chunk_size}_overlap{overlap}"
    
    # Run ingestion
    base_outdir = "optimal_test_indexes"
    if not run_ingestion("F1_33.pdf", base_outdir, chunk_size, overlap):
        print("❌ Failed to create optimal index")
        return
    
    # Find the actual index directory
    index_dir = None
    for root, dirs, files in os.walk(base_outdir):
        if "index.faiss" in files and "meta.json" in files and config_name in root:
            index_dir = root
            break
    
    if not index_dir:
        print("❌ Could not find optimal index files")
        return
    
    print(f"\n🧪 Testing optimal configuration: {config_name}")
    print("="*60)
    
    results = []
    correct_count = 0
    total_questions = len(questions)
    start_time = time.time()
    
    for i, (question, correct) in enumerate(zip(questions, correct_answers), 1):
        print(f"\n📝 Question {i}/{total_questions}")
        print(f"Question: {question['question']}")
        print(f"Expected: {'TRUE' if correct['answer'] else 'FALSE'}")
        
        response = query_single_question(question['question'], index_dir)
        parsed_answer = parse_model_response(response)
        
        expected = "TRUE" if correct['answer'] else "FALSE"
        is_correct = parsed_answer == expected
        
        if is_correct:
            correct_count += 1
            status = "✅ CORRECT"
        else:
            status = "❌ INCORRECT"
        
        print(f"Model Response: {parsed_answer}")
        print(f"Status: {status}")
        
        results.append({
            'question_id': i,
            'question': question['question'],
            'expected': expected,
            'model_response': parsed_answer,
            'correct': is_correct,
            'raw_response': response[:200] + "..." if len(response) > 200 else response
        })
        
        time.sleep(1)
    
    end_time = time.time()
    total_time = end_time - start_time
    average_time = total_time / total_questions
    
    accuracy = (correct_count / total_questions) * 100
    points_earned = correct_count * 2.5
    total_points = total_questions * 2.5
    
    print(f"\n📊 OPTIMAL CONFIGURATION RESULTS:")
    print(f"Configuration: {config_name}")
    print(f"Chunk Size: {chunk_size}")
    print(f"Overlap: {overlap}")
    print(f"Correct: {correct_count}/{total_questions}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Points: {points_earned:.1f}/{total_points}")
    print(f"Average Response Time: {average_time:.2f} seconds")
    print(f"Total Time: {total_time:.2f} seconds")
    
    # Save final report
    final_report = {
        "model": "llama3.2:1b",
        "test_type": "optimal_configuration_final",
        "configuration": {
            "chunk_size": chunk_size,
            "overlap": overlap,
            "config_name": config_name
        },
        "results": {
            "correct_count": correct_count,
            "total_questions": total_questions,
            "accuracy": accuracy,
            "points_earned": points_earned,
            "total_points": total_points,
            "average_time": average_time,
            "total_time": total_time
        },
        "detailed_results": results,
        "comparison": {
            "chunk_size_test_best": "size300_overlap100 (82.50%)",
            "overlap_test_best": "size300_overlap50 (85.00%)",
            "optimal_combination": f"{config_name} ({accuracy:.2f}%)"
        }
    }
    
    with open("llama3.2_optimal_final.json", "w") as f:
        json.dump(final_report, f, indent=2)
    
    print(f"\n💾 Final report saved to: llama3.2_optimal_final.json")
    print("="*80)
    
    if accuracy >= 85.0:
        print("🎉 SUCCESS: Optimal configuration achieved expected accuracy!")
    else:
        print("⚠️  WARNING: Accuracy below expected 85.00%")


if __name__ == "__main__":
    test_optimal_configuration()

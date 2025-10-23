#!/usr/bin/env python3
"""
Comprehensive evaluation for deepseek-r1:1.5b model
Tests different chunk sizes and overlaps to find optimal configuration
"""

import os
import json
import time
import subprocess
from typing import List, Dict
import argparse


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
        "--outdir", index_dir,
        "--chat-model", "deepseek-r1:1.5b"
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


def test_configuration(questions: List[Dict], correct_answers: List[Dict], 
                      chunk_size: int, overlap: int, config_name: str) -> Dict:
    print(f"\n🧪 Testing configuration: {config_name}")
    print("="*60)
    
    # Create index directory
    base_outdir = "deepseek_test_indexes"
    if not run_ingestion("F1_33.pdf", base_outdir, chunk_size, overlap):
        print(f"❌ Failed to create index for {config_name}")
        return None
    
    # Find the actual index directory (it's nested)
    index_dir = None
    for root, dirs, files in os.walk(base_outdir):
        if "index.faiss" in files and "meta.json" in files and config_name in root:
            index_dir = root
            break
    
    if not index_dir:
        print(f"❌ Could not find index files for {config_name}")
        return None
    
    # Test questions
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
    
    print(f"\n📊 Results for {config_name}:")
    print(f"Correct: {correct_count}/{total_questions}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Points: {points_earned:.1f}/{total_points}")
    print(f"Average Response Time: {average_time:.2f} seconds")
    print(f"Total Time: {total_time:.2f} seconds")
    
    return {
        'config_name': config_name,
        'chunk_size': chunk_size,
        'overlap': overlap,
        'correct_count': correct_count,
        'total_questions': total_questions,
        'accuracy': accuracy,
        'points_earned': points_earned,
        'total_points': total_points,
        'average_time': average_time,
        'total_time': total_time,
        'results': results
    }


def main():
    print("🧪 DEEPSEEK-R1:1.5B MODEL ACCURACY TEST")
    print("="*80)
    print("Testing True/False questions about Formula 1")
    print("Scoring: 2.5 points per correct answer, 0 points per incorrect answer")
    print("="*80)
    
    # Load questions and answers
    questions, correct_answers = load_questions_and_answers()
    print(f"✅ Loaded {len(questions)} questions")
    
    # Test configurations (chunk_size, overlap)
    configurations = [
        (300, 50, "size300_overlap50"),
        (300, 100, "size300_overlap100"),
        (500, 100, "size500_overlap100"),
        (800, 150, "size800_overlap150")
    ]
    
    all_results = []
    
    # Test each configuration
    for chunk_size, overlap, config_name in configurations:
        print(f"\n🔍 Testing deepseek-r1:1.5b model with {config_name}...")
        print("-" * 60)
        
        result = test_configuration(questions, correct_answers, chunk_size, overlap, config_name)
        if result:
            all_results.append(result)
    
    # Generate final report
    print("\n" + "="*80)
    print("📊 DEEPSEEK-R1:1.5B MODEL ACCURACY REPORT")
    print("="*80)
    
    best_config = None
    best_accuracy = 0
    
    for result in all_results:
        print(f"\n🔧 Configuration: {result['config_name']}")
        print(f"   Chunk Size: {result['chunk_size']}, Overlap: {result['overlap']}")
        print(f"   Accuracy: {result['accuracy']:.2f}%")
        print(f"   Points: {result['points_earned']:.1f}/{result['total_points']}")
        print(f"   Correct: {result['correct_count']}/{result['total_questions']}")
        print(f"   Average Time: {result['average_time']:.2f}s")
        
        if result['accuracy'] > best_accuracy:
            best_accuracy = result['accuracy']
            best_config = result
    
    if best_config:
        print(f"\n🏆 BEST CONFIGURATION: {best_config['config_name']}")
        print(f"   Chunk Size: {best_config['chunk_size']}, Overlap: {best_config['overlap']}")
        print(f"   Accuracy: {best_config['accuracy']:.2f}%")
        print(f"   Points: {best_config['points_earned']:.1f}/{best_config['total_points']}")
    else:
        print("\n❌ No successful configurations found")
    
    # Save detailed results
    report_data = {
        'model': 'deepseek-r1:1.5b',
        'best_configuration': best_config,
        'all_results': all_results,
        'summary': {
            'total_configurations': len(all_results),
            'best_accuracy': best_accuracy,
            'best_config_name': best_config['config_name'] if best_config else None,
            'best_chunk_size': best_config['chunk_size'] if best_config else None,
            'best_overlap': best_config['overlap'] if best_config else None
        }
    }
    
    with open("deepseek-r1_chunk.json", 'w') as f:
        json.dump(report_data, f, indent=2)
    
    print(f"\n💾 Detailed report saved to: deepseek-r1_chunk.json")
    print("="*80)


if __name__ == "__main__":
    main()

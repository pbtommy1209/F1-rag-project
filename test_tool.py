#!/usr/bin/env python3
"""
Test Tool for llama3.2:1b RAG System
Automated testing tool that runs through question sets and generates comprehensive reports
"""

import os
import json
import time
import subprocess
from typing import List, Dict
import argparse
from datetime import datetime


def load_questions_and_answers():
    """Load questions and correct answers"""
    with open('question_ask/example_questions.json', 'r') as f:
        questions = json.load(f)
    
    with open('correct_answer.json', 'r') as f:
        correct_answers = json.load(f)
    
    return questions, correct_answers


def query_single_question(question: str, index_dir: str, model: str = "llama3.2:1b") -> str:
    """Query a single question using the RAG system"""
    cmd = [
        "python", "fixed_ingest.py",
        "--query", question,
        "--outdir", index_dir,
        "--chat-model", model
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
    """Parse model response to extract TRUE/FALSE answer"""
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


def run_comprehensive_test(index_dir: str, model: str = "llama3.2:1b", output_file: str = None):
    """Run comprehensive test on all questions"""
    print("🧪 COMPREHENSIVE TEST MODE")
    print("="*60)
    print(f"🤖 Model: {model}")
    print(f"📁 Index: {index_dir}")
    print(f"⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    questions, correct_answers = load_questions_and_answers()
    print(f"📚 Loaded {len(questions)} questions")
    
    results = []
    correct_count = 0
    total_questions = len(questions)
    start_time = time.time()
    
    # Track performance metrics
    response_times = []
    error_count = 0
    timeout_count = 0
    
    for i, (question, correct) in enumerate(zip(questions, correct_answers), 1):
        print(f"\n📝 Question {i}/{total_questions}")
        print(f"Question: {question['question']}")
        print(f"Expected: {'TRUE' if correct['answer'] else 'FALSE'}")
        
        # Time the query
        query_start = time.time()
        response = query_single_question(question['question'], index_dir, model)
        query_end = time.time()
        query_time = query_end - query_start
        response_times.append(query_time)
        
        parsed_answer = parse_model_response(response)
        
        expected = "TRUE" if correct['answer'] else "FALSE"
        is_correct = parsed_answer == expected
        
        if is_correct:
            correct_count += 1
            status = "✅ CORRECT"
        else:
            status = "❌ INCORRECT"
        
        # Track errors
        if parsed_answer == "ERROR":
            error_count += 1
        elif parsed_answer == "TIMEOUT":
            timeout_count += 1
        
        print(f"Model Response: {parsed_answer}")
        print(f"Status: {status}")
        print(f"Response Time: {query_time:.2f}s")
        
        results.append({
            'question_id': i,
            'question': question['question'],
            'expected': expected,
            'model_response': parsed_answer,
            'correct': is_correct,
            'response_time': query_time,
            'raw_response': response[:300] + "..." if len(response) > 300 else response
        })
        
        time.sleep(0.5)  # Small delay between queries
    
    end_time = time.time()
    total_time = end_time - start_time
    average_time = sum(response_times) / len(response_times) if response_times else 0
    
    accuracy = (correct_count / total_questions) * 100
    points_earned = correct_count * 2.5
    total_points = total_questions * 2.5
    
    print(f"\n📊 COMPREHENSIVE TEST RESULTS:")
    print("="*60)
    print(f"✅ Correct: {correct_count}/{total_questions}")
    print(f"📈 Accuracy: {accuracy:.2f}%")
    print(f"🎯 Points: {points_earned:.1f}/{total_points}")
    print(f"⏱️  Average Response Time: {average_time:.2f} seconds")
    print(f"⏱️  Total Time: {total_time:.2f} seconds")
    print(f"❌ Errors: {error_count}")
    print(f"⏰ Timeouts: {timeout_count}")
    print(f"🏁 Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Generate comprehensive report
    report_data = {
        'test_info': {
            'model': model,
            'index_directory': index_dir,
            'test_type': 'comprehensive',
            'timestamp': datetime.now().isoformat(),
            'total_questions': total_questions
        },
        'performance_metrics': {
            'correct_count': correct_count,
            'accuracy': accuracy,
            'points_earned': points_earned,
            'total_points': total_points,
            'average_response_time': average_time,
            'total_time': total_time,
            'error_count': error_count,
            'timeout_count': timeout_count,
            'response_times': response_times
        },
        'detailed_results': results,
        'summary': {
            'best_response_time': min(response_times) if response_times else 0,
            'worst_response_time': max(response_times) if response_times else 0,
            'success_rate': ((total_questions - error_count - timeout_count) / total_questions) * 100
        }
    }
    
    # Save results
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        print(f"\n💾 Comprehensive report saved to: {output_file}")
    
    return report_data


def run_quick_test(index_dir: str, model: str = "llama3.2:1b", num_questions: int = 5):
    """Run quick test on first N questions"""
    print("⚡ QUICK TEST MODE")
    print("="*40)
    print(f"🤖 Model: {model}")
    print(f"📁 Index: {index_dir}")
    print(f"📝 Testing first {num_questions} questions")
    print("="*40)
    
    questions, correct_answers = load_questions_and_answers()
    test_questions = questions[:num_questions]
    test_answers = correct_answers[:num_questions]
    
    results = []
    correct_count = 0
    
    for i, (question, correct) in enumerate(zip(test_questions, test_answers), 1):
        print(f"\n📝 Question {i}/{num_questions}")
        print(f"Question: {question['question']}")
        print(f"Expected: {'TRUE' if correct['answer'] else 'FALSE'}")
        
        start_time = time.time()
        response = query_single_question(question['question'], index_dir, model)
        end_time = time.time()
        
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
        print(f"Response Time: {end_time - start_time:.2f}s")
        
        results.append({
            'question_id': i,
            'question': question['question'],
            'expected': expected,
            'model_response': parsed_answer,
            'correct': is_correct,
            'response_time': end_time - start_time
        })
    
    accuracy = (correct_count / num_questions) * 100
    print(f"\n📊 QUICK TEST RESULTS:")
    print(f"✅ Correct: {correct_count}/{num_questions}")
    print(f"📈 Accuracy: {accuracy:.2f}%")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Test Tool for llama3.2:1b RAG System")
    parser.add_argument("--index-dir", required=True, help="Path to the index directory")
    parser.add_argument("--model", default="llama3.2:1b", help="Model to use for testing")
    parser.add_argument("--mode", choices=['comprehensive', 'quick'], default='comprehensive',
                       help="Test mode: comprehensive or quick")
    parser.add_argument("--output", help="Output file for test results")
    parser.add_argument("--num-questions", type=int, default=5, help="Number of questions for quick test")
    args = parser.parse_args()
    
    if not os.path.exists(args.index_dir):
        print(f"❌ Index directory not found: {args.index_dir}")
        return 1
    
    if args.mode == 'comprehensive':
        run_comprehensive_test(args.index_dir, args.model, args.output)
    elif args.mode == 'quick':
        run_quick_test(args.index_dir, args.model, args.num_questions)
    
    return 0


if __name__ == "__main__":
    exit(main())

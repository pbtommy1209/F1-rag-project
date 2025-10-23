#!/usr/bin/env python3
"""
Development Query Tool for llama3.2:1b RAG System
Interactive tool for querying the RAG system with custom questions
"""

import os
import json
import time
import subprocess
from typing import List, Dict
import argparse


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


def interactive_query_mode(index_dir: str, model: str = "llama3.2:1b"):
    """Interactive query mode for development"""
    print("🔍 INTERACTIVE QUERY MODE")
    print("="*50)
    print("Enter your questions (type 'quit' to exit, 'help' for commands)")
    print("="*50)
    
    while True:
        try:
            question = input("\n❓ Your question: ").strip()
            
            if question.lower() == 'quit':
                print("👋 Goodbye!")
                break
            elif question.lower() == 'help':
                print("\n📋 Available commands:")
                print("  - Ask any True/False question about F1")
                print("  - Type 'quit' to exit")
                print("  - Type 'help' to see this message")
                continue
            elif not question:
                continue
            
            print(f"\n🔍 Querying: {question}")
            print("⏳ Processing...")
            
            start_time = time.time()
            response = query_single_question(question, index_dir, model)
            end_time = time.time()
            
            parsed_answer = parse_model_response(response)
            
            print(f"\n✅ Model Response: {parsed_answer}")
            print(f"⏱️  Response Time: {end_time - start_time:.2f} seconds")
            
            if parsed_answer == "UNKNOWN":
                print(f"📝 Raw Response: {response[:300]}...")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def batch_query_mode(index_dir: str, model: str = "llama3.2:1b", output_file: str = None):
    """Batch query mode for testing all questions"""
    print("🧪 BATCH QUERY MODE")
    print("="*50)
    
    questions, correct_answers = load_questions_and_answers()
    print(f"📚 Loaded {len(questions)} questions")
    
    results = []
    correct_count = 0
    total_questions = len(questions)
    start_time = time.time()
    
    for i, (question, correct) in enumerate(zip(questions, correct_answers), 1):
        print(f"\n📝 Question {i}/{total_questions}")
        print(f"Question: {question['question']}")
        print(f"Expected: {'TRUE' if correct['answer'] else 'FALSE'}")
        
        response = query_single_question(question['question'], index_dir, model)
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
        
        time.sleep(1)  # Small delay between queries
    
    end_time = time.time()
    total_time = end_time - start_time
    average_time = total_time / total_questions
    
    accuracy = (correct_count / total_questions) * 100
    points_earned = correct_count * 2.5
    total_points = total_questions * 2.5
    
    print(f"\n📊 BATCH RESULTS:")
    print(f"Correct: {correct_count}/{total_questions}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Points: {points_earned:.1f}/{total_points}")
    print(f"Average Response Time: {average_time:.2f} seconds")
    print(f"Total Time: {total_time:.2f} seconds")
    
    # Save results
    if output_file:
        report_data = {
            'model': model,
            'test_type': 'batch_query',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'results': {
                'correct_count': correct_count,
                'total_questions': total_questions,
                'accuracy': accuracy,
                'points_earned': points_earned,
                'total_points': total_points,
                'average_time': average_time,
                'total_time': total_time
            },
            'detailed_results': results
        }
        
        with open(output_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Query Tool for llama3.2:1b RAG System")
    parser.add_argument("--index-dir", required=True, help="Path to the index directory")
    parser.add_argument("--model", default="llama3.2:1b", help="Model to use for queries")
    parser.add_argument("--mode", choices=['interactive', 'batch'], default='interactive', 
                       help="Query mode: interactive or batch")
    parser.add_argument("--output", help="Output file for batch mode results")
    args = parser.parse_args()
    
    if not os.path.exists(args.index_dir):
        print(f"❌ Index directory not found: {args.index_dir}")
        return 1
    
    print(f"🤖 Using model: {args.model}")
    print(f"📁 Index directory: {args.index_dir}")
    
    if args.mode == 'interactive':
        interactive_query_mode(args.index_dir, args.model)
    elif args.mode == 'batch':
        batch_query_mode(args.index_dir, args.model, args.output)
    
    return 0


if __name__ == "__main__":
    exit(main())

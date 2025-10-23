#!/usr/bin/env python3
"""
Evaluate llama3.2:1b accuracy by varying overlap only (fixed chunk size = 300)
Saves a final report to llama3.2_overlap.json
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


def evaluate_config(questions: List[Dict], correct_answers: List[Dict], index_dir: str, config_name: str) -> Dict:
    print(f"\n🧪 Evaluating configuration: {config_name}")
    print("="*60)
    results = []
    correct_count = 0
    total_questions = len(questions)
    start = time.time()

    for i, (q, gold) in enumerate(zip(questions, correct_answers), start=1):
        print(f"\n📝 Question {i}/{total_questions}")
        print(f"Question: {q['question']}")
        print(f"Expected: {'TRUE' if gold['answer'] else 'FALSE'}")
        resp = query_single_question(q['question'], index_dir)
        pred = parse_model_response(resp)
        expected = "TRUE" if gold['answer'] else "FALSE"
        ok = pred == expected
        correct_count += 1 if ok else 0
        print(f"Model Response: {pred}")
        print(f"Status: {'✅ CORRECT' if ok else '❌ INCORRECT'}")
        results.append({
            "question_id": i,
            "question": q['question'],
            "expected": expected,
            "model_response": pred,
            "correct": ok,
            "raw_response": resp if len(resp) <= 300 else resp[:300] + "..."
        })
        time.sleep(1)

    elapsed = time.time() - start
    acc = (correct_count / total_questions) * 100.0
    points = correct_count * 2.5
    total_points = total_questions * 2.5

    print(f"\n📊 Results for {config_name}:")
    print(f"Correct: {correct_count}/{total_questions}")
    print(f"Accuracy: {acc:.2f}%")
    print(f"Points: {points:.1f}/{total_points}")

    return {
        "config_name": config_name,
        "correct_count": correct_count,
        "total_questions": total_questions,
        "accuracy": acc,
        "points_earned": points,
        "total_points": total_points,
        "results": results,
        "total_time": elapsed,
        "average_time": elapsed / total_questions if total_questions else 0.0
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate llama3.2:1b by varying overlap only")
    parser.add_argument("--pdf", default="F1_33.pdf")
    args = parser.parse_args()

    print("🧪 LLAMA3.2:1B OVERLAP EVALUATION")
    print("="*80)
    print("Fixed chunk size = 300; varying overlaps")
    print("Scoring: 2.5 points per correct answer")
    print("="*80)

    questions, correct_answers = load_questions_and_answers()
    print(f"Loaded {len(questions)} questions")

    chunk_size = 300
    overlaps = [50, 100, 150]

    all_results = []
    for overlap in overlaps:
        config_name = f"size{chunk_size}_overlap{overlap}"
        # Use a simpler base directory structure
        base_outdir = "llama3_overlap_indexes"
        if not run_ingestion(args.pdf, base_outdir, chunk_size, overlap):
            print(f"❌ Skipping {config_name} due to ingestion failure")
            continue
        
        # Find the actual index directory (it's nested)
        index_dir = None
        for root, dirs, files in os.walk(base_outdir):
            if "index.faiss" in files and "meta.json" in files and config_name in root:
                index_dir = root
                break
        
        if not index_dir:
            print(f"❌ Could not find index files for {config_name}")
            continue
            
        res = evaluate_config(questions, correct_answers, index_dir, config_name)
        all_results.append(res)

    print("\n" + "="*80)
    print("📊 LLAMA3.2:1B OVERLAP REPORT")
    print("="*80)

    best = None
    best_acc = -1.0
    for r in all_results:
        print(f"\n🔧 Configuration: {r['config_name']}")
        print(f"   Accuracy: {r['accuracy']:.2f}%")
        print(f"   Points: {r['points_earned']:.1f}/{r['total_points']}")
        if r['accuracy'] > best_acc:
            best = r
            best_acc = r['accuracy']

    if best:
        print(f"\n🏆 BEST CONFIGURATION: {best['config_name']}")
        print(f"   Accuracy: {best['accuracy']:.2f}%")
        print(f"   Points: {best['points_earned']:.1f}/{best['total_points']}")
    else:
        print("\n❌ No successful configurations")

    report = {
        "model": "llama3.2:1b",
        "fixed_chunk_size": chunk_size,
        "all_results": all_results,
        "best_configuration": best,
        "summary": {
            "num_configs": len(all_results),
            "best_config_name": best["config_name"] if best else None,
            "best_accuracy": best["accuracy"] if best else None
        }
    }

    with open("llama3.2_overlap.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\n💾 Saved report to llama3.2_overlap.json")


if __name__ == "__main__":
    main()

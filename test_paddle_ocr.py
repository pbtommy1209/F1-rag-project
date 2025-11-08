#!/usr/bin/env python3
"""
Test PaddleOCR functionality directly
Tests the OCR function with a PDF file
"""

import sys
import os
from fixed_ingest import _ocr_pdf_paddle, read_pdf_fallback, read_pdf

def test_ocr_direct(pdf_path: str):
    """Test OCR function directly"""
    print("="*60)
    print("🧪 DIRECT OCR TEST")
    print("="*60)
    print(f"Testing PDF: {pdf_path}")
    
    if not os.path.exists(pdf_path):
        print(f"❌ PDF file not found: {pdf_path}")
        return False
    
    print("\n1️⃣ Testing _ocr_pdf_paddle() directly...")
    ocr_text = _ocr_pdf_paddle(pdf_path)
    
    if ocr_text:
        print(f"✅ OCR extracted {len(ocr_text)} characters")
        print(f"\nPreview (first 300 chars):\n{ocr_text[:300]}...")
        return True
    else:
        print("❌ OCR returned no text")
        return False

def test_ocr_fallback(pdf_path: str):
    """Test OCR through the fallback chain"""
    print("\n" + "="*60)
    print("🧪 OCR FALLBACK CHAIN TEST")
    print("="*60)
    print(f"Testing PDF: {pdf_path}")
    
    if not os.path.exists(pdf_path):
        print(f"❌ PDF file not found: {pdf_path}")
        return False
    
    print("\n2️⃣ Testing read_pdf_fallback() (should trigger OCR if little text)...")
    text = read_pdf_fallback(pdf_path)
    
    if text and len(text.strip()) > 0:
        print(f"✅ Fallback extracted {len(text)} characters")
        print(f"\nPreview (first 300 chars):\n{text[:300]}...")
        return True
    else:
        print("❌ Fallback returned no text")
        return False

def test_full_chain(pdf_path: str):
    """Test the full read_pdf chain"""
    print("\n" + "="*60)
    print("🧪 FULL PDF READING CHAIN TEST")
    print("="*60)
    print(f"Testing PDF: {pdf_path}")
    
    if not os.path.exists(pdf_path):
        print(f"❌ PDF file not found: {pdf_path}")
        return False
    
    print("\n3️⃣ Testing read_pdf() (full chain: PyPDF → PyMuPDF → pdfplumber → OCR)...")
    text = read_pdf(pdf_path)
    
    if text and len(text.strip()) > 0:
        print(f"✅ Full chain extracted {len(text)} characters")
        print(f"\nPreview (first 300 chars):\n{text[:300]}...")
        return True
    else:
        print("❌ Full chain returned no text")
        return False

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test PaddleOCR functionality")
    parser.add_argument("--pdf", help="PDF file to test (scanned/image PDF recommended)")
    parser.add_argument("--test-all", action="store_true", help="Test all three methods")
    args = parser.parse_args()
    
    pdf_path = args.pdf
    
    # If no PDF specified, try to find F1_33.pdf
    if not pdf_path:
        if os.path.exists("F1_33.pdf"):
            pdf_path = "F1_33.pdf"
            print("ℹ️ No PDF specified, using F1_33.pdf (this is a digital PDF, OCR won't trigger)")
        else:
            print("❌ No PDF file specified and F1_33.pdf not found")
            print("Usage: python test_paddle_ocr.py --pdf <path_to_scanned_pdf>")
            return 1
    
    results = []
    
    if args.test_all:
        results.append(("Direct OCR", test_ocr_direct(pdf_path)))
        results.append(("OCR Fallback", test_ocr_fallback(pdf_path)))
        results.append(("Full Chain", test_full_chain(pdf_path)))
    else:
        # Default: test full chain
        results.append(("Full Chain", test_full_chain(pdf_path)))
    
    print("\n" + "="*60)
    print("📊 TEST RESULTS SUMMARY")
    print("="*60)
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    all_passed = all(result[1] for result in results)
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())

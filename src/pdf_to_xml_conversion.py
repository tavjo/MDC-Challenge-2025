#!/usr/bin/env python3
"""
PDF to XML Conversion Implementation
Following the MDC-Challenge-2025 checklist
"""

import pandas as pd
import os
import sys
from pathlib import Path
from requests import post
import pdfplumber
from tqdm import tqdm
import time
from datetime import datetime

def main():
    print("=== PDF to XML Conversion for MDC-Challenge-2025 ===")
    print(f"Started at: {datetime.now()}")
    
    # Step 1: Load & Validate Conversion Candidates
    print("\n📋 Step 1: Load & Validate Conversion Candidates")
    cand = load_conversion_candidates()
    
    # Step 2: Inventory Existing XML Files
    print("\n📋 Step 2: Inventory Existing XML Files")
    todo = inventory_existing_xml(cand)
    
    # Step 3: Implement Robust Conversion Function
    print("\n📋 Step 3: Conversion Functions Ready")
    
    # Step 4: Execute Batch Conversion
    print("\n📋 Step 4: Execute Batch Conversion")
    log = execute_batch_conversion(todo)
    
    # Step 5: Generate Conversion Report
    print("\n📋 Step 5: Generate Conversion Report")
    generate_conversion_report(log)
    
    # Step 6: Quality Validation
    print("\n📋 Step 6: Quality Validation")
    validate_conversions()
    
    print(f"\n✅ Conversion process completed at: {datetime.now()}")

def load_conversion_candidates():
    """Step 1: Load & Validate Conversion Candidates"""
    cand = pd.read_csv("Data/conversion_candidates.csv")
    
    # Expected: 124 rows based on analysis
    assert len(cand) == 124, f"Expected 124 candidates, got {len(cand)}"
    assert {'article_id', 'pdf_path'}.issubset(cand.columns), "Missing required columns"
    
    print(f"✅ Loaded {len(cand)} conversion candidates")
    print(f"   Columns: {list(cand.columns)}")
    
    return cand

def inventory_existing_xml(cand):
    """Step 2: Inventory Existing XML Files"""
    xml_dir = Path("Data/train/XML")
    have_xml = {p.stem for p in xml_dir.glob("*.xml")}
    
    cand['already_has_xml'] = cand['article_id'].map(lambda aid: aid in have_xml)
    todo = cand[~cand['already_has_xml']]
    
    print(f"✅ Found {len(have_xml)} existing XML files")
    print(f"✅ {len(todo)} PDFs still need conversion")
    
    return todo

def convert_one(pdf_path: Path, out_xml: Path):
    """Step 3: Convert single PDF to XML using Grobid with pdfplumber fallback"""
    GROBID_URL = os.getenv("GROBID_URL", "http://localhost:8070")
    
    try:
        # Try Grobid first
        with pdf_path.open('rb') as f:
            r = post(f"{GROBID_URL}/api/processFulltextDocument",
                     files={'input': f}, timeout=120)
        if r.status_code == 200 and r.text.strip():
            out_xml.write_text(r.text, encoding='utf-8')
            return "grobid"
    except Exception as e:
        print(f"Grobid failed for {pdf_path.name}: {e}")
    
#     # Fallback - pdfplumber text extraction
#     try:
#         text = []
#         with pdfplumber.open(str(pdf_path)) as pdf:
#             for page in pdf.pages:
#                 t = page.extract_text() or ''
#                 text.append(t)
        
#         # Create simple XML structure for fallback
#         xml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
# <document>
# <filename>{pdf_path.name}</filename>
# <text>
# {''.join(text)}
# </text>
# </document>"""
        
#         out_xml.write_text(xml_content, encoding='utf-8')
#         return "pdfplumber"
#     except Exception as e:
#         print(f"Both conversions failed for {pdf_path.name}: {e}")
#         return "failed"

def execute_batch_conversion(todo):
    """Step 4: Execute Batch Conversion"""
    xml_dir = Path("Data/train/XML")
    xml_dir.mkdir(exist_ok=True)
    log = []
    
    print(f"Converting {len(todo)} PDFs...")
    
    for _, row in tqdm(todo.iterrows(), total=len(todo), desc="Converting PDFs"):
        # Handle both absolute and relative paths from CSV
        pdf_path_str = row['pdf_path']
        if pdf_path_str.startswith('/'):
            # Convert absolute path to relative
            pdf_path = Path(pdf_path_str.split('Data/')[-1])
            pdf_path = Path("Data") / pdf_path
        else:
            pdf_path = Path(pdf_path_str)
        
        # Use article_id for XML filename
        xml_path = xml_dir / f"{row['article_id']}.xml"
        
        try:
            source = convert_one(pdf_path, xml_path)
            log.append({
                'article_id': row['article_id'],
                'pdf_path': row['pdf_path'],
                'xml_path': str(xml_path), 
                'source': source, 
                'error': None,
                'success': source != "failed",
                'has_primary': row['has_primary'],
                'has_secondary': row['has_secondary'],
                'label_count': row['label_count']
            })
        except Exception as e:
            log.append({
                'article_id': row['article_id'],
                'pdf_path': row['pdf_path'],
                'xml_path': None, 
                'source': None, 
                'error': str(e),
                'success': False,
                'has_primary': row['has_primary'],
                'has_secondary': row['has_secondary'],
                'label_count': row['label_count']
            })
        
        # Small delay to avoid overwhelming services
        time.sleep(0.1)
    
    return log

def generate_conversion_report(log):
    """Step 5: Generate Conversion Report"""
    conversion_log = pd.DataFrame(log)
    conversion_log.to_csv("Data/document_inventory_step4.csv", index=False)
    
    # Generate summary statistics
    successful = conversion_log[conversion_log['success'] == True]
    failed = conversion_log[conversion_log['success'] == False]
    grobid_success = conversion_log[conversion_log['source'] == 'grobid']
    fallback_success = conversion_log[conversion_log['source'] == 'pdfplumber']
    
    print(f"✅ {len(successful)} successful conversions")
    print(f"   📊 Grobid: {len(grobid_success)}")  
    print(f"   📊 Fallback: {len(fallback_success)}")
    print(f"⚠️ {len(failed)} failed conversions")
    
    # Coverage KPI
    calculate_coverage_kpi()

def calculate_coverage_kpi():
    """Calculate conversion coverage KPI"""
    original_candidates = pd.read_csv("Data/conversion_candidates.csv")
    total_candidates = len(original_candidates)
    
    # Count existing XML files that match our candidates
    existing_xml = {p.stem for p in Path("Data/train/XML").glob("*.xml")}
    covered_candidates = sum(1 for article_id in original_candidates['article_id'] 
                           if article_id in existing_xml)
    
    # Coverage KPI: covered_candidates / total_candidates >= 0.90
    coverage = covered_candidates / total_candidates
    
    print(f"📊 Coverage KPI: {coverage:.2%}")
    print(f"   Total candidates: {total_candidates}")
    print(f"   Covered candidates: {covered_candidates}")
    
    if coverage < 0.90:
        print("🚨 ALERT: Coverage below 90% threshold!")
    else:
        print("✅ Coverage meets 90% requirement")
    
    return coverage

def validate_conversions():
    """Step 6: Validation & Quality Checks"""
    xml_dir = Path("Data/train/XML")
    issues = []
    
    print("🔍 Validating XML files...")
    
    for xml_file in xml_dir.glob("*.xml"):
        if xml_file.stat().st_size < 1024:  # Less than 1KB (very small)
            issues.append(f"{xml_file.name}: File too small ({xml_file.stat().st_size} bytes)")
    
    if issues:
        print(f"⚠️ Found {len(issues)} small files that may need review:")
        for issue in issues[:10]:  # Show first 10
            print(f"   {issue}")
    else:
        print("✅ All XML files appear to have reasonable sizes")

def get_remaining_conversions():
    """Get list of PDFs that still need conversion (resumability)"""
    existing_xml = {p.stem for p in Path("Data/train/XML").glob("*.xml")}
    all_candidates = pd.read_csv("Data/conversion_candidates.csv")
    remaining = all_candidates[~all_candidates['article_id'].isin(existing_xml)]
    return remaining

if __name__ == "__main__":
    main() 